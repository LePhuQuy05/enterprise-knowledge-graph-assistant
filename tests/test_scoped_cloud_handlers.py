"""AWS handler boundaries exercised offline with real local state/queue."""

import json
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import boto3
import pytest
from botocore.stub import Stubber

from app.contracts import canonical_json, sha256_bytes
from app.query_auth import ApiError
from app.scoped_cloud_coordinator import ScopedSqsSender
from app.scoped_cloud_state import WORK_PARTITION, ScopedCloudDynamoRunStore
from app.scoped_run_cloud import (
    ROOT,
    ScopedCloudConfigV2,
    ScopedCloudDispatch,
    ScopedRunApi,
    capabilities,
    load_cloud_config,
    make_scoped_quote_controller,
)
from tests.test_authorization_evidence import auth_case as auth_case  # noqa: F401
from tests.test_query_api import PUBLIC, event
from tests.test_scoped_cloud import cloud as cloud  # noqa: F401
from tests.test_scoped_owned_pipeline import QUESTION, deliver, submit
from tests.test_scoped_owned_pipeline import scoped_stack as scoped_stack  # noqa: F401


def live_config(s):
    configs = {key: c.model_copy(update={"fixture_only": False}) for key, c in s.executor.release.configs.items()}
    release = s.executor.release.model_copy(update={"budget_run_id": "collection319-scoped-answer-v2", "configs": configs,
        "method_config_sha256": {key: c.fingerprint for key, c in configs.items()}})
    path = "configs/prices/external-direct-2026-11-05.json"
    return {"schema_version": 2, "queue_url": "https://sqs.us-east-1.amazonaws.com/123456789012/kg-r72-scoped-work",
        "results_bucket": "kg-r72-scoped-fixture", "assets": {"bucket": "kg-r72-assets", "version": "wb_"+"a"*64,
            "manifest_sha256": "b"*64, "manifest_bytes": 123, "tenant_id": "demo"},
        "graph_assets": {"bucket": "kg-r72-assets", "version": "sg_"+"c"*64, "manifest_sha256": "d"*64, "manifest_bytes": 456},
        "prices": {"path": path, "sha256": sha256_bytes((ROOT/path).read_bytes())},
        "release": release.model_dump(mode="json"), "public_config": PUBLIC}


def test_v2_api_submit_graph_q_qb_history_sources_and_v1_is_not_mapped(cloud):
    s = cloud
    api = ScopedRunApi(s.service, s.metadata, PUBLIC, capabilities=lambda auth: capabilities(live_config(s)))
    response = api.handler(event("POST", "/v2/scoped-runs", body={"original_question": QUESTION, "language": "en",
        "run_kind": "answer", "method_ids": ["graph_1hop", "hybrid_multi_query", "hybrid_multi_query_balanced"]},
        key=str(uuid4())))
    assert response["statusCode"] == 202, response
    run = json.loads(response["body"])["run_id"]
    deliver(s, run)
    read = api.handler(event("GET", f"/v2/scoped-runs/{run}/methods/graph_1hop"))
    assert read["statusCode"] == 200
    result = json.loads(read["body"])
    chunk = result["context_chunk_ids"][0]
    assert api.handler(event("GET", f"/v2/scoped-runs/{run}/methods/graph_1hop/sources/{chunk}"))["statusCode"] == 200
    assert len(json.loads(api.handler(event("GET", "/v2/scoped-runs"))["body"])["items"]) == 1
    cap = json.loads(api.handler(event("GET", "/v2/scoped-run-capabilities"))["body"])
    assert cap["semantic_verified"] is False and cap["graph"]["human_reviewed"] is False and len(cap["methods"]) == 8
    for path in ("/v1/runs", "/v2/scoped-runs-other", "/v2/scoped-runs/../v1/runs"):
        assert api.handler(event("GET", path))["statusCode"] == 404


def test_worker_terminal_redelivery_does_not_load_assets_or_secret(cloud, monkeypatch):
    import app.scoped_run_cloud as module

    s = cloud
    row = submit(s)
    deliver(s, row["run_id"])
    message = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    monkeypatch.setattr(module, "cloud_components", lambda: ({}, s.service, s.metadata, s.coordinator))
    monkeypatch.setattr(module, "cloud_assets", lambda *_: pytest.fail("terminal delivery loaded assets"))
    monkeypatch.setattr(module, "make_scoped_executor", lambda *_: pytest.fail("terminal delivery loaded secret"))
    result = module.worker_handler({"Records": [{"eventSource": "aws:sqs", "messageId": "fixture",
        "body": canonical_json(message).decode()}]}, SimpleNamespace(get_remaining_time_in_millis=lambda: 180000))
    assert result == {"batchItemFailures": []} and s.gateway.provider.calls == 1


def test_handler_lazy_composition_completes_actual_graph_mechanism(cloud, monkeypatch):
    import app.scoped_run_cloud as module

    s = cloud
    row = submit(s, ("graph_2hop",))
    message = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    opened, closed = [], []
    class Asset:
        def __enter__(self):
            opened.append(True)
            return self
        def __exit__(self, *args):
            closed.append(True)
    monkeypatch.setattr(module, "cloud_components", lambda: ({"release": s.executor.release.model_dump(mode="json")},
        s.service, s.metadata, s.coordinator))
    monkeypatch.setattr(module, "cloud_assets", lambda *_: Asset())
    monkeypatch.setattr(module, "make_scoped_executor", lambda *_: s.executor)
    result = module.worker_handler({"Records": [{"eventSource": "aws:sqs", "messageId": "fixture",
        "body": canonical_json(message).decode()}]}, SimpleNamespace(get_remaining_time_in_millis=lambda: 180000))
    assert result == {"batchItemFailures": []} and opened == closed == [True]
    assert s.service.public_result(s.auth, row["run_id"], "graph_2hop")["used_graph"]


def test_lazy_release_mismatch_fails_before_provider_and_closes_assets(cloud):
    closed = []
    class Asset:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            closed.append(True)
    dispatch = ScopedCloudDispatch(cloud.service, {"release": {}}, None, assets_factory=lambda _: Asset(),
        executor_factory=lambda *_: cloud.executor)
    row = submit(cloud)
    try:
        with pytest.raises(ValueError):
            dispatch.prepare(cloud.auth, cloud.repo.get(cloud.auth, row["run_id"]))
        assert dispatch.executor is None
    finally:
        dispatch.close()
    assert closed == [True] and cloud.gateway.provider.calls == 0


def test_sdk_wire_graph_queue_body_and_bounded_strong_work_query(cloud):
    sqs = boto3.client("sqs", region_name="us-east-1", aws_access_key_id="fixture", aws_secret_access_key="fixture")
    row = submit(cloud, ("graph_1hop",))
    message = cloud.repo.outboxes(cloud.auth, row["run_id"])[0]["message"]
    url = "https://sqs.us-east-1.amazonaws.com/123456789012/kg-r72-scoped-work"
    with Stubber(sqs) as stub:
        stub.add_response("send_message", {"MessageId": "fixture"}, {"QueueUrl": url, "MessageBody": canonical_json(message).decode()})
        ScopedSqsSender(sqs, url)(message)
        stub.assert_no_pending_responses()
    db = boto3.client("dynamodb", region_name="us-east-1", aws_access_key_id="fixture", aws_secret_access_key="fixture")
    store = ScopedCloudDynamoRunStore(db, "kg-r72-query-state")
    args = {"TableName": "kg-r72-query-state", "ConsistentRead": True, "Limit": 7,
        "KeyConditionExpression": "PK = :pk AND begins_with(SK, :prefix)",
        "ExpressionAttributeValues": {":pk": {"S": WORK_PARTITION}, ":prefix": {"S": "SLOT#"}}}
    with Stubber(db) as stub:
        stub.add_response("query", {"Items": []}, args)
        assert store.work_slots() == []
        stub.assert_no_pending_responses()


def test_packaged_config_is_pinned_and_does_not_put_release_in_environment(cloud, tmp_path, monkeypatch):
    import app.scoped_run_cloud as module

    value = ScopedCloudConfigV2.model_validate(live_config(cloud))
    raw = canonical_json(value.model_dump(mode="json"))
    digest = sha256_bytes(raw)
    relative = f"configs/releases/scoped-cloud-{digest}.json"
    target = tmp_path/relative
    target.parent.mkdir(parents=True)
    target.write_bytes(raw)
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setenv("KG_SCOPED_CONFIG_PATH", relative)
    monkeypatch.setenv("KG_SCOPED_CONFIG_SHA256", digest)
    assert load_cloud_config() == value.model_dump(mode="json")
    target.write_bytes(raw+b" ")
    with pytest.raises(ValueError, match="CONFIG_PIN"):
        load_cloud_config()
    monkeypatch.setenv("KG_SCOPED_CONFIG_PATH", "../../credentials")
    with pytest.raises(ValueError, match="CONFIG_PIN"):
        load_cloud_config()


def test_expired_prices_allow_read_only_composition_but_block_admission_quote(cloud):
    from datetime import UTC, datetime

    controller = make_scoped_quote_controller(cloud.repo.store, live_config(cloud)["prices"],
        clock=lambda: datetime(2026, 12, 1, tzinfo=UTC))
    assert controller.gateway is None and controller.live is True
    with pytest.raises(ValueError, match="stale"):
        controller._spec({"run_id": "run_"+"a"*64, "owner_id": cloud.auth.user_id, "tenant_id": "demo"},
            "GENERATE:hybrid", "0"*64)


@pytest.mark.parametrize("bad", [{}, {"Records": []}, {"Records": [{}, {}]}])
def test_malformed_worker_invocation_does_not_construct_cloud_clients(monkeypatch, bad):
    import app.scoped_run_cloud as module

    monkeypatch.setattr(module, "cloud_components", lambda: pytest.fail("malformed invocation touched AWS"))
    with pytest.raises(ValueError):
        module.worker_handler(bad, None)


def test_low_time_worker_and_invalid_watchdog_do_not_construct_clients(monkeypatch):
    import app.scoped_run_cloud as module

    monkeypatch.setattr(module, "cloud_components", lambda: pytest.fail("invalid invocation touched AWS"))
    with pytest.raises(ValueError, match="TIME_BUDGET"):
        module.worker_handler({"Records": [{}]}, SimpleNamespace(get_remaining_time_in_millis=lambda: 64000))
    with pytest.raises(ValueError, match="EVENT_INVALID"):
        module.watchdog_handler({"operation": "RUN_WATCHDOG"}, None)


def test_expired_parent_cancels_without_asset_load_and_watchdog_reports_unknown(cloud, monkeypatch):
    import app.scoped_run_cloud as module

    s = cloud
    row = submit(s)
    s.clock[0] += timedelta(seconds=601)
    monkeypatch.setattr(module, "cloud_components", lambda: ({}, s.service, s.metadata, s.coordinator))
    report = module.watchdog_handler({"operation": "SCOPED_WATCHDOG_V2"}, None)
    assert report["model_calls"] == 0 and s.gateway.provider.calls == 0
    assert s.repo.child(s.auth, row["run_id"], "hybrid")["status"] == "EXPIRED"
    monkeypatch.setattr(s.coordinator, "sweep", lambda: {"unknown": 1, "errors": 0})
    with pytest.raises(RuntimeError, match="NEEDS_ATTENTION"):
        module.watchdog_handler({"operation": "SCOPED_WATCHDOG_V2"}, None)


def test_disabled_runtime_rejects_before_aws_client_construction(monkeypatch):
    import app.scoped_run_cloud as module

    monkeypatch.delenv("KG_SCOPED_RUNS_ENABLED", raising=False)
    monkeypatch.setattr(boto3, "client", lambda *a, **k: pytest.fail("disabled handler touched AWS"))
    with pytest.raises(ApiError, match="DISABLED"):
        module.cloud_components()


def test_account_aware_credentials_do_not_change_the_approved_regional_ledger_endpoint(cloud, monkeypatch):
    import app.scoped_run_cloud as module

    calls = []
    def client(name, **kwargs):
        calls.append((name, kwargs))
        return SimpleNamespace(meta=SimpleNamespace(endpoint_url=kwargs.get("endpoint_url", "account-specific-unapproved")))
    monkeypatch.setenv("KG_SCOPED_RUNS_ENABLED", "true")
    monkeypatch.setenv("MODEL_ENABLED", "false")
    monkeypatch.setattr(module, "load_cloud_config", lambda: live_config(cloud))
    monkeypatch.setattr(boto3, "client", client)
    _, service, _, _ = module.cloud_components()
    assert service.repo.store.client.meta.endpoint_url == "https://dynamodb.us-east-1.amazonaws.com"
    assert [kwargs["endpoint_url"] for name, kwargs in calls if name == "dynamodb"] == ["https://dynamodb.us-east-1.amazonaws.com"]
    assert all(name != "secretsmanager" for name, _ in calls)
