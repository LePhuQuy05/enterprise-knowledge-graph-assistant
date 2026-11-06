import io
from types import SimpleNamespace

import boto3
import pytest
from botocore.response import StreamingBody
from botocore.stub import Stubber

from app.contracts import canonical_json, sha256_bytes
from app.owned_runs import RunMessageV1
from app.run_artifacts import (
    MAX_RESULT_BYTES,
    LocalRunBlobs,
    RunArtifactRepository,
    S3RunBlobs,
    result_key,
)
from app.run_execution import validate_result
from app.run_storage import DynamoRunStore, SQLiteRunQueue, SQLiteRunStore
from tests.test_collection_review import review_context as review_context
from tests.test_owned_runs import USER_A, USER_B
from tests.test_rag_runner import setup_runner as setup_runner
from tests.test_run_execution import deliver, submit
from tests.test_run_execution import run_stack as run_stack


@pytest.mark.parametrize("bad", ["traversal", "backslash", "absolute", "root"])
def test_local_artifacts_reject_paths_outside_their_root(tmp_path, bad):
    store = LocalRunBlobs(tmp_path / "private")
    key = {"traversal": "../secret", "backslash": "..\\secret", "absolute": str(tmp_path / "secret"), "root": "."}[bad]
    with pytest.raises(ValueError, match="PATH_INVALID"):
        store.get(key)


def test_content_addressed_results_are_immutable_size_bounded_and_owner_scoped(tmp_path):
    blobs = LocalRunBlobs(tmp_path / "private")
    artifacts = RunArtifactRepository(blobs)
    run_id = "run_" + "a" * 64
    ref = artifacts.save(USER_A, run_id, "hybrid", {"fixture": True})
    assert artifacts.read(USER_A, run_id, "hybrid", ref) == canonical_json({"fixture": True})
    assert artifacts.save(USER_A, run_id, "hybrid", {"fixture": True}) == ref
    with pytest.raises(FileNotFoundError):
        artifacts.read(USER_B, run_id, "hybrid", ref)
    with pytest.raises(ValueError, match="REFERENCE_INVALID"):
        artifacts.read(USER_A, run_id, "hybrid", ref | {"key": "forged"})
    with pytest.raises(ValueError, match="SIZE_LIMIT"):
        artifacts.save(USER_A, run_id, "hybrid", {"fixture": "x" * MAX_RESULT_BYTES})
    key = result_key(USER_A, run_id, "hybrid", ref["sha256"])
    with pytest.raises(ValueError, match="IMMUTABLE_ARTIFACT_CONFLICT"):
        blobs.put(key, b"different")


def test_s3_artifact_adapter_uses_conditional_put_and_readback_on_existing_object():
    client = boto3.client("s3", region_name="us-east-1", aws_access_key_id="fixture", aws_secret_access_key="fixture")
    raw, key, bucket = b'{"fixture":true}', "tenants/demo/fixture.json", "fixture-private-bucket"
    expected = {"Bucket": bucket, "Key": key, "Body": raw, "ContentType": "application/json",
        "IfNoneMatch": "*", "Metadata": {"sha256": sha256_bytes(raw)}}
    with Stubber(client) as stub:
        stub.add_response("put_object", {}, expected)
        stub.add_client_error("put_object", service_error_code="PreconditionFailed", http_status_code=412,
            expected_params=expected)
        stub.add_response("get_object", {"ContentLength": len(raw), "Body": StreamingBody(io.BytesIO(raw), len(raw))},
            {"Bucket": bucket, "Key": key})
        blobs = S3RunBlobs(client, bucket)
        blobs.put(key, raw)
        blobs.put(key, raw)
        stub.assert_no_pending_responses()


def test_s3_read_is_bounded_and_closes_body_on_oversize():
    class Body(io.BytesIO):
        pass
    body = Body(b"x")
    client = SimpleNamespace(get_object=lambda **kw: {"Body": body, "ContentLength": MAX_RESULT_BYTES + 1})
    with pytest.raises(ValueError, match="SIZE_LIMIT"):
        S3RunBlobs(client, "private").get("fixture")
    assert body.closed


def test_dynamodb_history_is_consistent_owner_partition_query_without_scan():
    client = boto3.client("dynamodb", region_name="us-east-1", aws_access_key_id="fixture", aws_secret_access_key="fixture")
    expected = {"TableName": "fixture-state", "ConsistentRead": True, "Limit": 2, "ScanIndexForward": False,
        "KeyConditionExpression": "PK = :pk AND begins_with(SK, :prefix)",
        "ExpressionAttributeValues": {":pk": {"S": USER_A.partition}, ":prefix": {"S": "RUNHIST#"}},
        "ProjectionExpression": "PK, SK", "ExclusiveStartKey": {"PK": {"S": USER_A.partition}, "SK": {"S": "RUNHIST#after"}}}
    with Stubber(client) as stub:
        stub.add_response("query", {"Items": [{"PK": {"S": USER_A.partition}, "SK": {"S": "RUNHIST#one"}}]}, expected)
        rows, cursor = DynamoRunStore(client, "fixture-state").run_history_keys(USER_A.partition, limit=2, after="RUNHIST#after")
        assert rows == [{"PK": USER_A.partition, "SK": "RUNHIST#one"}] and cursor is None
        stub.assert_no_pending_responses()


def test_local_queue_persists_duplicate_deliveries_until_each_is_acknowledged(tmp_path):
    path = tmp_path / "queue.sqlite3"
    first = SQLiteRunQueue(path)
    first.send({"fixture": 1})
    first.send({"fixture": 1})
    rows = SQLiteRunQueue(path).receive()
    assert len(rows) == 2 and rows[0][0] != rows[1][0]
    first.ack(rows[0][0])
    assert SQLiteRunQueue(path).receive() == [rows[1]]


@pytest.mark.parametrize("bad", ["config", "cost", "source_hash", "extra_source_field", "source_order", "source_offset"])
def test_method_results_reject_mismatched_pin_and_unvalidated_provenance(run_stack, bad):
    s = run_stack
    row = submit(s)
    deliver(s, row["run_id"])
    result = s.service.result(s.auth, row["run_id"], "hybrid").model_dump(mode="json")
    if bad == "config":
        result["config_sha256"] = "f" * 64
    elif bad == "cost":
        result["new_cost_micro_usd"] = 1
    elif bad == "source_hash":
        result["context_sources"][0]["text_sha256"] = "f" * 64
    elif bad == "extra_source_field":
        result["context_sources"][0]["private_config"] = "synthetic_secret"
    elif bad == "source_order":
        result["context_sources"][0]["source_id"] = "s99"
    else:
        result["context_sources"][0]["end_char"] += 1
    with pytest.raises(ValueError):
        validate_result(s.repo.get(s.auth, row["run_id"]), "hybrid", result)


def test_watchdog_expires_queued_and_never_restarts_inflight_jobs(run_stack):
    from datetime import timedelta

    s = run_stack
    row = submit(s, ("bm25", "hybrid"))
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    message = RunMessageV1.model_validate(msg)
    _, token = s.repo.claim(message)
    s.repo.start(s.auth, row["run_id"], "bm25", token)
    s.clock[0] += timedelta(seconds=601)
    result = s.repo.recover(s.auth, row["run_id"])
    assert result["status"] == "UNKNOWN"
    assert [m["status"] for m in result["methods"]] == ["UNKNOWN", "EXPIRED"]
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 1
    assert s.calls == []


def test_sqs_duplicate_json_and_oversize_messages_are_not_executed(run_stack):
    s = run_stack
    bad = ["{\"run_id\":\"a\",\"run_id\":\"b\"}", " " * 4097]
    result = s.worker.sqs_batch({"Records": [{"eventSource": "aws:sqs", "messageId": str(i), "body": body}
        for i, body in enumerate(bad)]})
    assert len(result["batchItemFailures"]) == 2 and s.calls == []


def test_read_only_auditor_cannot_create_or_mutate_state_and_artifacts(tmp_path):
    from app.ledger import Change

    path, root = tmp_path / "state.sqlite3", tmp_path / "private"
    with pytest.raises(ValueError, match="MISSING"):
        SQLiteRunStore(path, read_only=True)
    with pytest.raises(ValueError, match="MISSING"):
        LocalRunBlobs(root, read_only=True)
    store = SQLiteRunStore(path)
    store.commit([Change(("fixture", "row"), None, {"PK": "fixture", "SK": "row", "version": 1})])
    readonly = SQLiteRunStore(path, read_only=True)
    assert readonly.get(("fixture", "row"))["version"] == 1
    with pytest.raises(ValueError, match="READ_ONLY"):
        readonly.commit([])
    with readonly.connect() as db, pytest.raises(Exception):
        db.execute("CREATE TABLE forbidden (value TEXT)")
    blobs = LocalRunBlobs(root)
    blobs.put("fixture.json", b"fixture")
    readonly_blobs = LocalRunBlobs(root, read_only=True)
    assert readonly_blobs.get("fixture.json") == b"fixture"
    with pytest.raises(ValueError, match="READ_ONLY"):
        readonly_blobs.put("fixture.json", b"changed")
