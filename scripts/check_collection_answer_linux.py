"""Network-disabled native answer pipeline proof with simulated provider/state, actual ONNX."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, "/tmp/query-package")
sys.path.insert(0, "/")

import boto3  # noqa: E402

from app.cloud_owned_answers import CollectionAnswerPolicy  # noqa: E402
from app.contracts import canonical_json, sha256_bytes  # noqa: E402
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet  # noqa: E402
from app.ledger import Change, Ledger, MemoryLedgerStore  # noqa: E402
from app.owned_runs import OwnedRunRepository  # noqa: E402
from app.query_auth import require_current_membership  # noqa: E402
from app.run_answer_executor import CollectionOwnedAnswerExecutor  # noqa: E402
from app.run_artifacts import LocalRunBlobs, RunArtifactRepository  # noqa: E402
from app.run_budget import RunBudgetController  # noqa: E402
from app.run_cloud import (  # noqa: E402
    CloudRunCoordinator,
    CloudRunService,
    api_handler,
    make_retrieval_executor,
    worker_handler,
)
from app.run_storage import SQLiteRunQueue, SQLiteRunStore  # noqa: E402
from app.run_work_index import WorkIndexMixin  # noqa: E402
from app.workbench_assets import load_s3_workbench_assets  # noqa: E402

OWNER, OTHER = "00000000-0000-4000-8000-000000000005", "00000000-0000-4000-8000-000000000006"
QUESTION = "Which policies must permit cross-account use of an AWS KMS key?"


class IndexedState(WorkIndexMixin, SQLiteRunStore):
    pass


class FileObjects:
    def get_object(self, **kwargs):
        assert kwargs["Bucket"] == "native-no-aws"
        prefix = "workbench/" + bundle["version"] + "/"
        assert kwargs["Key"].startswith(prefix)
        path = Path("/public-assets") / kwargs["Key"][len(prefix):]
        assert path.resolve().is_relative_to(Path("/public-assets"))
        return {"Body": path.open("rb"), "ContentLength": path.stat().st_size}


class SimulatedProvider:
    live = False
    calls = 0

    def request(self, provider, method, path, body):
        assert (provider, method, path) == ("DEEPSEEK", "POST", "/chat/completions")
        self.calls += 1
        payload = json.loads(body)
        prompt = json.loads(payload["messages"][1]["content"])
        if "untrusted_sources" in prompt:
            source = prompt["untrusted_sources"][0]
            text = "Ví dụ mô phỏng, hãy đọc nguồn." if "Vietnamese" in payload["messages"][0]["content"] else "Simulated example; inspect the source."
            content = {"answer_status": "ANSWERED", "reason": "NONE", "claims": [{"text": text, "evidence": [
                {"source_id": source["source_id"], "span_id": source["spans"][0]["span_id"]}]}]}
        else:
            content = {"search_query": QUESTION}
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 30, "total_tokens": 130},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(content)}}]}


raw = Path("/public-assets/bundle.json").read_bytes()
bundle = json.loads(raw)
pins = {"bucket": "native-no-aws", "version": bundle["version"], "manifest_sha256": sha256_bytes(raw),
        "manifest_bytes": len(raw), "tenant_id": "demo"}
objects = FileObjects()
boto3.client = lambda name, **kwargs: objects if name == "s3" else (_ for _ in ()).throw(ValueError("NO_NATIVE_AWS_OR_SECRET_READ"))
with load_s3_workbench_assets(objects, **pins, cache_root=Path("/tmp/kg-workbench-assets")) as assets:
    base = make_retrieval_executor(assets)
release, routes = base.release, base.runtime["routes"]
prices = ExternalPriceSheet.model_validate_json(Path("/tmp/query-package/configs/prices/external-direct-2026-10-24.json").read_bytes()).model_copy(update={"purpose": "FIXTURE"})
store = IndexedState("/tmp/collection-answer-proof/state.sqlite3")
ledger, policy = Ledger(store), CollectionAnswerPolicy()
ledger.initialize_month("kg-r72", ledger._now().strftime("%Y-%m"))
ledger.initialize_run("kg-r72", "demo", policy.budget_run_id, cap_micro_usd=policy.run_cap_micro_usd)
ledger.initialize_external_funding("kg-r72", cap_micro_usd=policy.funding_cap_micro_usd)
provider = SimulatedProvider()
gateway = ExternalModelGateway(ledger, provider, routes, prices)
controller = RunBudgetController(ledger=ledger, routes=routes, prices=prices, quote_live=False, budget_run_id=policy.budget_run_id)
repo = OwnedRunRepository(store, answer_budget=controller)
metadata = MemoryLedgerStore()
for owner in (OWNER, OTHER):
    row = {"PK": "USER#" + owner, "SK": "MEMBERSHIP", "version": 1, "record_type": "MEMBERSHIP", "schema_version": 1,
           "status": "ACTIVE", "roles": ["reviewer"], "tenant_id": "demo"}
    metadata.commit([Change((row["PK"], row["SK"]), None, row)])
queue = SQLiteRunQueue("/tmp/collection-answer-proof/queue.sqlite3")
artifacts = RunArtifactRepository(LocalRunBlobs("/tmp/collection-answer-proof/results"))


def authorize(auth):
    require_current_membership(auth, metadata)


def check(pin):
    assert pin == release.model_dump(mode="json")


coordinator = CloudRunCoordinator(repo, queue.send, authorize)
service = CloudRunService(repo, artifacts, lambda: release, authorize=authorize, release_check=check,
                          coordinator=coordinator, answer_policy=policy)
config = {"public_config": {"issuer": "offline", "client_id": "offline"}, "assets": pins,
          "release": release.model_dump(mode="json"), "corpus": {"documents": 316, "chunks": 2978, "service_groups": 24}}
import app.cloud_owned_answers as composition  # noqa: E402
import app.run_cloud as cloud  # noqa: E402

cloud.cloud_components = lambda: (config, service, metadata, coordinator)


def simulated_factory(assets, service, config, context):
    controller.bind_gateway(gateway)
    runtime = make_retrieval_executor(assets).runtime | {"routes": routes, "run_id": policy.budget_run_id}
    return CollectionOwnedAnswerExecutor(assets.collection, assets.lexical, controller=controller,
        repository=repo, authorize=authorize, **runtime)


composition.make_cloud_answer_executor = simulated_factory


def call(method, path, *, body=None, key=None, owner=OWNER):
    event = {"requestContext": {"http": {"method": method}, "authorizer": {"jwt": {"claims": {
        "sub": owner, "iss": "offline", "client_id": "offline", "token_use": "access", "scope": "kg/read kg/query", "exp": "9999999999"}}}},
        "rawPath": path, "headers": {"content-type": "application/json", "idempotency-key": key or str(uuid4())},
        "body": json.dumps(body) if body is not None else None}
    return api_handler(event, None)


sources = 0
for language, method in (("en", "hybrid"), ("vi", "vector")):
    key = str(uuid4())
    payload = {"original_question": QUESTION if language == "en" else "Những chính sách nào cho phép dùng khóa KMS khác tài khoản?",
               "language": language, "method_ids": [method], "run_kind": "answer"}
    response = call("POST", "/v1/runs", body=payload, key=key)
    assert response["statusCode"] == 202, response
    parent = json.loads(response["body"])
    assert parent["budget"]["reserved_micro_usd"] > 0
    before = provider.calls
    for identifier, message in queue.receive():
        event = {"Records": [{"eventSource": "aws:sqs", "messageId": identifier, "body": canonical_json(message).decode()}]}
        context = SimpleNamespace(get_remaining_time_in_millis=lambda: 90_000)
        assert worker_handler(event, context) == {"batchItemFailures": []}
        settled = provider.calls
        assert worker_handler(event, context) == {"batchItemFailures": []}
        assert provider.calls == settled
        queue.ack(identifier)
    assert provider.calls - before == (1 if language == "en" else 2)
    run_id = parent["run_id"]
    final = json.loads(call("GET", "/v1/runs/" + run_id)["body"])
    assert final["status"] == "COMPLETED" and final["budget"]["reserved_micro_usd"] == 0
    result = json.loads(call("GET", f"/v1/runs/{run_id}/methods/{method}")["body"])
    assert result["response"]["answer_language"] == language and result["response"]["citations"]
    assert result["run_id"] == run_id
    for chunk in result["context_chunk_ids"]:
        source = json.loads(call("GET", f"/v1/runs/{run_id}/methods/{method}/sources/{chunk}")["body"])
        assert sha256_bytes(source["text"].encode()) == source["text_sha256"]
        sources += 1
    assert call("GET", f"/v1/runs/{run_id}/methods/{method}", owner=OTHER)["statusCode"] == 404
    assert json.loads(call("POST", "/v1/runs", body=payload, key=key)["body"])["idempotent_replay"]
assert provider.calls == 3 and all(s["status"] == "FREE" for s in store.work_slots())
assert ledger.store.get(ledger.run_key(policy.budget_run_id))["reserved_micro_usd"] == 0
report = {"status": "PASS_NATIVE_COLLECTION_ANSWER_FIXTURE", "actual_onnx": True, "source_appearances": sources,
          "languages": ["en", "vi"], "methods": ["hybrid", "vector"], "simulated_provider_dispatches": provider.calls,
          "raw_receipts_and_reservations": True, "terminal_duplicate_no_new_dispatch": True, "slots_released": True,
          "cross_owner_rejected": True, "aws_calls": 0, "paid_model_calls": 0, "semantic_accuracy": None,
          "state_queue_provider_auth": "SIMULATED_LOCAL_NOT_MANAGED_AWS_OR_JWT_SIGNATURE_PROOF"}
Path("/build/collection-answer-report.json").write_bytes(canonical_json(report) + b"\n")
print(json.dumps(report), flush=True)
