"""Offline native API→outbox→queue→worker→private sources proof; not managed AWS."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, "/tmp/query-package")
sys.path.insert(0, "/")

import boto3  # noqa: E402

from app.contracts import canonical_json, sha256_bytes  # noqa: E402
from app.ledger import Change, MemoryLedgerStore  # noqa: E402
from app.owned_runs import OwnedRunRepository  # noqa: E402
from app.query_auth import require_current_membership  # noqa: E402
from app.run_artifacts import LocalRunBlobs, RunArtifactRepository  # noqa: E402
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

OWNER = "00000000-0000-4000-8000-000000000003"
OTHER = "00000000-0000-4000-8000-000000000004"
KEY = str(uuid4())

class IndexedState(WorkIndexMixin, SQLiteRunStore):
    pass


class FileObjects:
    def get_object(self, **kwargs):
        assert kwargs["Bucket"] == "isolated-file-backed-s3-fixture"
        prefix = "workbench/" + bundle["version"] + "/"
        assert kwargs["Key"].startswith(prefix)
        path = Path("/public-assets") / kwargs["Key"][len(prefix):]
        assert path.resolve().is_relative_to(Path("/public-assets"))
        return {"Body": path.open("rb"), "ContentLength": path.stat().st_size}


def wire(method, path, *, body=None, owner=OWNER):
    return {"requestContext": {"http": {"method": method}, "authorizer": {"jwt": {"claims": {
        "sub": owner, "iss": "offline-issuer", "client_id": "offline-client", "token_use": "access", "scope": "kg/read kg/query", "exp": "9999999999"}}}},
        "rawPath": path, "headers": {"content-type": "application/json", "idempotency-key": KEY},
        "body": json.dumps(body) if body is not None else None}


def call(method, path, *, body=None, owner=OWNER):
    result = api_handler(wire(method, path, body=body, owner=owner), None)
    assert result["statusCode"] < 300, (path, result["statusCode"])
    return json.loads(result["body"])


raw = Path("/public-assets/bundle.json").read_bytes()
bundle = json.loads(raw)
pins = {"bucket": "isolated-file-backed-s3-fixture", "version": bundle["version"],
    "manifest_sha256": sha256_bytes(raw), "manifest_bytes": len(raw), "tenant_id": "demo"}
objects = FileObjects()
boto3.client = lambda name, **kwargs: objects if name == "s3" else (_ for _ in ()).throw(ValueError("NATIVE_AWS_CALL_FORBIDDEN"))
with load_s3_workbench_assets(objects, **pins, cache_root=Path("/tmp/kg-workbench-assets")) as assets:
    release = make_retrieval_executor(assets).release
metadata = MemoryLedgerStore()
for owner in (OWNER, OTHER):
    row = {"PK": "USER#" + owner, "SK": "MEMBERSHIP", "version": 1, "record_type": "MEMBERSHIP", "schema_version": 1,
        "status": "ACTIVE", "roles": ["reviewer"], "tenant_id": "demo"}
    metadata.commit([Change((row["PK"], row["SK"]), None, row)])
repo = OwnedRunRepository(IndexedState("/tmp/owned-cloud-proof/state.sqlite3"))
queue = SQLiteRunQueue("/tmp/owned-cloud-proof/queue.sqlite3")
artifacts = RunArtifactRepository(LocalRunBlobs("/tmp/owned-cloud-proof/results"))
def authorize(auth):
    require_current_membership(auth, metadata)
def check(pin):
    assert pin == release.model_dump(mode="json")
coordinator = CloudRunCoordinator(repo, queue.send, authorize)
service = CloudRunService(repo, artifacts, lambda: release, authorize=authorize, release_check=check, coordinator=coordinator)
config = {"public_config": {"issuer": "offline-issuer", "client_id": "offline-client"}, "assets": pins,
    "release": release.model_dump(mode="json"), "corpus": {"documents": 316, "chunks": 2978, "service_groups": 24}}
import app.run_cloud as cloud  # noqa: E402

cloud.cloud_components = lambda: (config, service, metadata, coordinator)
request = {"original_question": "Which policies must permit cross-account use of an AWS KMS key?", "language": "en", "method_ids": ["bm25", "vector", "hybrid"], "run_kind": "retrieval"}
parent = call("POST", "/v1/runs", body=request)
assert parent["status"] == "QUEUED"
assert len(queue.receive()) == 3
context = SimpleNamespace(get_remaining_time_in_millis=lambda: 90_000)
for identifier, message in queue.receive():
    event = {"Records": [{"eventSource": "aws:sqs", "messageId": identifier, "body": canonical_json(message).decode()}]}
    assert worker_handler(event, context) == {"batchItemFailures": []}
    assert worker_handler(event, context) == {"batchItemFailures": []}  # Terminal replay, no second execution.
    queue.ack(identifier)
run_id = parent["run_id"]
detail = call("GET", "/v1/runs/" + run_id)
assert detail["status"] == "COMPLETED"
replay = call("POST", "/v1/runs", body=request)
assert replay["idempotent_replay"] and not queue.receive()
source_count = 0
reference = json.loads(Path("/build/reference.json").read_bytes())["methods"]
for method in request["method_ids"]:
    result = call("GET", f"/v1/runs/{run_id}/methods/{method}")
    for key in ("config_sha256", "ranked_chunk_ids", "context_chunk_ids"):
        assert result[key] == reference["kms:" + method][key]
    for chunk in result["context_chunk_ids"]:
        source = call("GET", f"/v1/runs/{run_id}/methods/{method}/sources/{chunk}")
        assert sha256_bytes(source["text"].encode()) == source["text_sha256"]
        source_count += 1
    other = api_handler(wire("GET", f"/v1/runs/{run_id}/methods/{method}", owner=OTHER), None)
    assert other["statusCode"] == 404
assert all(slot["status"] == "FREE" for slot in repo.store.work_slots())
assert len(call("GET", f"/v1/runs/{run_id}/trace")["methods"]) == 3
report = {"status": "PASS_NATIVE_OWNED_WORKFLOW", "actual_onnx": True, "methods": 3, "source_appearances": source_count,
    "http_api_signature_verification": False, "api_authorization_checks": True, "cross_owner_rejected": True,
    "duplicate_messages_acked": True, "idempotent_submit_no_new_queue_messages": True, "slots_released": True,
    "rank_config_context_parity": True, "state": "SQLite transactional work index", "queue": "SQLite at-least-once fixture",
    "assets": "file-backed S3 client", "model_calls": 0, "aws_calls": 0, "shared_production_state_changed": False}
Path("/build/owned-report.json").write_bytes(canonical_json(report) + b"\n")
print(json.dumps(report), flush=True)
