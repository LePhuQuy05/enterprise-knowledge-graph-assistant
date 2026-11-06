import copy
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ledger import Change, MemoryLedgerStore
from app.owned_runs import OwnedRunRepository, RunMessageV1
from app.query_auth import ApiError, AuthContext, require_current_membership
from app.query_state import QueryRepository
from app.run_artifacts import LocalRunBlobs, RunArtifactRepository
from app.run_collection_adapter import CollectionRetrievalExecutor
from app.run_execution import OwnedRunWorker, RunOutboxDispatcher, SqsRunSender
from app.run_service import OwnedRunApi, OwnedRunService
from app.run_storage import SQLiteRunQueue, SQLiteRunStore
from tests.test_collection_review import review_context as review_context
from tests.test_query_api import PUBLIC, USER_A, USER_B, event
from tests.test_rag_runner import setup_runner as setup_runner


@pytest.fixture
def run_stack(setup_runner, tmp_path):
    collection, lexical, kwargs = setup_runner
    kwargs = {k: v for k, v in kwargs.items() if k not in {"tenant_id", "owner_id"}}
    executor = CollectionRetrievalExecutor(collection, lexical, **kwargs)
    metadata = MemoryLedgerStore()
    for user in (USER_A, USER_B):
        item = {"PK": "USER#" + user, "SK": "MEMBERSHIP", "version": 1, "record_type": "MEMBERSHIP",
            "schema_version": 1, "tenant_id": collection.manifest.tenant_id, "status": "ACTIVE", "roles": ["reviewer"]}
        metadata.commit([Change((item["PK"], item["SK"]), None, item)])
    clock = [datetime(2026, 10, 25, tzinfo=UTC)]
    repo = OwnedRunRepository(SQLiteRunStore(tmp_path / "state.sqlite3"), clock=lambda: clock[0])
    queue = SQLiteRunQueue(tmp_path / "queue.sqlite3")
    artifacts = RunArtifactRepository(LocalRunBlobs(tmp_path / "private"))
    flags, calls = {"revoked": False}, []
    def check(pin):
        if flags["revoked"]:
            raise ApiError(503, "COLLECTION_UNAVAILABLE")
        executor.check_release(pin)
    def execute(auth, parent, method, guard):
        calls.append(method)
        return executor(auth, parent, method, guard)
    def authorize(auth):
        return require_current_membership(auth, metadata)
    service = OwnedRunService(repo, artifacts, lambda: executor.release, authorize=authorize, release_check=check)
    worker = OwnedRunWorker(repo, artifacts, execute, authorize=authorize, release_check=check)
    return SimpleNamespace(repo=repo, queue=queue, artifacts=artifacts, executor=executor, worker=worker,
        dispatcher=RunOutboxDispatcher(repo, queue.send), service=service, api=OwnedRunApi(service, metadata, PUBLIC),
        auth=AuthContext(collection.manifest.tenant_id, USER_A), other=AuthContext(collection.manifest.tenant_id, USER_B),
        clock=clock, calls=calls, flags=flags, metadata=metadata)


def submit(s, methods=("hybrid",), key=None, **extra):
    return s.service.submit(s.auth, {"original_question": "Service B", "method_ids": methods} | extra, key or str(uuid4()))


def deliver(s, run):
    s.dispatcher.publish(s.auth, run)
    for identifier, msg in s.queue.receive():
        if s.worker.process(msg) == "ACK":
            s.queue.ack(identifier)


def test_real_runner_three_method_results_sources_trace_and_owner_bound_replay(run_stack):
    s = run_stack
    key = str(uuid4())
    row = submit(s, ("bm25", "vector", "hybrid"), key)
    assert s.calls == [] and s.queue.receive() == []  # No notification before committed outbox.
    deliver(s, row["run_id"])
    assert s.service.detail(s.auth, row["run_id"])["status"] == "COMPLETED"
    assert s.calls == ["bm25", "vector", "hybrid"]
    for method in ("bm25", "vector", "hybrid"):
        result = s.service.public_result(s.auth, row["run_id"], method)
        assert result["quality_metrics"] is None and result["new_cost_micro_usd"] == 0
        assert result["context_chunk_ids"]
        assert all("text" not in source for source in result["context_sources"])
        source = s.service.source(s.auth, row["run_id"], method, result["context_chunk_ids"][0])
        assert source["text"]
        with pytest.raises(ApiError, match="RUN_NOT_FOUND"):
            s.service.source(s.other, row["run_id"], method, result["context_chunk_ids"][0])
        with pytest.raises(ApiError, match="SOURCE_NOT_FOUND"):
            s.service.source(s.auth, row["run_id"], method, "chk_" + "f" * 64)
    assert len(s.service.trace(s.auth, row["run_id"])["methods"]) == 3
    replay = submit(s, ("bm25", "vector", "hybrid"), key)
    assert replay["idempotent_replay"] and replay["run_id"] == row["run_id"]
    assert s.dispatcher.publish(s.auth, row["run_id"]) == 0
    assert len(s.calls) == 3 and QueryRepository(s.repo.store, clock=s.repo.clock).quota_remaining() == 297
    assert s.service.history(s.other)["items"] == []
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 0


def test_send_succeeded_ack_failed_duplicate_outbox_never_computes_twice(run_stack):
    s = run_stack
    row = submit(s)
    def ambiguous_send(msg):
        s.queue.send(msg)
        raise TimeoutError("synthetic send success followed by lost ack")
    with pytest.raises(TimeoutError):
        RunOutboxDispatcher(s.repo, ambiguous_send).publish(s.auth, row["run_id"])
    assert s.repo.outboxes(s.auth, row["run_id"])[0]["status"] == "PENDING"
    deliver(s, row["run_id"])
    assert s.calls == ["hybrid"] and s.service.detail(s.auth, row["run_id"])["status"] == "COMPLETED"


def test_concurrent_messages_have_one_executor_and_terminal_duplicates_are_acks(run_stack):
    s = run_stack
    row = submit(s)
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    entered, release = threading.Event(), threading.Event()
    original = s.worker.executor
    def blocked(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    s.worker.executor = blocked
    with ThreadPoolExecutor(max_workers=6) as pool:
        first = pool.submit(s.worker.process, msg)
        assert entered.wait(5)
        peers = list(pool.map(lambda _: s.worker.process(msg), range(5)))
        assert peers == ["RETRY"] * 5
        release.set()
        assert first.result() == "ACK"
    assert s.worker.process(msg) == "ACK" and s.calls == ["hybrid"]


def test_hard_crash_after_start_stays_unknown_without_adapter_retry(run_stack):
    s = run_stack
    row = submit(s)
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    class ProcessDeath(BaseException):
        pass
    def die(*args):
        s.calls.append("crashed")
        raise ProcessDeath()
    s.worker.executor = die
    with pytest.raises(ProcessDeath):
        s.worker.process(msg)
    s.clock[0] += timedelta(seconds=121)
    assert s.worker.process(msg) == "ACK"
    assert s.service.detail(s.auth, row["run_id"])["status"] == "UNKNOWN" and s.calls == ["crashed"]
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 1


def test_crash_before_start_can_safely_resume_after_lease(run_stack, monkeypatch):
    s = run_stack
    row = submit(s)
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    class ProcessDeath(BaseException):
        pass
    original = s.repo.start
    monkeypatch.setattr(s.repo, "start", lambda *a: (_ for _ in ()).throw(ProcessDeath()))
    with pytest.raises(ProcessDeath):
        s.worker.process(msg)
    assert not s.calls
    monkeypatch.setattr(s.repo, "start", original)
    s.clock[0] += timedelta(seconds=121)
    assert s.worker.process(msg) == "ACK" and s.calls == ["hybrid"]


@pytest.mark.parametrize("stage", ["adapter", "artifact", "completed_write"])
def test_failure_after_execution_gate_is_durable_unknown_with_no_retry(run_stack, monkeypatch, stage):
    s = run_stack
    row = submit(s)
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    if stage == "adapter":
        monkeypatch.setattr(s.worker, "executor", lambda *a: (_ for _ in ()).throw(TimeoutError("secret")))
    elif stage == "artifact":
        monkeypatch.setattr(s.artifacts, "save", lambda *a: (_ for _ in ()).throw(TimeoutError("secret")))
    else:
        finish = s.repo.finish
        def fail_complete(*args, **kwargs):
            if args[3] == "COMPLETED":
                raise RuntimeError("secret")
            return finish(*args, **kwargs)
        monkeypatch.setattr(s.repo, "finish", fail_complete)
    assert s.worker.process(msg) == "ACK"
    assert s.service.detail(s.auth, row["run_id"])["status"] == "UNKNOWN"
    count = len(s.calls)
    assert s.worker.process(msg) == "ACK" and len(s.calls) == count
    assert "secret" not in json.dumps(s.service.detail(s.auth, row["run_id"]))


def test_cancel_revocation_and_forged_nonce_never_run_compute(run_stack):
    s = run_stack
    row = submit(s)
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    with pytest.raises(ApiError):
        s.worker.process(msg | {"admission_token": "0" * 32})
    assert not s.calls
    s.service.cancel(s.auth, row["run_id"])
    assert s.worker.process(msg) == "ACK" and not s.calls
    row = submit(s)
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]["status"] = "DISABLED"
    assert s.worker.process(msg) == "ACK" and not s.calls
    assert s.repo.summary(s.auth, row["run_id"])["status"] == "CANCELLED"


def test_revoked_release_and_worker_guard_after_compute_fail_closed(run_stack):
    s = run_stack
    row = submit(s)
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    original = s.worker.executor
    def revoke(*args):
        result = original(*args)
        s.flags["revoked"] = True
        return result
    s.worker.executor = revoke
    assert s.worker.process(msg) == "ACK"
    assert s.repo.summary(s.auth, row["run_id"])["status"] == "UNKNOWN"
    with pytest.raises(ApiError):
        s.service.detail(s.auth, row["run_id"])


def test_artifact_checksum_and_tampered_pin_do_not_leak_result(run_stack):
    s = run_stack
    row = submit(s)
    deliver(s, row["run_id"])
    result = s.service.result(s.auth, row["run_id"], "hybrid")
    assert result.context_chunk_ids
    parent = s.repo.get(s.auth, row["run_id"])
    child = s.repo.child(s.auth, row["run_id"], "hybrid")
    from app.run_artifacts import result_key
    path = s.artifacts.blobs.path(result_key(s.auth, row["run_id"], "hybrid", child["result_ref"]["sha256"]))
    raw = path.read_bytes()
    path.write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="CHECKSUM"):
        s.service.result(s.auth, row["run_id"], "hybrid")
    path.write_bytes(raw)
    altered = copy.deepcopy(parent["release"])
    altered["method_config_sha256"]["hybrid"] = "0" * 64
    with pytest.raises(ValueError, match="RELEASE_CHANGED"):
        s.executor.check_release(altered)


def test_sqs_partial_batch_and_sender_use_bounded_internal_envelopes(run_stack):
    s = run_stack
    row = submit(s)
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    calls = []
    sender = SqsRunSender(SimpleNamespace(send_message=lambda **kwargs: calls.append(kwargs)), "fixture-queue")
    sender(msg)
    assert RunMessageV1.model_validate_json(calls[0]["MessageBody"]).run_id == row["run_id"]
    assert "question" not in calls[0]["MessageBody"] and "token_use" not in calls[0]["MessageBody"]
    output = s.worker.sqs_batch({"Records": [
        {"eventSource": "aws:sqs", "messageId": "good", "body": json.dumps(msg)},
        {"eventSource": "aws:sqs", "messageId": "bad", "body": "not JSON"}]})
    assert output == {"batchItemFailures": [{"itemIdentifier": "bad"}]}


def test_history_cursor_is_owner_bound_expires_and_preserves_pagination(run_stack):
    s = run_stack
    for _ in range(3):
        row = submit(s)
        deliver(s, row["run_id"])
        s.clock[0] += timedelta(seconds=1)
    first = s.service.history(s.auth, limit=1)
    assert len(first["items"]) == 1 and first["next_cursor"]
    second = s.service.history(s.auth, limit=1, cursor=first["next_cursor"])
    assert first["items"][0]["run_id"] != second["items"][0]["run_id"]
    with pytest.raises(ApiError, match="INVALID_CURSOR"):
        s.service.history(s.other, cursor=first["next_cursor"])
    s.clock[0] += timedelta(hours=2)
    with pytest.raises(ApiError, match="INVALID_CURSOR"):
        s.service.history(s.auth, cursor=first["next_cursor"])


def test_api_creation_returns_202_read_projection_and_never_sends_from_admission(run_stack):
    s = run_stack
    request = event("POST", "/v1/runs", body={"original_question": "Service B", "method_ids": ["hybrid"]})
    result = s.api.handler(request)
    assert result["statusCode"] == 202 and result["headers"]["cache-control"] == "no-store"
    data = json.loads(result["body"])
    assert data["status"] == "QUEUED" and result["headers"]["retry-after"] == "2"
    assert s.queue.receive() == [] and s.calls == []
    deliver(s, data["run_id"])
    base = "/v1/runs/" + data["run_id"]
    for path in (base, base + "/trace", base + "/methods/hybrid"):
        assert s.api.handler(event("GET", path))["statusCode"] == 200
        assert s.api.handler(event("GET", path, user=USER_B))["statusCode"] == 404
    assert s.api.handler(event("GET", "/v1/runs"))["statusCode"] == 200


@pytest.mark.parametrize("bad", ["missing_auth", "wrong_scope", "params", "extra_identity", "answer", "graph", "no_key", "duplicate_json", "oversize"])
def test_api_rejects_bad_requests_without_admission_or_compute(run_stack, bad):
    s = run_stack
    req = event("POST", "/v1/runs", body={"original_question": "Service B"})
    if bad == "missing_auth":
        req["requestContext"].pop("authorizer")
    elif bad == "wrong_scope":
        req["requestContext"]["authorizer"]["jwt"]["claims"]["scope"] = "kg/read"
    elif bad == "params":
        req["queryStringParameters"] = {"enable_answer": "true"}
    elif bad in {"extra_identity", "answer", "graph"}:
        req["body"] = json.dumps({"original_question": "Service B"} | {
            "extra_identity": {"owner_id": USER_B}, "answer": {"run_kind": "answer"},
            "graph": {"method_ids": ["graph_2hop"]}}[bad])
    elif bad == "no_key":
        req["headers"].pop("idempotency-key")
    elif bad == "duplicate_json":
        req["body"] = '{"original_question":"a","original_question":"b"}'
    else:
        req["body"] = " " * 17000
    assert s.api.handler(req)["statusCode"] in {400, 401, 403}
    assert not s.calls and not s.queue.receive()
    assert QueryRepository(s.repo.store, clock=s.repo.clock).quota_remaining() == 300


def test_api_final_reauthorization_and_safe_error_projection(run_stack, monkeypatch):
    s = run_stack
    row = submit(s)
    original = s.service.detail
    def revoke(*args):
        result = original(*args)
        s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]["status"] = "DISABLED"
        return result
    monkeypatch.setattr(s.service, "detail", revoke)
    assert s.api.handler(event("GET", "/v1/runs/" + row["run_id"]))["statusCode"] == 403


@pytest.mark.parametrize("question,language,diagnostic", [("Service B", "en", False), ("Dịch vụ B là gì?", "vi", True)])
def test_retrieval_never_claims_rewrite_or_hides_direct_vietnamese_limitation(run_stack, question, language, diagnostic):
    s = run_stack
    row = s.service.submit(s.auth, {"original_question": question, "language": language}, str(uuid4()))
    assert row["query_transform"] == "NONE" and "RETRIEVAL_ONLY_NO_GENERATION" in row["warnings"]
    assert ("DIRECT_VI_RETRIEVAL_DIAGNOSTIC_ENGLISH_ENCODER_ASCII_BM25" in row["warnings"]) == diagnostic
    assert s.calls == []
