import copy
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.ledger import ConcurrentWrite
from app.owned_runs import OwnedRunRequestV1
from app.query_auth import ApiError
from app.run_cloud import CloudRunCoordinator, CloudRunService, worker_handler
from app.run_storage import SQLiteRunStore
from app.run_work_index import WorkIndexMixin, slot_key
from tests.test_collection_review import review_context as review_context
from tests.test_rag_runner import setup_runner as setup_runner
from tests.test_run_execution import deliver, submit
from tests.test_run_execution import run_stack as run_stack


class IndexedSQLiteStore(WorkIndexMixin, SQLiteRunStore):
    pass


@pytest.fixture
def cloud_stack(run_stack):
    s = run_stack
    s.repo.store = IndexedSQLiteStore(s.repo.store.path)
    s.coordinator = CloudRunCoordinator(s.repo, s.queue.send, s.service.authorize)
    s.service = CloudRunService(s.repo, s.artifacts, lambda: s.executor.release,
        authorize=s.service.authorize, release_check=s.service.release_check, coordinator=s.coordinator)
    return s


def test_committed_admission_survives_send_failure_and_watchdog_completes_real_runner(cloud_stack):
    s = cloud_stack
    def lost_send(message):
        raise TimeoutError("synthetic unavailable queue")
    s.coordinator.dispatcher.send = lost_send
    row = submit(s, ("bm25", "vector", "hybrid"))
    assert row["publication"] == "RECOVERY_PENDING"
    assert len(s.repo.store.work_slots()) == 3 and not s.queue.receive()
    assert all(c["status"] == "QUEUED" for c in s.repo.children(s.auth, row["run_id"]))
    s.coordinator.dispatcher.send = s.queue.send
    report = s.coordinator.sweep()
    assert report["pending_published"] == 3 and report["errors"] == 0
    deliver(s, row["run_id"])
    assert s.service.detail(s.auth, row["run_id"])["status"] == "COMPLETED"
    assert len(s.calls) == 3
    slots = s.repo.store.work_slots()
    assert all(slot["status"] == "FREE" and "owner_partition" not in slot for slot in slots)
    assert s.coordinator.sweep()["examined"] == 3


def test_unknown_keeps_work_slot_and_watchdog_never_reexecutes(cloud_stack):
    s = cloud_stack
    row = submit(s)
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    from app.owned_runs import RunMessageV1

    outcome, token = s.repo.claim(RunMessageV1.model_validate(msg))
    assert outcome == "CLAIMED"
    s.repo.start(s.auth, row["run_id"], "hybrid", token)
    s.clock[0] += timedelta(seconds=121)
    report = s.coordinator.sweep()
    assert report["unknown"] == 1 and not s.calls
    assert s.repo.child(s.auth, row["run_id"], "hybrid")["status"] == "UNKNOWN"
    assert s.repo.store.get(slot_key(0))["status"] == "ACTIVE"
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 1
    assert s.coordinator.sweep()["safe_redeliveries"] == 0


def test_stale_claim_can_redeliver_but_old_token_cannot_start(cloud_stack):
    from app.owned_runs import RunMessageV1

    s = cloud_stack
    row = submit(s)
    msg = RunMessageV1.model_validate(s.repo.outboxes(s.auth, row["run_id"])[0]["message"])
    _, old_token = s.repo.claim(msg)
    s.clock[0] += timedelta(seconds=121)
    assert s.coordinator.sweep()["safe_redeliveries"] == 1
    deliver(s, row["run_id"])
    assert s.calls == ["hybrid"]
    with pytest.raises(ApiError, match="RUN_LEASE_LOST"):
        s.repo.start(s.auth, row["run_id"], "hybrid", old_token)


def test_revocation_and_expiry_release_slots_without_retrieval(cloud_stack):
    s = cloud_stack
    first = submit(s, ("bm25", "vector"))
    member = s.metadata.get(("USER#" + s.auth.user_id, "MEMBERSHIP"))
    from app.ledger import Change

    s.metadata.commit([Change((member["PK"], member["SK"]), member["version"],
        member | {"status": "DISABLED", "version": member["version"] + 1})])
    assert s.coordinator.sweep()["errors"] == 0
    assert all(c["status"] == "CANCELLED" for c in s.repo.children(s.auth, first["run_id"]))
    assert not s.calls and all(slot["status"] == "FREE" for slot in s.repo.store.work_slots())
    row, _ = s.repo.admit(s.other, str(uuid4()), OwnedRunRequestV1(original_question="Service B"), s.executor.release)
    s.clock[0] += timedelta(seconds=601)
    s.coordinator.sweep()
    assert s.repo.child(s.other, row["run_id"], "hybrid")["status"] == "EXPIRED"
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 0


def test_slot_and_counters_do_not_free_on_failed_transaction(cloud_stack, monkeypatch):
    s = cloud_stack
    row = submit(s)
    slot_before = copy.deepcopy(s.repo.store.get(slot_key(0)))
    def fail_commit(self, changes):
        raise ConcurrentWrite()
    monkeypatch.setattr(SQLiteRunStore, "commit", fail_commit)
    with pytest.raises(ApiError, match="STATE_BUSY"):
        s.repo.finish(s.auth, row["run_id"], "hybrid", "FAILED", error="SYNTHETIC")
    assert s.repo.store.get(slot_key(0)) == slot_before
    assert s.repo.child(s.auth, row["run_id"], "hybrid")["status"] == "QUEUED"
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 1


def test_slot_capacity_matches_six_child_global_limit_and_is_reusable(cloud_stack):
    s = cloud_stack
    a = submit(s, ("bm25", "vector", "hybrid"))
    b, _ = s.repo.admit(s.other, str(uuid4()), OwnedRunRequestV1(original_question="Service B",
        method_ids=("bm25", "vector", "hybrid")), s.executor.release)
    assert len(s.repo.store.work_slots()) == 6
    with pytest.raises(ApiError, match="RUN_CONCURRENCY_LIMIT"):
        submit(s)
    s.repo.cancel(s.auth, a["run_id"])
    c = submit(s)
    assert s.repo.child(s.auth, c["run_id"], "hybrid")["work_slot"] < 3
    assert all(c["status"] == "QUEUED" for c in s.repo.children(s.other, b["run_id"]))


def test_duplicate_worker_message_never_loads_cloud_assets_when_already_terminal(cloud_stack, monkeypatch):
    import app.run_cloud as module
    s = cloud_stack
    row = submit(s)
    deliver(s, row["run_id"])
    msg = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    from app.contracts import canonical_json

    monkeypatch.setattr(module, "cloud_components", lambda: ({}, s.service, s.metadata, s.coordinator))
    monkeypatch.setattr(module, "make_retrieval_executor", lambda *a: pytest.fail("terminal redelivery must not initialize ONNX"))
    result = worker_handler({"Records": [{"eventSource": "aws:sqs", "messageId": "fixture-id",
        "body": canonical_json(msg).decode()}]}, SimpleNamespace(get_remaining_time_in_millis=lambda: 90_000))
    assert result == {"batchItemFailures": []} and s.calls == ["hybrid"]


def test_answer_admission_still_rejects_before_any_work_slot_or_publish(cloud_stack):
    with pytest.raises(ApiError, match="ANSWER_RUNS_NOT_ENABLED"):
        submit(cloud_stack, run_kind="answer")
    assert not cloud_stack.repo.store.work_slots() and not cloud_stack.queue.receive()
