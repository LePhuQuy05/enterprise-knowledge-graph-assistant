import copy
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from app.ledger import Change
from app.owned_runs import OwnedRunReleaseV1, OwnedRunRepository, OwnedRunRequestV1, RunMessageV1
from app.query_auth import ApiError, AuthContext
from app.query_state import QueryRepository
from app.run_storage import MemoryRunStore, SQLiteRunStore

USER_A = AuthContext("demo", "11111111-1111-1111-1111-111111111111")
USER_B = AuthContext("demo", "22222222-2222-2222-2222-222222222222")


@pytest.fixture(params=["memory", "sqlite"])
def owned_repo(request, tmp_path):
    store = MemoryRunStore() if request.param == "memory" else SQLiteRunStore(tmp_path / "state.sqlite3")
    clock = [datetime(2026, 10, 25, tzinfo=UTC)]
    return OwnedRunRepository(store, clock=lambda: clock[0]), clock


def release():
    return OwnedRunReleaseV1(collection_version="collection_" + "a" * 64, collection_manifest_sha256="b" * 64,
        method_config_sha256={"bm25": "c" * 64, "vector": "d" * 64, "hybrid": "e" * 64})


def admit(repo, *, auth=USER_A, key=None, methods=("hybrid",), **extra):
    return repo.admit(auth, key or str(uuid4()), OwnedRunRequestV1(original_question="Service B", method_ids=methods,
        **extra), release())


def message(repo, run, method="hybrid", auth=USER_A):
    return RunMessageV1.model_validate(next(row["message"] for row in repo.outboxes(auth, run)
        if row["message"]["method_id"] == method))


def test_atomic_admission_pins_all_children_outbox_and_shared_monthly_quota(owned_repo):
    repo, _ = owned_repo
    row, fresh = admit(repo, methods=("bm25", "vector", "hybrid"))
    assert fresh and repo.summary(USER_A, row["run_id"])["status"] == "QUEUED"
    assert len(repo.children(USER_A, row["run_id"])) == len(repo.outboxes(USER_A, row["run_id"])) == 3
    assert QueryRepository(repo.store, clock=repo.clock).quota_remaining() == 297
    for key in repo.active_keys(USER_A):
        assert repo.store.get(key)["count"] == 3
    assert row["release"] == release().model_dump(mode="json")


def test_concurrent_duplicate_is_one_admission_and_conflict_never_reuses_key(owned_repo):
    repo, _ = owned_repo
    key = str(uuid4())
    with ThreadPoolExecutor(max_workers=8) as pool:
        result = list(pool.map(lambda _: admit(repo, key=key), range(8)))
    assert sum(fresh for _, fresh in result) == 1
    assert len({row["run_id"] for row, _ in result}) == 1
    assert QueryRepository(repo.store, clock=repo.clock).quota_remaining() == 299
    with pytest.raises(ApiError, match="IDEMPOTENCY_CONFLICT"):
        admit(repo, key=key, methods=("bm25",))
    # A new current config does not replace the original pin under an already admitted key.
    newer = release().model_copy(update={"collection_manifest_sha256": "f" * 64})
    row, fresh = repo.admit(USER_A, key, OwnedRunRequestV1(original_question="Service B"), newer)
    assert not fresh and row["release"] == release().model_dump(mode="json")


def test_global_quota_is_shared_with_sync_and_counts_children(owned_repo):
    repo, clock = owned_repo
    key = QueryRepository.quota_key(clock[0].strftime("%Y-%m"))
    repo.store.commit([Change(key, None, {"PK": key[0], "SK": key[1], "version": 1, "count": 298,
        "record_type": "QUERY_QUOTA", "expires_at": int(clock[0].timestamp()) + 86400})])
    with pytest.raises(ApiError, match="QUERY_QUOTA_EXCEEDED"):
        admit(repo, methods=("bm25", "vector", "hybrid"))
    admit(repo, methods=("vector", "hybrid"))
    assert repo.store.get(key)["count"] == 300
    with pytest.raises(ApiError, match="QUERY_QUOTA_EXCEEDED"):
        admit(repo, auth=USER_B)


def test_owner_global_active_limits_and_cancellation_release_slots_once(owned_repo):
    repo, _ = owned_repo
    first, _ = admit(repo, methods=("bm25", "vector", "hybrid"))
    second, _ = admit(repo, auth=USER_B, methods=("bm25", "vector", "hybrid"))
    with pytest.raises(ApiError, match="RUN_CONCURRENCY_LIMIT"):
        admit(repo)
    repo.cancel(USER_A, first["run_id"])
    repo.cancel(USER_A, first["run_id"])
    assert repo.summary(USER_A, first["run_id"])["status"] == "CANCELLED"
    assert repo.store.get(repo.active_keys(USER_A)[0])["count"] == 3
    assert repo.store.get(repo.active_keys(USER_A)[1])["count"] == 0
    repo.cancel(USER_B, second["run_id"])
    assert repo.store.get(repo.active_keys(USER_A)[0])["count"] == 0


def test_claim_before_execution_can_recover_but_old_token_is_fenced(owned_repo):
    repo, clock = owned_repo
    row, _ = admit(repo)
    msg = message(repo, row["run_id"])
    outcome, old_token = repo.claim(msg)
    assert outcome == "CLAIMED" and repo.claim(msg) == ("RETRY", None)
    clock[0] += timedelta(seconds=121)
    _, new_token = repo.claim(msg)
    assert new_token != old_token
    with pytest.raises(ApiError, match="RUN_LEASE_LOST"):
        repo.start(USER_A, row["run_id"], "hybrid", old_token)
    repo.start(USER_A, row["run_id"], "hybrid", new_token)
    assert repo.child(USER_A, row["run_id"], "hybrid")["status"] == "EXECUTING"


def test_after_execution_gate_timeout_becomes_unknown_and_never_taken_over(owned_repo):
    repo, clock = owned_repo
    row, _ = admit(repo)
    msg = message(repo, row["run_id"])
    _, token = repo.claim(msg)
    repo.start(USER_A, row["run_id"], "hybrid", token)
    clock[0] += timedelta(seconds=121)
    assert repo.claim(msg) == ("ACK", None)
    assert repo.summary(USER_A, row["run_id"])["status"] == "UNKNOWN"
    assert repo.claim(msg) == ("ACK", None)
    assert repo.store.get(repo.active_keys(USER_A)[0])["count"] == 1
    assert repo.finish(USER_A, row["run_id"], "hybrid", "FAILED", token=token)["status"] == "UNKNOWN"


def test_deadline_cancel_forged_message_and_cross_owner(owned_repo):
    repo, clock = owned_repo
    row, _ = admit(repo)
    msg = message(repo, row["run_id"])
    with pytest.raises(ApiError, match="RUN_NOT_FOUND"):
        repo.claim(msg.model_copy(update={"admission_token": "0" * 32}))
    with pytest.raises(ApiError, match="RUN_NOT_FOUND"):
        repo.get(USER_B, row["run_id"])
    clock[0] += timedelta(seconds=601)
    assert repo.claim(msg) == ("ACK", None)
    assert repo.summary(USER_A, row["run_id"])["status"] == "EXPIRED"
    assert repo.store.get(repo.active_keys(USER_A)[0])["count"] == 0


def test_cancel_during_execution_is_best_effort_and_does_not_clear_unknown_obligation(owned_repo):
    repo, _ = owned_repo
    row, _ = admit(repo)
    _, token = repo.claim(message(repo, row["run_id"]))
    repo.start(USER_A, row["run_id"], "hybrid", token)
    repo.cancel(USER_A, row["run_id"])
    assert repo.child(USER_A, row["run_id"], "hybrid")["status"] == "EXECUTING"
    with pytest.raises(ApiError, match="RUN_STOP_REQUESTED"):
        repo.guard(USER_A, row["run_id"], "hybrid", token)
    repo.guard(USER_A, row["run_id"], "hybrid", token, allow_cancelled=True)


def test_keys_are_single_use_even_when_idempotency_row_expires(owned_repo):
    repo, clock = owned_repo
    key = str(uuid4())
    row, _ = admit(repo, key=key)
    repo.cancel(USER_A, row["run_id"])
    clock[0] += timedelta(days=8)
    with pytest.raises(ApiError, match="IDEMPOTENCY_KEY_EXPIRED"):
        admit(repo, key=key)
    clock[0] += timedelta(days=23)
    with pytest.raises(ApiError, match="RUN_NOT_FOUND"):
        repo.get(USER_A, row["run_id"])


@pytest.mark.parametrize("changes", [{"run_kind": "answer"}, {"method_ids": ("hybrid_multi_query",)},
    {"method_ids": ("hybrid_multi_query_balanced",)}])
def test_unreleased_paid_paths_are_rejected_before_admission(owned_repo, changes):
    repo, _ = owned_repo
    request = OwnedRunRequestV1(original_question="Service B", **changes)
    with pytest.raises(ApiError):
        repo.admit(USER_A, str(uuid4()), request, release())
    assert QueryRepository(repo.store, clock=repo.clock).quota_remaining() == 300


@pytest.mark.parametrize("changes", [{"original_question": " "}, {"method_ids": ("hybrid", "hybrid")},
    {"method_ids": ("bm25", "vector", "hybrid", "hybrid_reranked")}, {"source_scope": "aws_official"},
    {"tenant_id": "forged"}, {"method_ids": ("graph_2hop",)}])
def test_request_contract_is_bounded_and_has_no_client_identity_or_feature_override(changes):
    with pytest.raises(ValidationError):
        OwnedRunRequestV1.model_validate({"original_question": "Service B"} | changes)


def test_failed_admission_is_atomic_and_does_not_publish_or_consume_quota(owned_repo, monkeypatch):
    repo, _ = owned_repo
    original = repo.store.commit
    def fail(changes):
        raise RuntimeError("synthetic transaction failure")
    monkeypatch.setattr(repo.store, "commit", fail)
    with pytest.raises(RuntimeError):
        admit(repo)
    monkeypatch.setattr(repo.store, "commit", original)
    assert QueryRepository(repo.store, clock=repo.clock).quota_remaining() == 300
    assert repo.store.run_history_keys(USER_A.partition, limit=20) == ([], None)


def test_sqlite_survives_reopening_and_cas_conflict_is_atomic(tmp_path):
    store = SQLiteRunStore(tmp_path / "state.sqlite3")
    repo = OwnedRunRepository(store)
    row, _ = admit(repo)
    reopened = OwnedRunRepository(SQLiteRunStore(store.path))
    assert reopened.get(USER_A, row["run_id"]) == row
    key = (row["PK"], row["SK"])
    original = copy.deepcopy(row)
    with pytest.raises(Exception):
        store.commit([Change(key, None, row)])
    assert store.get(key) == original


def test_duplicate_committed_between_idempotency_and_parent_reads_is_not_expired(owned_repo, monkeypatch):
    repo, _ = owned_repo
    key = str(uuid4())
    original, once = repo.store.get, [False]
    def interleaved(item_key):
        value = original(item_key)
        if item_key == (USER_A.partition, "RUNIDEMP#" + key) and not once[0]:
            once[0] = True
            admit(repo, key=key)
        return value
    monkeypatch.setattr(repo.store, "get", interleaved)
    _, fresh = admit(repo, key=key)
    assert not fresh and QueryRepository(repo.store, clock=repo.clock).quota_remaining() == 299
