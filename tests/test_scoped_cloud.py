"""Offline V2 queue recovery/retention contracts; no AWS/model calls."""

from copy import deepcopy
from datetime import timedelta
from uuid import uuid4

import pytest

from app.answer_cache import AnswerCacheBlocked
from app.ledger import Change, ConcurrentWrite
from app.query_auth import ApiError
from app.run_storage import SQLiteRunStore
from app.scoped_cloud_coordinator import ScopedCloudCoordinator, ScopedCloudService, slot_message
from app.scoped_cloud_state import READY_RETENTION_SECONDS, ScopedWorkIndexMixin, slot_key
from app.scoped_run_contracts import ScopedRunMessageV2
from tests.test_authorization_evidence import auth_case as auth_case  # noqa: F401
from tests.test_scoped_owned_pipeline import QUESTION, deliver, submit  # noqa: F401
from tests.test_scoped_owned_pipeline import scoped_stack as scoped_stack


class Indexed(ScopedWorkIndexMixin, SQLiteRunStore):
    pass


@pytest.fixture
def cloud(scoped_stack):
    s = scoped_stack
    s.clock = [s.ledger._now()]
    s.ledger.clock = lambda: s.clock[0]
    s.repo.clock = s.ledger.clock
    store = Indexed(s.repo.store.path)
    store._epoch = lambda: int(s.clock[0].timestamp())
    s.repo.store = s.controller.store = s.ledger.store = store
    s.coordinator = ScopedCloudCoordinator(s.repo, s.queue.send, s.service.authorize)
    s.service = ScopedCloudService(s.repo, s.artifacts, lambda: s.executor.release,
        authorize=s.service.authorize, release_check=s.service.release_check, coordinator=s.coordinator)
    return s


def test_graph_q_qb_discovery_is_v2_only_and_slots_free_without_owner_data(cloud):
    s = cloud
    row = submit(s, ("graph_1hop", "hybrid_multi_query", "hybrid_multi_query_balanced"))
    slots = s.repo.store.work_slots()
    assert len(slots) == 3 and {slot_message(r).schema_version for r in slots} == {2}
    assert all(s.repo.store.get(("RUNWORK#kg-r72", f"SLOT#{i}")) is None for i in range(6))
    deliver(s, row["run_id"])
    assert s.service.detail(s.auth, row["run_id"])["status"] == "COMPLETED", s.failures
    assert all(r["status"] == "FREE" and set(r) == {"PK", "SK", "version", "record_type", "schema_version", "status"}
        for r in s.repo.store.work_slots())
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 0


def test_send_succeeds_but_ack_is_lost_then_sweep_and_duplicates_do_not_repeat_model(cloud):
    s = cloud
    def lost_ack(message):
        s.queue.send(message)
        raise TimeoutError("synthetic ack lost")
    s.coordinator.dispatcher.send = lost_ack
    row = submit(s)
    assert row["publication"] == "RECOVERY_PENDING" and len(s.queue.receive()) == 1
    s.coordinator.dispatcher.send = s.queue.send
    report = s.coordinator.sweep()
    assert report["pending_published"] == 1 and report["errors"] == 0
    deliver(s, row["run_id"])
    assert s.gateway.provider.calls == 1 and not s.queue.receive()


def test_unknown_keeps_slot_and_cost_reservation_without_resending(cloud):
    s = cloud
    row = submit(s)
    msg = ScopedRunMessageV2.model_validate(s.repo.outboxes(s.auth, row["run_id"])[0]["message"])
    outcome, token = s.repo.claim(msg)
    assert outcome == "CLAIMED"
    s.repo.start(s.auth, row["run_id"], "hybrid", token)
    s.clock[0] += timedelta(seconds=121)
    before = len(s.queue.receive())
    report = s.coordinator.sweep()
    assert report["unknown"] == 1 and report["safe_redeliveries"] == 0
    assert len(s.queue.receive()) == before and s.gateway.provider.calls == 0
    assert s.repo.store.get(slot_key(0))["status"] == "ACTIVE"
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 1


def test_stale_preexecution_claim_redelivers_and_old_token_cannot_start(cloud):
    s = cloud
    row = submit(s)
    msg = ScopedRunMessageV2.model_validate(s.repo.outboxes(s.auth, row["run_id"])[0]["message"])
    _, token = s.repo.claim(msg)
    s.clock[0] += timedelta(seconds=121)
    assert s.coordinator.sweep()["safe_redeliveries"] == 1
    deliver(s, row["run_id"])
    assert s.gateway.provider.calls == 1
    with pytest.raises(ApiError, match="RUN_LEASE_LOST"):
        s.repo.start(s.auth, row["run_id"], "hybrid", token)


def test_membership_revocation_cancels_without_assets_or_model(cloud):
    s = cloud
    row = submit(s, ("graph_1hop", "graph_2hop"))
    member = s.metadata.get(("USER#"+s.auth.user_id, "MEMBERSHIP"))
    s.metadata.commit([Change((member["PK"], member["SK"]), member["version"],
        member | {"status": "DISABLED", "version": member["version"]+1})])
    assert s.coordinator.sweep()["errors"] == 0
    assert all(r["status"] == "CANCELLED" for r in s.repo.children(s.auth, row["run_id"]))
    assert all(r["status"] == "FREE" for r in s.repo.store.work_slots()) and s.gateway.provider.calls == 0


def test_failed_transaction_cannot_release_slot_or_active_counter(cloud, monkeypatch):
    s = cloud
    row = submit(s)
    before = deepcopy(s.repo.store.get(slot_key(0)))
    def conflict(self, changes):
        raise ConcurrentWrite()
    monkeypatch.setattr(SQLiteRunStore, "commit", conflict)
    with pytest.raises(ApiError, match="STATE_BUSY"):
        s.repo.finish(s.auth, row["run_id"], "hybrid", "FAILED", error="SYNTHETIC")
    assert s.repo.store.get(slot_key(0)) == before
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 1


def test_ready_receipt_gets_bounded_ttl_incomplete_receipts_do_not(cloud):
    s = cloud
    row = submit(s)
    deliver(s, row["run_id"])
    with s.repo.store.connect() as db:
        keys = db.execute("SELECT pk,sk FROM run_items WHERE sk LIKE 'NATIVEANSWER#%'").fetchall()
    assert len(keys) == 1
    receipt = s.repo.store.get(keys[0])
    assert receipt["state"] == "READY" and receipt["expires_at"] == int(s.clock[0].timestamp())+READY_RETENTION_SECONDS
    s.clock[0] += timedelta(seconds=READY_RETENTION_SECONDS)
    with pytest.raises(AnswerCacheBlocked, match="EXPIRED_RECONCILE"):
        s.repo.store.get(keys[0])
    key = ("TENANT#demo#USER#"+s.auth.user_id, "NATIVEANSWER#FIXTURE#"+str(uuid4()))
    for version, state in enumerate(("IN_PROGRESS", "FAILED"), start=1):
        s.repo.store.commit([Change(key, version-1 or None, {"PK": key[0], "SK": key[1], "version": version,
            "record_type": "NATIVE_ANSWER_RECEIPT_V1", "state": state, "expires_at": 1})])
        assert "expires_at" not in s.repo.store.get(key)


def test_v1_and_v2_share_one_allowance_and_active_limit_but_separate_discovery(cloud):
    from app.owned_runs import OwnedRunReleaseV1, OwnedRunRepository, OwnedRunRequestV1
    from app.query_state import QueryRepository
    from app.run_work_index import WorkIndexMixin

    class Legacy(WorkIndexMixin, SQLiteRunStore):
        pass

    s = cloud
    legacy = OwnedRunRepository(Legacy(s.repo.store.path), clock=s.ledger.clock)
    release = OwnedRunReleaseV1(collection_version=s.executor.release.collection_version,
        collection_manifest_sha256=s.executor.release.collection_manifest_sha256,
        method_config_sha256={key: s.executor.release.method_config_sha256[key] for key in ("bm25", "vector", "hybrid")})
    old, _ = legacy.admit(s.other, str(uuid4()), OwnedRunRequestV1(original_question=QUESTION,
        method_ids=("bm25", "vector", "hybrid")), release)
    new = submit(s, ("graph_1hop", "hybrid_multi_query", "hybrid_multi_query_balanced"))
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 6
    quota = QueryRepository.quota_key(s.clock[0].strftime("%Y-%m"))
    assert s.repo.store.get(quota)["count"] == 6
    with pytest.raises(ApiError, match="CONCURRENCY_LIMIT"):
        submit(s)
    assert len(legacy.store.work_slots()) == len(s.repo.store.work_slots()) == 3
    legacy.cancel(s.other, old["run_id"])
    s.repo.cancel(s.auth, new["run_id"])
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 0
    assert s.repo.store.get(quota)["count"] == 6


def test_revoke_during_provider_call_cannot_publish_claims(cloud):
    s = cloud
    row = submit(s)
    def revoke():
        member = s.metadata.get(("USER#"+s.auth.user_id, "MEMBERSHIP"))
        s.metadata.commit([Change((member["PK"], member["SK"]), member["version"],
            member | {"status": "DISABLED", "version": member["version"]+1})])
    s.flags["after_call"] = revoke
    deliver(s, row["run_id"])
    child = s.repo.child(s.auth, row["run_id"], "hybrid")
    assert child["status"] in {"FAILED", "UNKNOWN"} and child.get("result_ref") is None
    with pytest.raises(ApiError, match="ACTIVE_MEMBERSHIP_REQUIRED"):
        s.service.public_result(s.auth, row["run_id"], "hybrid")
    assert s.gateway.provider.calls == 1


@pytest.mark.parametrize("field,value", [("schema_version", 1), ("record_type", "OWNED_RUN_WORK_SLOT"),
    ("owner_partition", "TENANT#demo#USER#not-a-uuid"), ("status", "FREE")])
def test_invalid_slot_rejected_before_dispatch(cloud, field, value):
    submit(cloud)
    with pytest.raises(ValueError):
        slot_message(cloud.repo.store.get(slot_key(0)) | {field: value})
