import json
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from app.answer_cache import AnswerCacheBlocked
from app.ledger import MemoryLedgerStore
from app.qualified_answer_service import resolve_qualified_answer
from app.run_storage import MemoryRunStore
from app.transactional_answer_cache import TransactionalAnswerCache
from tests.test_qualified_answer_service import BaseGateway, params


def data(tmp_path, store=None):
    p = params(tmp_path)
    p["raw_cache"] = TransactionalAnswerCache(store or MemoryRunStore(), tenant_id="demo", owner_id="owner-a", namespace="FIXTURE")
    return p


def test_service_can_reopen_durable_store_and_replay_qualification_without_model(tmp_path):
    store = MemoryRunStore()
    p, gateway = data(tmp_path, store), BaseGateway()
    first = resolve_qualified_answer(**p, gateway=gateway)
    p["raw_cache"] = TransactionalAnswerCache(store, tenant_id="demo", owner_id="owner-a", namespace="FIXTURE")
    second = resolve_qualified_answer(**p)
    assert first.receipt == second.receipt and second.cache_hit and gateway.calls == 1
    # Native raw-cache rows must not become entries in user-facing run history.
    assert store.run_history_keys(p["raw_cache"].partition, limit=10)[0] == []


def test_two_workers_cannot_both_dispatch_the_same_operation(tmp_path):
    entered, release = Event(), Event()

    class BlockingGateway(BaseGateway):
        def _call(self, *args):
            entered.set()
            assert release.wait(10)
            return super()._call(*args)

    p, gateway = data(tmp_path), BlockingGateway()
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(resolve_qualified_answer, **p, gateway=gateway)
        assert entered.wait(10)
        try:
            with pytest.raises(AnswerCacheBlocked):
                resolve_qualified_answer(**p, gateway=gateway)
        finally:
            release.set()
        result = future.result(timeout=10)
    assert result.qualification.status == "NO_MECHANICAL_FLAGS" and gateway.calls == 1


def test_crash_after_claim_cannot_trigger_automatic_second_model_call(tmp_path):
    class ClaimThenTimeout(MemoryLedgerStore):
        def commit(self, changes):
            super().commit(changes)
            raise TimeoutError("fixture ambiguous claim acknowledgement")

    p, gateway = data(tmp_path, ClaimThenTimeout()), BaseGateway()
    with pytest.raises(TimeoutError):
        resolve_qualified_answer(**p, gateway=gateway)
    with pytest.raises(AnswerCacheBlocked):
        resolve_qualified_answer(**p, gateway=gateway)
    assert gateway.calls == 0


def test_bad_native_json_is_durably_failed_with_raw_receipt_and_not_retried(tmp_path):
    p, gateway = data(tmp_path), BaseGateway(raw={"invalid": True})
    with pytest.raises(ValueError):
        resolve_qualified_answer(**p, gateway=gateway)
    with pytest.raises(AnswerCacheBlocked):
        resolve_qualified_answer(**p, gateway=gateway)
    key, _ = p["raw_cache"]._key(p["call"])
    row = p["raw_cache"].store.get(key)
    assert row["state"] == "FAILED" and json.loads(row["payload_json"])["raw_output"] == '{"invalid": true}'
    assert gateway.calls == 1


def test_payload_tampering_cannot_be_published(tmp_path):
    p, gateway = data(tmp_path), BaseGateway()
    resolve_qualified_answer(**p, gateway=gateway)
    store = p["raw_cache"].store
    key, _ = p["raw_cache"]._key(p["call"])
    row = store.get(key)
    from app.ledger import Change

    store.commit([Change(key, row["version"], row | {"version": row["version"]+1, "payload_json": "tampered"})])
    with pytest.raises(ValueError, match="CHECKSUM"):
        resolve_qualified_answer(**p)


def test_owner_partition_cannot_read_another_users_receipt(tmp_path):
    store = MemoryRunStore()
    p, gateway = data(tmp_path, store), BaseGateway()
    resolve_qualified_answer(**p, gateway=gateway)
    p["binding"] = p["binding"].model_copy(update={"owner_id": "owner-b"})
    p["raw_cache"] = TransactionalAnswerCache(store, tenant_id="demo", owner_id="owner-b", namespace="FIXTURE")
    with pytest.raises(ValueError, match="MISS_NO_GATEWAY"):
        resolve_qualified_answer(**p)


def test_misconfigured_cache_owner_is_rejected_before_any_storage_or_provider_access(tmp_path):
    class NoReads(MemoryRunStore):
        def get(self, key):
            raise AssertionError("misbound cache must not be read")

    p, gateway = data(tmp_path), BaseGateway()
    p["raw_cache"] = TransactionalAnswerCache(NoReads(), tenant_id="demo", owner_id="owner-b", namespace="FIXTURE")
    with pytest.raises(ValueError, match="OWNER_OR_TENANT_BINDING"):
        resolve_qualified_answer(**p, gateway=gateway)
    assert gateway.calls == 0


def test_live_dispatch_cannot_use_memory_persistence(tmp_path):
    p, gateway = data(tmp_path), BaseGateway()
    p["raw_cache"] = TransactionalAnswerCache(MemoryRunStore(), tenant_id="demo", owner_id="owner-a")
    gateway.provider = type("LiveFlag", (), {"live": True})()
    with pytest.raises(ValueError, match="MANAGED_STORE"):
        resolve_qualified_answer(**p, gateway=gateway)
    assert gateway.calls == 0


def test_dynamo_wire_adapter_writes_conditional_claim_raw_and_ready_then_replays(tmp_path):
    from copy import deepcopy
    from unittest.mock import patch

    from botocore.stub import ANY, Stubber

    from app.ledger import DynamoLedgerStore
    from tests.test_ledger import dynamo_client

    client = dynamo_client()  # Dummy keys; every SDK request is intercepted below.
    p, gateway = params(tmp_path), BaseGateway()
    cache = TransactionalAnswerCache(DynamoLedgerStore(client, "FixtureTable"), tenant_id="demo",
        owner_id="owner-a", namespace="FIXTURE")
    p["raw_cache"] = cache
    key, _ = cache._key(p["call"])
    expected_read = {"TableName": "FixtureTable", "Key": {"PK": {"S": key[0]}, "SK": {"S": key[1]}},
        "ConsistentRead": True}
    writes = []
    client.meta.events.register("before-parameter-build.dynamodb.TransactWriteItems",
        lambda params, **kwargs: writes.append(deepcopy(params)))
    with patch("socket.socket.connect", side_effect=AssertionError("wire test forbids network")), Stubber(client) as stub:
        stub.add_response("get_item", {}, expected_read)
        for expected in (None, 1, 2):
            put = {"TableName": "FixtureTable", "Item": ANY}
            if expected is None:
                put["ConditionExpression"] = "attribute_not_exists(PK) AND attribute_not_exists(SK)"
            else:
                put |= {"ConditionExpression": "#v = :expected", "ExpressionAttributeNames": {"#v": "version"},
                    "ExpressionAttributeValues": {":expected": {"N": str(expected)}}}
            stub.add_response("transact_write_items", {}, {"TransactItems": [{"Put": put}], "ClientRequestToken": ANY})
        first = resolve_qualified_answer(**p, gateway=gateway)
        items = [request["TransactItems"][0]["Put"]["Item"] for request in writes]
        assert [item["version"]["N"] for item in items] == ["1", "2", "3"]
        assert [item["state"]["S"] for item in items] == ["IN_PROGRESS", "IN_PROGRESS", "READY"]
        assert "payload_json" not in items[0] and "payload_json" in items[1]
        assert all(item["owner_id"] == {"S": "owner-a"} for item in items)
        stub.add_response("get_item", {"Item": items[-1]}, expected_read)
        second = resolve_qualified_answer(**p)
        stub.assert_no_pending_responses()
    assert first.receipt == second.receipt and second.cache_hit and gateway.calls == 1
