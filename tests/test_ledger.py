from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import boto3
import pytest
from botocore.exceptions import ClientError
from botocore.stub import ANY, Stubber

from app.contracts import text_hash
from app.cost import TokenUsage
from app.ledger import (
    AlreadyDispatched,
    BudgetDenied,
    Change,
    ConcurrentWrite,
    DynamoLedgerStore,
    IdempotencyConflict,
    Ledger,
    LedgerError,
    MemoryLedgerStore,
    OperationSpec,
)

NOW = datetime(2026, 10, 16, tzinfo=UTC)
USAGE = TokenUsage(input_tokens=10, output_tokens=5)


def spec(request="req1", **changes):
    return OperationSpec(**(dict(
        project_id="project", tenant_id="tenant", run_id="run1", request_id=request,
        stage="SMOKE", attempt=1, input_sha256=text_hash("synthetic input"),
        model_id="fixture-model", invocation_id="fixture-model", source_region="us-east-1",
        price_sheet_version="fixture-prices", max_cost_micro_usd=60,
    ) | changes))


@pytest.fixture
def ledger():
    value = Ledger(MemoryLedgerStore(), clock=lambda: NOW)
    value.initialize_month("project", "2026-10", cap_micro_usd=100,
                           other_cost_estimate_micro_usd=0, gross_stop_micro_usd=100)
    value.initialize_run("project", "tenant", "run1", cap_micro_usd=100)
    return value


def counts(ledger, period="2026-10"):
    return ledger.store.get(ledger.month_key("project", period))


def test_reservation_replay_and_double_settlement(ledger):
    operation = ledger.reserve(spec())
    assert ledger.reserve(spec()) == operation
    assert counts(ledger)["reserved_micro_usd"] == 60
    ledger.begin_call(operation.operation_id)
    assert ledger.settle(operation.operation_id, 25, USAGE).status == "SETTLED"
    ledger.settle(operation.operation_id, 25, USAGE)
    assert (counts(ledger)["reserved_micro_usd"], counts(ledger)["settled_micro_usd"]) == (0, 25)
    with pytest.raises(IdempotencyConflict):
        ledger.settle(operation.operation_id, 26, USAGE)
    with pytest.raises(AlreadyDispatched):
        ledger.begin_call(operation.operation_id)


def test_same_operation_different_payload_rejected(ledger):
    ledger.reserve(spec())
    with pytest.raises(IdempotencyConflict):
        ledger.reserve(spec(input_sha256=text_hash("changed input")))
    with pytest.raises(IdempotencyConflict):
        ledger.reserve(spec(invocation_id="other-model"))


def test_cancel_only_before_dispatch_and_idempotent(ledger):
    operation = ledger.reserve(spec())
    ledger.cancel_before_call(operation.operation_id)
    ledger.cancel_before_call(operation.operation_id)
    assert counts(ledger)["reserved_micro_usd"] == 0
    with pytest.raises(AlreadyDispatched):
        ledger.begin_call(operation.operation_id)
    with pytest.raises(LedgerError):
        ledger.settle(operation.operation_id, 1, USAGE)


def test_unknown_spend_keeps_allowance_until_actual_reconciliation(ledger):
    operation = ledger.reserve(spec())
    ledger.begin_call(operation.operation_id)
    ledger.mark_unknown(operation.operation_id)
    ledger.mark_unknown(operation.operation_id)
    assert counts(ledger)["reserved_micro_usd"] == 60
    with pytest.raises(LedgerError):
        ledger.cancel_before_call(operation.operation_id)
    with pytest.raises(BudgetDenied):
        ledger.reserve(spec("req2"))
    ledger.settle(operation.operation_id, 30, USAGE)
    assert (counts(ledger)["reserved_micro_usd"], counts(ledger)["settled_micro_usd"]) == (0, 30)


def test_actual_overrun_is_recorded_truthfully_and_halts(ledger):
    operation = ledger.reserve(spec())
    ledger.begin_call(operation.operation_id)
    settled = ledger.settle(operation.operation_id, 120, USAGE)
    assert settled.bound_exceeded is True
    assert counts(ledger)["settled_micro_usd"] == 120
    assert counts(ledger)["halted"] is True
    assert ledger.store.get(ledger.run_key("run1"))["halted"] is True
    with pytest.raises(BudgetDenied):
        ledger.reserve(spec("req2", max_cost_micro_usd=1))


def test_month_rollover_settles_original_counter(ledger):
    operation = ledger.reserve(spec())
    ledger.begin_call(operation.operation_id)
    ledger.clock = lambda: datetime(2026, 11, 1, tzinfo=UTC)
    ledger.initialize_month("project", "2026-11", cap_micro_usd=100)
    ledger.settle(operation.operation_id, 20, USAGE)
    assert counts(ledger)["settled_micro_usd"] == 20
    assert counts(ledger, "2026-11")["settled_micro_usd"] == 0
    assert ledger.reserve(spec()).period == "2026-10"


def test_atomic_concurrent_reservation_does_not_overspend(ledger):
    def reserve(index):
        try:
            return ledger.reserve(spec(f"request{index}", max_cost_micro_usd=60))
        except BudgetDenied:
            return None

    with ThreadPoolExecutor(max_workers=12) as executor:
        results = list(executor.map(reserve, range(24)))
    assert sum(result is not None for result in results) == 1
    assert counts(ledger)["reserved_micro_usd"] == 60


def test_concurrent_same_operation_and_dispatch_authorize_one_call(ledger):
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: ledger.reserve(spec()), range(16)))
    assert len({result.operation_id for result in results}) == 1
    operation_id = results[0].operation_id

    def begin(_):
        try:
            ledger.begin_call(operation_id)
            return True
        except AlreadyDispatched:
            return False

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(begin, range(16))) == 1
    assert counts(ledger)["reserved_micro_usd"] == 60


def test_initialization_never_resets_spend_or_cap(ledger):
    ledger.reserve(spec())
    ledger.initialize_run("project", "tenant", "run1", cap_micro_usd=100)
    assert ledger.store.get(ledger.run_key("run1"))["reserved_micro_usd"] == 60
    with pytest.raises(IdempotencyConflict):
        ledger.initialize_run("project", "tenant", "run1", cap_micro_usd=200)
    with pytest.raises(BudgetDenied):
        ledger.reserve(spec("req2", tenant_id="other"))


def test_missing_budget_and_pre_dispatch_settle_fail_closed():
    ledger = Ledger(MemoryLedgerStore(), clock=lambda: NOW)
    with pytest.raises(BudgetDenied):
        ledger.reserve(spec())
    ledger.initialize_month("project", "2026-10")
    ledger.initialize_run("project", "tenant", "run1")
    operation = ledger.reserve(spec())
    with pytest.raises(LedgerError):
        ledger.settle(operation.operation_id, 10, USAGE)
    with pytest.raises(LedgerError):
        ledger.mark_unknown(operation.operation_id)


def test_naive_clock_and_noncanonical_month_are_rejected(ledger):
    ledger.clock = lambda: datetime(2026, 10, 16)
    with pytest.raises(ValueError):
        ledger.reserve(spec())
    with pytest.raises(ValueError):
        ledger.initialize_month("project", "2026-1")


def test_partial_atomic_write_failure_does_not_change_memory_counters(ledger):
    counter = counts(ledger)
    stale = Change((counter["PK"], counter["SK"]), 99,
                   counter | {"version": 100, "settled_micro_usd": 99})
    create = Change(("OP#fixture", "STATE"), None,
                    {"PK": "OP#fixture", "SK": "STATE", "version": 1})
    with pytest.raises(ConcurrentWrite):
        ledger.store.commit([create, stale])
    assert ledger.store.get(create.key) is None
    assert counts(ledger)["settled_micro_usd"] == 0


def test_gross_stop_includes_other_cost_estimate():
    ledger = Ledger(MemoryLedgerStore(), clock=lambda: NOW)
    ledger.initialize_month("project", "2026-10", cap_micro_usd=100,
                            other_cost_estimate_micro_usd=90, gross_stop_micro_usd=100)
    ledger.initialize_run("project", "tenant", "run1")
    with pytest.raises(BudgetDenied):
        ledger.reserve(spec(max_cost_micro_usd=11))
    assert ledger.reserve(spec(max_cost_micro_usd=10))


def dynamo_client():
    # Explicit dummy credentials; Stubber intercepts every request, no credential discovery/network.
    return boto3.client("dynamodb", region_name="us-east-1", aws_access_key_id="fixture",
                        aws_secret_access_key="fixture")


def test_dynamo_read_is_strong_and_converts_only_integral_numbers():
    client = dynamo_client()
    with Stubber(client) as stub:
        stub.add_response("get_item", {"Item": {"PK": {"S": "P"}, "SK": {"S": "S"},
                                                "version": {"N": "1"}}},
                          {"TableName": "Ledger", "Key": {"PK": {"S": "P"}, "SK": {"S": "S"}},
                           "ConsistentRead": True})
        assert DynamoLedgerStore(client, "Ledger").get(("P", "S"))["version"] == 1


def test_dynamo_atomic_put_plan_and_conditional_failure():
    client = dynamo_client()
    change = Change(("P", "S"), 1, {"PK": "P", "SK": "S", "version": 2, "reserved": 4})
    expected = {"TransactItems": [{"Put": {
        "TableName": "Ledger", "Item": {"PK": {"S": "P"}, "SK": {"S": "S"},
                                         "version": {"N": "2"}, "reserved": {"N": "4"}},
        "ConditionExpression": "#v = :expected", "ExpressionAttributeNames": {"#v": "version"},
        "ExpressionAttributeValues": {":expected": {"N": "1"}},
    }}], "ClientRequestToken": ANY}
    with Stubber(client) as stub:
        stub.add_response("transact_write_items", {}, expected)
        stub.add_client_error("transact_write_items", "TransactionCanceledException",
                              modeled_fields={"CancellationReasons": [{"Code": "ConditionalCheckFailed"}]},
                              expected_params=expected)
        store = DynamoLedgerStore(client, "Ledger")
        store.commit([change])
        with pytest.raises(ConcurrentWrite):
            store.commit([change])


def test_service_failure_propagates_and_duplicate_item_transaction_rejected():
    client = dynamo_client()
    change = Change(("P", "S"), None, {"PK": "P", "SK": "S", "version": 1})
    with Stubber(client) as stub:
        stub.add_client_error("transact_write_items", "InternalServerError")
        store = DynamoLedgerStore(client, "Ledger")
        with pytest.raises(ClientError):
            store.commit([change])
        with pytest.raises(ValueError):
            store.commit([change, change])
