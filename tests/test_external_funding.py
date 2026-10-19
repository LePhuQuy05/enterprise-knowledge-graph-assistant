"""External budget protocol tests; no credentials, provider calls or AWS requests."""

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from app.contracts import text_hash
from app.cost import TokenUsage
from app.ledger import (
    BudgetDenied,
    Change,
    ConcurrentWrite,
    ExternalOperationSpec,
    IdempotencyConflict,
    Ledger,
    MemoryLedgerStore,
    Operation,
    OperationSpec,
)

NOW = datetime(2026, 10, 17, tzinfo=UTC)
USAGE = TokenUsage(input_tokens=12, output_tokens=4)
# Exact pre-migration shape. Its digest must not acquire defaults from the v2 DTO.
LEGACY_SPEC = {
    "project_id": "project", "tenant_id": "tenant", "run_id": "run1", "request_id": "legacy",
    "stage": "SMOKE", "attempt": 1, "input_sha256": text_hash("synthetic input"),
    "model_id": "fixture-model", "invocation_id": "fixture-model", "source_region": "us-east-1",
    "price_sheet_version": "fixture-prices", "max_cost_micro_usd": 60,
}


def spec(request="req1", provider="DEEPSEEK", **changes):
    return ExternalOperationSpec(**(LEGACY_SPEC | {
        "request_id": request, "provider": provider, "source_region": "external-default",
        "model_id": "fixture-" + provider.lower(), "invocation_id": "fixture-" + provider.lower(),
    } | changes))


@pytest.fixture
def ledger():
    value = Ledger(MemoryLedgerStore(), clock=lambda: NOW)
    value.initialize_month("project", "2026-10", cap_micro_usd=1_000,
                           other_cost_estimate_micro_usd=0, gross_stop_micro_usd=1_000)
    value.initialize_run("project", "tenant", "run1", cap_micro_usd=1_000)
    value.initialize_external_funding("project", cap_micro_usd=100)
    return value


def funding(ledger):
    return ledger.store.get(ledger.external_funding_key("project"))


def all_counters(ledger, run="run1", month="2026-10"):
    return [ledger.store.get(key) for key in (
        ledger.month_key("project", month), ledger.run_key(run),
        ledger.external_funding_key("project"),
    )]


def test_legacy_serialized_identity_and_operation_survive_without_new_defaults(ledger):
    expected_digest = hashlib.sha256(json.dumps(
        LEGACY_SPEC, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode()).hexdigest()
    legacy = OperationSpec(**LEGACY_SPEC)
    assert legacy.fingerprint == expected_digest
    operation = ledger.reserve(legacy)
    saved = ledger.store.get((operation.PK, operation.SK))
    assert saved["spec"] == LEGACY_SPEC
    assert saved["spec_sha256"] == expected_digest
    reopened = Operation.model_validate_json(json.dumps(saved))
    assert type(reopened.spec) is OperationSpec
    assert reopened.spec.fingerprint == expected_digest
    assert funding(ledger)["reserved_micro_usd"] == 0


def test_external_spec_survives_nested_storage_and_provider_change_conflicts(ledger):
    external = spec()
    operation = ledger.reserve(external)
    reopened = ledger.operation(operation.operation_id)
    assert type(reopened.spec) is ExternalOperationSpec
    assert reopened.spec.model_dump() == external.model_dump()
    assert reopened.spec.fingerprint == external.fingerprint
    assert ledger.reserve(external) == operation
    assert funding(ledger)["reserved_micro_usd"] == 60
    with pytest.raises(IdempotencyConflict):
        ledger.reserve(spec(provider="VOYAGE"))
    # Removing the provider/funding fields cannot reclassify this op as a legacy call.
    with pytest.raises(ValidationError):
        OperationSpec.model_validate(external.model_dump())


@pytest.mark.parametrize("changes", [
    {"funding_id": "another-run"}, {"schema_version": 1}, {"billing_source": "AWS_CREDIT"},
    {"provider": "BEDROCK"}, {"source_region": "us-east-1"},
])
def test_external_identity_rejects_funding_and_billing_relabeling(changes):
    with pytest.raises(ValidationError):
        spec(**changes)


def test_external_counter_required_before_any_budget_is_debited():
    value = Ledger(MemoryLedgerStore(), clock=lambda: NOW)
    value.initialize_month("project", "2026-10")
    value.initialize_run("project", "tenant", "run1")
    with pytest.raises(BudgetDenied, match="not initialized"):
        value.reserve(spec())
    assert value.store.get(value.month_key("project", "2026-10"))["reserved_micro_usd"] == 0
    assert value.store.get(value.run_key("run1"))["reserved_micro_usd"] == 0
    assert value.store.get(("OP#" + spec().operation_id, "STATE")) is None


def test_two_providers_share_funding_and_settlement_releases_only_unused_bound(ledger):
    first = ledger.reserve(spec(max_cost_micro_usd=60))
    second = ledger.reserve(spec("req2", "VOYAGE", max_cost_micro_usd=40))
    assert [c["reserved_micro_usd"] for c in all_counters(ledger)] == [100, 100, 100]
    with pytest.raises(BudgetDenied):
        ledger.reserve(spec("req3", max_cost_micro_usd=1))
    ledger.begin_call(first.operation_id)
    ledger.settle(first.operation_id, 25, USAGE)
    ledger.settle(first.operation_id, 25, USAGE)
    assert [(c["reserved_micro_usd"], c["settled_micro_usd"]) for c in all_counters(ledger)] == [
        (40, 25), (40, 25), (40, 25),
    ]
    ledger.cancel_before_call(second.operation_id)
    ledger.cancel_before_call(second.operation_id)
    assert funding(ledger)["reserved_micro_usd"] == 0
    assert funding(ledger)["settled_micro_usd"] == 25


def test_shared_funding_does_not_reset_across_run_tenant_or_month(ledger):
    first = ledger.reserve(spec())
    ledger.begin_call(first.operation_id)
    ledger.settle(first.operation_id, 60, USAGE)
    ledger.clock = lambda: datetime(2026, 11, 1, tzinfo=UTC)
    ledger.initialize_month("project", "2026-11")
    ledger.initialize_run("project", "tenant2", "run2")
    ledger.initialize_external_funding("project", cap_micro_usd=100)
    second = spec("req2", "VOYAGE", tenant_id="tenant2", run_id="run2", max_cost_micro_usd=41)
    with pytest.raises(BudgetDenied):
        ledger.reserve(second)
    assert funding(ledger)["settled_micro_usd"] == 60
    assert ledger.store.get(ledger.month_key("project", "2026-11"))["reserved_micro_usd"] == 0
    assert ledger.store.get(ledger.run_key("run2"))["reserved_micro_usd"] == 0
    with pytest.raises(IdempotencyConflict):
        ledger.initialize_external_funding("project", cap_micro_usd=99)


def test_unknown_retains_all_reservations_and_reconciles_original_month(ledger):
    operation = ledger.reserve(spec())
    ledger.begin_call(operation.operation_id)
    ledger.mark_unknown(operation.operation_id)
    assert [c["reserved_micro_usd"] for c in all_counters(ledger)] == [60, 60, 60]
    ledger.clock = lambda: datetime(2026, 11, 1, tzinfo=UTC)
    ledger.initialize_month("project", "2026-11")
    ledger.settle(operation.operation_id, 30, USAGE)
    assert [(c["reserved_micro_usd"], c["settled_micro_usd"]) for c in all_counters(ledger)] == [
        (0, 30), (0, 30), (0, 30),
    ]
    assert ledger.store.get(ledger.month_key("project", "2026-11"))["settled_micro_usd"] == 0


def test_overrun_records_true_amount_and_halts_external_funding_for_other_runs(ledger):
    operation = ledger.reserve(spec())
    ledger.begin_call(operation.operation_id)
    ledger.settle(operation.operation_id, 120, USAGE)
    assert all(c["halted"] for c in all_counters(ledger))
    assert funding(ledger)["settled_micro_usd"] == 120
    ledger.initialize_run("project", "tenant", "run2")
    with pytest.raises(BudgetDenied):
        ledger.reserve(spec("req2", "VOYAGE", run_id="run2", max_cost_micro_usd=1))


def test_explicit_halt_includes_external_counter_even_without_money_overrun(ledger):
    operation = ledger.reserve(spec())
    ledger.begin_call(operation.operation_id)
    ledger.settle(operation.operation_id, 1, USAGE)
    ledger.halt_operation_budgets(operation.operation_id)
    ledger.halt_operation_budgets(operation.operation_id)
    assert all(c["halted"] for c in all_counters(ledger))
    assert funding(ledger)["settled_micro_usd"] == 1


@pytest.mark.parametrize("cap", [0, -1, 10_000_001, True, 1.0])
def test_external_allowance_cannot_exceed_authorization_or_coerce_types(cap):
    value = Ledger(MemoryLedgerStore())
    with pytest.raises(ValueError):
        value.initialize_external_funding("project", cap_micro_usd=cap)
    assert value.store.get(value.external_funding_key("project")) is None


def test_parallel_providers_and_runs_admit_only_one_shared_reservation(ledger):
    ledger.initialize_run("project", "tenant", "run2", cap_micro_usd=1_000)

    def reserve(index):
        try:
            return ledger.reserve(spec(
                f"request{index}", "VOYAGE" if index % 2 else "DEEPSEEK",
                run_id="run2" if index % 2 else "run1",
            ))
        except BudgetDenied:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        admitted = [result for result in executor.map(reserve, range(24)) if result is not None]
    assert len(admitted) == 1
    assert funding(ledger)["reserved_micro_usd"] == 60
    assert ledger.store.get(ledger.month_key("project", "2026-10"))["reserved_micro_usd"] == 60
    assert sum(ledger.store.get(ledger.run_key(run))["reserved_micro_usd"] for run in (
        "run1", "run2",
    )) == 60


def test_conflict_on_fourth_transaction_item_cannot_partially_debit_other_counters(ledger):
    class FundingConflictStore(MemoryLedgerStore):
        def commit(self, changes):
            if len(changes) == 4:
                assert sum(change.key[0].startswith("FUNDING#") for change in changes) == 1
                assert sum(change.key[0].startswith("OP#") for change in changes) == 1
                raise ConcurrentWrite()
            super().commit(changes)

    failed = FundingConflictStore()
    failed._items = ledger.store._items.copy()
    ledger.store = failed
    with pytest.raises(BudgetDenied, match="bounded retries"):
        ledger.reserve(spec())
    assert [c["reserved_micro_usd"] for c in all_counters(ledger)] == [0, 0, 0]
    assert ledger.store.get(("OP#" + spec().operation_id, "STATE")) is None


def test_corrupt_external_cap_or_scope_fails_before_reservation(ledger):
    counter = funding(ledger)
    ledger.store.commit([Change(
        (counter["PK"], counter["SK"]), counter["version"],
        counter | {"version": counter["version"] + 1, "tenant_id": "wrong-tenant"},
    )])
    with pytest.raises(BudgetDenied, match="scope or cap"):
        ledger.reserve(spec())
    assert ledger.store.get(ledger.month_key("project", "2026-10"))["reserved_micro_usd"] == 0
