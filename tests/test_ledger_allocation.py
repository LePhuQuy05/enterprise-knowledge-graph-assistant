import pytest

from app.ledger import Change, IdempotencyConflict, Ledger, MemoryLedgerStore
from app.ledger_allocation import increase_external_allocation, restore_compatible_external_cap


def setup():
    ledger = Ledger(MemoryLedgerStore())
    ledger.initialize_external_funding("kg-r72", cap_micro_usd=2000000)
    ledger.initialize_month("kg-r72", "2026-10")
    key = ledger.external_funding_key("kg-r72")
    row = ledger.store.get(key)
    ledger.store.commit([Change(key, row["version"], row | {"version": row["version"] + 1,
        "settled_micro_usd": 385830, "reserved_micro_usd": 640})])
    return ledger


def change(ledger, **kwargs):
    return increase_external_allocation(ledger, **(dict(project="kg-r72", old_cap=2000000, new_cap=8000000,
        authorization_id="user-authorized-v1", maximum_authorized_cap=8000000) | kwargs))


def test_operator_increase_preserves_spend_unknown_reservations_and_month_limit():
    ledger = setup()
    month_before = ledger.store.get(ledger.month_key("kg-r72", "2026-10"))
    result = change(ledger)
    assert result["changed"] and result["counter"]["cap_micro_usd"] == 8000000
    assert result["counter"]["settled_micro_usd"] == 385830 and result["counter"]["reserved_micro_usd"] == 640
    assert ledger.store.get(ledger.month_key("kg-r72", "2026-10")) == month_before
    assert not change(ledger)["changed"]


def test_cannot_exceed_authorized_cap_or_reuse_authorization_for_another_amount():
    ledger = setup()
    with pytest.raises(ValueError, match="AUTHORIZED_CAP"):
        change(ledger, new_cap=9000000)
    change(ledger)
    with pytest.raises(IdempotencyConflict):
        change(ledger, new_cap=7000000)


def test_allocation_never_unhalts_or_erases_a_counter_needing_reconciliation():
    ledger = setup()
    key = ledger.external_funding_key("kg-r72")
    row = ledger.store.get(key)
    ledger.store.commit([Change(key, row["version"], row | {"version": row["version"] + 1, "halted": True})])
    with pytest.raises(ValueError, match="HALTED"):
        change(ledger)
    assert ledger.store.get(key)["reserved_micro_usd"] == 640


def test_compatibility_restore_preserves_accounting_halt_and_authorization_audit():
    ledger = setup()
    change(ledger)
    key = ledger.external_funding_key("kg-r72")
    row = ledger.store.get(key)
    ledger.store.commit([Change(key, row["version"], row | {"version": row["version"] + 1, "halted": True})])
    result = restore_compatible_external_cap(ledger, project="kg-r72", expected_cap=8000000,
        compatible_cap=2000000, correction_id="compat-v1")
    assert result["counter"]["halted"]
    assert result["counter"]["cap_micro_usd"] == 2000000
    assert result["counter"]["settled_micro_usd"] == 385830 and result["counter"]["reserved_micro_usd"] == 640
    assert ledger.store.get((key[0], "ALLOCATION#user-authorized-v1"))["new_cap"] == 8000000
    assert not restore_compatible_external_cap(ledger, project="kg-r72", expected_cap=8000000,
        compatible_cap=2000000, correction_id="compat-v1")["changed"]


def test_restore_never_lowers_cap_below_settled_plus_reserved():
    ledger = setup()
    change(ledger)
    with pytest.raises(ValueError, match="COMMITTED_SPENDING"):
        restore_compatible_external_cap(ledger, project="kg-r72", expected_cap=8000000,
            compatible_cap=100000, correction_id="bad-low")
    assert ledger.store.get(ledger.external_funding_key("kg-r72"))["cap_micro_usd"] == 8000000


def test_legacy_query_gateway_accepts_restored_cap_before_any_provider_boundary(monkeypatch):
    from app import query_handler
    from app.query_auth import ApiError
    from app.query_service import RUN

    ledger = setup()
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
    change(ledger)
    monkeypatch.setenv("MODEL_ENABLED", "true")
    monkeypatch.setattr(query_handler, "clients", lambda: {"dynamodb": None, "secretsmanager": None})
    monkeypatch.setattr(query_handler, "DynamoLedgerStore", lambda *args: ledger.store)
    monkeypatch.setenv("KG_DEEPSEEK_SECRET_ARN", "fixture-secret-never-read")
    def boundary(*args, **kwargs):
        raise RuntimeError("SAFE_TEST_PROVIDER_BOUNDARY_REACHED")
    monkeypatch.setattr(query_handler, "read_deepseek_secret", boundary)
    with pytest.raises(ApiError, match="BUDGET_LIMIT"):
        query_handler.gateway(1)
    restore_compatible_external_cap(ledger, project="kg-r72", expected_cap=8000000,
        compatible_cap=2000000, correction_id="gateway-compat")
    with pytest.raises(RuntimeError, match="SAFE_TEST_PROVIDER_BOUNDARY_REACHED"):
        query_handler.gateway(1)
