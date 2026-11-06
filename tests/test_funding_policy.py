import pytest

from app.funding_policy import (
    COMPATIBLE_AUTHORIZATION_ID,
    LEGACY_PROFILE,
    TRANSITION_PROFILE,
    FundingPolicyError,
    validate_external_funding,
)
from app.ledger import Change
from app.ledger_allocation import increase_external_allocation
from tests.test_ledger_allocation import setup


def increase(ledger, authorization=COMPATIBLE_AUTHORIZATION_ID):
    return increase_external_allocation(ledger, project="kg-r72", old_cap=2000000, new_cap=8000000,
        authorization_id=authorization, maximum_authorized_cap=8000000)


def test_default_and_transition_both_accept_existing_cap_without_any_write():
    ledger = setup()
    key = ledger.external_funding_key("kg-r72")
    before = ledger.store.get(key)
    assert validate_external_funding(ledger).cap_micro_usd == 2000000
    assert validate_external_funding(ledger, profile=TRANSITION_PROFILE).cap_micro_usd == 2000000
    assert ledger.store.get(key) == before


def test_eight_requires_opt_in_and_exact_new_authorization_not_historical_audit():
    ledger = setup()
    increase(ledger, "historical-pre-compatibility")
    with pytest.raises(FundingPolicyError, match="UNAPPROVED_CAP"):
        validate_external_funding(ledger, profile=LEGACY_PROFILE)
    with pytest.raises(FundingPolicyError, match="AUDIT_REQUIRED"):
        validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    valid = setup()
    increase(valid)
    row = validate_external_funding(valid, profile=TRANSITION_PROFILE)
    assert row.cap_micro_usd == 8000000 and row.settled_micro_usd == 385830 and row.reserved_micro_usd == 640


@pytest.mark.parametrize("change", [{"halted": True}, {"cap_micro_usd": 9000000}, {"tenant_id": "other"}, {"project_id": "other"}])
def test_transition_still_rejects_halt_scope_and_unapproved_cap(change):
    ledger = setup()
    increase(ledger)
    key = ledger.external_funding_key("kg-r72")
    row = ledger.store.get(key)
    ledger.store.commit([Change(key, row["version"], row | change | {"version": row["version"] + 1})])
    with pytest.raises(FundingPolicyError):
        validate_external_funding(ledger, profile=TRANSITION_PROFILE)


def test_owned_answer_policy_v1_shape_stays_legacy_and_v2_has_explicit_ceiling():
    from app.cloud_owned_answers import (
        CollectionAnswerPolicy,
        CollectionAnswerPolicyV2,
        load_collection_answer_policy,
    )

    old = CollectionAnswerPolicy().model_dump(mode="json")
    assert "funding_reader_profile" not in old and old["funding_cap_micro_usd"] == 2000000
    assert type(load_collection_answer_policy(old)) is CollectionAnswerPolicy
    new = CollectionAnswerPolicyV2().model_dump(mode="json")
    assert new["funding_cap_micro_usd"] == 8000000 and new["funding_reader_profile"] == TRANSITION_PROFILE
    assert type(load_collection_answer_policy(new)) is CollectionAnswerPolicyV2


def test_real_query_gateway_transition_is_explicit_and_never_reads_provider_before_audit(monkeypatch):
    from app import query_handler
    from app.query_auth import ApiError
    from app.query_service import RUN

    ledger = setup()
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
    increase(ledger)
    monkeypatch.setenv("MODEL_ENABLED", "true")
    monkeypatch.delenv("KG_FUNDING_POLICY", raising=False)
    monkeypatch.setattr(query_handler, "clients", lambda: {"dynamodb": None, "secretsmanager": None})
    monkeypatch.setattr(query_handler, "DynamoLedgerStore", lambda *args: ledger.store)
    monkeypatch.setenv("KG_DEEPSEEK_SECRET_ARN", "fixture-never-read")
    def boundary(*args, **kwargs):
        raise RuntimeError("SAFE_PROVIDER_BOUNDARY")
    monkeypatch.setattr(query_handler, "read_deepseek_secret", boundary)
    with pytest.raises(ApiError, match="BUDGET_LIMIT"):
        query_handler.gateway(1)
    monkeypatch.setenv("KG_FUNDING_POLICY", TRANSITION_PROFILE)
    with pytest.raises(RuntimeError, match="SAFE_PROVIDER_BOUNDARY"):
        query_handler.gateway(1)
