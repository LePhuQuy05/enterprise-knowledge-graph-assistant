"""Explicit reader profiles for a two-phase external funding migration.

Loading this code never changes a counter. Legacy readers stay at 2 USD until
an operator deploys the transition profile and later writes a matching audit.
"""

from app.ledger import Counter

LEGACY_PROFILE = "LEGACY_2_USD"
TRANSITION_PROFILE = "AUDITED_2_TO_8_USD_V2"
COMPATIBLE_AUTHORIZATION_ID = "user-2026-10-26-full-graph-up-to-8-usd-compatible-v2"


class FundingPolicyError(ValueError):
    pass


def validate_external_funding(ledger, *, profile=LEGACY_PROFILE, project="kg-r72"):
    if profile not in {LEGACY_PROFILE, TRANSITION_PROFILE} or project != "kg-r72":
        raise FundingPolicyError("FUNDING_READER_PROFILE")
    key = ledger.external_funding_key(project)
    raw = ledger.store.get(key)
    try:
        row = Counter.model_validate(raw)
    except (ValueError, TypeError):
        raise FundingPolicyError("FUNDING_COUNTER_INVALID") from None
    if (row.project_id != project or row.tenant_id is not None or row.period is not None
            or (row.PK, row.SK) != key or row.halted):
        raise FundingPolicyError("FUNDING_COUNTER_SCOPE_OR_HALTED")
    if row.cap_micro_usd == 2000000:
        return row
    if profile != TRANSITION_PROFILE or row.cap_micro_usd != 8000000:
        raise FundingPolicyError("FUNDING_UNAPPROVED_CAP")
    audit_key = (key[0], "ALLOCATION#" + COMPATIBLE_AUTHORIZATION_ID)
    audit = ledger.store.get(audit_key)
    if (not audit or (audit.get("PK"), audit.get("SK")) != audit_key
            or audit.get("record_type") != "AUTHORIZED_ALLOCATION_CHANGE" or audit.get("project_id") != project
            or audit.get("authorization_id") != COMPATIBLE_AUTHORIZATION_ID or audit.get("old_cap") != 2000000
            or audit.get("new_cap") != 8000000 or audit.get("maximum_authorized_cap") != 8000000):
        raise FundingPolicyError("FUNDING_AUTHORIZATION_AUDIT_REQUIRED")
    return row
