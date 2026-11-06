"""Operator-only, audited cap increase after explicit user authorization.

This does not initialize/reset counters, settle reservations, unhalt spending,
increase monthly limits or expose an allocation API to public users.
"""

from app.contracts import canonical_json, sha256_bytes
from app.ledger import Change, ConcurrentWrite, Counter, IdempotencyConflict


def increase_external_allocation(ledger, *, project, old_cap, new_cap, authorization_id, maximum_authorized_cap):
    if (type(old_cap) is not int or type(new_cap) is not int or type(maximum_authorized_cap) is not int
            or not 0 < old_cap < new_cap <= maximum_authorized_cap <= 10000000
            or not authorization_id or len(authorization_id) > 128):
        raise ValueError("ALLOCATION_AUTHORIZED_CAP_REQUIRED")
    key = ledger.external_funding_key(project)
    audit_key = (key[0], "ALLOCATION#" + authorization_id)
    for _ in range(4):
        raw = ledger.store.get(key)
        if raw is None:
            raise ValueError("ALLOCATION_EXISTING_FUNDING_REQUIRED")
        current = Counter.model_validate(raw)
        if (current.project_id != project or current.tenant_id is not None or current.period is not None
                or current.halted or (current.PK, current.SK) != key):
            raise ValueError("ALLOCATION_SCOPE_OR_HALTED")
        audit = ledger.store.get(audit_key)
        if audit is not None:
            if (audit.get("old_cap") != old_cap or audit.get("new_cap") != new_cap
                    or audit.get("maximum_authorized_cap") != maximum_authorized_cap or current.cap_micro_usd != new_cap):
                raise IdempotencyConflict("ALLOCATION_AUTHORIZATION_CONFLICT")
            return {"changed": False, "counter": current.model_dump(mode="json"), "audit": audit}
        if current.cap_micro_usd != old_cap:
            raise IdempotencyConflict("ALLOCATION_PRIOR_CAP_CHANGED")
        updated = current.model_copy(update={"version": current.version + 1, "cap_micro_usd": new_cap})
        updated = Counter.model_validate(updated.model_dump())
        audit = {"PK": audit_key[0], "SK": audit_key[1], "version": 1, "record_type": "AUTHORIZED_ALLOCATION_CHANGE",
            "project_id": project, "authorization_id": authorization_id, "old_cap": old_cap, "new_cap": new_cap,
            "maximum_authorized_cap": maximum_authorized_cap, "prior_counter_sha256": sha256_bytes(canonical_json(raw)),
            "new_counter_sha256": sha256_bytes(canonical_json(updated.model_dump(mode="json"))),
            "settled_preserved": updated.settled_micro_usd, "reserved_preserved": updated.reserved_micro_usd}
        try:
            ledger.store.commit([Change(key, current.version, updated.model_dump(mode="json")), Change(audit_key, None, audit)])
            return {"changed": True, "before": raw, "counter": updated.model_dump(mode="json"), "audit": audit}
        except ConcurrentWrite:
            continue
    raise ConcurrentWrite("ALLOCATION_CONCURRENT_UPDATE_RETRY_EXHAUSTED")


def restore_compatible_external_cap(ledger, *, project, expected_cap, compatible_cap, correction_id):
    """Undo an incompatible increase without erasing the authorization or spend.

    Operator repair only. Never lowers the cap below committed/unknown spending,
    never alters halt state, and writes a separate immutable correction audit.
    """
    if (type(expected_cap) is not int or type(compatible_cap) is not int
            or not 0 < compatible_cap < expected_cap <= 10000000 or not correction_id or len(correction_id) > 128):
        raise ValueError("ALLOCATION_CORRECTION_SCOPE")
    key = ledger.external_funding_key(project)
    audit_key = (key[0], "ALLOCATION_CORRECTION#" + correction_id)
    for _ in range(4):
        raw = ledger.store.get(key)
        if raw is None:
            raise ValueError("ALLOCATION_EXISTING_FUNDING_REQUIRED")
        current = Counter.model_validate(raw)
        if (current.project_id != project or current.tenant_id is not None or current.period is not None
                or (current.PK, current.SK) != key):
            raise ValueError("ALLOCATION_CORRECTION_SCOPE")
        prior = ledger.store.get(audit_key)
        if prior is not None:
            if prior.get("to_cap") != compatible_cap or prior.get("from_cap") != expected_cap or current.cap_micro_usd != compatible_cap:
                raise IdempotencyConflict("ALLOCATION_CORRECTION_CONFLICT")
            return {"changed": False, "counter": raw, "audit": prior}
        if current.cap_micro_usd != expected_cap:
            raise IdempotencyConflict("ALLOCATION_CORRECTION_PRIOR_CAP_CHANGED")
        if current.settled_micro_usd + current.reserved_micro_usd > compatible_cap:
            raise ValueError("ALLOCATION_CORRECTION_BELOW_COMMITTED_SPENDING")
        updated = current.model_copy(update={"cap_micro_usd": compatible_cap, "version": current.version + 1})
        audit = {"PK": audit_key[0], "SK": audit_key[1], "version": 1, "record_type": "ALLOCATION_COMPATIBILITY_CORRECTION",
            "project_id": project, "correction_id": correction_id, "from_cap": expected_cap, "to_cap": compatible_cap,
            "reason": "LEGACY_SERVING_REQUIRES_EXACT_2_USD_CAP_AUTHORIZED_8_USD_REQUIRES_COMPATIBLE_RELEASE",
            "prior_counter_sha256": sha256_bytes(canonical_json(raw)), "settled_preserved": current.settled_micro_usd,
            "reserved_preserved": current.reserved_micro_usd, "halted_preserved": current.halted}
        try:
            ledger.store.commit([Change(key, current.version, updated.model_dump(mode="json")), Change(audit_key, None, audit)])
            return {"changed": True, "before": raw, "counter": updated.model_dump(mode="json"), "audit": audit}
        except ConcurrentWrite:
            continue
    raise ConcurrentWrite("ALLOCATION_CORRECTION_CONCURRENT_UPDATE_RETRY_EXHAUSTED")
