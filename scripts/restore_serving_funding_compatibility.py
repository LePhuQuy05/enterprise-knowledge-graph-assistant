"""Repair the incompatible shared cap change, preserving all spending/history."""

import argparse
from datetime import UTC, datetime

from app.contracts import canonical_json
from app.ledger import DynamoLedgerStore, Ledger
from app.ledger_allocation import restore_compatible_external_cap
from scripts.run_collection_benchmark import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("PLANNED_COMPATIBILITY_RESTORE_8_TO_2_USD_NO_COUNTER_RESET")
        return
    import boto3
    from botocore.config import Config

    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("ALLOCATION_CORRECTION_OPERATOR_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    month_key = ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m"))
    before_month = ledger.store.get(month_key)
    result = restore_compatible_external_cap(ledger, project="kg-r72", expected_cap=8000000, compatible_cap=2000000,
        correction_id="2026-10-26-legacy-gateway-exact-cap")
    after = ledger.store.get(ledger.external_funding_key("kg-r72"))
    if after["cap_micro_usd"] != 2000000 or before_month != ledger.store.get(month_key):
        raise ValueError("ALLOCATION_CORRECTION_READBACK_OR_MONTH_CHANGED")
    report = {"status": "LEGACY_SERVING_CAP_RESTORED", "result": result, "readback": after,
        "authorized_total_ceiling_micro_usd": 8000000, "active_counter_cap_micro_usd": 2000000,
        "additional_authorized_capacity_pending_compatible_release_micro_usd": 6000000,
        "available_under_active_cap_micro_usd": 2000000 - after["settled_micro_usd"] - after["reserved_micro_usd"],
        "month_unchanged": True, "new_model_calls": 0, "production_code_deployed": False,
        "checked_at": datetime.now(UTC).isoformat()}
    output = ROOT / "results/method-correctness-v2/allocation-compatibility-restore-v1.json"
    if not output.exists():
        output.write_bytes(canonical_json(report) + b"\n")
    print(report["status"], "active_cap=2USD", "authorized_ceiling=8USD", "spending_and_reservations_preserved")


if __name__ == "__main__":
    main()
