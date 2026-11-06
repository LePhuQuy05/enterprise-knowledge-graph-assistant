"""Read-only reconciliation of this checkpoint's settled model operations."""

import json
from datetime import UTC, datetime

from app.contracts import canonical_json
from scripts.run_collection_benchmark import ROOT


def main():
    import boto3
    from botocore.config import Config

    from app.ledger import DynamoLedgerStore, Ledger

    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("STRICT_RECONCILE_OPERATOR_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    records = []
    for run_id, operations, cost in (("strict-method-preparation-v2", 8, 1121), ("strict-method-preparation-v3", 2, 251),
            ("collection316-full-graph-v2-qualification", 45, 22468), ("strict-method-answers-v1", 3, 3775)):
        row = ledger.store.get(ledger.run_key(run_id))
        if not row or row["halted"] or row["reserved_micro_usd"] or row["settled_micro_usd"] != cost:
            raise ValueError("STRICT_CHECKPOINT_RUN_RECONCILIATION")
        records.append({"run_id": run_id, "new_operations": operations, "settled_micro_usd": cost, "counter": row})
    funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    if (not funding or funding["halted"] or funding["cap_micro_usd"] != 2000000
            or not month or month["halted"] or month["cap_micro_usd"] != 12000000):
        raise ValueError("STRICT_CHECKPOINT_POLICY_CHANGED")
    report = {"status": "RUN_COUNTERS_MATCH_RECORDED_RECEIPT_TOTALS", "new_operations": sum(r["new_operations"] for r in records),
        "new_cost_micro_usd": sum(r["settled_micro_usd"] for r in records), "runs": records,
        "funding": funding, "month": month, "remaining_funding_micro_usd": funding["cap_micro_usd"] - funding["reserved_micro_usd"] - funding["settled_micro_usd"],
        "authorized_total_ceiling_micro_usd": 8000000, "active_funding_cap_micro_usd": 2000000,
        "additional_authorized_capacity_pending_compatible_release_micro_usd": 6000000,
        "note": "Counter totals match cached experiment receipts. This command does not read or certify every historic operation or invoice.",
        "checked_at": datetime.now(UTC).isoformat(), "aws_writes": 0, "provider_calls": 0}
    (ROOT / "results/method-correctness-v2/cost-reconciliation-v2.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "new_operations", "new_cost_micro_usd", "remaining_funding_micro_usd")}))


if __name__ == "__main__":
    main()
