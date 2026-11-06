"""Apply the user's 8 USD total allocation without resetting shared spending."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config

from app.contracts import canonical_json, sha256_bytes
from app.ledger import DynamoLedgerStore, Ledger
from app.ledger_allocation import increase_external_allocation

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    raw = (ROOT / "configs/budgets/full-graph-v2-allocation.json").read_bytes()
    auth = json.loads(raw)
    if (auth["project_id"] != "kg-r72" or auth["authorized_external_funding_cap_micro_usd"] != 8000000
            or auth["previous_external_funding_cap_micro_usd"] != 2000000
            or auth["includes_previously_settled_and_reserved"] is not True):
        raise ValueError("FULL_GRAPH_ALLOCATION_SCOPE_CHANGED")
    if not args.execute:
        print(json.dumps({"status": "AUTHORIZED_PLAN_ONLY", "total_cap_usd": 8, "aws_writes": 0}))
        return
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("FULL_GRAPH_ALLOCATION_DEVELOPER_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    month_key = ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m"))
    before_month = ledger.store.get(month_key)
    if not before_month or before_month["cap_micro_usd"] != 12000000 or before_month["halted"]:
        raise ValueError("FULL_GRAPH_EXISTING_MONTH_NOT_READY")
    result = increase_external_allocation(ledger, project="kg-r72", old_cap=2000000, new_cap=8000000,
        maximum_authorized_cap=8000000, authorization_id=auth["authorization_id"])
    after = ledger.store.get(ledger.external_funding_key("kg-r72"))
    if after["cap_micro_usd"] != 8000000:
        raise ValueError("FULL_GRAPH_ALLOCATION_READBACK_FAILED")
    after_month = ledger.store.get(month_key)
    # Other active callers may settle, but this operation must not change month policy.
    for field in ("cap_micro_usd", "gross_stop_micro_usd", "other_cost_estimate_micro_usd", "halted"):
        if before_month.get(field) != after_month.get(field):
            raise ValueError("FULL_GRAPH_MONTH_POLICY_CHANGED")
    output = ROOT / "results/method-correctness-v2/allocation-8usd-v1.json"
    report = {"status": "AUTHORIZED_ALLOCATION_VERIFIED", "authorization_sha256": sha256_bytes(raw),
        "changed": result["changed"], "result": result, "readback": after, "month_policy_unchanged": True,
        "new_model_calls": 0, "external_top_up": False, "production_deployed": False}
    if not output.exists():
        output.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({"status": report["status"], "changed": result["changed"],
        "total_cap_micro_usd": after["cap_micro_usd"], "settled_micro_usd": after["settled_micro_usd"],
        "reserved_micro_usd": after["reserved_micro_usd"]}))


if __name__ == "__main__":
    main()
