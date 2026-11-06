"""Freeze two development demos, then optionally run at most three model stages.

Uses the existing 0.50 USD local run and shared 2 USD funding without reset/raise.
No infrastructure deployment or automatic paid retries.
"""

import argparse
import json
from datetime import UTC, datetime

from app.contracts import canonical_json, sha256_bytes
from app.external_http import deepseek_preflight
from scripts.answer_local import ROOT, RUN, managed_gateway
from scripts.curate_collection_graph import write_immutable
from scripts.reconcile_collection_graph_pilot import require
from scripts.run_reviewed_collection_graph import main as run_case

CASES = (
    {"id": "nat-en-g1", "question": "What protocols does a NAT gateway support, and what is NAT64 used for?", "hops": 1, "language": "en"},
    {"id": "privatelink-vi-g2", "question": "Để dùng AWS PrivateLink, cần tạo tài nguyên nào và VPC endpoint service liên quan đến PrivateLink như thế nào?", "hops": 2, "language": "vi"},
)
FILES = ("app/collection_graph.py", "app/collection_graph_runtime.py", "app/collection_graph_runner.py",
    "scripts/run_reviewed_collection_graph.py", "scripts/pilot_reviewed_graph_answers.py")


def freeze():
    curation_raw = (ROOT / "results/collection-graph/curated-subset-v1.json").read_bytes()
    curation = json.loads(curation_raw)
    plan = {"status": "FROZEN_SOURCE_INSPECTION_DEMOS_NOT_QUALITY_BENCHMARK", "run_id": RUN,
        "graph_receipt_sha256": sha256_bytes(curation_raw), "graph_file_sha256": curation["graph_file_sha256"],
        "collection_manifest_sha256": curation["collection_manifest_sha256"], "cases": CASES,
        "maximum_model_dispatches": 3, "existing_run_cap_micro_usd": 500000, "existing_funding_cap_micro_usd": 2000000,
        "observed_stop_between_cases_micro_usd": 50000, "initialize_allowance": False, "funding_cap_raise": False,
        "transport_retries": 0, "automatic_paid_retries": False, "human_reviewed": False, "cloud_deployments": 0,
        "selection_reason": "Development demonstrations chosen for source inspection, not random held-out performance estimates.",
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "implementation_sha256": {name: sha256_bytes((ROOT / name).read_bytes()) for name in FILES}}
    raw = canonical_json(plan) + b"\n"
    write_immutable(ROOT / "results/collection-graph/answer-pilot-plan-v1.json", raw)
    return plan, raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan, plan_raw = freeze()
    if not args.execute:
        print(json.dumps({"status": plan["status"], "model_calls": 0, "aws_calls": 0}))
        return
    marker = ROOT / "results/collection-graph/answer-pilot-started-v1.json"
    require(not marker.exists(), "GRAPH_ANSWER_PILOT_ALREADY_STARTED_NO_AUTOMATIC_RETRY")
    gateway = managed_gateway(prices_path=ROOT / "configs/prices/external-direct-2026-10-24.json")
    provider = deepseek_preflight(gateway.provider)
    require(provider["metadata_sha256"] == gateway.routes.account_model_metadata_sha256, "GRAPH_ANSWER_PROVIDER_CHANGED")
    ledger = gateway.ledger
    before = {name: ledger.store.get(key) for name, key in (
        ("run", ledger.run_key(RUN)), ("funding", ledger.external_funding_key("kg-r72")),
        ("month", ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m"))))}
    require(before["month"] and not before["month"]["halted"] and before["month"]["cap_micro_usd"] == 12000000,
        "GRAPH_ANSWER_MONTH_REQUIRED")
    write_immutable(marker, canonical_json({"plan_sha256": sha256_bytes(plan_raw), "started_at": datetime.now(UTC).isoformat(),
        "before": before, "provider": provider, "automatic_retry": False}) + b"\n")
    summary = {"status": "RUNNING", "plan_sha256": sha256_bytes(plan_raw), "cases": [],
        "new_cost_micro_usd": 0, "model_dispatches": 0, "cloud_deployments": 0}
    output = ROOT / "results/collection-graph/answer-pilot-summary-v1.json"
    try:
        for case in plan["cases"]:
            require(summary["new_cost_micro_usd"] < plan["observed_stop_between_cases_micro_usd"], "GRAPH_ANSWER_OBSERVED_STOP")
            receipt = ROOT / "results/collection-graph" / ("answer-" + case["id"] + "-v1.json")
            run_case([case["question"], "--hops", str(case["hops"]), "--language", case["language"], "--run-kind", "answer",
                "--allow-experimental", "--execute", "--idempotency-key", "graph-demo-" + case["id"] + "-v1", "--output", str(receipt)])
            row = json.loads(receipt.read_bytes())
            require(row["status"] == "COMPLETED", "GRAPH_ANSWER_CASE_NOT_COMPLETE")
            summary["model_dispatches"] += row["model_dispatch_attempts_this_invocation"]
            summary["new_cost_micro_usd"] += row["new_cost_micro_usd"]
            require(summary["model_dispatches"] <= plan["maximum_model_dispatches"], "GRAPH_ANSWER_DISPATCH_LIMIT")
            summary["cases"].append({"id": case["id"], "receipt_sha256": sha256_bytes(receipt.read_bytes()),
                "requested": row["requested_method"], "effective": row["effective_method"],
                "answer_status": row["response"]["answer_status"], "claims": len(row["response"]["claims"]),
                "citations": len(row["response"]["citations"]), "cost_micro_usd": row["new_cost_micro_usd"]})
            output.write_bytes(canonical_json(summary) + b"\n")
        summary["status"] = "COMPLETED_SOURCE_AND_COST_RECONCILIATION_PENDING"
        summary["after"] = {name: ledger.store.get((row["PK"], row["SK"])) for name, row in before.items()}
    except Exception as exc:
        from app.model_gateway import safe_failure_code

        summary["status"], summary["error_category"] = "STOPPED_NO_AUTOMATIC_RETRY", safe_failure_code(exc)
    finally:
        output.write_bytes(canonical_json(summary) + b"\n")
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
