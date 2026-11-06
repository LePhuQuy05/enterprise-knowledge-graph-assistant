"""One explicit, audited recovery of tranche001/job3 after zero dispatch.

Original FAILED cache claim and attempt remain intact. No retry on any other
failure. Default is preflight/read-only; --execute has an isolated one-shot DB.
"""

import argparse
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime

from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes, stable_id
from app.semantic_extraction_cache_v7 import SemanticExtractionCache
from app.semantic_extraction_v8_review import review_output
from app.source_candidate_policy import retained
from app.source_graph_jobs import context_for, managed_source_gateway
from app.source_job_recovery import validate_zero_dispatch_proof
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

DIRECTORY = ROOT / "results/semantic-graph-v7/tranche-001"
RECOVERY = DIRECTORY / "recovery-v1"


def failed_claim(unit):
    path = ROOT / "data/cache/semantic-extraction-v7/semantic_extractions_v7.sqlite3"
    rows = []
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
        for key, binding, owner, state, payload, psha, error in db.execute(
                "SELECT cache_key,binding,owner,state,payload,payload_sha,error FROM extractions WHERE state='FAILED'"):
            if json.loads(binding)["source"] == unit["text_sha256"]:
                rows.append({"cache_key": key, "binding_sha256": sha256_bytes(binding), "owner": owner,
                    "state": state, "payload_present": payload is not None, "payload_sha256": psha, "error": error})
    return rows


def recover(*, execute=False):
    import boto3
    from botocore.config import Config

    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    job = DIRECTORY / "job-3"
    plan = json.loads((job / "plan.json").read_bytes())
    attempt = json.loads((job / "attempt-001.json").read_bytes())
    if (attempt["error"], attempt["processed"], attempt["planned"], attempt["new_cost_micro_usd"]) != ("BudgetDenied", 22, 33, 15131):
        raise ValueError("SOURCE_RECOVERY_EXACT_PRIOR_ATTEMPT_REQUIRED")
    for path, expected in plan["code_sha256"].items():
        if sha256_bytes((ROOT / path).read_bytes()) != expected:
            raise ValueError("SOURCE_RECOVERY_FROZEN_CODE_CHANGED")
    _, chunks, _ = load_collection_inputs()
    recipe = ExtractionRecipe.model_validate_json(canonical_json(plan["recipe"]))
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=20)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("SOURCE_RECOVERY_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    context = context_for(plan["selection"][22], plan["run_id"])
    op = stable_id("op", **context.model_dump())
    proof = {"checked_at": datetime.now(UTC).isoformat(), "context": context.model_dump(mode="json"), "operation_id": op,
        "operation": ledger.store.get(("OP#" + op, "STATE")), "failed_cache": failed_claim(plan["selection"][22]),
        "runs": {"3": ledger.store.get(ledger.run_key(plan["run_id"]))},
        "funding": validate_external_funding(ledger, profile=TRANSITION_PROFILE).model_dump(mode="json"),
        "prior_attempt_sha256": sha256_bytes((job / "attempt-001.json").read_bytes()),
        "recovery_code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in
            ("app/source_job_recovery.py", "scripts/recover_source_graph_tranche.py")},
        "cause": "RESERVATION_DENIED_ZERO_DISPATCH; EXACT_CONCURRENT_RETRY_VS_ALLOWANCE_MESSAGE_NOT_RECORDED"}
    validate_zero_dispatch_proof(proof, plan, 22)
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v7", namespace="LIVE")
    previous = []
    for unit in plan["selection"][:22]:
        lookup = cache.resolve(context_for(unit, plan["run_id"]), text=chunks[unit["occurrences"][0]["chunk_id"]].text, recipe=recipe)
        operation = ledger.operation(lookup.receipt["operation_id"])
        if operation.status != "SETTLED" or operation.actual_micro_usd != lookup.receipt["cost_micro_usd"]:
            raise ValueError("SOURCE_RECOVERY_PRIOR_RECEIPT_LEDGER_MISMATCH")
        previous.append(lookup.receipt)
    if sum(r["cost_micro_usd"] for r in previous) != proof["runs"]["3"]["settled_micro_usd"]:
        raise ValueError("SOURCE_RECOVERY_PRIOR_RUN_COST_MISMATCH")
    RECOVERY.mkdir(exist_ok=True)
    if not execute:
        (RECOVERY / "preflight.json").write_bytes(canonical_json(proof) + b"\n")
        return {"status": "ZERO_DISPATCH_PREFLIGHT_PASS", "model_calls": 0, "remaining_sources": 11}
    proof_path = RECOVERY / "execution-proof.json"
    with proof_path.open("xb") as handle:
        handle.write(canonical_json(proof) + b"\n")
    # Operator-authorized replacement cache for the ONE proven zero-dispatch
    # source only; its old FAILED row remains unchanged and is bound above.
    replacement = SemanticExtractionCache(RECOVERY / "replacement-cache", namespace="LIVE")
    subplan = dict(plan, selection=plan["selection"][22:], maximum_new_calls=11)
    gateway, ledger = managed_source_gateway(ROOT, subplan, chunks)
    validate_zero_dispatch_proof(proof | {"operation": ledger.store.get(("OP#" + op, "STATE"))}, plan, 22)
    rows = []
    for index, unit in enumerate(subplan["selection"]):
        text = chunks[unit["occurrences"][0]["chunk_id"]].text
        if sha256_bytes(text.encode()) != unit["text_sha256"]:
            raise ValueError("SOURCE_RECOVERY_SOURCE_CHANGED")
        lookup = (replacement if index == 0 else cache).resolve(context_for(unit, plan["run_id"]),
            text=text, recipe=recipe, gateway=gateway)
        decisions = []
        for decision in review_output(lookup.validation.output, text):
            allowed, reason = retained(decision)
            decisions.append(dict(decision, retention_status="RETAINED_PENDING_SOURCE_REVIEW" if allowed else "QUARANTINED", retention_reason=reason))
        row = {"unit_id": unit["v7_unit_id"], "text_sha256": unit["text_sha256"], "occurrences": unit["occurrences"],
            "receipt": lookup.receipt, "coverage_note": lookup.validation.output.coverage_note, "decisions": decisions,
            "composite_recipe_sha256": sha256_bytes(canonical_json(plan["composite_recipe"])), "semantic_review": "PENDING_NOT_SERVING"}
        with (job / "units" / (unit["v7_unit_id"] + ".json")).open("xb") as handle:
            handle.write(canonical_json(row) + b"\n")
        rows.append(row)
        print(json.dumps({"recovered_sources": len(rows), "remaining": 11 - len(rows), "new_cost_micro_usd": gateway.cost}), flush=True)
    run = ledger.store.get(ledger.run_key(plan["run_id"]))
    receipts = previous + [r["receipt"] for r in rows]
    if (run["reserved_micro_usd"] or run["settled_micro_usd"] != sum(r["cost_micro_usd"] for r in receipts)
            or failed_claim(plan["selection"][22]) != proof["failed_cache"]):
        raise ValueError("SOURCE_RECOVERY_FINAL_RECONCILIATION")
    report = {"status": "RECOVERED_11_COMPLETE_JOB_33_NOT_SERVING", "processed": 33, "new_sources": len(rows),
        "new_cost_micro_usd": gateway.cost, "provider_reservation_attempts": len(gateway.seen), "run_counter": run,
        "operation_ids": [r["operation_id"] for r in receipts], "original_failed_claim_preserved": True,
        "coverage_notes_new_sources": dict(Counter(r["coverage_note"] for r in rows)), "serving_records_added": 0,
        "funding": validate_external_funding(ledger, profile=TRANSITION_PROFILE).model_dump(mode="json"),
        "finished_at": datetime.now(UTC).isoformat()}
    (RECOVERY / "completion.json").write_bytes(canonical_json(report) + b"\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(recover(execute=args.execute)))
