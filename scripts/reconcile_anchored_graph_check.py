"""Reconcile both frozen graph experiments without model calls or ledger writes."""

import json
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.external_gateway import ExternalPriceSheet
from app.extraction import canonicalize, extraction_request, raw_cache_key, validate_output
from app.ledger import Operation
from scripts.check_anchored_graph_extraction import RUN, freeze
from scripts.reconcile_collection_graph_pilot import (
    audit,
    load_inputs,
    managed_ledger,
    read_claims,
    require,
    verify_operation,
)
from scripts.run_collection_graph_pilot import ROOT


def reconcile():
    loaded = load_inputs()
    _, _, docs, chunks = loaded
    ledger = managed_ledger()
    original = audit(ledger, loaded=loaded, include_continuation=True)
    plan_raw, plan, recipe, aliases, _ = freeze(loaded)
    report_raw = (ROOT / "results/collection-graph/anchored-v4-check.json").read_bytes()
    report = json.loads(report_raw)
    require(report["status"] == "DEVELOPMENT_CHECK_COMPLETE_SEMANTIC_REVIEW_PENDING"
        and report["plan_sha256"] == sha256_bytes(plan_raw) and report["model_dispatch_attempts"] == 4,
        "GRAPH_V4_REPORT_BINDING")
    canonical_raw = Path(report["canonical_path"]).read_bytes()
    require(sha256_bytes(canonical_raw) == report["canonical_sha256"], "GRAPH_V4_CANONICAL_HASH")
    canonical = {r["chunk_id"]: r for r in map(json.loads, canonical_raw.splitlines())}
    accepted = {r["chunk_id"] for r in report["extractions"]}
    rejected = {r["chunk_id"] for r in report["quarantined"]}
    require(accepted == set(canonical) and not accepted & rejected, "GRAPH_V4_PARTITION")
    claims = read_claims(run_id=RUN, request_prefix="graph316-v4-")
    require(set(claims) == {r["chunk_id"] for r in plan["selection"]} == accepted | rejected,
        "GRAPH_V4_CLAIMS_SCOPE")
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    operations = []
    for key, claim in claims.items():
        context, receipts = claim["context"], claim["receipts"]
        require(len(receipts) == 1 and claim["cache_key"] == raw_cache_key("demo", chunks[key].text, recipe), "GRAPH_V4_CACHE_IDENTITY")
        opid = stable_id("op", **context.model_dump())
        operation = Operation.model_validate_json(canonical_json(ledger.store.get(("OP#" + opid, "STATE"))))
        system, prompt = extraction_request(chunks[key].text, recipe)
        body = canonical_json({"model": "deepseek-flash", "messages": [{"role": "system", "content": system},
            {"role": "user", "content": prompt}], "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": 2048, "stream": False, "response_format": {"type": "json_object"}})
        verify_operation(operation, context, receipts[0], sha256_bytes(body), prices, run_id=RUN)
        if key in accepted:
            require(claim["state"] == "READY", "GRAPH_V4_ACCEPTED_STATE")
            validated = validate_output(receipts[0].text, chunk=chunks[key], document=docs[chunks[key].document_version_id],
                stop_reason=receipts[0].stop_reason, recipe=recipe)
            restored = canonicalize(validated, chunk=chunks[key], document=docs[chunks[key].document_version_id], recipe=recipe, aliases=aliases)
            require(all([r.model_dump(mode="json") for r in getattr(restored, name)] == canonical[key][name]
                for name in ("entities", "mentions", "assertions")), "GRAPH_V4_CANONICAL_REPLAY")
        else:
            require(claim["state"] == "FAILED", "GRAPH_V4_REJECTED_STATE")
        operations.append({"operation_id": opid, "chunk_id": key, "status": operation.status,
            "actual_micro_usd": operation.actual_micro_usd,
            "receipt_sha256": sha256_bytes(canonical_json(receipts[0].model_dump(mode="json")))})
    total = sum(o["actual_micro_usd"] for o in operations)
    run = ledger.store.get(ledger.run_key(RUN))
    require(total == report["new_cost_micro_usd"] == run["settled_micro_usd"] and run["reserved_micro_usd"] == 0
        and run["cap_micro_usd"] == 500000 and not run["halted"], "GRAPH_V4_FINAL_COUNTERS")
    return {"status": "PASS_BOTH_GRAPH_EXPERIMENTS_SOURCE_RECEIPT_USAGE_MANAGED_LEDGER",
        "checked_at": datetime.now(UTC).isoformat(), "original_v3": original,
        "anchored_v4": {"run": run, "operations": operations, "accepted_chunks": len(accepted),
            "quarantined_chunks": len(rejected), "assertions": sum(len(r["assertions"]) for r in canonical.values()),
            "plan_sha256": sha256_bytes(plan_raw), "report_sha256": sha256_bytes(report_raw),
            "canonical_sha256": sha256_bytes(canonical_raw), "known_development_errors_only": True},
        "total_settled_micro_usd": original["settled_micro_usd"] + total,
        "new_model_calls": 0, "aws_reads": True, "aws_writes": False, "cache_modified": False,
        "semantic_review": "PENDING", "cloud_published": False}


def main():
    path = ROOT / "results/collection-graph/reconciliation-final-v1.json"
    require(not path.exists(), "GRAPH_FINAL_AUDIT_EXISTS")
    result = reconcile()
    with path.open("xb") as handle:
        handle.write(canonical_json(result) + b"\n")
    print(json.dumps({"status": result["status"], "settled_model_calls": len(result["original_v3"]["operations"])
        + len(result["anchored_v4"]["operations"]), "total_settled_micro_usd": result["total_settled_micro_usd"],
        "new_model_calls": 0}))


if __name__ == "__main__":
    main()
