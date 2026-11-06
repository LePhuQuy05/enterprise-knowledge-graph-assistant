"""Offline coverage/error summary; accepted structure is never reported as answer accuracy."""

import json
from collections import Counter
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.evidence_extraction import EvidenceExtractionOutput
from scripts.reconcile_collection_graph_pilot import load_inputs, read_claims, require
from scripts.run_collection_graph_pilot import ROOT


def occurrences(text, quote):
    positions, position = 0, -1
    if not quote:
        return 0
    while (position := text.find(quote, position + 1)) >= 0:
        positions += 1
    return positions


def anchor_errors(raw, text):
    try:
        output = EvidenceExtractionOutput.model_validate_json(raw)
    except ValueError:
        return ["OUTPUT_NOT_COMPLETE_V3_SCHEMA"]
    errors = []
    for relation in output.relationships:
        if occurrences(text, relation.evidence_quote) != 1:
            errors.append("RELATION_EVIDENCE_MISSING_OR_AMBIGUOUS")
        for endpoint in (relation.subject, relation.object):
            count = occurrences(relation.evidence_quote, endpoint.name)
            if count != 1:
                errors.append("RELATION_ENDPOINT_MISSING" if count == 0 else "RELATION_ENDPOINT_REPEATED")
        if any(q.text not in relation.evidence_quote for q in relation.qualifiers):
            errors.append("QUALIFIER_OUTSIDE_EVIDENCE")
    for mention in output.standalone_entities:
        if occurrences(text, mention.context_quote) != 1:
            errors.append("STANDALONE_CONTEXT_MISSING_OR_AMBIGUOUS")
        if occurrences(mention.context_quote, mention.name) != 1:
            errors.append("STANDALONE_NAME_MISSING_OR_AMBIGUOUS")
    return errors


def summarize():
    _, plan, _, chunks = load_inputs()
    audit_raw = (ROOT / "results/collection-graph/reconciliation-final-v1.json").read_bytes()
    audit = json.loads(audit_raw)
    require(audit["status"] == "PASS_BOTH_GRAPH_EXPERIMENTS_SOURCE_RECEIPT_USAGE_MANAGED_LEDGER", "GRAPH_SUMMARY_RECONCILIATION_REQUIRED")
    reports = [json.loads((ROOT / "results/collection-graph" / name).read_bytes()) for name in
        ("pilot-v1.json", "pilot-continuation-v1.json", "anchored-v4-check.json")]
    rows = []
    for report in reports:
        content = Path(report["canonical_path"]).read_bytes()
        if "canonical_sha256" in report:
            require(sha256_bytes(content) == report["canonical_sha256"], "GRAPH_SUMMARY_CANONICAL_CHANGED")
        rows.append([json.loads(line) for line in content.splitlines()])
    original_rows = rows[0] + rows[1]
    errors, error_inputs, groups = Counter(), [], {}
    claims = read_claims()
    for selected in plan["selection"]:
        key = selected["chunk_id"]
        claim = claims[key]
        group = groups.setdefault(selected["group"], {"selected": 0, "settled": 0, "structurally_accepted": 0,
            "quarantined_outputs": 0, "zero_dispatch_skips": 0})
        group["selected"] += 1
        group["settled"] += bool(claim["receipts"])
        group["structurally_accepted"] += claim["state"] == "READY"
        group["quarantined_outputs"] += claim["state"] == "FAILED" and bool(claim["receipts"])
        group["zero_dispatch_skips"] += not claim["receipts"]
        if claim["state"] == "FAILED" and claim["receipts"]:
            receipt = claim["receipts"][0]
            categories = anchor_errors(receipt.text, chunks[key].text)
            errors.update(categories)
            error_inputs.append({"chunk_id": key, "source_uri": selected["source_uri"],
                "raw_output_sha256": sha256_bytes(receipt.text.encode()), "anchor_errors": categories,
                "whole_chunk_accepted": False})
    assertions = [a for row in original_rows for a in row["assertions"]]
    provisional = {
        "v3": {"selected_chunks": 48, "settled_model_calls": len(audit["original_v3"]["operations"]),
            "structurally_accepted_chunks": len(original_rows), "quarantined_outputs": len(error_inputs),
            "zero_dispatch_skips": len(audit["original_v3"]["zero_dispatch_skips"]),
            "source_bound_assertion_candidates": len(assertions),
            "relation_source_documents": len({a["document_version_id"] for a in assertions}),
            "relation_source_chunks": len({a["chunk_id"] for a in assertions}),
            "accepted_empty_relation_chunks": sum(not r["assertions"] for r in original_rows),
            "settled_micro_usd": audit["original_v3"]["settled_micro_usd"], "groups": groups,
            "anchor_error_instances": dict(sorted(errors.items())), "failed_inputs": error_inputs},
        "v4_development_check": {"selected_known_failure_chunks": 4,
            "structurally_accepted_chunks": len(rows[2]), "source_bound_assertion_candidates": sum(len(r["assertions"]) for r in rows[2]),
            "accepted_empty_relation_chunks": sum(not r["assertions"] for r in rows[2]),
            "settled_micro_usd": reports[2]["new_cost_micro_usd"], "unbiased_quality_comparison": False,
            "explanation": "Four examples were selected after seeing V3 failures. A valid empty result is abstention, not recovered relation recall."}}
    return {"status": "RECONCILED_GRAPH_PILOT_NOT_SEMANTIC_APPROVAL", "documents": 316, "chunks": 2978,
        "collection_manifest_sha256": plan["collection_manifest_sha256"], "financial_audit_sha256": sha256_bytes(audit_raw),
        **provisional, "total_model_calls_all_graph_experiments": 51, "total_settled_micro_usd": audit["total_settled_micro_usd"],
        "full_relation_extraction": False, "graph_answering_cloud_enabled": False, "semantic_precision": None,
        "relation_recall": None, "answer_accuracy": None, "human_reviewed": False,
        "new_model_calls": 0, "aws_calls": 0, "cloud_published": False}


def main():
    output = ROOT / "results/collection-graph/pilot-summary-v1.json"
    require(not output.exists(), "GRAPH_SUMMARY_EXISTS")
    result = summarize()
    with output.open("xb") as handle:
        handle.write(canonical_json(result) + b"\n")
    print(json.dumps({"status": result["status"], "v3": {key: result["v3"][key] for key in
        ("settled_model_calls", "structurally_accepted_chunks", "quarantined_outputs", "source_bound_assertion_candidates",
         "relation_source_documents", "accepted_empty_relation_chunks")}, "v4": result["v4_development_check"]}))


if __name__ == "__main__":
    main()
