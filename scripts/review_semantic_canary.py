"""Cache-only replay and V6 diagnostic review of unchanged native V5 outputs."""

import json
from collections import Counter

from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction_cache import SemanticExtractionCache
from app.semantic_extraction_v6 import recipe_v6, review_output_v6
from scripts.run_semantic_graph_canary import OUTPUT, ROOT, context_for, freeze


def main():
    plan, chunks, recipe = freeze()
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v5", namespace="LIVE")
    rows = []
    for unit in plan["selection"]:
        text = chunks[unit["occurrences"][0]["chunk_id"]].text
        lookup = cache.resolve(context_for(unit), text=text, recipe=recipe)
        if not lookup.cache_hit:
            raise ValueError("SEMANTIC_REVIEW_MUST_BE_CACHE_ONLY")
        decisions = review_output_v6(lookup.validation.output, text)
        rows.append({"unit_id": unit["v5_unit_id"], "cohort": unit["cohort"],
            "native_receipt_sha256": sha256_bytes(canonical_json(lookup.receipt)),
            "source_text_sha256": unit["text_sha256"], "occurrences": unit["occurrences"], "decisions": decisions,
            "original_records": len(lookup.validation.output.relationships) + len(lookup.validation.output.permission_requirements),
            "empty_output": not lookup.validation.output.relationships and not lookup.validation.output.permission_requirements})
    decisions = [d for row in rows for d in row["decisions"]]
    report = {"status": "NATIVE_V5_CACHE_REPLAY_AND_V6_DIAGNOSTIC_REVIEW_COMPLETE",
        "source_units": len(rows), "original_records": len(decisions), "v6_policy": recipe_v6(),
        "statuses": dict(Counter(d["status"] for d in decisions)),
        "rejected_reasons": dict(Counter(d["reason"] for d in decisions if d["reason"])),
        "normalized_triggers": sum(d.get("trigger_normalized", False) for d in decisions),
        "candidate_relationships": sum(d["kind"] == "RELATION" and d["status"] == "CANDIDATE_SOURCE_GUARDS_PASS" for d in decisions),
        "candidate_permissions": sum(d["kind"] == "PERMISSION" and d["status"] == "CANDIDATE_SOURCE_GUARDS_PASS" for d in decisions),
        "empty_outputs": sum(r["empty_output"] for r in rows), "provider_calls": 0, "aws_calls": 0,
        "new_native_v6_inference": False, "human_reviewed": 0, "serving_records_added": 0,
        "semantic_acceptance": "PENDING_SOURCE_TYPE_CONDITION_AND_EMPTY_RECALL_REVIEW"}
    directory = OUTPUT / "v6-review-v1"
    directory.mkdir(exist_ok=True)
    for name, raw in (("report.json", canonical_json(report) + b"\n"),
            ("decisions.jsonl", b"".join(canonical_json(row) + b"\n" for row in rows))):
        path = directory / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("SEMANTIC_REVIEW_ARTIFACT_CHANGED")
        path.write_bytes(raw)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
