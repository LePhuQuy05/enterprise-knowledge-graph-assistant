"""Offline V8 review of immutable native V7 outputs; no provider/AWS calls."""

import json
from collections import Counter

from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction_v7 import SemanticOutputV7
from app.semantic_extraction_v8_review import review_output, review_recipe
from scripts.run_semantic_graph_canary import load_collection_inputs
from scripts.run_typed_graph_qualification import OUTPUT


def main():
    plan = json.loads((OUTPUT / "plan.json").read_bytes())
    _, chunks, _ = load_collection_inputs()
    rows = []
    for unit in plan["selection"]:
        row = json.loads((OUTPUT / "units" / (unit["v7_unit_id"] + ".json")).read_bytes())
        text = chunks[unit["occurrences"][0]["chunk_id"]].text
        if sha256_bytes(text.encode()) != row["text_sha256"]:
            raise ValueError("TYPED_REVIEW_SOURCE_CHECKSUM")
        output = SemanticOutputV7.model_validate_json(row["receipt"]["raw_output"])
        rows.append({"unit_id": unit["v7_unit_id"], "cohort": unit["cohort"], "text_sha256": row["text_sha256"],
            "occurrences": row["occurrences"], "native_receipt_sha256": sha256_bytes(canonical_json(row["receipt"])),
            "coverage_note": output.coverage_note, "decisions": review_output(output, text)})
    decisions = [d for row in rows for d in row["decisions"]]
    report = {"status": "NATIVE_V7_WITH_DETERMINISTIC_V8_REVIEW_COMPLETE_SOURCE_REVIEW_PENDING",
        "source_units": len(rows), "raw_records": len(decisions), "review_recipe": review_recipe(),
        "statuses": dict(Counter(d["status"] for d in decisions)),
        "rejections": dict(Counter(d["reason"] for d in decisions if d["reason"])),
        "candidate_relationships": sum(d["kind"] == "RELATION" and d["status"] == "CANDIDATE_SOURCE_GUARDS_PASS" for d in decisions),
        "candidate_permissions": sum(d["kind"] == "PERMISSION" and d["status"] == "CANDIDATE_SOURCE_GUARDS_PASS" for d in decisions),
        "normalizations": dict(Counter(n for d in decisions for n in d["normalizations"])),
        "provider_calls": 0, "aws_calls": 0, "native_v8_inference": False, "human_reviewed": 0,
        "raw_outputs_modified": False, "serving_records_added": 0, "full_batch_gate": "PENDING_SOURCE_REVIEW_AND_FRESH_COMPOSITE_QUALIFICATION"}
    out = OUTPUT / "v8-review-v1"
    out.mkdir(exist_ok=True)
    for name, payload in (("report.json", canonical_json(report) + b"\n"),
        ("decisions.jsonl", b"".join(canonical_json(row) + b"\n" for row in rows))):
        path = out / name
        if path.exists() and path.read_bytes() != payload:
            raise ValueError("TYPED_REVIEW_FROZEN_REPORT_CHANGED")
        path.write_bytes(payload)
    print(json.dumps({k: report[k] for k in ("status", "source_units", "raw_records", "statuses",
        "candidate_relationships", "candidate_permissions", "rejections", "normalizations")}))


if __name__ == "__main__":
    main()
