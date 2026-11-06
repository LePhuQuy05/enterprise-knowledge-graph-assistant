"""Revalidate existing V9 facts locally; not native V10 inference or new gold."""

import json
from collections import Counter
from copy import deepcopy

from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction_v10 import validate_semantic_output
from scripts.run_clause_graph_qualification import DIRECTORY as V9_DIRECTORY
from scripts.run_clause_graph_qualification import freeze as freeze_v9
from scripts.run_source_actions_qualification import DIRECTORY, ROOT


def project_record(record, *, kind):
    """Structural annotation migration only; names/predicate/action remain exact."""
    projected, edits = deepcopy(record), []
    fields = ("subject", "object") if kind == "RELATION" else ("principal", "resource")
    for field in fields:
        endpoint = projected[field]
        endpoint["scope_qualifiers"] = []
        if endpoint["semantic_type"] not in {"RESOURCE_TYPE", "RESOURCE_INSTANCE"} and endpoint["restrictions"]:
            endpoint["scope_qualifiers"] = endpoint["restrictions"]
            endpoint["restrictions"] = []
            edits.append({"path": field, "reason": "NONRESOURCE_ANNOTATION_MOVED_TO_SCOPE_NO_NAME_CHANGE"})
    return projected, edits


def main():
    plan, chunks, _ = freeze_v9()
    rows, counts, reasons = [], Counter(), Counter()
    for unit in plan["selection"]:
        native = json.loads((V9_DIRECTORY / "units" / (unit["v9_unit_id"] + ".json")).read_bytes())
        raw = json.loads(native["receipt"]["raw_output"])
        text = chunks[unit["occurrences"][0]["chunk_id"]].text
        for kind, key, accepted_field in (("RELATION", "relationships", "accepted_relationships"),
                ("PERMISSION", "permission_requirements", "accepted_permissions")):
            for index, original in enumerate(raw[key]):
                projected, edits = project_record(original, kind=kind)
                request = {"relationships": [], "permission_requirements": [], "actions": [], "deferred_statements": [], "coverage_note": "HAS_CANDIDATES"}
                request[key] = [projected]
                result = validate_semantic_output(canonical_json(request).decode(), text, stop_reason="end_turn")
                accepted = getattr(result, accepted_field)
                was_native_accepted = any(c["index"] == index for c in native[accepted_field])
                status = "RETAINED_GUARD_PASS" if accepted and was_native_accepted else "RECOVERED_MECHANICAL_GUARD_PASS" if accepted else "REJECTED"
                counts[status] += 1
                reasons.update(r["reason"] for r in result.rejected_records)
                rows.append({"unit_id": unit["v9_unit_id"], "kind": kind, "native_record_index": index, "status": status,
                    "native_receipt_sha256": sha256_bytes(canonical_json(native["receipt"])), "raw_record": original,
                    "structural_projection_edits": edits, "projected_record_sha256": sha256_bytes(canonical_json(projected)),
                    "candidate": accepted[0] if accepted else None, "rejected_records": result.rejected_records,
                    "native_v10_inference": False, "semantic_review": "PENDING_NOT_SERVING"})
    report = {"status": "V9_RECEIPTS_COMPOSITE_V10_REPLAY_NOT_NATIVE_V10", "sources": 32, "records": len(rows),
        "counts": dict(counts), "rejection_reasons": dict(reasons), "model_calls": 0, "aws_calls": 0, "new_cost_micro_usd": 0,
        "serving_records_added": 0, "rows": rows, "implementation_sha256": sha256_bytes((ROOT / "scripts/replay_source_addressing_v10.py").read_bytes())}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path = DIRECTORY / "v9-composite-source-replay.json"
    if path.exists():
        raise ValueError("COMPOSITE_REPLAY_REPORT_ALREADY_EXISTS")
    path.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
