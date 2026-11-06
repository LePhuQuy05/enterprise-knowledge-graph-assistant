"""Compare pinned dependency witnesses with frozen V10 on existing receipts."""

import json
from collections import Counter

from app.contracts import canonical_json, sha256_bytes
from app.extraction import ExtractionError
from app.semantic_extraction_v10 import RelationV10, SourceActionV1
from app.source_normalization_v10 import normalize_record
from app.source_syntax import SyntaxDocument, policy_recipe, validate_syntax_record
from scripts.freeze_source_syntax_queue import OUTPUT
from scripts.run_clause_graph_qualification import immutable


def main():
    queue_raw, parse_raw = (OUTPUT / "queue.json").read_bytes(), (OUTPUT / "parsed.json").read_bytes()
    queue, parsed = json.loads(queue_raw), json.loads(parse_raw)
    documents = {d["text"]: SyntaxDocument(d) for d in parsed["documents"]}
    if len(documents) != len(parsed["documents"]) or parsed["installation"]["model_wheel_sha256"] != "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85":
        raise ValueError("SYNTAX_PARSE_DUPLICATE_OR_UNPINNED_MODEL")
    destination = OUTPUT / "comparison-v3"
    destination.mkdir(exist_ok=True)
    plan = {"version": "native-v10-plus-local-syntax-comparison-v3", "queue_sha256": sha256_bytes(queue_raw),
        "parser_output_sha256": sha256_bytes(parse_raw), "policy": policy_recipe(), "native_v10_mutated": False,
        "model_api_calls": 0, "human_reviewed": 0, "serving_published": False}
    immutable(destination / "plan.json", canonical_json(plan) + b"\n")
    rows = []
    for source in queue["rows"]:
        for observation in source["observations"]:
            if observation["kind"] not in {"ACTION", "RELATION"}:
                continue
            kind, raw = observation["kind"], observation["record"]
            candidate, error = None, None
            try:
                normalized, audit = normalize_record(raw, source["source_text"], kind=kind)
                record = (RelationV10 if kind == "RELATION" else SourceActionV1).model_validate_json(canonical_json(normalized))
                candidate = validate_syntax_record(record, source["source_text"], documents[record.clause_quote], kind=kind)
                candidate["normalization"] = audit
            except ExtractionError as exc:
                error = str(exc)
            rows.append({"unit_id": source["unit_id"], "source_sha256": source["source_sha256"], "kind": kind,
                "record_index": observation["index"], "native_guard_admitted": observation["native_guard_admitted"],
                "syntax_guard_admitted": candidate is not None, "reason": error, "candidate": candidate,
                "candidate_sha256": sha256_bytes(canonical_json(candidate)) if candidate else None})
    report = {"version": plan["version"], "status": "LOCAL_COMPARISON_SOURCE_REVIEW_PENDING", "source_units": len(queue["rows"]),
        "compared_action_relation_records": len(rows), "native_admitted": sum(r["native_guard_admitted"] for r in rows),
        "syntax_admitted": sum(r["syntax_guard_admitted"] for r in rows),
        "recovered": sum(r["syntax_guard_admitted"] and not r["native_guard_admitted"] for r in rows),
        "regressions": sum(r["native_guard_admitted"] and not r["syntax_guard_admitted"] for r in rows),
        "rejections": dict(Counter(r["reason"] for r in rows if r["reason"])), "rows": rows,
        "load_seconds": parsed["load_seconds"], "parse_seconds": parsed["parse_seconds"],
        "model_api_calls": 0, "aws_calls": 0, "new_cost_micro_usd": 0, "serving_records_added": 0}
    immutable(destination / "report.json", canonical_json(report) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}))


if __name__ == "__main__":
    main()
