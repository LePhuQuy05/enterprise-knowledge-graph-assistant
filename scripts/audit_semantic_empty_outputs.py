"""Validate 20 explicit AI source judgments; no claim of independent recall."""

import json
from collections import Counter
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction import validate_semantic_output as validate_v5
from app.semantic_extraction_v7 import TypedFrameV7, project_endpoint, validate_frame_v7

ROOT = Path(__file__).resolve().parents[1]


def audit():
    directory = ROOT / "results/semantic-graph-v7/development-source-audit-v1"
    raw = (directory / "sources.jsonl").read_bytes()
    config_path = ROOT / "configs/extraction/semantic-v5-empty-source-review-v1.json"
    config = json.loads(config_path.read_bytes())
    if sha256_bytes(raw) != config["sources_sha256"]:
        raise ValueError("EMPTY_AUDIT_SOURCE_EXPORT_CHANGED")
    rows = {r["unit_id"]: r for r in map(json.loads, raw.splitlines()) if r["empty"]}
    if len(rows) != 20 or len(config["reviews"]) != 20 or len({r["unit_id"] for r in config["reviews"]}) != 20:
        raise ValueError("EMPTY_AUDIT_EXPECTED_TWENTY_UNIQUE_JUDGMENTS")
    results = []
    for review in config["reviews"]:
        row = rows[review["unit_id"]]
        if (row["text_sha256"] != review["text_sha256"] or sha256_bytes(row["text"].encode()) != row["text_sha256"]
                or row["native_receipt_sha256"] != review["native_receipt_sha256"] or review["witness_quote"] not in row["text"]):
            raise ValueError("EMPTY_AUDIT_SOURCE_RECEIPT_BINDING")
        result = dict(review, source_uris=row["source_uris"])
        if review["status"] == "MISSED_ELIGIBLE_RELATION":
            frame = TypedFrameV7.model_validate_json(canonical_json(review["expected_v7_frame"]))
            accepted = validate_frame_v7(frame, row["text"])
            old = frame.model_dump(mode="json")
            old.update(subject=project_endpoint(frame.subject), object=project_endpoint(frame.object))
            old_validation = validate_v5(json.dumps({"relationships": [old], "permission_requirements": []}), row["text"], stop_reason="end_turn")
            if len(old_validation.accepted_relationships) != 1:
                raise ValueError("EMPTY_AUDIT_EXPECTED_FRAME_NOT_ELIGIBLE_UNDER_V5")
            result.update(local_guard_witness_v5="PASS", local_guard_witness_v7="PASS",
                          expected_frame_sha256=sha256_bytes(canonical_json(accepted)))
        results.append(result)
    report = {"status": "TWENTY_AI_EMPTY_SOURCE_JUDGMENTS_BOUND", "reviews": results,
        "categories": dict(Counter(r["status"] for r in results)), "human_reviewed": 0,
        "scope": config["scope"], "recall_estimate": "NOT_ESTIMATED",
        "reason": "Twenty development empty outputs, single AI review, one demonstrated missed eligible relation; not a complete independently annotated corpus sample",
        "config_sha256": sha256_bytes(config_path.read_bytes()), "provider_calls": 0, "aws_calls": 0,
        "raw_outputs_modified": False, "serving_records_added": 0}
    payload, path = canonical_json(report) + b"\n", directory / "empty-review.json"
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("EMPTY_AUDIT_FROZEN_REPORT_CHANGED")
    path.write_bytes(payload)
    return report


if __name__ == "__main__":
    report = audit()
    print(json.dumps({k: report[k] for k in ("status", "categories", "recall_estimate", "provider_calls", "aws_calls")}))
