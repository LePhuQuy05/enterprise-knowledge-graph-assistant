"""Bind the ten explicit AI source judgments to immutable V6 candidate records."""

import json
from collections import Counter

from app.contracts import canonical_json, sha256_bytes
from scripts.run_semantic_graph_canary import OUTPUT, ROOT


def main():
    config_path = ROOT / "configs/extraction/semantic-canary-v6-source-review-v1.json"
    config = json.loads(config_path.read_bytes())
    candidates = {}
    for line in (OUTPUT / "v6-review-v1/decisions.jsonl").read_bytes().splitlines():
        row = json.loads(line)
        for decision in row["decisions"]:
            if decision["status"] == "CANDIDATE_SOURCE_GUARDS_PASS":
                candidates[(row["unit_id"], decision["kind"], decision["index"])] = (row, decision)
    reviews = []
    for item in config["decisions"]:
        key = ("semantic-v5-unit_" + item["unit_suffix"], item["kind"], item["index"])
        if key not in candidates:
            raise ValueError("SEMANTIC_SOURCE_REVIEW_CANDIDATE_BINDING")
        row, candidate = candidates[key]
        reviews.append({**item, "unit_id": key[0], "candidate_sha256": sha256_bytes(canonical_json(candidate)),
            "source_text_sha256": row["source_text_sha256"],
            "evidence_quote_sha256": sha256_bytes(candidate["normalized_record"]["evidence_quote"].encode()),
            "native_receipt_sha256": row["native_receipt_sha256"], "source_uris": sorted({o["source_uri"] for o in row["occurrences"]})})
    if len(reviews) != len(candidates) or len({(r["unit_id"], r["kind"], r["index"]) for r in reviews}) != len(candidates):
        raise ValueError("SEMANTIC_SOURCE_REVIEW_INCOMPLETE_OR_DUPLICATE")
    report = {"status": "TEN_AI_SOURCE_JUDGMENTS_BOUND_FULL_BATCH_GATE_FAILED", "reviewed_candidates": len(reviews),
        "statuses": dict(Counter(r["status"] for r in reviews)), "reviews": reviews,
        "review_config_sha256": sha256_bytes(config_path.read_bytes()), "human_reviewed": 0,
        "serving_records_added": 0, "raw_outputs_modified": False, "provider_calls": 0, "aws_calls": 0,
        "full_batch_gate": config["full_batch_gate"], "empty_output_recall": "NOT_MEASURED"}
    path = OUTPUT / "v6-review-v1/source-review.json"
    raw = canonical_json(report) + b"\n"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("SEMANTIC_SOURCE_REVIEW_CHANGED")
    path.write_bytes(raw)
    print(json.dumps({k: report[k] for k in ("status", "reviewed_candidates", "statuses", "serving_records_added")}))


if __name__ == "__main__":
    main()
