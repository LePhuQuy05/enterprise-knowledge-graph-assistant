"""Bind explicit AI source judgments to composite candidates and quarantine.

This is source review evidence, not independent gold or an accuracy estimate.
No candidate is automatically published to a serving graph.
"""

import argparse
import json
from collections import Counter

from app.contracts import canonical_json, sha256_bytes
from app.source_candidate_policy import policy_recipe, retained
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs


def audit(phase):
    if phase == "development":
        directory = ROOT / "results/semantic-graph-v7/qualification-v1/v8-review-v1"
        rows = [json.loads(line) for line in (directory / "decisions.jsonl").read_bytes().splitlines()]
        config_path = ROOT / "configs/extraction/semantic-v7-v8-development-source-review-v1.json"
    else:
        directory = ROOT / "results/semantic-graph-v7/composite-qualification-v1"
        plan = json.loads((directory / "plan.json").read_bytes())
        if plan["composite_recipe"]["retention"] != policy_recipe():
            raise ValueError("SOURCE_REVIEW_RETENTION_POLICY_CHANGED")
        rows = [json.loads((directory / "units" / (u["v7_unit_id"] + ".json")).read_bytes()) for u in plan["selection"]]
        config_path = ROOT / "configs/extraction/semantic-v7-v8-fresh-source-review-v1.json"
    config = json.loads(config_path.read_bytes())
    _, chunks, _ = load_collection_inputs()
    candidates = {(r["unit_id"], d["kind"], d["index"]): (r, d) for r in rows for d in r["decisions"]
                  if d["status"] == "CANDIDATE_SOURCE_GUARDS_PASS"}
    judgments = []
    for review in config["reviews"]:
        key = review["unit_id"], review["kind"], review["index"]
        row, candidate = candidates[key]
        if sha256_bytes(canonical_json(candidate)) != review["candidate_sha256"]:
            raise ValueError("SOURCE_REVIEW_CANDIDATE_CHECKSUM")
        quote = candidate["normalized_record"]["evidence_quote"]
        text = chunks[row["occurrences"][0]["chunk_id"]].text
        if sha256_bytes(text.encode()) != row["text_sha256"] or quote not in text:
            raise ValueError("SOURCE_REVIEW_QUOTE_SOURCE_BINDING")
        admitted, retention_reason = retained(candidate)
        supported = review["status"] == "AI_SOURCE_SUPPORTED_SOURCE_SCOPED"
        judgments.append(dict(review, text_sha256=row["text_sha256"], evidence_sha256=sha256_bytes(quote.encode()),
            source_uris=sorted({o["source_uri"] for o in row["occurrences"]}),
            retention_admitted=admitted, retention_reason=retention_reason,
            decision="AI_REVIEWED_CANDIDATE_NOT_SERVING" if admitted and supported else "QUARANTINED_BY_POLICY_OR_SOURCE_REVIEW"))
    if len(judgments) != len(candidates) or len({(r["unit_id"], r["kind"], r["index"]) for r in judgments}) != len(candidates):
        raise ValueError("SOURCE_REVIEW_INCOMPLETE_OR_DUPLICATE")
    report = {"status": "ALL_COMPOSITE_CANDIDATES_HAVE_BOUND_AI_SOURCE_JUDGMENTS", "phase": phase,
        "reviewed_candidates": len(judgments), "judgments": judgments,
        "review_statuses": dict(Counter(r["status"] for r in judgments)),
        "decisions": dict(Counter(r["decision"] for r in judgments)),
        "config_sha256": sha256_bytes(config_path.read_bytes()), "retention_recipe": policy_recipe(),
        "human_reviewed": 0, "provider_calls": 0, "aws_calls": 0, "serving_records_added": 0,
        "evaluation_scope": "AI_SINGLE_PASS_SOURCE_QUALIFICATION_NOT_INDEPENDENT_GOLD",
        "precision_recall": "NOT_ESTIMATED", "empty_output_recall": "NOT_ESTIMATED",
        "next_batch_scope": "CANDIDATE_EXTRACTION_ONLY_REVIEW_EVERY_CANDIDATE_BEFORE_SERVING"}
    raw, path = canonical_json(report) + b"\n", directory / "source-review.json"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("SOURCE_REVIEW_REPORT_FROZEN")
    path.write_bytes(raw)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("development", "fresh"))
    args = parser.parse_args()
    result = audit(args.phase)
    print(json.dumps({k: result[k] for k in ("status", "phase", "reviewed_candidates", "review_statuses", "decisions")}))
