"""Review completed source units only; preserve failed/pending tranche states."""

import json
from collections import Counter

from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes
from app.semantic_extraction_cache_v7 import SemanticExtractionCache
from app.semantic_extraction_v8_review import review_output
from app.source_candidate_policy import retained
from app.source_graph_jobs import context_for
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

DIRECTORY = ROOT / "results/semantic-graph-v7/tranche-001"
CONFIG = ROOT / "configs/extraction/semantic-v7-v8-tranche-001-source-review-v1.json"


def completed_rows(*, include_recovery=False):
    _, chunks, _ = load_collection_inputs()
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v7", namespace="LIVE")
    recovery = DIRECTORY / "recovery-v1"
    rows, plans = [], []
    for directory in sorted(DIRECTORY.glob("job-*")):
        plan = json.loads((directory / "plan.json").read_bytes())
        recipe = ExtractionRecipe.model_validate_json(canonical_json(plan["recipe"]))
        plans.append(plan)
        selection = plan["selection"]
        if directory.name == "job-3" and not include_recovery:
            selection = selection[:22]  # Frozen v1 review covers the first completed 89 only.
        for unit in selection:
            path = directory / "units" / (unit["v7_unit_id"] + ".json")
            if not path.exists():
                continue
            row = json.loads(path.read_bytes())
            text = chunks[unit["occurrences"][0]["chunk_id"]].text
            use_replacement = directory.name == "job-3" and unit == plan["selection"][22]
            selected_cache = cache
            if use_replacement:
                proof = json.loads((recovery / "execution-proof.json").read_bytes())
                completion = json.loads((recovery / "completion.json").read_bytes())
                from app.source_job_recovery import validate_zero_dispatch_proof
                from scripts.recover_source_graph_tranche import failed_claim

                validate_zero_dispatch_proof(proof, plan, 22)
                if (completion["status"] != "RECOVERED_11_COMPLETE_JOB_33_NOT_SERVING"
                        or failed_claim(unit) != proof["failed_cache"]):
                    raise ValueError("TRANCHE_RECOVERY_EVIDENCE_MISMATCH")
                selected_cache = SemanticExtractionCache(recovery / "replacement-cache", namespace="LIVE")
            lookup = selected_cache.resolve(context_for(unit, plan["run_id"]), text=text, recipe=recipe)
            decisions = []
            for decision in review_output(lookup.validation.output, text):
                allowed, reason = retained(decision)
                decisions.append(dict(decision, retention_status="RETAINED_PENDING_SOURCE_REVIEW" if allowed else "QUARANTINED",
                    retention_reason=reason))
            expected = {"unit_id": unit["v7_unit_id"], "text_sha256": unit["text_sha256"], "occurrences": unit["occurrences"],
                "receipt": lookup.receipt, "coverage_note": lookup.validation.output.coverage_note, "decisions": decisions,
                "composite_recipe_sha256": sha256_bytes(canonical_json(plan["composite_recipe"])), "semantic_review": "PENDING_NOT_SERVING"}
            if canonical_json(row) != canonical_json(expected) or sha256_bytes(text.encode()) != row["text_sha256"]:
                raise ValueError("TRANCHE_SOURCE_RECEIPT_REPLAY_MISMATCH")
            rows.append(row)
    return rows, plans, chunks


def audit(*, version="v1"):
    rows, plans, chunks = completed_rows(include_recovery=version == "v2")
    config_path = CONFIG if version == "v1" else CONFIG.with_name(CONFIG.name.replace("-v1.json", "-v2.json"))
    config = json.loads(config_path.read_bytes())
    candidates = {(r["unit_id"], d["kind"], d["index"]): (r, d) for r in rows for d in r["decisions"]
        if d["status"] == "CANDIDATE_SOURCE_GUARDS_PASS"}
    judgments, seen = [], set()
    for judgment in config["reviews"]:
        key = judgment["unit_id"], judgment["kind"], judgment["index"]
        if key in seen or key not in candidates:
            raise ValueError("TRANCHE_UNKNOWN_OR_DUPLICATE_SOURCE_JUDGMENT")
        seen.add(key)
        row, decision = candidates[key]
        quote = decision["normalized_record"]["evidence_quote"]
        text = chunks[row["occurrences"][0]["chunk_id"]].text
        if judgment["candidate_sha256"] != sha256_bytes(canonical_json(decision)) or quote not in text:
            raise ValueError("TRANCHE_SOURCE_JUDGMENT_BINDING")
        allowed, reason = retained(decision)
        supported = judgment["status"] == "AI_SOURCE_SUPPORTED_SOURCE_SCOPED"
        judgments.append(dict(judgment, text_sha256=row["text_sha256"], evidence_sha256=sha256_bytes(quote.encode()),
            source_uris=sorted({o["source_uri"] for o in row["occurrences"]}), retention_admitted=allowed,
            retention_reason=reason, decision="AI_REVIEWED_CANDIDATE_NOT_SERVING" if allowed and supported else "QUARANTINED_BY_SOURCE_REVIEW"))
    if seen != set(candidates):
        raise ValueError("TRANCHE_SOURCE_JUDGMENTS_INCOMPLETE")
    report = {"status": "ALL_COMPLETED_COMPOSITE_CANDIDATES_HAVE_BOUND_AI_SOURCE_JUDGMENTS",
        "processed_units": len(rows), "planned_units": sum(len(p["selection"]) for p in plans),
        "tranche_complete": len(rows) == sum(len(p["selection"]) for p in plans),
        "reviewed_candidates": len(judgments), "judgments": judgments,
        "review_statuses": dict(Counter(j["status"] for j in judgments)),
        "config_sha256": sha256_bytes(config_path.read_bytes()), "source_replay": "MATCH_ALL_COMPLETED_UNITS",
        "provider_calls": 0, "aws_calls": 0, "serving_records_added": 0, "human_reviewed": 0,
        "precision_recall": "NOT_ESTIMATED", "independent_gold": False}
    out = DIRECTORY / ("review-completed-" + version)
    out.mkdir(exist_ok=True)
    raw = canonical_json(report) + b"\n"
    path = out / "source-review.json"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("TRANCHE_COMPLETED_REVIEW_REPORT_FROZEN")
    path.write_bytes(raw)
    return report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", choices=("v1", "v2"), default="v1")
    result = audit(version=parser.parse_args().version)
    print(json.dumps({k: v for k, v in result.items() if k != "judgments"}))
