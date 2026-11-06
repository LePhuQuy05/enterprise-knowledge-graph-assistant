"""Freeze/run native V9 on eight development sources and 24 fresh sources.

No serving, full batch, automatic repair, or changes to frozen V7/V8 artifacts.
"""

import argparse
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.extraction import AliasDictionary
from app.model_gateway import safe_failure_code
from app.semantic_extraction_cache_v9 import SemanticExtractionCache
from app.semantic_extraction_v9 import semantic_recipe, semantic_request
from app.source_graph_jobs_v9 import context_for, managed_gateway
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs
from scripts.run_typed_graph_qualification import source_selection

DIRECTORY = ROOT / "results/semantic-graph-v9/qualification-v1"
RUN = "collection316-native-v9-qualification-v1"


def immutable(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("V9_FROZEN_ARTIFACT_CHANGED:" + path.name)
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(raw)


def freeze():
    collection, chunks, units = load_collection_inputs()
    review = json.loads((ROOT / "configs/extraction/semantic-v7-v8-tranche-001-source-review-v2.json").read_bytes())
    dev_ids = {j["unit_id"] for j in review["reviews"]}
    v7_rows = [json.loads(p.read_bytes()) for p in (ROOT / "results/semantic-graph-v7/tranche-001").glob("job-*/units/*.json")]
    dev_hashes = {r["text_sha256"] for r in v7_rows if r["unit_id"] in dev_ids}
    if len(dev_hashes) != 8:
        raise ValueError("V9_EIGHT_KNOWN_DEVELOPMENT_SOURCES_REQUIRED")
    statuses = ROOT / "results/semantic-graph-v7/typed-export-v1/source-statuses.jsonl"
    excluded = {r["text_sha256"] for r in map(json.loads, statuses.read_bytes().splitlines()) if r["status"] == "NATIVE_V7_RECEIPT_READY"}
    excluded.update(r["text_sha256"] for r in map(json.loads,
        (ROOT / "results/semantic-graph-v7/development-source-audit-v1/sources.jsonl").read_bytes().splitlines()))
    dbpath = ROOT / "data/cache/external-models/extraction.sqlite3"
    with sqlite3.connect(f"file:{dbpath.as_posix()}?mode=ro", uri=True) as db:
        excluded.update(json.loads(row[0])["input_sha256"] for row in db.execute("SELECT claim FROM extraction WHERE namespace='LIVE'"))
    selected = [dict(u, cohort="KNOWN_DEVELOPMENT_V7_OBJECTION_OR_POSITIVE") for u in units if u["text_sha256"] in dev_hashes]
    selected.extend(source_selection(units, chunks, excluded)[:24])
    if len(selected) != 32 or len({u["text_sha256"] for u in selected}) != 32:
        raise ValueError("V9_EXPECTED_32_DISTINCT_SOURCES")
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = semantic_recipe(AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v2.json"),
        metadata_sha256=routes["account_model_metadata_sha256"])
    for unit in selected:
        unit["v9_unit_id"] = stable_id("semantic-v9-unit", source=collection.manifest_sha256,
            text=unit["text_sha256"], recipe=recipe.fingerprint)
    plan = {"version": "native-v9-clause-head-qualification-v1", "run_id": RUN, "recipe": recipe.model_dump(mode="json"),
        "selection": selected, "source_manifest_sha256": collection.manifest_sha256,
        "maximum_new_calls": 32, "run_cap_micro_usd": 500000, "actual_cost_stop_between_calls_micro_usd": 100000,
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "funding_profile": "AUDITED_2_TO_8_USD_V2", "funding_cap_changes": False, "full_batch_enabled": False,
        "serving_promotion": False, "human_reviewed": 0,
        "development_review_sha256": sha256_bytes(canonical_json(review)),
        "excluded_source_hashes_sha256": sha256_bytes(canonical_json(sorted(excluded))),
        "input_proxy_tokens": sum((len(s) + len(p) + 3) // 4 for s, p in
            (semantic_request(chunks[u["occurrences"][0]["chunk_id"]].text) for u in selected)),
        "selection_policy": "8development + one relation-cue-preferred unclaimed text per24URIgroups; not independent benchmark holdout",
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in
            ("app/semantic_extraction_v9.py", "app/semantic_extraction_cache_v9.py", "app/source_graph_jobs_v9.py",
             "scripts/run_clause_graph_qualification.py")}}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    immutable(DIRECTORY / "plan.json", canonical_json(plan) + b"\n")
    return plan, chunks, recipe


def run(*, replay=False):
    plan, chunks, recipe = freeze()
    gateway, ledger = (None, None) if replay else managed_gateway(ROOT, plan, chunks)
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v9", namespace="LIVE")
    output = DIRECTORY / "units"
    output.mkdir(exist_ok=True)
    rows, error, detail = [], None, None
    try:
        for unit in plan["selection"]:
            text = chunks[unit["occurrences"][0]["chunk_id"]].text
            if sha256_bytes(text.encode()) != unit["text_sha256"]:
                raise ValueError("V9_SOURCE_CHANGED")
            lookup = cache.resolve(context_for(unit, RUN), text=text, recipe=recipe, gateway=gateway)
            validated = lookup.validation
            row = {"unit_id": unit["v9_unit_id"], "text_sha256": unit["text_sha256"], "cohort": unit["cohort"],
                "occurrences": unit["occurrences"], "receipt": lookup.receipt, "coverage_note": validated.output.coverage_note,
                "accepted_relationships": validated.accepted_relationships, "accepted_permissions": validated.accepted_permissions,
                "deferred_statements": validated.deferred_statements, "rejected_records": validated.rejected_records,
                "record_cap_reached": len(validated.output.relationships) == 3 or len(validated.output.permission_requirements) == 2 or len(validated.output.deferred_statements) == 3,
                "semantic_review": "PENDING_NOT_SERVING"}
            immutable(output / (unit["v9_unit_id"] + ".json"), canonical_json(row) + b"\n")
            rows.append(row)
            print(json.dumps({"processed": len(rows), "planned": 32, "relations": len(validated.accepted_relationships),
                "permissions": len(validated.accepted_permissions), "deferred": len(validated.deferred_statements),
                "rejected": len(validated.rejected_records), "new_cost_micro_usd": gateway.cost if gateway else 0}), flush=True)
    except Exception as exc:
        error = safe_failure_code(exc)
        from app.ledger import BudgetDenied

        if isinstance(exc, BudgetDenied) and str(exc) in {"Concurrent updates exhausted bounded retries", "Budget allowance exhausted or halted"}:
            detail = str(exc)
    run_counter = ledger.store.get(ledger.run_key(RUN)) if ledger else None
    own_cost = sum(row["receipt"]["cost_micro_usd"] for row in rows if row["receipt"]["context"]["run_id"] == RUN)
    if run_counter and not error and (run_counter["reserved_micro_usd"] or run_counter["settled_micro_usd"] != own_cost):
        error = "V9_RECEIPT_RUN_COST_MISMATCH"
    report = {"status": "STOPPED_NO_AUTOMATIC_RETRY" if error else "REPLAY_MATCH" if replay else "NATIVE_V9_COMPLETE_SOURCE_REVIEW_PENDING",
        "error": error, "admission_detail": detail, "processed": len(rows), "planned": 32,
        "relations": sum(len(r["accepted_relationships"]) for r in rows), "permissions": sum(len(r["accepted_permissions"]) for r in rows),
        "deferred": sum(len(r["deferred_statements"]) for r in rows), "rejected": sum(len(r["rejected_records"]) for r in rows),
        "coverage_notes": dict(Counter(r["coverage_note"] for r in rows)), "record_cap_reached_sources": sum(r["record_cap_reached"] for r in rows),
        "provider_reservation_attempts": len(gateway.seen) if gateway else 0, "new_cost_micro_usd": gateway.cost if gateway else 0,
        "own_run_receipt_cost_micro_usd": own_cost, "run_counter": run_counter, "human_reviewed": 0, "serving_records_added": 0,
        "full_batch_enabled": False, "finished_at": datetime.now(UTC).isoformat()}
    path = DIRECTORY / ("replay.json" if replay else f"attempt-{len(list(DIRECTORY.glob('attempt-*.json'))) + 1:03d}.json")
    path.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))
    return 0 if not error else 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.execute or args.replay:
        raise SystemExit(run(replay=args.replay))
    plan, _, _ = freeze()
    print(json.dumps({"status": "FROZEN_NATIVE_V9_32_SOURCES_NOT_EXECUTED", "source_units": len(plan["selection"]), "model_calls": 0, "aws_calls": 0}))
