"""Native V10: eight known controls +24 fresh source texts, not a QA holdout.

Default freezes only. --execute permits one bounded dispatch per cache claim
under the authorized total8USD ledger; failed/unknown calls are never retried.
"""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.extraction import AliasDictionary
from app.model_gateway import safe_failure_code
from app.semantic_extraction_cache_v10 import SemanticExtractionCache
from app.semantic_extraction_v10 import semantic_recipe, semantic_request
from app.source_graph_jobs_v10 import context_for, managed_gateway
from scripts.run_clause_graph_qualification import immutable
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

DIRECTORY = ROOT / "results/semantic-graph-v10/qualification-v1"
RUN = "collection316-source-actions-v10-qualification-v1"
CONTROLS = ("0df376cb", "249dbbda", "2d2d86c0", "8fc134a9", "bf82bf71", "ccf04481", "058a2853", "caeed015")
CODE = ("app/semantic_extraction_v10.py", "app/source_normalization_v10.py", "app/semantic_extraction_cache_v10.py",
    "app/source_graph_jobs_v10.py", "app/source_action_graph.py", "scripts/run_source_actions_qualification.py")


def select_sources(units, chunks):
    controls = [json.loads(p.read_bytes()) for p in (ROOT / "results/semantic-graph-v9/qualification-v1/units").glob("*.json")
        if any(p.stem.split("_")[-1].startswith(prefix) for prefix in CONTROLS)]
    hashes = {r["text_sha256"] for r in controls}
    if len(hashes) != 8:
        raise ValueError("V10_EIGHT_KNOWN_SOURCE_CONTROLS_REQUIRED")
    excluded = set(hashes)
    for version in (7, 9):
        path = ROOT / f"data/cache/semantic-extraction-v{version}/semantic_extractions_v{version}.sqlite3"
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
            excluded.update(json.loads(row[0])["source"] for row in db.execute("SELECT binding FROM extractions WHERE namespace='LIVE'"))
    path = ROOT / "data/cache/external-models/extraction.sqlite3"
    with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
        excluded.update(json.loads(row[0])["input_sha256"] for row in db.execute("SELECT claim FROM extraction WHERE namespace='LIVE'"))
    groups = defaultdict(list)
    for unit in units:
        if unit["text_sha256"] not in excluded:
            for group in {o["service_group"] for o in unit["occurrences"]}:
                groups[group].append(unit)
    selected = [dict(u, cohort="KNOWN_DEVELOPMENT_CONTROL") for u in units if u["text_sha256"] in hashes]
    seen = set(hashes)
    cues = re.compile(r"\b(?:invokes?|triggers?|sends?|delivers?|publishes?|reads?|writes?|encrypts?|decrypts?|assumes?)\b", re.I)
    for index, (group, candidates) in enumerate(sorted(groups.items())):
        def order(unit):
            text = chunks[unit["occurrences"][0]["chunk_id"]].text
            preferred = len(text) >= 200 and "```" not in text and bool(cues.search(text))
            return (not preferred if index % 2 == 0 else False, unit["text_sha256"])
        unit = next(u for u in sorted(candidates, key=order) if u["text_sha256"] not in seen)
        seen.add(unit["text_sha256"])
        selected.append(dict(unit, cohort="FRESH_SOURCE_SELECTED", selected_group=group,
            selection_stratum="ACTION_CUE_PREFERRED" if index % 2 == 0 else "HASH_ORDER_CONTROL"))
    if len(selected) != 32 or len(seen) != 32 or len(groups) != 24:
        raise ValueError("V10_EXPECTED_8_CONTROLS_PLUS_24_GROUPS")
    return selected, excluded


def freeze():
    collection, chunks, units = load_collection_inputs()
    selected, excluded = select_sources(units, chunks)
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = semantic_recipe(AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v2.json"),
        metadata_sha256=routes["account_model_metadata_sha256"])
    for unit in selected:
        unit["v10_unit_id"] = stable_id("semantic-v10-unit", source=collection.manifest_sha256,
            text=unit["text_sha256"], recipe=recipe.fingerprint)
    full_proxy = sum((len(s) + len(p) + 3) // 4 for s, p in
        (semantic_request(chunks[u["occurrences"][0]["chunk_id"]].text) for u in units))
    plan = {"version": "native-v10-source-actions-qualification-v1", "run_id": RUN,
        "recipe": recipe.model_dump(mode="json"), "selection": selected, "source_manifest_sha256": collection.manifest_sha256,
        "maximum_new_calls": 32, "run_cap_micro_usd": 500000, "actual_cost_stop_between_calls_micro_usd": 100000,
        "max_output_tokens": 2048, "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "excluded_source_hashes_sha256": sha256_bytes(canonical_json(sorted(excluded))),
        "full_inventory_unique_texts": len(units), "full_inventory_input_proxy_tokens": full_proxy,
        "forecast_micro_usd_by_output_tokens_per_unit": {str(n): (full_proxy * 300000 + len(units) * n * 1200000 + 999999) // 1000000 for n in (256, 512, 1024, 2048)},
        "forecast_notes": "4 characters/token proxy; peak cache-miss rates; not provider tokens or admission bound; excludes review/repair/answers",
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "selection_policy": "8previously seen controls +24fresh source-only URIgroups;12action-cue preference/12hash controls; no questions/gold/scores",
        "funding_profile": "AUDITED_2_TO_8_USD_V2", "funding_cap_changes": False, "full_batch_enabled": False,
        "serving_promotion": False, "human_reviewed": 0}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    immutable(DIRECTORY / "plan.json", canonical_json(plan) + b"\n")
    return plan, chunks, recipe


def run(*, replay=False):
    plan, chunks, recipe = freeze()
    gateway, ledger = (None, None) if replay else managed_gateway(ROOT, plan, chunks)
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v10", namespace="LIVE")
    output = DIRECTORY / "units"
    output.mkdir(exist_ok=True)
    rows, error = [], None
    try:
        for unit in plan["selection"]:
            text = chunks[unit["occurrences"][0]["chunk_id"]].text
            if sha256_bytes(text.encode()) != unit["text_sha256"]:
                raise ValueError("V10_SOURCE_CHANGED")
            lookup = cache.resolve(context_for(unit, RUN), text=text, recipe=recipe, gateway=gateway)
            result = lookup.validation
            row = {"unit_id": unit["v10_unit_id"], "text_sha256": unit["text_sha256"], "cohort": unit["cohort"],
                "occurrences": unit["occurrences"], "receipt": lookup.receipt, "coverage_note": result.output.coverage_note,
                "accepted_relationships": result.accepted_relationships, "accepted_permissions": result.accepted_permissions,
                "accepted_actions": result.accepted_actions, "deferred_statements": result.deferred_statements, "rejected_records": result.rejected_records,
                "fact_cap_reached": len(result.output.relationships) + len(result.output.permission_requirements) + len(result.output.actions) == 4,
                "semantic_review": "PENDING_NOT_SERVING"}
            immutable(output / (unit["v10_unit_id"] + ".json"), canonical_json(row) + b"\n")
            rows.append(row)
            print(json.dumps({"processed": len(rows), "planned": 32, "relations": len(result.accepted_relationships),
                "actions": len(result.accepted_actions), "permissions": len(result.accepted_permissions),
                "rejected": len(result.rejected_records), "new_cost_micro_usd": gateway.cost if gateway else 0}), flush=True)
    except Exception as exc:
        error = safe_failure_code(exc)
    counter = ledger.store.get(ledger.run_key(RUN)) if ledger else None
    own_cost = sum(r["receipt"]["cost_micro_usd"] for r in rows if r["receipt"]["context"]["run_id"] == RUN)
    if counter and not error and (counter["reserved_micro_usd"] or counter["settled_micro_usd"] != own_cost):
        error = "V10_RECEIPT_RUN_COST_MISMATCH"
    report = {"status": "STOPPED_NO_AUTOMATIC_RETRY" if error else "REPLAY_MATCH" if replay else "NATIVE_V10_COMPLETE_SOURCE_REVIEW_PENDING",
        "error": error, "processed": len(rows), "planned": 32,
        **{field: sum(len(r[field]) for r in rows) for field in ("accepted_relationships", "accepted_permissions", "accepted_actions", "deferred_statements", "rejected_records")},
        "coverage_notes": dict(Counter(r["coverage_note"] for r in rows)), "fact_cap_reached_sources": sum(r["fact_cap_reached"] for r in rows),
        "provider_reservation_attempts": len(gateway.seen) if gateway else 0, "new_cost_micro_usd": gateway.cost if gateway else 0,
        "own_run_receipt_cost_micro_usd": own_cost, "run_counter": counter, "human_reviewed": 0, "serving_records_added": 0,
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
    print(json.dumps({"status": "FROZEN_NATIVE_V10_32_SOURCES_NOT_EXECUTED", "source_units": len(plan["selection"]), "model_calls": 0, "aws_calls": 0,
        "forecast": plan["forecast_micro_usd_by_output_tokens_per_unit"]}))
