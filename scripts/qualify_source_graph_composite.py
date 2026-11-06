"""Freeze the composite pipeline before selecting 24 further source texts."""

import argparse
import json

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.extraction import AliasDictionary
from app.semantic_extraction_v7 import semantic_recipe
from app.semantic_extraction_v8_review import review_recipe
from app.source_candidate_policy import policy_recipe
from app.source_graph_jobs import execute_source_plan
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs
from scripts.run_typed_graph_qualification import source_selection

DIRECTORY = ROOT / "results/semantic-graph-v7/composite-qualification-v1"
RUN = "collection316-source-composite-qualification-v1"


def freeze():
    old = json.loads((ROOT / "results/semantic-graph-v7/qualification-v1/plan.json").read_bytes())
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v2.json")
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = semantic_recipe(aliases, metadata_sha256=routes["account_model_metadata_sha256"])
    if recipe.model_dump(mode="json") != old["recipe"]:
        raise ValueError("COMPOSITE_NATIVE_V7_CHANGED")
    composite = {"native_recipe_sha256": recipe.fingerprint, "review": review_recipe(), "retention": policy_recipe()}
    collection, chunks, units = load_collection_inputs()
    excluded = {r["text_sha256"] for r in old["selection"]}
    original_sources = [json.loads(line) for line in (ROOT / "results/semantic-graph-v7/development-source-audit-v1/sources.jsonl").read_bytes().splitlines()]
    excluded.update(r["text_sha256"] for r in original_sources)
    # Use the complete legacy exclusion set as well, without benchmark labels.
    import sqlite3

    dbpath = ROOT / "data/cache/external-models/extraction.sqlite3"
    with sqlite3.connect(f"file:{dbpath.as_posix()}?mode=ro", uri=True) as db:
        excluded.update(json.loads(row[0])["input_sha256"] for row in db.execute("SELECT claim FROM extraction WHERE namespace='LIVE'"))
    selected = list(source_selection(units, chunks, excluded)[:24])
    for unit in selected:
        unit["v7_unit_id"] = stable_id("semantic-v7-unit", source=collection.manifest_sha256,
            text=unit["text_sha256"], recipe=recipe.fingerprint)
    if len(selected) != 24 or len({u["selected_group"] for u in selected}) != 24:
        raise ValueError("COMPOSITE_EXPECTED_24_GROUPS")
    plan = {"version": "source-composite-qualification-v1", "run_id": RUN, "recipe": recipe.model_dump(mode="json"),
        "composite_recipe": composite, "selection": selected, "source_manifest_sha256": collection.manifest_sha256,
        "maximum_new_calls": 24, "run_cap_micro_usd": 500000, "actual_cost_stop_between_calls_micro_usd": 50000,
        "price_sheet_sha256": old["price_sheet_sha256"], "funding_profile": "AUDITED_2_TO_8_USD_V2",
        "excluded_source_hashes_sha256": sha256_bytes(canonical_json(sorted(excluded))), "excluded_unique_texts": len(excluded),
        "source_selection": "One relation-cue preferred unclaimed text per URI group; not benchmark questions or independent holdout",
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in
            ("app/source_graph_jobs.py", "scripts/qualify_source_graph_composite.py")},
        "human_reviewed": 0, "full_batch_enabled": False, "serving_promotion": False,
        "notes": ["V8 and retention policy were developed after inspecting V7 outputs.",
            "These 24 extractions use the same native V7 prompt and separately frozen deterministic review/retention."]}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    path, raw = DIRECTORY / "plan.json", canonical_json(plan) + b"\n"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("COMPOSITE_FROZEN_PLAN_CHANGED")
    path.write_bytes(raw)
    return plan, chunks, recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    plan, chunks, recipe = freeze()
    if args.execute or args.replay:
        return execute_source_plan(ROOT, DIRECTORY, plan, chunks, recipe, replay=args.replay)
    print(json.dumps({"status": "COMPOSITE_FROZEN_24_SOURCES_NOT_EXECUTED", "source_units": 24, "model_calls": 0, "aws_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
