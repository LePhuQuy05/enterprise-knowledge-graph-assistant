"""Freeze/run one 100-source candidate tranche as three disjoint bounded jobs.

No graph publication: every retained candidate still needs source review.
Selection is source-only, balanced over 24 groups, with no benchmark labels.
"""

import argparse
import json
import re
import xml.etree.ElementTree as ET
from collections import defaultdict

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.extraction import AliasDictionary
from app.semantic_extraction_v6 import VERBS
from app.semantic_extraction_v7 import semantic_recipe
from app.semantic_extraction_v8_review import review_recipe
from app.source_candidate_policy import policy_recipe
from app.source_graph_jobs import execute_source_plan
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

DIRECTORY = ROOT / "results/semantic-graph-v7/tranche-001"


def source_gate():
    paths = [ROOT / "results/semantic-graph-v7/qualification-v1/v8-review-v1/source-review.json",
             ROOT / "results/semantic-graph-v7/composite-qualification-v1/source-review.json"]
    reports = [json.loads(p.read_bytes()) for p in paths]
    if any(r["status"] != "ALL_COMPOSITE_CANDIDATES_HAVE_BOUND_AI_SOURCE_JUDGMENTS" for r in reports):
        raise ValueError("TRANCHE_COMPLETE_SOURCE_JUDGMENTS_REQUIRED")
    if any(r["retention_recipe"] != policy_recipe() for r in reports):
        raise ValueError("TRANCHE_REVIEWED_RETENTION_POLICY_CHANGED")
    tests_path = ROOT / "results/semantic-graph-v7/full-tests.xml"
    suites = list(ET.parse(tests_path).getroot())
    if not suites or sum(int(s.attrib.get("tests", 0)) - int(s.attrib.get("skipped", 0)) for s in suites) < 2201 or any(
            int(s.attrib.get("failures", 0)) or int(s.attrib.get("errors", 0)) for s in suites):
        raise ValueError("TRANCHE_PASSING_FULL_REGRESSION_REQUIRED")
    for stage in ("qualification-v1", "composite-qualification-v1"):
        directory = ROOT / "results/semantic-graph-v7" / stage
        attempt = json.loads((directory / "attempt-001.json").read_bytes())
        replay = json.loads((directory / "replay.json").read_bytes())
        if (attempt["error"] is not None or attempt["processed"] != attempt["planned"]
                or attempt["run_counter"]["reserved_micro_usd"] or replay["status"] != "REPLAY_MATCH"
                or replay["provider_calls_this_invocation"]):
            raise ValueError("TRANCHE_NATIVE_RUN_AND_REPLAY_RECONCILIATION_REQUIRED")
    return {"status": "BOUNDED_CANDIDATE_TRANCHE_ONLY_NOT_SERVING_OR_QUALITY_ACCEPTANCE",
        "source_review_sha256": {str(p.relative_to(ROOT)): sha256_bytes(p.read_bytes()) for p in paths},
        "full_tests_sha256": sha256_bytes(tests_path.read_bytes()), "human_reviewed": 0,
        "independent_gold": False, "require_source_review_every_retained_candidate": True}


def select_balanced(units, chunks, excluded, *, maximum=100):
    trigger = re.compile(r"\b(?:" + "|".join(VERBS.values()) + r"|must have permissions?)\b", re.I)
    groups = defaultdict(list)
    for unit in units:
        if unit["text_sha256"] in excluded or not trigger.search(chunks[unit["occurrences"][0]["chunk_id"]].text):
            continue
        for group in {o["service_group"] for o in unit["occurrences"]}:
            groups[group].append(unit)
    selected, seen, group_docs = [], set(), defaultdict(set)
    while len(selected) < maximum:
        added = False
        for group, rows in sorted(groups.items()):
            def ordering(unit):
                docs = {o["document_version_id"] for o in unit["occurrences"]}
                return bool(docs & group_docs[group]), unit["text_sha256"]
            row = next((u for u in sorted(rows, key=ordering) if u["text_sha256"] not in seen), None)
            if row is None:
                continue
            selected.append(dict(row, selected_group=group))
            seen.add(row["text_sha256"])
            group_docs[group].update(o["document_version_id"] for o in row["occurrences"])
            added = True
            if len(selected) == maximum:
                break
        if not added:
            raise ValueError("TRANCHE_INSUFFICIENT_DISTINCT_TRIGGER_ELIGIBLE_SOURCES")
    return selected


def freeze():
    gate = source_gate()
    collection, chunks, units = load_collection_inputs()
    old = json.loads((ROOT / "results/semantic-graph-v7/composite-qualification-v1/plan.json").read_bytes())
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v2.json")
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = semantic_recipe(aliases, metadata_sha256=routes["account_model_metadata_sha256"])
    composite = {"native_recipe_sha256": recipe.fingerprint, "review": review_recipe(), "retention": policy_recipe()}
    if recipe.model_dump(mode="json") != old["recipe"] or composite != old["composite_recipe"]:
        raise ValueError("TRANCHE_FROZEN_COMPOSITE_CHANGED")
    excluded = set()
    for stage in ("qualification-v1", "composite-qualification-v1"):
        p = json.loads((ROOT / "results/semantic-graph-v7" / stage / "plan.json").read_bytes())
        excluded.update(u["text_sha256"] for u in p["selection"])
    selected = select_balanced(units, chunks, excluded)
    for unit in selected:
        unit["v7_unit_id"] = stable_id("semantic-v7-unit", source=collection.manifest_sha256,
            text=unit["text_sha256"], recipe=recipe.fingerprint)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    code = {p: sha256_bytes((ROOT / p).read_bytes()) for p in
            ("app/source_graph_jobs.py", "scripts/run_source_graph_tranche.py")}
    plans = []
    for i in range(3):
        job_dir = DIRECTORY / f"job-{i + 1}"
        job_dir.mkdir(exist_ok=True)
        plan = {"version": "source-graph-tranche-001", "run_id": f"collection316-source-tranche-001-job-{i + 1}",
            "recipe": recipe.model_dump(mode="json"), "composite_recipe": composite,
            "selection": selected[i::3], "source_manifest_sha256": collection.manifest_sha256,
            "maximum_new_calls": len(selected[i::3]), "run_cap_micro_usd": 500000,
            "actual_cost_stop_between_calls_micro_usd": 100000, "price_sheet_sha256": old["price_sheet_sha256"],
            "code_sha256": code, "gate": gate, "full_batch_enabled": False, "serving_promotion": False}
        raw, path = canonical_json(plan) + b"\n", job_dir / "plan.json"
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("TRANCHE_FROZEN_PLAN_CHANGED")
        path.write_bytes(raw)
        plans.append(plan)
    return plans, chunks, recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=int, choices=(1, 2, 3))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    plans, chunks, recipe = freeze()
    if args.execute or args.replay:
        if args.job is None:
            raise ValueError("TRANCHE_EXPLICIT_JOB_REQUIRED")
        return execute_source_plan(ROOT, DIRECTORY / f"job-{args.job}", plans[args.job - 1], chunks, recipe, replay=args.replay)
    print(json.dumps({"status": "FROZEN_100_SOURCE_CANDIDATE_TRANCHE_NOT_EXECUTED", "source_units": 100,
        "job_sizes": [len(p["selection"]) for p in plans], "model_calls": 0, "aws_calls": 0,
        "automatic_serving_promotion": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
