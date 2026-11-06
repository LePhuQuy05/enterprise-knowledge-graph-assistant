"""Read-only second-pass source/path/fairness and pool-reachability diagnostics."""

import argparse
import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes
from app.graph_collection_migration import collection_records
from app.ingestion.text import ProxyTokenizer
from app.question_coverage_review import PRESENT
from app.source_dependency_audit import missing_intervals
from scripts import run_context_profiles as experiment
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable

OUTPUT = experiment.OUTPUT / "source-audit-v1"


def freeze():
    paths = [experiment.OUTPUT / n for n in ("scope.json", "pools.json", "pool-manifest.json", "receipts.json", "report.json", "replay.json")]
    immutable(OUTPUT / "scope.json", {"version": "profile-source-path-audit-v1",
        "input_sha256": {str(p.relative_to(experiment.parent.ROOT)).replace("\\", "/"): sha256_bytes(p.read_bytes()) for p in paths},
        "auditor_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "new_model_api_calls": 0, "new_aws_calls": 0})


def audit():
    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    if (scope["auditor_sha256"] != sha256_bytes(Path(__file__).read_bytes())
            or any(sha256_bytes((experiment.parent.ROOT / p).read_bytes()) != h for p, h in scope["input_sha256"].items())):
        raise ValueError("PROFILE_SOURCE_AUDIT_PIN")
    experiment.check_pins()
    collection, _, graph = load_migrated()
    docs, chunks, _ = collection_records(collection)
    pools = json.loads((experiment.OUTPUT / "pools.json").read_bytes())["rows"]
    receipts = json.loads((experiment.OUTPUT / "receipts.json").read_bytes())["cells"]
    rubric, witnesses, control_scope = experiment.load_review()
    facts = {f.assertion_id: f for f in graph.facts}
    appearances, spans, selected_paths, failed = 0, 0, 0, 0
    grouped = defaultdict(list)
    for r in receipts:
        key = (r["family"], r["task"]["task_id"], r["task"]["language"], r["method"])
        grouped[key].append(r)
        if r["mechanism_status"] == "FAILED_NO_FALLBACK":
            failed += 1
            continue
        prompt = json.loads(r["source_message"])
        keys = r["context_chunk_ids"]
        if (prompt["untrusted_question"] != r["task"]["question"] or len(keys) > r["profile_chunk_cap"]
                or len(keys) != len(set(keys)) or r["user_message_proxy_tokens"] != ProxyTokenizer().count(r["source_message"])
                or r["user_message_proxy_tokens"] > 4096
                or [s["source_id"] for s in prompt["untrusted_sources"]] != [f"s{i+1}" for i in range(len(keys))]):
            raise ValueError("PROFILE_SOURCE_AUDIT_QUESTION_IDS_OR_BUDGET")
        for key, source in zip(keys, prompt["untrusted_sources"], strict=True):
            c = chunks[key]
            d = docs[c.document_version_id]
            if (d.normalized_text[c.start_char:c.end_char] != c.text or sha256_bytes(c.text.encode()) != c.text_sha256
                    or source["untrusted_document_title"] != d.title):
                raise ValueError("PROFILE_SOURCE_AUDIT_SOURCE_BYTES")
            cursor = 0
            for i, s in enumerate(source["spans"], 1):
                pos = c.text.find(s["text"], cursor)
                if pos < cursor or c.text[cursor:pos].strip() or s["span_id"] != f"{source['source_id']}p{i}":
                    raise ValueError("PROFILE_SOURCE_AUDIT_INTERIOR_LOSS")
                cursor = pos+len(s["text"])
                spans += 1
            if c.text[cursor:].strip():
                raise ValueError("PROFILE_SOURCE_AUDIT_TRUNCATION")
            appearances += 1
        packed_ids = r["packing"]["context_assertion_ids"]
        if (r["packing"]["used_graph"] != bool(packed_ids) or packed_ids and not r["method"].startswith("graph_")
                or r["packing"]["hidden_fallbacks"]
                or r["packing"]["partial_source_truncations"] or r["answer_generation_executed"]):
            raise ValueError("PROFILE_SOURCE_AUDIT_METHOD_OR_TRUNCATION")
        for d in r["packing"]["decisions"]:
            if d["reason"] != "SELECTED":
                continue
            if not set(d["chunk_ids"]) <= set(keys):
                raise ValueError("PROFILE_SOURCE_AUDIT_NONATOMIC_BUNDLE")
            if d["assertion_ids"]:
                if {facts[k].chunk_id for k in d["assertion_ids"]} != set(d["chunk_ids"]):
                    raise ValueError("PROFILE_SOURCE_AUDIT_NONATOMIC_PATH")
                selected_paths += 1
    for pair in grouped.values():
        if len(pair) != 2 or {r["profile_chunk_cap"] for r in pair} != {8, 12}:
            raise ValueError("PROFILE_SOURCE_AUDIT_PAIRED_PROFILES")
        if pair[0]["mechanism_status"] == "FAILED_NO_FALLBACK":
            continue
        if any(pair[0][k] != pair[1][k] for k in ("pool_fingerprint", "system_sha256", "generation_recipe_sha256")):
            raise ValueError("PROFILE_SOURCE_AUDIT_UNPAIRED_RECIPE_OR_POOL")
    reachability = []
    for pool in pools:
        if pool["mechanism_status"] == "FAILED_NO_FALLBACK":
            continue
        task = pool["task"]
        scoped = experiment.control_review(rubric, witnesses, control_scope, task["task_id"]) if pool["family"] == "new-development-controls" else (rubric, witnesses)
        if scoped is None:
            continue
        own_rubric, own_witnesses = scoped
        intervals = defaultdict(list)
        for key in pool["pool_ids"]:
            c = chunks[key]
            intervals[c.document_version_id].append((c.start_char, c.end_char))
        pool_present = {}
        relevant_intent = task["task_id"]
        for req in (r for r in own_rubric.requirements if r.intent == relevant_intent):
            if req.kind == "QUALIFIED_INFERENCE":
                pool_present[req.id] = all(pool_present[k] for k in req.requires)
            else:
                pool_present[req.id] = any(w.requirement_id == req.id and w.evidence_kind in {"EXPLICIT_SUPPORT", "QUALIFIED_INFERENCE"}
                    and all(not missing_intervals(s.span.start_char, s.span.end_char, intervals[s.document_version_id]) for s in w.spans)
                    for w in own_witnesses)
        pair = grouped[(pool["family"], task["task_id"], task["language"], pool["method"])]
        for r in pair:
            for req in r["review"]["requirements"]:
                present = req["status"] in PRESENT
                if present and not pool_present[req["requirement_id"]]:
                    raise ValueError("PROFILE_SOURCE_AUDIT_EVIDENCE_OUTSIDE_POOL")
                reachability.append({"family": pool["family"], "task_id": task["task_id"], "language": task["language"],
                    "method": pool["method"], "profile_chunk_cap": r["profile_chunk_cap"], "requirement_id": req["requirement_id"],
                    "stage": "SELECTED_REVIEWED_WITNESS" if present else "REVIEWED_WITNESS_IN_POOL_NOT_SELECTED"
                        if pool_present[req["requirement_id"]] else "REVIEWED_WITNESS_NOT_COMPLETE_IN_POOL"})
    return {"status": "SOURCE_PATH_BUDGET_AND_PROFILE_PAIRING_PASS", "cells": len(receipts), "failed_cells": failed,
        "profile_pairs": len(grouped), "rendered_source_appearances": appearances, "literal_spans": spans,
        "selected_graph_path_bundle_appearances": selected_paths, "reachability": reachability,
        "reachability_summary": dict(Counter(r["stage"] for r in reachability)),
        "review_annotations_used_only_in_evaluator": True, "all_equivalent_sources_discovered": False,
        "new_model_api_calls": 0, "new_aws_calls": 0, "semantic_answer_accuracy_measured": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.create_connection", "socket.getaddrinfo"):
            stack.enter_context(patch(target, side_effect=AssertionError("PROFILE_AUDIT_NETWORK_FORBIDDEN")))
        if args.freeze:
            freeze()
            print("Read-only profile audit frozen.")
            return
        result = audit()
        if args.replay:
            if canonical_json(result) != canonical_json(json.loads((OUTPUT / "audit.json").read_bytes())):
                raise ValueError("PROFILE_AUDIT_REPLAY_MISMATCH")
        else:
            immutable(OUTPUT / "audit.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "reachability"}))


if __name__ == "__main__":
    main()
