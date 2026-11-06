"""Local C8/C12 same-pool comparison and eight new development control groups.

Stage 1 freezes code/tasks; stage 2 produces question-only core-six pools;
stage 3 packs/reviews sources. Replay reruns packing/review, NOT retrieval/ONNX.
Q/QB controls without native preparation remain NOT_RUN rather than substituting.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.graph_collection_migration import collection_records
from app.ingestion.text import ProxyTokenizer
from app.model_gateway import safe_failure_code
from app.profiled_generation import ProfiledGenerationRecipe, profiled_generation_request
from app.profiled_retrieval import CORE, ProfileRetrievalTask, retrieve_profile_pool
from app.profiled_source_context import (
    SourceContextProfile,
    pack_profiled_context,
    render_profiled_context,
)
from app.question_coverage_review import (
    QuestionRubric,
    Requirement,
    ReviewedWitness,
    load_rubric,
    make_witness,
)
from scripts import amend_question_coverage as source_review
from scripts import review_question_coverage as parent
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.profiled_coverage_audit import audit_profiled_coverage
from scripts.run_collection_benchmark import OfflineRuntime
from scripts.run_facet_recall_v5 import inputs as recorded_tasks

OUTPUT = parent.ROOT / "results/context-profiles-v1"
CONTROL_SCOPE = "configs/evaluation/context-profile-controls-v1.json"
RECORDED = "results/collection319-methods-v1/receipts.json"
FAMILIES = ("parent-section-recall-v1", "bounded-ancestor-recall-v1")
CODE = ("app/profiled_source_context.py", "app/profiled_generation.py", "app/profiled_retrieval.py",
    "scripts/run_context_profiles.py", "scripts/profiled_coverage_audit.py", "app/question_coverage_review.py",
    "app/authorization_retrieval.py", "app/collection_answering.py", "app/collection_lexical.py", "app/collections.py",
    "app/reranking.py", "app/local_bge.py", "app/generation.py", "app/context.py", "app/source_scope.py",
    "app/answer_assessment.py", "app/graph_collection_migration.py", "scripts/migrate_collection319_graph.py",
    "scripts/run_collection_benchmark.py", "scripts/run_facet_recall_v5.py", "app/embeddings.py", "app/ingestion/text.py",
    "app/contracts.py", "scripts/review_question_coverage.py", "scripts/amend_question_coverage.py")


def load_review():
    rubric = load_rubric((parent.ROOT / parent.RUBRIC).read_bytes(), trusted_sha256=parent.RUBRIC_SHA)
    data = json.loads((source_review.OUTPUT / "witnesses.json").read_bytes())
    witnesses = tuple(ReviewedWitness.model_validate_json(canonical_json(w)) for w in data["witnesses"])
    controls = json.loads((parent.ROOT / CONTROL_SCOPE).read_bytes())
    if controls["parent_rubric_sha256"] != parent.RUBRIC_SHA:
        raise ValueError("PROFILE_CONTROL_PARENT_RUBRIC_PIN")
    return rubric, witnesses, controls


def control_review(rubric, witnesses, scope, control_id):
    ids = scope["control_requirements"][control_id]
    if not ids:
        return None
    lookup = {r.id: r for r in rubric.requirements}
    lookup.update({r["id"]: Requirement.model_validate_json(canonical_json(r)) for r in scope["additional_requirements"]})
    reqs = tuple(lookup[key].model_copy(update={"intent": control_id}) for key in ids)
    selected = tuple(w for w in witnesses if w.requirement_id in ids)
    if "C6_KEY_TYPE" in ids:
        source = next(s for w in witnesses if w.requirement_id == "S6" for s in w.spans
            if s.source_uri.endswith("/UsingKMSEncryption.html"))
        selected += (make_witness("C6_KEY_TYPE", "EXPLICIT_SUPPORT", (source,),
            "Key-type-only control: explicit AWS-managed versus customer-managed cross-account sharing paragraph; policy configuration is not asked here."),)
    scoped = QuestionRubric.model_validate_json(canonical_json(rubric.model_dump(mode="json") | {
        "requirements": [r.model_dump(mode="json") for r in reqs], "controls_not_yet_retrieved": [],
        "exposure": "AI authored development controls; requirement scope refined before their first retrieval, with exposed related source domain."}))
    return scoped, selected


def tasks():
    rubric, _, _ = load_review()
    return [ProfileRetrievalTask(task_id=c.id, question=getattr(c, lang), search=c.en, language=lang,
        query_track="ORIGINAL_EN" if lang == "en" else "AUTHOR_SUPPLIED_EN_REFERENCE")
        for c in rubric.controls_not_yet_retrieved for lang in ("en", "vi")]


def recorded_pools():
    task_rows, raw = recorded_tasks()  # Checks original manifest/code/receipt lineage.
    lookup = {(t["intent"], t["language"]): t for t in task_rows}
    rows = []
    for r in raw["cells"]:
        if (r["case"], r["language"]) not in lookup:
            continue
        task = lookup[(r["case"], r["language"])]
        decisions = r["trace"]["source_message_packing"]["decisions"]
        bundles = []
        for d in decisions:
            support = tuple(dict.fromkeys((d["candidate_chunk_id"], *d["supporting_chunk_ids"])))
            facts = d["path_assertion_ids"]
            bundles.append({"bundle_id": stable_id("profile-recorded-bundle", chunk=d["candidate_chunk_id"], facts=facts),
                "chunk_ids": list(support), "assertion_ids": facts, "query_index": 0})
        order = [d["candidate_chunk_id"] for d in decisions]
        pool_ids = r["trace"]["multi_query"]["ranked_chunk_ids"] if r["method"].startswith("hybrid_multi_query") else order
        if set(pool_ids) != set(order):
            raise ValueError("PROFILE_RECORDED_POOL_OR_SELECTION_CHANGED")
        rows.append({"task": {"task_id": r["case"], "question": task["question"], "search": task["search"],
            "language": r["language"], "query_track": "RECORDED_NATIVE_V3_REWRITE" if r["language"] == "vi" else "ORIGINAL_EN"},
            "method": r["method"], "mechanism_status": r["outcome"]["mechanism_status"], "pool_ids": pool_ids,
            "bundles": bundles, "balanced": False, "query_order": [0], "source_annotations_used_for_ranking": False,
            "hidden_fallbacks": 0, "family": "recorded-eight-methods", "recorded_qb_allocator_used": r["method"] == "hybrid_multi_query_balanced",
            "pool_fingerprint": sha256_bytes(canonical_json({"pool": pool_ids,
                "bundles": sorted(bundles, key=lambda b: b["bundle_id"])})),
            "native_semantic_approval": "NOT_PROMOTED_EXISTING_NATIVE_PLANS_RETAINED", "trace": {}})
    for family in FAMILIES:
        directory = parent.ROOT / "results" / family
        data = json.loads((directory / "receipts.json").read_bytes())
        report = json.loads((directory / "report.json").read_bytes())
        if report["receipts_sha256"] != sha256_bytes((directory / "receipts.json").read_bytes()):
            raise ValueError("PROFILE_RECORDED_FAMILY_PIN")
        for r in data["cells"]:
            if "RRF" not in r["arm"]:
                continue
            rows.append({"task": {"task_id": r["intent"], "question": r["question"], "search": r["search"],
                "language": r["language"], "query_track": "RECORDED_NATIVE_V3_REWRITE" if r["language"] == "vi" else "ORIGINAL_EN"},
                "method": r["arm"], "mechanism_status": "RECORDED_LITERAL_PROBE_DERIVED_POOL_NOT_NEW_NATIVE_PLAN",
                "pool_ids": r["pool_ids"], "pool_fingerprint": r["pool_fingerprint"], "bundles": r["bundles"],
                "balanced": r["arm"].endswith("_QB"), "query_order": r["pool_trace"]["query_order"],
                "source_annotations_used_for_ranking": False, "hidden_fallbacks": 0, "family": family, "trace": {}})
    if len(rows) != 64:
        raise ValueError("PROFILE_RECORDED_MATRIX_BOUND")
    return rows


def freeze():
    collection, runtime_paths, graph = load_migrated()
    artifact_paths = [parent.RUBRIC, CONTROL_SCOPE, RECORDED, "configs/models/external-direct-v1.json",
        "configs/models/bge-small-en-v1.5-onnx.json", "configs/models/ms-marco-MiniLM-L2-v2-onnx.json",
        "results/question-coverage-v1/semantic-amendment-v2/witnesses.json",
        "results/question-coverage-v1/budget-feasibility-v1/report.json",
        *(f"results/{f}/receipts.json" for f in FAMILIES)]
    immutable(OUTPUT / "scope.json", {"version": "same-pool-source-profiles-v1", "profiles": [8, 12],
        "user_message_proxy_token_cap": 4096, "output_protocol": "V12 common 8 claims / 16 citations / 1600 output tokens, not yet invoked",
        "collection_sha256": collection.manifest_sha256, "graph_release_sha256": graph.release_sha256,
        "lexical_manifest_sha256": sha256_bytes(Path(runtime_paths["lexical_manifest_path"]).read_bytes()),
        "graph_stored_facts": len(graph.facts), "tasks": [t.model_dump(mode="json") for t in tasks()],
        "recorded_pools": 64, "new_core_control_pools": 96, "context_cells": 320,
        "new_control_native_q_qb": "NOT_RUN_NEEDS_SEPARATE_NATIVE_PREPARATION",
        "code_sha256": {p: sha256_bytes((parent.ROOT / p).read_bytes()) for p in CODE},
        "input_sha256": {p: sha256_bytes((parent.ROOT / p).read_bytes()) for p in artifact_paths},
        "oracle_used_for_retrieval": False, "new_api_calls": 0, "aws_calls": 0, "serving_published": False})


def check_pins():
    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    for pins in (scope["code_sha256"], scope["input_sha256"]):
        if any(sha256_bytes((parent.ROOT / p).read_bytes()) != h for p, h in pins.items()):
            raise ValueError("PROFILE_SCOPE_PIN")
    return scope


def prepare_pools():
    scope = check_pins()
    if (OUTPUT / "pools.json").exists():
        raise ValueError("PROFILE_POOLS_ALREADY_EXIST_USE_REPLAY")
    collection, paths, graph = load_migrated()
    if collection.manifest_sha256 != scope["collection_sha256"] or graph.release_sha256 != scope["graph_release_sha256"]:
        raise ValueError("PROFILE_COLLECTION_OR_GRAPH_PIN")
    rows = recorded_pools()
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection,
            trusted_manifest_sha256=scope["lexical_manifest_sha256"]) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
        immutable(OUTPUT / "runtime-configuration.json", runtime.configuration() | {
            "active_profile_graph_sha256": graph.release_sha256, "active_profile_graph_facts": len(graph.facts),
            "query_track": "EN original / VI with author-supplied EN reference, not native rewrite qualification",
            "legacy_recipe_used_for_pool_retrieval_only": True, "generation_dispatched": False})
        for task in tasks():
            for method in CORE:
                try:
                    row = retrieve_profile_pool(collection, lexical, graph, runtime, task, method)
                except Exception as exc:
                    row = {"task": task.model_dump(mode="json"), "method": method, "mechanism_status": "FAILED_NO_FALLBACK",
                        "failure_category": safe_failure_code(exc), "source_annotations_used_for_ranking": False, "hidden_fallbacks": 0}
                row["family"] = "new-development-controls"
                rows.append(row)
            print(json.dumps({"prepared_control": task.task_id, "language": task.language, "methods": len(CORE)}), flush=True)
    immutable(OUTPUT / "pools.json", {"scope_sha256": sha256_bytes((OUTPUT / "scope.json").read_bytes()), "rows": rows,
        "not_run": [{"task_id": t.task_id, "language": t.language, "method": m, "status": "NOT_RUN_NEEDS_NATIVE_PREPARATION"}
            for t in tasks() for m in ("hybrid_multi_query", "hybrid_multi_query_balanced")],
        "new_model_api_calls": 0, "new_aws_calls": 0})
    immutable(OUTPUT / "pool-manifest.json", {"pools_sha256": sha256_bytes((OUTPUT / "pools.json").read_bytes()),
        "runtime_configuration_sha256": sha256_bytes((OUTPUT / "runtime-configuration.json").read_bytes()),
        "rows": len(rows), "new_control_inputs": 16, "new_core_control_pools": 96,
        "onnx_retrieval_replay_claimed": False, "new_model_api_calls": 0, "new_aws_calls": 0})


def evaluate():
    scope = check_pins()
    raw = (OUTPUT / "pools.json").read_bytes()
    manifest = json.loads((OUTPUT / "pool-manifest.json").read_bytes())
    if manifest["pools_sha256"] != sha256_bytes(raw):
        raise ValueError("PROFILE_POOL_MANIFEST_PIN")
    if manifest["runtime_configuration_sha256"] != sha256_bytes((OUTPUT / "runtime-configuration.json").read_bytes()):
        raise ValueError("PROFILE_RUNTIME_CONFIGURATION_PIN")
    pools = json.loads(raw)
    if pools["scope_sha256"] != sha256_bytes((OUTPUT / "scope.json").read_bytes()):
        raise ValueError("PROFILE_POOL_SCOPE_PIN")
    collection, _, graph = load_migrated()
    if collection.manifest_sha256 != scope["collection_sha256"] or graph.release_sha256 != scope["graph_release_sha256"]:
        raise ValueError("PROFILE_COLLECTION_PIN")
    docs, chunks, _ = collection_records(collection)
    records = {f.assertion_id: json.loads(f.context_json) for f in graph.facts}
    rubric, witnesses, control_scope = load_review()
    metadata = json.loads((parent.ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = ProfiledGenerationRecipe(model_metadata_sha256=metadata["account_model_metadata_sha256"])
    rows = []
    for pool in pools["rows"]:
        t = pool["task"]
        if pool["source_annotations_used_for_ranking"] or pool["hidden_fallbacks"]:
            raise ValueError("PROFILE_RUNTIME_LABELS_OR_FALLBACK")
        for cap in (8, 12):
            if pool["mechanism_status"] == "FAILED_NO_FALLBACK":
                scoped = control_review(rubric, witnesses, control_scope, t["task_id"])
                n = 0 if scoped is None else len(scoped[0].requirements)
                rows.append({"family": pool["family"], "task": t, "method": pool["method"], "profile_chunk_cap": cap,
                    "mechanism_status": "FAILED_NO_FALLBACK", "failure_category": pool["failure_category"],
                    "review": None if not n else {"requirements_total": n, "reviewed_requirements_present": 0,
                        "all_reviewed_requirements_present": False, "review_executed": False, "failed_included_in_denominator": True},
                    "packing": {"used_graph": False, "hidden_fallbacks": 0}, "user_message_proxy_tokens": 0,
                    "answer_generation_executed": False, "human_reviewers": 0})
                continue
            context, packing = pack_profiled_context(docs, chunks, tuple(pool["pool_ids"]), pool["bundles"], question=t["question"],
                language=t["language"], tenant_id="demo", collection_sha256=collection.manifest_sha256,
                profile=SourceContextProfile(max_chunks=cap), balanced=pool["balanced"], query_order=pool["query_order"], graph_records=records)
            system, prompt = profiled_generation_request(context, docs, chunks, recipe)
            review = None
            if pool["family"] == "new-development-controls":
                scoped = control_review(rubric, witnesses, control_scope, t["task_id"])
                if scoped is not None:
                    review = audit_profiled_coverage(*scoped, docs, chunks, context, intent=t["task_id"])
            else:
                review = audit_profiled_coverage(rubric, witnesses, docs, chunks, context, intent=t["task_id"])
            mechanism = ("USED_GRAPH" if packing["used_graph"] else "GRAPH_ZERO_EXPANSION_NOT_READY_FOR_ANSWER"
                if pool["method"].startswith("graph_") else pool["mechanism_status"])
            rows.append({"family": pool["family"], "task": t, "method": pool["method"], "profile_chunk_cap": cap,
                "pool_fingerprint": pool["pool_fingerprint"], "source_message": prompt,
                "context_chunk_ids": context.chunk_ids, "context_fingerprint": context.fingerprint,
                "source_provenance": json.loads(context.source_audit_json), "packing": packing, "review": review,
                "user_message_proxy_tokens": context.proxy_tokens, "system_proxy_tokens": ProxyTokenizer().count(system),
                "system_sha256": sha256_bytes(system.encode()), "generation_recipe_sha256": recipe.fingerprint,
                "mechanism_status": mechanism, "answer_generation_executed": False, "human_reviewers": 0})
    oracle = json.loads((parent.ROOT / "results/question-coverage-v1/budget-feasibility-v1/report.json").read_bytes())
    old_tasks, _ = recorded_tasks()
    oracle_rows = []
    for o in oracle["intents"]:
        keys = tuple(s["chunk_id"] for s in o["example_annotation_oracle_sources"])
        for t in (t for t in old_tasks if t["intent"] == o["intent"]):
            context = render_profiled_context(docs, chunks, keys, question=t["question"], language=t["language"],
                tenant_id="demo", collection_sha256=collection.manifest_sha256, profile=SourceContextProfile(max_chunks=12))
            review = audit_profiled_coverage(rubric, witnesses, docs, chunks, context, intent=t["intent"])
            oracle_rows.append({"intent": t["intent"], "language": t["language"], "oracle_not_runtime": True,
                "chunks": len(keys), "user_message_proxy_tokens": context.proxy_tokens,
                "all_reviewed_requirements_present": review["all_reviewed_requirements_present"]})
    if len(rows) != scope["context_cells"]:
        raise ValueError("PROFILE_CONTEXT_MATRIX_BOUND")
    return {"version": "same-pool-source-profiles-v1", "scope_sha256": sha256_bytes((OUTPUT / "scope.json").read_bytes()),
        "pools_sha256": manifest["pools_sha256"], "cells": rows, "oracle_feasibility": oracle_rows,
        "not_run": pools["not_run"], "new_model_api_calls": 0, "new_aws_calls": 0, "independent_accuracy_measured": False}


def summarize(receipts):
    rows = receipts["cells"]
    summaries = []
    for family, method, cap in sorted({(r["family"], r["method"], r["profile_chunk_cap"]) for r in rows}):
        own = [r for r in rows if (r["family"], r["method"], r["profile_chunk_cap"]) == (family, method, cap)]
        eligible = [r["review"] for r in own if r["review"] is not None]
        summaries.append({"family": family, "method": method, "profile_chunk_cap": cap, "inputs": len(own),
            "review_eligible_inputs": len(eligible), "all_requirements_present_inputs": sum(r["all_reviewed_requirements_present"] for r in eligible),
            "reviewed_requirements_present": sum(r["reviewed_requirements_present"] for r in eligible),
            "requirements_total": sum(r["requirements_total"] for r in eligible),
            "user_message_tokens_max": max(r["user_message_proxy_tokens"] for r in own),
            "used_graph_inputs": sum(r["packing"]["used_graph"] for r in own),
            "graph_zero_inputs": sum(r["mechanism_status"] == "GRAPH_ZERO_EXPANSION_NOT_READY_FOR_ANSWER" for r in own)})
        summaries[-1]["failed_inputs_included"] = sum(r["mechanism_status"] == "FAILED_NO_FALLBACK" for r in own)
    return {"status": "LOCAL_PROFILE_AND_REVIEW_RESULTS_NOT_ANSWER_ACCURACY", "context_cells": len(rows),
        "underlying_intents": 10, "original_language_inputs": 20, "new_control_groups": 8,
        "new_control_native_q_qb_not_run": len(receipts["not_run"]), "summaries": summaries,
        "oracle_feasibility": receipts["oracle_feasibility"], "new_model_api_calls": 0, "new_aws_calls": 0,
        "human_reviewers": 0, "serving_published": False, "output_protocol_native_calls": 0,
        "replay_scope": "Source packing and review from frozen pools; not fresh ONNX retrieval or native query preparation"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for flag in ("freeze", "prepare-pools", "run", "replay"):
        group.add_argument("--"+flag, action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.create_connection", "socket.getaddrinfo"):
            stack.enter_context(patch(target, side_effect=AssertionError("PROFILE_EXPERIMENT_NETWORK_FORBIDDEN")))
        if args.freeze:
            freeze()
            print("Profile comparison scope frozen; no inference or retrieval.")
        elif args.prepare_pools:
            prepare_pools()
        else:
            receipts = evaluate()
            report = summarize(receipts)
            if args.run:
                immutable(OUTPUT / "receipts.json", receipts)
                immutable(OUTPUT / "report.json", report)
            else:
                if (canonical_json(receipts) != canonical_json(json.loads((OUTPUT / "receipts.json").read_bytes()))
                        or report != json.loads((OUTPUT / "report.json").read_bytes())):
                    raise ValueError("PROFILE_PACKING_REVIEW_REPLAY_MISMATCH")
                immutable(OUTPUT / "replay.json", {"status": "EXACT_PACKING_AND_REVIEW_REPLAY_PASS",
                    "receipts_sha256": sha256_bytes((OUTPUT / "receipts.json").read_bytes()), "cells": len(receipts["cells"]),
                    "new_model_api_calls": 0, "new_aws_calls": 0, "socket_access_blocked": True,
                    "fresh_onnx_retrieval_executed": False})
            print(json.dumps({k: v for k, v in report.items() if k != "summaries"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
