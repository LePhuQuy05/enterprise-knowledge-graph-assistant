"""Predeclare/replay same-pool allocation arms on exposed source controls.

All comparisons are local context diagnostics, not independent answer-quality
measurements. No provider or AWS calls are made. Existing paid code is pinned.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.authorization_evidence import load_authorization_facts
from app.authorization_retrieval import combine_evidence_graph
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.generation import generation_request
from app.graph_source_allocation import reallocate_graph_sources
from app.ingestion.text import ProxyTokenizer
from app.method_execution import ReviewedGraphExecutionV3, StrictPreparedMethod
from app.reviewed_source_graph import load_reviewed_source_graph
from app.source_message_runtime import prepare_source_message_graph
from scripts.check_authorization_evidence import CONTROLS
from scripts.pilot_reviewed_source_answers import immutable
from scripts.pilot_source_message_answers_v2 import code_hashes, prepare_cases
from scripts.prepare_authorization_evidence import ROOT, load_sources
from scripts.run_collection_benchmark import OfflineRuntime, load_inputs

OUTPUT = ROOT / "results/graph-source-allocation-v1"
DISTRACTORS = (
    ("event-metadata-qualified", "Does EventBridge encrypt the event metadata or just the event body?"),
    ("s3-pricing", "What are the pricing tiers for Amazon S3 Intelligent-Tiering?"),
    ("neptune-storage", "Which storage classes does Neptune Analytics use for snapshots?"),
    ("lambda-console-font", "How do I configure the font size in the Lambda console?"),
    ("no-catalog-entity", "How can I grow tomatoes in winter?"),
    ("kms-key-rotation", "How often does AWS KMS rotate imported key material automatically?"),
)
CODE = ("app/graph_source_allocation.py", "scripts/compare_graph_source_allocation.py")
IMMUTABLE_TRACE = ("paths", "graph_ranking", "vector_ranking", "lexical_ranking", "baseline_ranking",
    "seed_entity_ids", "catalog_links", "assertion_visits", "condition_compatibility", "release_sha256")


def load_graph():
    collection, chunks, documents = load_sources()
    config = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    reviewed = load_reviewed_source_graph(ROOT / config["release_path"], collection,
        trusted_sha256=config["release_sha256"], allow_experimental=True)
    config = json.loads((ROOT / "configs/graph/authorization-evidence-local-v1.json").read_bytes())
    auth = load_authorization_facts(ROOT / config["release_path"], collection, chunks, documents,
        trusted_sha256=config["release_sha256"], allow_experimental=True)
    return collection, combine_evidence_graph(collection, reviewed, auth)


def compare(*, replay=False):
    parent = json.loads((ROOT / "results/source-message-answers-v2/scope.json").read_bytes())
    if parent["code_sha256"] != code_hashes():
        raise ValueError("GRAPH_ALLOCATION_FROZEN_PAID_CODE_CHANGED")
    collection, graph = load_graph()
    plan = {"version": "same-pool-source-allocation-comparison-v1",
        "purpose": "EXPOSED_DEVELOPMENT_CONTEXT_ALLOCATION_NOT_ANSWER_ACCURACY",
        "controls": CONTROLS, "additional_distractors": DISTRACTORS,
        "arms": ["PREDECESSOR_TWO_CANDIDATES", "PRIORITY_2_UNIQUE_SOURCES", "PRIORITY_4_UNIQUE_SOURCES"],
        "limits": {"max_chunks": 8, "user_message_proxy_tokens": 4096, "same_candidate_pool_per_case": True},
        "selection_rule": "Evaluate all arms; no target IDs, gold evidence, source labels or service-specific boosts in allocator",
        "native_matrix": "All eight graph cells from the frozen two-intent EN/VI matrix; Q/QB/native prep unchanged",
        "graph_release_sha256": graph.release_sha256, "collection_manifest_sha256": collection.manifest_sha256,
        "parent_paid_code_sha256": parent["code_sha256"], "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "new_model_calls": 0, "aws_calls": 0, "human_reviewed": 0, "deploy": False}
    # The plan is persisted before executing any comparison.
    if replay:
        if (OUTPUT / "plan.json").read_bytes() != canonical_json(plan) + b"\n":
            raise ValueError("GRAPH_ALLOCATION_REPLAY_PLAN_CHANGED")
    else:
        immutable(OUTPUT / "plan.json", plan)
    _, _, _, _, pipeline = load_inputs()
    rows = []
    tokenizer = ProxyTokenizer()
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))

        def vector(query):
            record = runtime.cache.resolve_local("demo", text=query_embedding_input(query),
                recipe=runtime.query_recipe, encoder=runtime.encoder).record
            return np.asarray(record.vector, dtype=np.float32)

        def record(case_id, section, predecessor):
            before = predecessor.prepared
            arms = [("PREDECESSOR_TWO_CANDIDATES", predecessor)]
            arms += [(f"PRIORITY_{cap}_UNIQUE_SOURCES", reallocate_graph_sources(collection, lexical, graph,
                predecessor, priority_source_cap=cap)) for cap in (2, 4)]
            for name, result in arms:
                p = result.prepared
                system, prompt = generation_request(p.question, result.view, p.packed, "demo", p.recipe,
                    answer_language=result.answer_language)
                if (set(p.retrieved_chunk_ids) != set(before.retrieved_chunk_ids)
                        or len(p.retrieved_chunk_ids) != len(before.retrieved_chunk_ids)
                        or len(p.packed.chunk_ids) > 8 or tokenizer.count(prompt) > 4096
                        or prompt != p.packed.serialized or p.packed.proxy_tokens != tokenizer.count(prompt)
                        or any(canonical_json(result.trace[k]) != canonical_json(predecessor.trace[k]) for k in IMMUTABLE_TRACE)
                        or result.outcome.requested_method != predecessor.outcome.requested_method
                        or result.outcome.effective_method != predecessor.outcome.effective_method):
                    raise ValueError("GRAPH_ALLOCATION_EXPERIMENT_INVARIANT")
                if (predecessor.outcome.zero_expansion_reason in {"NO_LINKED_ENTITY", "NO_RELEVANT_PATH"}
                        and result.outcome.mechanism_status != "GRAPH_ZERO_EXPANSION"):
                    raise ValueError("GRAPH_ALLOCATION_ZERO_STATUS_CHANGED")
                rows.append({"id": case_id, "section": section, "arm": name, "language": result.answer_language,
                    "method": result.outcome.requested_method, "question": p.question,
                    "outcome": result.outcome.model_dump(mode="json"), "trace": result.trace,
                    "source_count": len(p.packed.chunk_ids), "context_chunk_ids": p.packed.chunk_ids,
                    "user_message_proxy_tokens": p.packed.proxy_tokens, "system_sha256": sha256_bytes(system.encode()),
                    "user_message_sha256": sha256_bytes(prompt.encode()), "request_id": p.binding.request_id,
                    "sources_added": sorted(set(p.packed.chunk_ids) - set(before.packed.chunk_ids)),
                    "sources_removed": sorted(set(before.packed.chunk_ids) - set(p.packed.chunk_ids)),
                    "same_candidate_pool": True, "answer_generated": False})
            print(json.dumps({"case": case_id, "compared_arms": 3}), flush=True)

        for name, question in (*CONTROLS, *DISTRACTORS):
            for hops in (1, 2):
                before = prepare_source_message_graph(collection, lexical, graph, question, vector(question),
                    tenant_id="demo", owner_id="graph-allocation-local", run_id="graph-source-allocation-v1",
                    recipe=runtime.recipe, hops=hops, allow_experimental=True)
                if before.prepared is None:
                    raise ValueError("GRAPH_ALLOCATION_CONTROL_NOT_READY")
                record(f"{name}-g{hops}", "SOURCE_CONTROLS" if name in dict(CONTROLS) else "ADDITIONAL_DISTRACTORS", before)
        cases, preparations = prepare_cases(collection, lexical, graph, runtime, None)
        for case in cases:
            if case["method"].startswith("graph_"):
                record(case["id"], "NATIVE_EN_VI_GRAPH_CELLS", StrictPreparedMethod(
                    # prepare_cases saves an outcome dict; retain the actual V3 type.
                    ReviewedGraphExecutionV3.model_validate_json(
                        canonical_json(case["outcome"])), case["view"], case["prepared"], case["trace"], case["language"]))
    raw = b"".join(canonical_json(r) + b"\n" for r in rows)
    report = {"status": "SAME_POOL_CONTEXT_COMPARISON_COMPLETE_NOT_ANSWER_QUALITY", "cases": len(rows) // 3,
        "arm_rows": len(rows), "new_model_calls": 0, "aws_calls": 0, "new_cost_micro_usd": 0,
        "human_reviewed": 0, "native_preparation_receipts_reused": len(preparations), "serving_published": False,
        "all_same_pool": all(r["same_candidate_pool"] for r in rows), "receipts_sha256": sha256_bytes(raw),
        "arms": {arm: {"cases": sum(r["arm"] == arm for r in rows),
            "changed_contexts": sum(bool(r["sources_added"] or r["sources_removed"]) for r in rows if r["arm"] == arm),
            "added_source_occurrences": sum(len(r["sources_added"]) for r in rows if r["arm"] == arm),
            "removed_source_occurrences": sum(len(r["sources_removed"]) for r in rows if r["arm"] == arm),
            "zero_expansion_cases": sum(r["outcome"]["mechanism_status"] == "GRAPH_ZERO_EXPANSION" for r in rows if r["arm"] == arm),
            "max_user_message_proxy_tokens": max(r["user_message_proxy_tokens"] for r in rows if r["arm"] == arm)}
            for arm in sorted({r["arm"] for r in rows})}}
    if replay:
        if (OUTPUT / "receipts.jsonl").read_bytes() != raw:
            raise ValueError("GRAPH_ALLOCATION_EXACT_REPLAY_MISMATCH")
        immutable(OUTPUT / "replay.json", {"status": "ALL_CONTEXTS_PATHS_AND_ALLOCATION_REPLAY_MATCH",
            "cases": len(rows) // 3, "arm_rows": len(rows), "receipts_sha256": sha256_bytes(raw),
            "network_blocked": True, "new_model_calls": 0, "aws_calls": 0})
    else:
        immutable(OUTPUT / "report.json", report)
        (OUTPUT / "receipts.jsonl").write_bytes(raw)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=RuntimeError("ALLOCATION_COMPARISON_NETWORK_BLOCKED")))
        compare(replay=args.replay)
