"""Exercise source-only graph controls and recorded native Q/QB without API calls.

These are exposed development mechanism checks, not independent quality scores.
Replay requires the same release/code/controls and exact saved contexts/traces.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.external_gateway import ExternalRoutes
from app.generation import generation_request
from app.query_preparation_v3 import QueryPlanCacheV3, QueryRewriteCacheV3
from app.reviewed_source_graph import load_reviewed_source_graph
from app.strict_collection_methods import (
    prepare_strict_graph_answer,
    prepare_strict_multi_query_answer,
)
from scripts.prepare_strict_method_canary import build_plan, context_for
from scripts.run_collection_benchmark import ROOT, OfflineRuntime, load_inputs

CONTROLS = (
    ("cloudfront-metrics", "How does CloudFront send metrics to CloudWatch for distributions and edge functions?"),
    ("rds-metrics", "How often does Amazon RDS send metrics to CloudWatch for each active database?"),
    ("secrets-native-events", "How does Secrets Manager publish native events to EventBridge without CloudTrail?"),
    ("s3-decrypt", "How does Amazon S3 decrypt encrypted data and remove the plaintext data key from memory?"),
    ("dynamodb-disabled-stream", "After disabling a streaming destination, can DynamoDB still send in-flight records until propagation completes?"),
    ("s3-lambda-permissions", "Why can S3 invoke Lambda but the function cannot read an SSE-KMS object?"),
    ("eventbridge-metadata-negative", "Does EventBridge encrypt event metadata?"),
    ("outside-reviewed-coverage", "How does Neptune Analytics handle a graph snapshot?"),
    ("two-source-evidence-not-workflow-proof", "How does Secrets Manager publish native events to EventBridge, and how does EventBridge encrypt events when the target is another event bus in a different Region?"),
)
CODE_PATHS = ("app/reviewed_source_graph.py", "app/reviewed_graph_runtime.py", "app/context.py",
    "app/method_execution.py", "app/strict_collection_methods.py", "app/collection_graph_v2.py",
    "app/collection_answering.py", "app/collection_lexical.py", "app/query_preparation_v3.py",
    "app/query_planning_v2.py", "app/query_rewrite_v2.py", "scripts/check_reviewed_source_runtime.py")


def code_hashes():
    return {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE_PATHS}


def check(output, *, replay=False):
    _, _, collection, _, pipeline = load_inputs()
    selection_path = ROOT / "configs/graph/reviewed-source-runtime-v1.json"
    selection = json.loads(selection_path.read_bytes())
    graph = load_reviewed_source_graph(ROOT / selection["release_path"], collection,
        trusted_sha256=selection["release_sha256"], allow_experimental=True)
    plan = {"version": "reviewed-source-runtime-controls-v1", "purpose": "EXPOSED_DEVELOPMENT_MECHANISMS_NOT_QUALITY_BENCHMARK",
        "source_manifest_sha256": collection.manifest_sha256, "reviewed_release_sha256": graph.release_sha256,
        "selection_sha256": sha256_bytes(selection_path.read_bytes()), "controls": CONTROLS,
        "code_sha256": code_hashes(), "provider_calls": 0, "aws_calls": 0, "generation_calls": 0}
    raw_plan = canonical_json(plan) + b"\n"
    if replay:
        if (output / "plan.json").read_bytes() != raw_plan:
            raise ValueError("REVIEWED_RUNTIME_REPLAY_PLAN_CHANGED")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "plan.json").write_bytes(raw_plan)
    rows, durations = [], []
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ())
        vectors = {}

        def vector(text):
            if text not in vectors:
                lookup = runtime.cache.resolve_local(collection.manifest.tenant_id, text=query_embedding_input(text),
                    recipe=runtime.query_recipe, encoder=runtime.encoder)
                vectors[text] = np.asarray(lookup.record.vector, dtype=np.float32)
            return vectors[text]

        def record(name, result):
            if result.prepared is None:
                raise ValueError("REVIEWED_RUNTIME_CONTROL_NOT_COMPLETED")
            system, prompt = generation_request(result.prepared.question, result.view, result.prepared.packed,
                collection.manifest.tenant_id, result.prepared.recipe, answer_language=result.answer_language)
            rows.append({"case": name, "outcome": result.outcome.model_dump(mode="json"), "trace": result.trace,
                "packed": json.loads(result.prepared.packed.serialized), "system_sha256": sha256_bytes(system.encode()),
                "prompt_sha256": sha256_bytes(prompt.encode()), "generation_executed": False})

        for name, question in CONTROLS:
            query_vector = vector(question)
            for hops in (1, 2):
                started = perf_counter()
                result = prepare_strict_graph_answer(collection, lexical, graph, question, query_vector,
                    tenant_id=collection.manifest.tenant_id, owner_id="source-runtime-local", run_id="source-runtime-v1",
                    recipe=runtime.recipe, hops=hops, allow_experimental=True)
                zero_controls = {"s3-lambda-permissions", "eventbridge-metadata-negative", "outside-reviewed-coverage"}
                if (result.outcome.mechanism_status == "GRAPH_ZERO_EXPANSION") != (name in zero_controls):
                    raise ValueError("REVIEWED_RUNTIME_EXPOSED_CONTROL_RELEVANCE_REGRESSION")
                negative_ids = {e.assertion_id for e in graph.edges if e.polarity != "POSITIVE"}
                if negative_ids & set(result.outcome.context_assertion_ids):
                    raise ValueError("REVIEWED_RUNTIME_NEGATIVE_EDGE_EXPANDED")
                if name == "two-source-evidence-not-workflow-proof" and hops == 2 and not any(
                        p["hop"] == 2 for p in result.trace["packed_paths"]):
                    raise ValueError("REVIEWED_RUNTIME_TWO_HOP_SUPPORT_MISSING")
                durations.append({"case": name, "method": f"graph_{hops}hop", "duration_ms": round((perf_counter() - started) * 1000, 3)})
                record(name, result)
            print(json.dumps({"case": name, "receipts": len(rows)}), flush=True)
        # Real V3 provider receipts are read from the existing cache. Cache miss
        # raises explicitly because no gateway is supplied; no synthetic plan.
        graph_rows = tuple(rows)
        tasks, _ = build_plan(3)
        task = tasks[0]
        cache_dir = ROOT / "data/cache/strict-method-preparation-v3"
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        rewrite = QueryRewriteCacheV3(cache_dir).resolve(context_for(task, "REWRITE", 3), question=task.question,
            language=task.language, routes=routes)
        search = rewrite.rewrite.search_query
        lookup = QueryPlanCacheV3(cache_dir).resolve(context_for(task, "PLAN_QUERY", 3), question=task.question,
            search_query=search, language=task.language, routes=routes)
        if not rewrite.cache_hit or not lookup.cache_hit or lookup.plan.origin != "NATIVE_V3":
            raise ValueError("REVIEWED_RUNTIME_NATIVE_PREPARATION_CACHE_REQUIRED")
        paired = []
        for mode in ("hybrid_multi_query", "hybrid_multi_query_balanced"):
            result = prepare_strict_multi_query_answer(collection, lexical, mode=mode, question=task.question,
                search_query=search, query_vector=vector(search), subquery_vectors=tuple(vector(q) for q in lookup.plan.subqueries),
                plan=lookup.plan, tenant_id=collection.manifest.tenant_id, owner_id="source-runtime-local",
                run_id="source-runtime-v1", recipe=runtime.recipe, language=task.language)
            record("native-v3-step-functions-vi", result)
            paired.append(result)
        if paired[0].trace["multi_query"] != paired[1].trace["multi_query"]:
            raise ValueError("REVIEWED_RUNTIME_Q_QB_POOL_MISMATCH")
    if plan["code_sha256"] != code_hashes():
        raise ValueError("REVIEWED_RUNTIME_CODE_CHANGED_DURING_RUN")
    raw_rows = b"".join(canonical_json(row) + b"\n" for row in rows)
    report = {"status": "LOCAL_SOURCE_MECHANISMS_COMPLETE_NOT_SERVING_OR_QUALITY_EVALUATION", "receipts": len(rows),
        "graph_statuses": dict(Counter(r["outcome"]["mechanism_status"] for r in graph_rows)),
        "graph_controls": [{"case": r["case"], "method": r["outcome"]["requested_method"],
            "status": r["outcome"]["mechanism_status"], "packed_paths": len(r["trace"]["packed_paths"]),
            "max_packed_hops": max((p["hop"] for p in r["trace"]["packed_paths"]), default=0),
            "new_context_chunks_vs_base": len(r["trace"]["graph_added_context_chunk_ids"]),
            "packed_actions": sum(a.get("kind") == "ACTION" for a in r["packed"]["untrusted_graph_assertions"])} for r in graph_rows],
        "native_q_qb_same_pool": True, "native_planner_origin": lookup.plan.origin,
        "native_subqueries": list(lookup.plan.subqueries), "native_preparation_cache_hits": 2,
        "native_q_qb_selection_order_changed": paired[1].trace["selection_order_changed"],
        "native_q_qb_context_selection_decisions": len(paired[1].trace["context_selection"]["decisions"]),
        "receipts_sha256": sha256_bytes(raw_rows), "human_reviewed": 0, "new_model_calls": 0, "aws_calls": 0,
        "hidden_fallbacks": 0, "full_relation_extraction": False, "semantic_accuracy_measured": False,
        "known_graph_coverage_gaps": ["READ", "ASSUME_ROLE", "S3_LAMBDA_PERMISSION_TRIADS"],
        "new_local_embedding_onnx_calls": runtime.encoder.inference_calls}
    if replay:
        if (output / "receipts.jsonl").read_bytes() != raw_rows:
            raise ValueError("REVIEWED_RUNTIME_REPLAY_TRACE_MISMATCH")
        (output / "replay.json").write_bytes(canonical_json({"status": "EXACT_CONTEXT_TRACE_AND_PROMPT_REPLAY_MATCH",
            "receipts": len(rows), "receipts_sha256": sha256_bytes(raw_rows), "new_model_calls": 0, "aws_calls": 0}) + b"\n")
    else:
        (output / "receipts.jsonl").write_bytes(raw_rows)
        (output / "report.json").write_bytes(canonical_json(report) + b"\n")
        (output / "timing.json").write_bytes(canonical_json({"backend": "LOCAL_WINDOWS_CPU", "durations": durations,
            "includes_query_embedding": False, "note": "Development measurements, not cloud latency SLA"}) + b"\n")
    print(json.dumps(report, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    check(args.output_dir, replay=args.replay)


if __name__ == "__main__":
    main()
