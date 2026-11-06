"""Reproduce retained planner failures and the paired graph-use regression locally.

Read-only diagnosis of v1; no gateway, AWS client, generation or artifact repair.
Exit 2 means the specific method-readiness symptoms are still present.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from app.benchmark_queries import query_context
from app.collection_benchmark import RetrievalTask
from app.collection_graph_runtime import prepare_collection_graph_answer
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.query_planning import PlannerRecipe, QueryPlanError, _parse_subqueries
from app.retrieval import reciprocal_rank_fusion
from scripts.prepare_benchmark_queries import tasks_from
from scripts.run_augmented_collection_benchmark import load_shared_queries
from scripts.run_collection_benchmark import OfflineRuntime, load_inputs


def diagnose():
    protocol_raw, dataset, collection, _, pipeline = load_inputs()
    bundle, bundle_sha, _ = load_shared_queries(dataset, protocol_raw)
    inputs = {(t.scenario_id, t.language): t for t in tasks_from(dataset)}
    rows = {(r.scenario_id, r.language): r for r in bundle.rows}
    checks = []
    for key in (("aws100-019", "en"), ("aws100-066", "vi"), ("aws100-086", "en")):
        row, task = rows[key], inputs[key]
        try:
            queries = _parse_subqueries(row.planner_receipt,
                query_context(task, sha256_bytes(protocol_raw), "PLAN_QUERY"), row.search_query, PlannerRecipe())
            checks.append({"case": ".".join(key), "mechanism": "planner",
                "status": "PASS", "subqueries": queries})
        except QueryPlanError as exc:
            checks.append({"case": ".".join(key), "mechanism": "planner", "status": "FAIL",
                "error": str(exc), "raw_plan": json.loads(row.planner_receipt.raw_output)})
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("graph_1hop",))
        for key in (("aws100-068", "en"), ("aws100-068", "vi"), ("aws100-001", "en")):
            row, original = rows[key], inputs[key]
            task = RetrievalTask(original.scenario_id, original.language, original.question)
            embedded = runtime.cache.resolve_local(collection.manifest.tenant_id,
                text=query_embedding_input(row.search_query), recipe=runtime.query_recipe, encoder=runtime.encoder)
            vector = np.asarray(embedded.record.vector, dtype=np.float32)
            _, prepared, trace = prepare_collection_graph_answer(collection, lexical, runtime.graph,
                task.question, vector, search_query=row.search_query, tenant_id=collection.manifest.tenant_id,
                owner_id="benchmark-local", run_id="diagnose-methods-v1", recipe=runtime.recipe,
                hops=1, allow_experimental=True, clock=lambda: 0.0)
            names = {e.entity_id: e.canonical_name for e in runtime.graph.overlay.entities}
            mentions = {}
            for mention in runtime.graph.overlay.mentions:
                mentions.setdefault(mention.chunk_id, set()).add(mention.entity_id)
            ranked = trace["final_ranking"]
            prepack = tuple(reciprocal_rank_fusion([trace["vector_ranking"], trace["lexical_ranking"],
                [c["chunk_id"] for c in trace["graph_candidates"]]]))[:40]
            checks.append({"case": ".".join(key), "mechanism": "graph", "status":
                "PASS" if prepared.binding.used_graph else "FAIL", "search_query": row.search_query,
                "warnings": prepared.binding.warnings,
                "query_literal_entities": [e.canonical_name for e in runtime.graph.overlay.entities
                    if e.canonical_name.casefold() in row.search_query.casefold()],
                "ranked_mention_entities": [{"rank": i + 1, "entities": sorted(names[e] for e in mentions.get(chunk, ()))}
                    for i, chunk in enumerate(trace["baseline_ranking"][:20]) if mentions.get(chunk)],
                "relation_candidate_ranks": [{"rank": ranked.index(c["chunk_id"]) + 1
                    if c["chunk_id"] in ranked else None, "hops": len(c["assertion_path"]),
                    "prepack_rank": prepack.index(c["chunk_id"]) + 1 if c["chunk_id"] in prepack else None,
                    "support_survives_candidate_filter": set(c["supporting_chunk_ids"]) <= set(prepack),
                    "support_ranks": [ranked.index(s) + 1 if s in ranked else None for s in c["supporting_chunk_ids"]]}
                    for c in trace["graph_candidates"] if c["assertion_path"]],
                "trace": {name: trace.get(name) for name in (
                    "seed_entity_ids", "assertion_visits", "visited_entities", "timed_out", "truncated",
                    "graph_candidates", "baseline_ranking", "context_assertion_ids")}})
    return {"status": "READINESS_RED" if any(c["status"] == "FAIL" for c in checks) else "READINESS_GREEN",
        "scope": "PINNED_V1_DIAGNOSIS_NOT_A_NEW_BENCHMARK", "bundle_sha256": bundle_sha,
        "provider_calls": 0, "aws_calls": 0, "local_embedding_onnx_calls": runtime.encoder.inference_calls,
        "checks": checks}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = diagnose()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("xb") as handle:
            handle.write(canonical_json(result) + b"\n")
    print(json.dumps({"status": result["status"], "provider_calls": result["provider_calls"],
        "aws_calls": result["aws_calls"], "checks": [
            {k: c[k] for k in ("case", "mechanism", "status")} for c in result["checks"]]}))
    return 2 if result["status"] == "READINESS_RED" else 0


if __name__ == "__main__":
    raise SystemExit(main())
