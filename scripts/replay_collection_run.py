"""Replay cached answers without AWS/paid calls; R can recompute local reranking."""

import argparse
import json
from pathlib import Path

from app.answer_cache import AnswerCache
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context_selection import ContextSelectionRecipe
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalRoutes
from app.generation import GenerationRecipeV9
from app.local_bge import BgeEncoder
from app.query_planning import QueryPlanCache
from app.rag_methods import RunInputV1
from app.rag_runner import run_methods
from app.reranking import OnnxCrossEncoder
from app.rewrite_cache import RewriteCache
from scripts.answer_local import ROOT, RUN


def assert_replay(original, replay):
    """Compare meaningful persisted results, excluding timestamps/timings/new-cost counters."""
    fields = ("run_identity", "input", "collection_version", "collection_manifest_sha256", "search_query", "query_transform",
              "context_selection_recipe_sha256", "planner_sharing")
    if (original.get("status") != "COMPLETED" or replay.get("status") != "COMPLETED"
            or any(original.get(key) != replay.get(key) for key in fields)
            or not replay.get("cost_complete") or replay.get("new_known_cost_micro_usd") != 0
            or len(original["methods"]) != len(replay["methods"])):
        raise ValueError("COLLECTION_REPLAY_RUN_MISMATCH")
    for before, after in zip(original["methods"], replay["methods"], strict=True):
        equal = ("requested_method", "effective_method", "config", "config_sha256", "ranked_chunk_ids", "context_chunk_ids",
                 "context_sources", "response", "usage", "original_receipt_cost_micro_usd", "multi_query_trace",
                 "context_selection_trace")
        if (after["status"] != "COMPLETED" or not after["cache_hit"] or after["new_cost_micro_usd"] != 0
                or any(before.get(key) != after.get(key) for key in equal)):
            raise ValueError("COLLECTION_REPLAY_METHOD_MISMATCH")
        stable_traces = []
        for row in (before, after):
            trace = row.get("rerank_trace")
            stable_traces.append(None if trace is None else {
                key: value for key, value in trace.items() if key != "duration_ms"})
        if stable_traces[0] != stable_traces[1]:
            raise ValueError("COLLECTION_REPLAY_RERANK_TRACE_MISMATCH")
        planning_before, planning_after = before.get("planning_trace"), after.get("planning_trace")
        if planning_before is not None:
            if (planning_after is None or not planning_after["cache_hit"] or planning_after["new_cost_micro_usd"] != 0
                    or {k: v for k, v in planning_before.items() if k not in {"cache_hit", "new_cost_micro_usd"}}
                    != {k: v for k, v in planning_after.items() if k not in {"cache_hit", "new_cost_micro_usd"}}):
                raise ValueError("COLLECTION_REPLAY_PLANNING_TRACE_MISMATCH")
        elif planning_after is not None:
            raise ValueError("COLLECTION_REPLAY_PLANNING_TRACE_MISMATCH")
    if original.get("rewrite_receipt") and (not replay.get("rewrite_cache_hit")
            or replay.get("rewrite_receipt") != original["rewrite_receipt"]):
        raise ValueError("COLLECTION_REPLAY_REWRITE_MISMATCH")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    parser.add_argument("--pipeline", type=Path, default=ROOT / "results/corpus-expansion-v2/pipeline.json")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve() == args.receipt.resolve():
        raise ValueError("REPLAY_CANNOT_OVERWRITE_ORIGINAL_RECEIPT")
    if not 1 <= args.receipt.stat().st_size <= 4 * 1024 * 1024:
        raise ValueError("REPLAY_REPORT_SIZE_LIMIT")
    raw = args.receipt.read_bytes()
    original = json.loads(raw)
    request = RunInputV1.model_validate_json(canonical_json(original["input"]))
    if request.run_kind != "answer" or original.get("status") != "COMPLETED":
        raise ValueError("REPLAY_REQUIRES_COMPLETED_ANSWER_RUN")
    paths = json.loads(args.pipeline.read_bytes())
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    needs_vector = any(method != "bm25" for method in request.method_ids)
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5",
                         ROOT / "configs/models/bge-small-en-v1.5-onnx.json") if needs_vector else None
    reranker = OnnxCrossEncoder(ROOT / "data/models/ms-marco-MiniLM-L2-v2",
        ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json") if "hybrid_reranked" in request.method_ids else None
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    if (collection.manifest_sha256 != original["collection_manifest_sha256"]
            or collection.manifest.collection_version != original["collection_version"]
            or any(method["config"]["generation_recipe_sha256"] != recipe.fingerprint for method in original["methods"])):
        raise ValueError("REPLAY_PINNED_ARTIFACT_MISMATCH")
    selector = None
    if "hybrid_multi_query_balanced" in request.method_ids:
        config = next(row["config"] for row in original["methods"]
                      if row["effective_method"] == "hybrid_multi_query_balanced")
        selector = ContextSelectionRecipe.model_validate_json(canonical_json(config["context_selection_recipe"]))
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        replay = run_methods(collection, lexical, request, tenant_id="demo", owner_id="local-ldh-owner", run_id=RUN,
            recipe=recipe, routes=routes, encoder=encoder,
            embedding_cache=EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE") if encoder else None,
            cache=AnswerCache(ROOT / "data/cache/collection-answers", namespace="LIVE"),
            rewrite_cache=RewriteCache(ROOT / "data/cache/collection-rewrites", namespace="LIVE"),
            reranker=reranker, query_plan_cache=QueryPlanCache(ROOT / "data/cache/collection-query-plans", namespace="LIVE")
                if any(method in {"hybrid_multi_query", "hybrid_multi_query_balanced"} for method in request.method_ids) else None,
            context_selection_recipe=selector, gateway=None)
    assert_replay(original, replay)
    replay.update(replay_verified_against_sha256=sha256_bytes(raw),
                  model_dispatch_attempts_this_invocation=0, aws_api_calls_this_invocation=0,
                  local_query_onnx_calls_this_invocation=encoder.inference_calls if encoder else 0)
    if reranker is not None:
        replay["local_reranker_onnx_calls_this_invocation"] = reranker.inference_calls
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(replay) + b"\n")
    print(json.dumps({"status": "PASS_OFFLINE_REPLAY", "methods": len(replay["methods"]),
                      "new_cost_micro_usd": 0, "model_dispatches": 0, "aws_api_calls": 0, "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
