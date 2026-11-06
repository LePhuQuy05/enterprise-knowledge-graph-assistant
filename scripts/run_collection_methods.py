"""Compare local collection methods; default retrieval-only, paid answers explicit."""

import argparse
import json
from pathlib import Path

from app.answer_cache import AnswerCache
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context_selection import ContextSelectionRecipe
from app.contracts import canonical_json
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalRoutes
from app.generation import GenerationRecipeV9
from app.local_bge import BgeEncoder
from app.model_gateway import safe_failure_code
from app.query_planning import QueryPlanCache
from app.rag_methods import RunInputV1
from app.rag_runner import run_methods
from app.reranking import OnnxCrossEncoder
from app.rewrite_cache import RewriteCache
from scripts.answer_local import ROOT, RUN, managed_gateway


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--pipeline", type=Path, default=ROOT / "results/corpus-expansion-v2/pipeline.json")
    parser.add_argument("--methods", nargs="+", choices=("bm25", "vector", "hybrid", "hybrid_reranked", "hybrid_multi_query", "hybrid_multi_query_balanced"),
                        default=("bm25", "vector", "hybrid"))
    parser.add_argument("--language", choices=("auto", "en", "vi"), default="auto")
    parser.add_argument("--run-kind", choices=("retrieval", "answer"), default="retrieval")
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--execute", action="store_true", help="Allow budgeted cache misses; never automatically retry failures")
    parser.add_argument("--price-sheet", type=Path, help="Explicit separately verified price sheet for a new paid experiment")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.execute and args.output.exists():
        raise ValueError("METHOD_LIVE_RECEIPT_IMMUTABLE_USE_REPLAY")
    if args.execute and args.run_kind != "answer":
        parser.error("--execute is available only with --run-kind answer; retrieval-only never calls a model")
    request = RunInputV1(original_question=args.question, language=args.language,
        method_ids=tuple(args.methods), run_kind=args.run_kind, idempotency_key=args.idempotency_key)
    pipeline = json.loads(args.pipeline.read_bytes())
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    # A BM25-only run does not load ONNX or read any external embedding key.
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5",
                         ROOT / "configs/models/bge-small-en-v1.5-onnx.json") if any(
                             method != "bm25" for method in request.method_ids) else None
    embedding_cache = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE") if encoder else None
    model_id = encoder.recipe("document").model_id if encoder else "BAAI/bge-small-en-v1.5"
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
                                 embedding_model_id=model_id, cache_shards=4)
    reranker = OnnxCrossEncoder(ROOT / "data/models/ms-marco-MiniLM-L2-v2",
        ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json") if "hybrid_reranked" in request.method_ids else None
    gateway = managed_gateway(prices_path=args.price_sheet) if args.execute else None
    before = gateway.provider.dispatch_attempts if gateway else 0
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        report = run_methods(collection, lexical, request, tenant_id="demo", owner_id="local-ldh-owner",
            run_id=RUN, recipe=recipe, cache=AnswerCache(ROOT / "data/cache/collection-answers", namespace="LIVE"),
            routes=routes, encoder=encoder, embedding_cache=embedding_cache,
            rewrite_cache=RewriteCache(ROOT / "data/cache/collection-rewrites", namespace="LIVE"),
            reranker=reranker, query_plan_cache=QueryPlanCache(ROOT / "data/cache/collection-query-plans", namespace="LIVE")
                if any(method in {"hybrid_multi_query", "hybrid_multi_query_balanced"} for method in request.method_ids) else None,
            context_selection_recipe=ContextSelectionRecipe() if "hybrid_multi_query_balanced" in request.method_ids else None,
            gateway=gateway)
    report["model_dispatch_attempts_this_invocation"] = gateway.provider.dispatch_attempts - before if gateway else 0
    report["local_query_onnx_calls_this_invocation"] = encoder.inference_calls if encoder else 0
    if reranker is not None:
        report["local_reranker_onnx_calls_this_invocation"] = reranker.inference_calls
    raw = canonical_json(report) + b"\n"
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("METHOD_REPORT_SIZE_LIMIT")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(raw)
    print(json.dumps({"status": report["status"], "output": str(args.output),
        "methods": [{"method": result["effective_method"], "status": result["status"],
                     "cache_hit": result["cache_hit"]} for result in report["methods"]],
        "model_dispatch_attempts": report["model_dispatch_attempts_this_invocation"],
        "new_known_cost_micro_usd": report["new_known_cost_micro_usd"], "cloud_deployed": False}))
    return 0 if report["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_category": safe_failure_code(exc),
                          "instruction": "Inspect receipts; never retry failed/uncertain calls automatically."}))
        raise SystemExit(1) from None
