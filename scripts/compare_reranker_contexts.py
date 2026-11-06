"""Offline same-pool H/R context experiment; no answer generation, API or AWS access."""

import argparse
import json
import platform
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.collection_answering import prepare_collection_answer
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context import pack_context
from app.contracts import LocalBgeRecipe, canonical_json, sha256_bytes
from app.embeddings import EmbeddingCache, query_embedding_input
from app.external_gateway import ExternalRoutes
from app.generation import GenerationRecipeV9
from app.graph_retrieval import GraphRetrievalResult
from app.local_bge import BgeEncoder
from app.model_gateway import safe_failure_code
from app.reranking import OnnxCrossEncoder, RerankerRecipe
from scripts.answer_local import ROOT
from scripts.benchmark_collection import process_memory
from scripts.compare_citation_recipes import ComparisonCase, pin_comparison_plan

VERSION = "same-pool-windowed-reranker-development-v1"


def pack_order(view, ordered):
    result = GraphRetrievalResult(tuple(ordered), tuple(ordered), (), (), "hybrid", False, (), 0, 0)
    return pack_context(view, result)


def describe_context(view, packed):
    sources = []
    for key in packed.chunk_ids:
        chunk = view.chunks[key]
        document = view.documents[chunk.document_version_id]
        if document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text:
            raise ValueError("RERANKER_CONTEXT_SOURCE_SLICE")
        sources.append({"chunk_id": key, "document_version_id": chunk.document_version_id,
            "text_sha256": chunk.text_sha256, "source_uri": document.source_uri, "title": document.title,
            "start_char": chunk.start_char, "end_char": chunk.end_char, "text": chunk.text,
            "shard_id": view.source_refs[key].shard_id,
            "shard_manifest_sha256": view.source_refs[key].manifest_sha256})
    return {"chunk_ids": list(packed.chunk_ids), "proxy_tokens": packed.proxy_tokens,
            "serialized_sha256": sha256_bytes(packed.serialized.encode()),
            "warnings": list(packed.warnings), "sources": sources}


def compare_prepared(view, prepared, reranker, *, search_query, tenant_id, repeats=3):
    """Inputs are actual candidates only; labels and target IDs are never accepted."""
    if type(repeats) is not int or not 1 <= repeats <= 5:
        raise ValueError("RERANKER_EXPERIMENT_REPEAT_BOUND")
    candidates = prepared.retrieved_chunk_ids
    if len(candidates) > 40 or len(set(candidates)) != len(candidates):
        raise ValueError("RERANKER_EXPERIMENT_CANDIDATE_BOUND")
    chunks = tuple(view.chunks[key] for key in candidates)
    times, result = [], None
    first_ids, first_scores = None, None
    before = reranker.inference_calls
    memory_before = process_memory()
    for _ in range(repeats):
        started = time.perf_counter()
        current = reranker.rerank(search_query, chunks, tenant_id=tenant_id)
        times.append((time.perf_counter() - started) * 1000)
        if (len(current.ranked_chunk_ids) != len(candidates)
                or set(current.ranked_chunk_ids) != set(candidates)
                or len(set(current.ranked_chunk_ids)) != len(candidates)):
            raise ValueError("RERANKER_EXPERIMENT_CHANGED_POOL")
        scores = [(item.chunk_id, item.raw_logit) for item in current.scores]
        if first_ids is None:
            first_ids, first_scores, result = current.ranked_chunk_ids, scores, current
        elif current.ranked_chunk_ids != first_ids or scores != first_scores:
            raise ValueError("RERANKER_EXPERIMENT_NONDETERMINISTIC_REPEAT")
    packed = pack_order(view, result.ranked_chunk_ids)
    baseline = describe_context(view, prepared.packed)
    reranked = describe_context(view, packed)
    return {"status": "COMPLETED_RETRIEVAL_ONLY_NOT_ANSWER_QUALITY",
        "same_candidate_pool": True, "candidate_count": len(candidates),
        "candidate_chunk_ids": list(candidates), "reranked_chunk_ids": list(result.ranked_chunk_ids),
        "scores": [asdict(item) for item in result.scores], "windows": [asdict(item) for item in result.windows],
        "H": baseline, "R": reranked,
        "context_added_ids": [key for key in packed.chunk_ids if key not in prepared.packed.chunk_ids],
        "context_removed_ids": [key for key in prepared.packed.chunk_ids if key not in packed.chunk_ids],
        "reranker_recipe_sha256": result.recipe_sha256,
        "reranker_onnx_batch_calls_this_case": reranker.inference_calls - before,
        "timing": {"observations_ms": times, "first_case_run_ms": times[0],
            "repeated_run_ms": times[1:], "median_ms": float(np.median(times)),
            "note": "Local loaded-model repeated scoring, not Lambda cold/warm latency or an SLA"},
        "process_memory_before": memory_before, "process_memory_after": process_memory(),
        "repeat_rank_and_score_equal": True, "generation_calls": 0, "paid_api_calls": 0,
        "quality_metrics": None, "human_reviewed": False}


def verify_saved(directory):
    """Rebuild ranks/packed sources from stored logits without rerunning any encoder/model."""
    plan = json.loads((directory / "plan.json").read_bytes())
    result_raw = (directory / "experiment.json").read_bytes()
    results = json.loads(result_raw)
    identity = {key: value for key, value in plan.items() if key not in {"created_at", "plan_sha256"}}
    if sha256_bytes(canonical_json(identity)) != plan["plan_sha256"] or results["plan_sha256"] != plan["plan_sha256"]:
        raise ValueError("RERANKER_VERIFY_PLAN_HASH")
    recipe = RerankerRecipe.model_validate_json(canonical_json(plan["reranker_recipe"]))
    if recipe.fingerprint != plan["reranker_recipe_sha256"]:
        raise ValueError("RERANKER_VERIFY_RECIPE")
    collection = load_collection(Path(plan["collection_path"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    if collection.manifest_sha256 != plan["collection_manifest_sha256"]:
        raise ValueError("RERANKER_VERIFY_COLLECTION")
    collection.validate_all()
    count = 0
    with load_global_lexical_index(Path(plan["lexical_manifest_path"]), collection) as lexical:
        from app.collection_answering import _evidence_view

        for before, case in zip(plan["cases"], results["cases"], strict=True):
            if before["input"]["id"] != case["case_id"] or case["status"] != "COMPLETED_RETRIEVAL_ONLY_NOT_ANSWER_QUALITY":
                raise ValueError("RERANKER_VERIFY_INCOMPLETE_CASE")
            ids = tuple(before["candidate_chunk_ids"])
            if case["candidate_chunk_ids"] != list(ids) or len(case["scores"]) != len(ids):
                raise ValueError("RERANKER_VERIFY_CANDIDATE_BINDING")
            view = _evidence_view(collection, lexical, ids)
            windows = case["windows"]
            if (len(windows) > 160 or any(row["chunk_id"] not in ids for row in windows)
                    or sha256_bytes(canonical_json(list(ids))) != before["candidate_pool_sha256"]):
                raise ValueError("RERANKER_VERIFY_WINDOW_POOL_OR_BOUND")
            maximum = {}
            for key in ids:
                own = sorted((row for row in windows if row["chunk_id"] == key), key=lambda row: row["window_index"])
                cursor = 0
                if not 1 <= len(own) <= 8:
                    raise ValueError("RERANKER_VERIFY_WINDOW_COUNT")
                for index, row in enumerate(own, 1):
                    if (row["window_index"] != index or row["start_char"] != cursor
                            or not cursor < row["end_char"] <= len(view.chunks[key].text)
                            or row["offset_basis"] != "CHUNK_LOCAL_UNICODE"
                            or not 1 <= row["input_tokens"] <= 512 or not np.isfinite(row["raw_logit"])):
                        raise ValueError("RERANKER_VERIFY_WINDOW_COVERAGE")
                    cursor = row["end_char"]
                if cursor != len(view.chunks[key].text):
                    raise ValueError("RERANKER_VERIFY_UNCOVERED_SOURCE")
                maximum[key] = max(row["raw_logit"] for row in own)
                score = next((row for row in case["scores"] if row["chunk_id"] == key), None)
                if (score is None or score["original_rank"] != ids.index(key) + 1
                        or score["window_count"] != len(own) or score["raw_logit"] != maximum[key]
                        or any(row["original_rank"] != ids.index(key) + 1 for row in own)):
                    raise ValueError("RERANKER_VERIFY_SCORE_AGGREGATION")
            ordered = sorted(ids, key=lambda key: (-maximum[key], ids.index(key), key))
            if (case["reranked_chunk_ids"] != ordered or [row["chunk_id"] for row in case["scores"]] != ordered
                    or case["reranker_recipe_sha256"] != recipe.fingerprint):
                raise ValueError("RERANKER_VERIFY_RANKS")
            if describe_context(view, pack_order(view, ids)) != case["H"] or describe_context(view, pack_order(view, ordered)) != case["R"]:
                raise ValueError("RERANKER_VERIFY_CONTEXT_OR_PROVENANCE")
            if case["H"]["serialized_sha256"] != before["baseline_context_sha256"]:
                raise ValueError("RERANKER_VERIFY_BASELINE_PIN")
            count += 1
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "PASS_SOURCE_RANK_CONTEXT_REBUILD",
        "experiment_sha256": sha256_bytes(result_raw), "plan_sha256": plan["plan_sha256"], "cases": count,
        "model_encoder_calls": 0, "paid_api_calls": 0, "aws_calls": 0,
        "quality_metrics": None, "semantic_review": "NOT_HUMAN_REVIEWED"}
    (directory / "verification.json").write_bytes(canonical_json(report) + b"\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reranker/development-v1")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        print(json.dumps(verify_saved(args.output_dir)))
        return 0
    if not args.cases:
        parser.error("--cases is required unless --verify is used")
    if (args.output_dir / "experiment.json").exists():
        raise ValueError("RERANKER_EXPERIMENT_IMMUTABLE_USE_VERIFY")
    cases = [ComparisonCase.model_validate_json(canonical_json(row)) for path in args.cases
             for row in json.loads(path.read_bytes())["cases"]]
    if not 1 <= len(cases) <= 12 or len({case.id for case in cases}) != len(cases):
        raise ValueError("RERANKER_EXPERIMENT_CASE_BOUND")
    paths = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
        embedding_model_id=encoder.recipe("document").model_id, cache_shards=4)
    if collection.manifest.embedding_recipe != encoder.recipe("document"):
        raise ValueError("RERANKER_EXPERIMENT_QUERY_RECIPE_MISMATCH")
    cache = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE")
    query_recipe = LocalBgeRecipe.model_validate_json(canonical_json(encoder.recipe("query").model_dump(mode="json")))
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    loaded = time.perf_counter()
    reranker = OnnxCrossEncoder(ROOT / "data/models/ms-marco-MiniLM-L2-v2", ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json")
    load_ms = (time.perf_counter() - loaded) * 1000
    ready, case_plans = [], []
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        for case in cases:
            vector, cache_key = None, None
            if case.mode != "bm25":
                lookup = cache.resolve_local("demo", text=query_embedding_input(case.search_query), recipe=query_recipe, encoder=encoder)
                vector, cache_key = np.asarray(lookup.record.vector, dtype=np.float32), lookup.record.cache_key
            view, prepared = prepare_collection_answer(collection, lexical, case.question, vector, tenant_id="demo",
                owner_id="local-reranker-development", run_id="offline-reranker-development", recipe=recipe,
                mode=case.mode, request_id="reranker-" + case.id, search_query=case.search_query)
            ready.append((case, view, prepared))
            case_plans.append({"input": case.model_dump(mode="json"), "query_embedding_cache_key": cache_key,
                "candidate_chunk_ids": list(prepared.retrieved_chunk_ids),
                "candidate_pool_sha256": sha256_bytes(canonical_json(list(prepared.retrieved_chunk_ids))),
                "baseline_context_sha256": sha256_bytes(prepared.packed.serialized.encode())})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plan = pin_comparison_plan(args.output_dir / "plan.json", {
        "version": VERSION, "purpose": "DEVELOPMENT_SAME_POOL_NOT_GOLD_OR_HOLDOUT",
        "created_at": datetime.now(UTC).isoformat(), "collection_path": paths["collection_path"],
        "lexical_manifest_path": paths["lexical_manifest_path"], "collection_manifest_sha256": collection.manifest_sha256,
        "lexical_index_version": lexical.manifest.index_version, "query_recipe_sha256": query_recipe.fingerprint,
        "query_policy": "FIXED_ENGLISH_SEARCH_NOT_END_TO_END_VI_REWRITE", "generation_recipe_unused": recipe.model_dump(mode="json"),
        "reranker_recipe": reranker.recipe.model_dump(mode="json"), "reranker_recipe_sha256": reranker.recipe.fingerprint,
        "reranker_code_sha256": sha256_bytes((ROOT / "app/reranking.py").read_bytes()),
        "repeats": 3, "same_pool_max_candidates": 40, "max_context_chunks": 8, "context_proxy_cap": 4096,
        "cases": case_plans, "target_annotations_used_by_ranker": False, "paid_api_calls": 0})
    results = []
    for case, view, prepared in ready:
        try:
            results.append({"case_id": case.id, **compare_prepared(view, prepared, reranker,
                search_query=case.search_query, tenant_id="demo", repeats=plan["repeats"])})
        except Exception as exc:
            failure = {"case_id": case.id, "status": "FAILED_RERANKER_NO_FALLBACK",
                       "error_category": safe_failure_code(exc)}
            if isinstance(exc, ValueError) and str(exc).startswith("RERANKER_"):
                failure["failed_check"] = str(exc)
            results.append(failure)
        report = {"version": VERSION, "plan_sha256": plan["plan_sha256"], "cases": results,
            "platform": platform.platform(), "reranker_load_including_hash_validation_ms": load_ms,
            "bge_query_onnx_calls_this_invocation": encoder.inference_calls,
            "reranker_onnx_batch_calls_this_invocation": reranker.inference_calls,
            "model_api_calls": 0, "aws_calls": 0, "new_provider_cost_micro_usd": 0,
            "quality_metrics": None, "human_reviewed": False, "cloud_deployed": False}
        (args.output_dir / "experiment.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({"cases": len(results), "completed": sum(row["status"].startswith("COMPLETED") for row in results),
                      "reranker_onnx_calls": reranker.inference_calls, "model_api_calls": 0}))
    return 0 if all(row["status"].startswith("COMPLETED") for row in results) else 2


if __name__ == "__main__":
    raise SystemExit(main())
