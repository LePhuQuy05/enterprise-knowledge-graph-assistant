"""Post-hoc Q-then-R context ablation, not a new answer method or independent test."""

import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from app.collection_answering import _evidence_view
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context import pack_context
from app.contracts import sha256_bytes
from app.graph_retrieval import GraphRetrievalResult
from app.reranking import OnnxCrossEncoder
from scripts.answer_local import ROOT
from scripts.run_reranker_answer_pilot import write_once


def main():
    directory = ROOT / "results/multi-query/q-then-r-diagnostic-v1"
    if (directory / "plan.json").exists():
        raise ValueError("Q_R_DIAGNOSTIC_ALREADY_PLANNED_OR_EXECUTED_NO_OVERWRITE")
    sources = [ROOT / f"results/multi-query/pilot-v1/case-{index}.json" for index in range(1, 4)]
    reports = [json.loads(path.read_bytes()) for path in sources]
    if any(report["status"] != "COMPLETED" for report in reports):
        raise ValueError("Q_R_REQUIRES_COMPLETED_IMMUTABLE_Q_INPUTS")
    paths = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    reranker = OnnxCrossEncoder(ROOT / "data/models/ms-marco-MiniLM-L2-v2",
                               ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json")
    plan = {"kind": "POST_HOC_Q_THEN_R_CONTEXT_DIAGNOSTIC_NOT_ANSWER_METHOD", "created_at": datetime.now(UTC).isoformat(),
        "selection_status": "POST_HOC_AFTER_H_Q_PILOT_OBSERVED_NOT_INDEPENDENT_OR_HOLDOUT",
        "inputs": [{"path": str(path), "sha256": sha256_bytes(path.read_bytes())} for path in sources],
        "collection_manifest_sha256": collection.manifest_sha256, "reranker_recipe": reranker.recipe.model_dump(mode="json"),
        "reranker_recipe_sha256": reranker.recipe.fingerprint, "query": "UNCHANGED_ACTUAL_BASE_SEARCH_QUERY",
        "pool": "EXACT_FROZEN_Q_FINAL_POOL_MAX40", "packer": "COMMON_8_CHUNKS_4096_PROXY_TOKENS",
        "generation_calls": 0, "planner_calls": 0, "aws_calls": 0, "paid_calls": 0,
        "human_reviewed": False, "gold": False, "holdout": False, "answer_quality_metrics": None}
    write_once(directory / "plan.json", plan)
    results = []
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        for report in reports:
            method = report["methods"][1]
            if (method["effective_method"] != "hybrid_multi_query"
                    or report["collection_manifest_sha256"] != collection.manifest_sha256):
                raise ValueError("Q_R_METHOD_OR_COLLECTION_MISMATCH")
            original = tuple(method["ranked_chunk_ids"])
            view = _evidence_view(collection, lexical, original)
            before = pack_context(view, GraphRetrievalResult(original, original, (), (), "hybrid_multi_query", False, (), 0, 0))
            if list(before.chunk_ids) != method["context_chunk_ids"] or before.proxy_tokens != method["context_proxy_tokens"]:
                raise ValueError("Q_R_FROZEN_CONTEXT_MISMATCH")
            start = perf_counter()
            ranked = reranker.rerank(report["search_query"], tuple(view.chunks[key] for key in original), tenant_id="demo")
            duration_ms = (perf_counter() - start) * 1000
            after = pack_context(view, GraphRetrievalResult(ranked.ranked_chunk_ids, ranked.ranked_chunk_ids, (), (),
                                                          "hybrid_multi_query", False, (), 0, 0))
            changed = tuple(dict.fromkeys((*before.chunk_ids, *after.chunk_ids)))
            sources_out = []
            for key in changed:
                chunk = view.chunks[key]
                doc = view.documents[chunk.document_version_id]
                ref = view.source_refs[key]
                sources_out.append({"chunk_id": key, "document_version_id": doc.document_version_id,
                    "source_uri": doc.source_uri, "title": doc.title, "section_path": list(chunk.section_path),
                    "start_char": chunk.start_char, "end_char": chunk.end_char, "text": chunk.text,
                    "text_sha256": sha256_bytes(chunk.text.encode()), "shard_id": ref.shard_id,
                    "shard_manifest_sha256": ref.manifest_sha256})
            results.append({"case_id": report["pilot_case_id"], "base_search_query": report["search_query"],
                "frozen_q_ranked_chunk_ids": list(original), "reranked_chunk_ids": list(ranked.ranked_chunk_ids),
                "scores": [score.__dict__ for score in ranked.scores], "windows": [window.__dict__ for window in ranked.windows],
                "rerank_duration_ms": round(duration_ms, 3), "before_context_ids": list(before.chunk_ids),
                "after_context_ids": list(after.chunk_ids), "before_proxy_tokens": before.proxy_tokens,
                "after_proxy_tokens": after.proxy_tokens, "source_appearances": sources_out})
    result = {"status": "COMPLETED_CONTEXT_ONLY_NOT_ANSWER_QUALITY", "created_at": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_bytes((directory / "plan.json").read_bytes()), "results": results,
        "local_reranker_onnx_batches": reranker.inference_calls, "paid_calls": 0, "aws_calls": 0,
        "planner_calls": 0, "query_encoder_calls": 0, "generation_calls": 0, "new_api_cost_micro_usd": 0,
        "quality_metrics": None, "human_reviewed": False, "cloud_deployed": False}
    write_once(directory / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "local_reranker_onnx_batches", "paid_calls", "aws_calls")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
