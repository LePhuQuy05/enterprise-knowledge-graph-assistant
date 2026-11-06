"""Question-only core-six pool producer for profile experiments.

No rubric, expected behavior, oracle source IDs or evaluator is accepted.
Q/QB need their own approved native preparation; this producer rejects them.
Graph pools come from actual typed paths, never a guessed permission edge.
"""

from typing import Annotated, Literal

import numpy as np
from pydantic import StringConstraints

from app.authorization_retrieval import prepare_authorization_answer
from app.collection_answering import prepare_collection_answer
from app.contracts import Record, canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.generation import GenerationError
from app.retrieval import reciprocal_rank_fusion

CORE = ("bm25", "vector", "hybrid", "hybrid_reranked", "graph_1hop", "graph_2hop")


class ProfileRetrievalTask(Record):
    task_id: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    search: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    language: Literal["en", "vi"]
    query_track: Literal["ORIGINAL_EN", "AUTHOR_SUPPLIED_EN_REFERENCE"]


def retrieve_profile_pool(collection, lexical, graph, runtime, task, method):
    if type(task) is not ProfileRetrievalTask or method not in CORE:
        raise GenerationError("PROFILE_RETRIEVAL_TYPED_CORE_TASK_REQUIRED")
    ProfileRetrievalTask.model_validate_json(task.model_dump_json())
    if (task.language == "en" and (task.search != task.question or task.query_track != "ORIGINAL_EN")
            or task.language == "vi" and task.query_track != "AUTHOR_SUPPLIED_EN_REFERENCE"):
        raise GenerationError("PROFILE_RETRIEVAL_QUERY_TRACK")
    vector = None
    if method != "bm25":
        lookup = runtime.cache.resolve_local(collection.manifest.tenant_id, text=query_embedding_input(task.search),
            recipe=runtime.query_recipe, encoder=runtime.encoder)
        vector = np.asarray(lookup.record.vector, dtype=np.float32)
    trace = {}
    if not method.startswith("graph_"):
        reranks = []
        _, prepared = prepare_collection_answer(collection, lexical, task.question, vector,
            tenant_id=collection.manifest.tenant_id, owner_id="profile-controls-local", run_id="context-profiles-v1",
            recipe=runtime.recipe, mode=method, search_query=task.search, reranker=runtime.reranker,
            rerank_trace=reranks)
        ranking = prepared.retrieved_chunk_ids
        bundles = [{"bundle_id": f"rank-{i}", "chunk_ids": [k], "query_index": 0} for i, k in enumerate(ranking)]
        status = "READY_RETRIEVAL_ONLY"
        trace = {"reranking": [r.model_dump(mode="json", exclude={"duration_ms"}) for r in reranks]}
    else:
        result = prepare_authorization_answer(collection, lexical, graph, task.question, vector,
            tenant_id=collection.manifest.tenant_id, owner_id="profile-controls-local", run_id="context-profiles-v1",
            recipe=runtime.recipe, hops=int(method[6]), search_query=task.search, allow_experimental=True,
            answer_language=task.language)
        trace = result.trace
        if trace.get("runtime_status") == "BUDGET_EXCEEDED":
            raise GenerationError("PROFILE_GRAPH_TRAVERSAL_BUDGET_NO_FALLBACK")
        facts = {f.assertion_id: f for f in graph.facts}
        eligible = sorted(trace["paths"], key=lambda p: (-p["query_entities_covered"], -len(p["facet_terms"]),
            -p["hop"], p["assertion_ids"], p["entity_ids"]))
        paths, representatives = [], set()
        for p in eligible:
            representative = facts[p["assertion_ids"][-1]].chunk_id
            if representative not in representatives:
                paths.append((representative, p))
                representatives.add(representative)
        graph_ranking = tuple(k for k, _ in paths)
        fused = reciprocal_rank_fusion([trace["vector_ranking"], trace["lexical_ranking"], graph_ranking])
        prefix = tuple(dict.fromkeys(k for _, p in paths[:2] for k in p["supporting_chunk_ids"]))
        ranking = tuple(dict.fromkeys((*prefix, *fused)))[:40]
        by_rep = {k: p for k, p in paths if set(p["supporting_chunk_ids"]) <= set(ranking)}
        bundles = []
        for i, key in enumerate(ranking):
            path = by_rep.get(key)
            bundles.append({"bundle_id": f"rank-{i}", "chunk_ids": list(path["supporting_chunk_ids"]) if path else [key],
                "assertion_ids": list(path["assertion_ids"]) if path else [], "query_index": 0})
        status = "GRAPH_PATH_POOL_READY" if by_rep else "GRAPH_ZERO_EXPANSION_NOT_READY_FOR_ANSWER"
        trace = {"typed_graph_trace": trace, "pool_rebuilt_from_actual_paths_before_legacy_8_source_packing": True,
            "legacy_context_or_zero_substitution_not_used": True, "release_sha256": graph.release_sha256}
    return {"task": task.model_dump(mode="json"), "method": method, "mechanism_status": status,
        "pool_ids": list(ranking), "bundles": bundles, "balanced": False, "query_order": [0],
        "pool_fingerprint": sha256_bytes(canonical_json({"ranking": ranking, "bundles": bundles})),
        "trace": trace, "source_annotations_used_for_ranking": False, "hidden_fallbacks": 0,
        "answer_generation_executed": False}
