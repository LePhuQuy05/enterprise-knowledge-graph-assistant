"""Question -> actual retrieval -> atomic V12 context -> durable answer.

Opt-in local slice for collection319; no evaluator, rubric or oracle inputs.
Native extractive Q/QB share a pool and differ only in bundle scheduling.
Graph zero expansion or unselected paths are explicit non-answerable states.
"""

import json
from dataclasses import dataclass

import numpy as np

from app.authorization_retrieval import prepare_authorization_answer
from app.collection_answering import prepare_collection_answer, validate_collection_lexical_binding
from app.contracts import QueryRequest, canonical_json, sha256_bytes, stable_id
from app.embeddings import query_embedding_input
from app.generation import AnswerBinding, GenerationError
from app.graph_collection_migration import collection_records
from app.model_gateway import CallContext
from app.profiled_generation import ProfiledGenerationRecipe
from app.profiled_source_context import SourceContextProfile, pack_profiled_context
from app.retrieval import reciprocal_rank_fusion
from app.scoped_query_v6 import scoped_queries

METHODS = ("bm25", "vector", "hybrid", "hybrid_reranked", "graph_1hop", "graph_2hop",
    "hybrid_multi_query", "hybrid_multi_query_balanced")
Q_METHODS = METHODS[-2:]


def _vector(collection, runtime, text):
    lookup = runtime.cache.resolve_local(collection.manifest.tenant_id, text=query_embedding_input(text),
        recipe=runtime.query_recipe, encoder=runtime.encoder)
    return np.asarray(lookup.record.vector, dtype=np.float32)


def _ranking(collection, lexical, runtime, question, search, mode):
    vector = None if mode == "bm25" else _vector(collection, runtime, search)
    reranks = []
    _, prepared = prepare_collection_answer(collection, lexical, question, vector,
        tenant_id=collection.manifest.tenant_id, owner_id="scoped-flow-pool", run_id="scoped-flow-pool",
        recipe=runtime.recipe, mode=mode, search_query=search,
        reranker=runtime.reranker if mode == "hybrid_reranked" else None, rerank_trace=reranks)
    return tuple(prepared.retrieved_chunk_ids), [r.model_dump(mode="json", exclude={"duration_ms"}) for r in reranks]


def shared_query_pool(rankings):
    """Bound final admission, preserving a small prefix from every actual query."""
    if not 1 <= len(rankings) <= 4 or any(len(r) > 40 or len(r) != len(set(r)) for r in rankings):
        raise GenerationError("SCOPED_FLOW_QUERY_RANKINGS_BOUND")
    # This changes admission, not corpus size or final pool/context caps. Expose
    # it in the recipe trace; it is not a pure legacy-Q rerun.
    prefix = tuple(dict.fromkeys(key for depth in range(2) for row in rankings for key in row[depth:depth+1]))
    fused = tuple(reciprocal_rank_fusion(rankings))
    pool = tuple(dict.fromkeys((*prefix, *fused)))[:40]
    bundles = []
    for i, key in enumerate(pool):
        owner = min((q for q, row in enumerate(rankings) if key in row), key=lambda q:(rankings[q].index(key), q))
        bundles.append({"bundle_id":f"scoped-query-rank-{i}", "chunk_ids":[key], "query_index":owner})
    return pool, bundles


def _graph_pool(collection, lexical, graph, runtime, question, search, language, hops):
    result = prepare_authorization_answer(collection, lexical, graph, question, _vector(collection, runtime, search),
        tenant_id=collection.manifest.tenant_id, owner_id="scoped-flow-pool", run_id="scoped-flow-pool",
        recipe=runtime.recipe, hops=hops, search_query=search, allow_experimental=True, answer_language=language)
    trace = result.trace
    if trace.get("runtime_status") == "BUDGET_EXCEEDED":
        raise GenerationError("SCOPED_FLOW_GRAPH_BUDGET_NO_FALLBACK")
    facts = {f.assertion_id:f for f in graph.facts}
    paths, seen = [], set()
    for path in sorted(trace["paths"], key=lambda p:(-p["query_entities_covered"], -len(p["facet_terms"]),
            -p["hop"], p["assertion_ids"], p["entity_ids"])):
        key = facts[path["assertion_ids"][-1]].chunk_id
        if key not in seen:
            paths.append((key, path))
            seen.add(key)
    graph_ranking = tuple(k for k, _ in paths)
    fused = reciprocal_rank_fusion([trace["vector_ranking"], trace["lexical_ranking"], graph_ranking])
    prefix = tuple(dict.fromkeys(k for _, path in paths[:2] for k in path["supporting_chunk_ids"]))
    pool = tuple(dict.fromkeys((*prefix, *fused)))[:40]
    by_rep = {k:path for k, path in paths if set(path["supporting_chunk_ids"]) <= set(pool)}
    bundles = []
    for i, key in enumerate(pool):
        path = by_rep.get(key)
        bundles.append({"bundle_id":f"scoped-graph-rank-{i}", "query_index":0,
            "chunk_ids":list(path["supporting_chunk_ids"]) if path else [key],
            "assertion_ids":list(path["assertion_ids"]) if path else []})
    return pool, bundles, {"typed_graph_trace":trace, "release_sha256":graph.release_sha256,
        "path_count":len(trace["paths"]), "complete_path_bundles_in_pool":len(by_rep),
        "support_is_atomic":True}


@dataclass(frozen=True)
class PreparedScopedFlow:
    method: str
    status: str
    context: object
    binding: AnswerBinding
    call_context: CallContext
    recipe: ProfiledGenerationRecipe
    documents: dict
    chunks: dict
    trace: dict

    @property
    def ready(self):
        return self.status in {"READY", "READY_NO_EVIDENCE_ABSTENTION"}


def prepare_scoped_flow(collection, lexical, graph, runtime, *, question, search, language, method,
    run_id, owner_id, recipe, profile=None, plan=None):
    if (method not in METHODS or language not in {"en", "vi"}
            or QueryRequest(question=question).question != question
            or QueryRequest(question=search).question != search
            or language == "en" and question != search
            or type(recipe) is not ProfiledGenerationRecipe):
        raise GenerationError("SCOPED_FLOW_TASK_METHOD_OR_RECIPE")
    validate_collection_lexical_binding(collection, lexical)
    profile = SourceContextProfile(max_chunks=12) if profile is None else profile
    docs, chunks, _ = collection_records(collection)
    trace = {"version":"scoped-method-flow-v1", "method_requested":method, "method_used":method,
        "question":question, "english_search_query":search, "language":language,
        "collection_sha256":collection.manifest_sha256, "source_annotations_used_for_ranking":False,
        "hidden_fallbacks":0, "generation_executed":False}
    if method in Q_METHODS:
        queries = scoped_queries(plan, collection=collection, question=question, search=search, language=language)
        rankings = [_ranking(collection, lexical, runtime, question, query, "hybrid")[0] for query in queries]
        pool, bundles = shared_query_pool(rankings)
        trace |= {"native_v6_plan":plan.model_dump(mode="json"), "queries":queries,
            "query_rankings":rankings, "admission_policy":"TOP2_EACH_QUERY_THEN_RRF_CAP40",
            "pool_kind":"SHARED_NATIVE_EXTRACTIVE_MULTI_QUERY", "native_additional_queries":len(queries)-1}
        balanced, order = method == Q_METHODS[1], tuple((*range(1, len(queries)), 0))
        records = {}
    elif method.startswith("graph_"):
        pool, bundles, graph_trace = _graph_pool(collection, lexical, graph, runtime, question, search, language, int(method[6]))
        trace |= graph_trace | {"pool_kind":method, "queries":(search,)}
        balanced, order = False, (0,)
        records = {f.assertion_id:json.loads(f.context_json) for f in graph.facts}
    else:
        pool, reranks = _ranking(collection, lexical, runtime, question, search, method)
        bundles = [{"bundle_id":f"scoped-core-rank-{i}", "chunk_ids":[key], "query_index":0}
            for i, key in enumerate(pool)]
        trace |= {"pool_kind":method, "queries":(search,), "reranking":reranks}
        balanced, order, records = False, (0,), {}
    fingerprint = sha256_bytes(canonical_json({"collection":collection.manifest_sha256,
        "question":question, "search":search, "language":language, "kind":trace["pool_kind"],
        "pool":pool, "bundles":bundles, "queries":trace["queries"],
        "plan":trace.get("native_v6_plan"), "graph_release":trace.get("release_sha256")}))
    context, packing = pack_profiled_context(docs, chunks, pool, bundles, question=question, language=language,
        tenant_id=collection.manifest.tenant_id, collection_sha256=collection.manifest_sha256,
        profile=profile, balanced=balanced, query_order=order, graph_records=records)
    status = "READY" if context.chunk_ids else "READY_NO_EVIDENCE_ABSTENTION"
    if method in Q_METHODS and not plan.facets:
        status = "NOT_READY_NO_DISTINCT_NATIVE_FACETS"
    if method.startswith("graph_"):
        if not trace["complete_path_bundles_in_pool"]:
            status = "NOT_READY_GRAPH_ZERO_EXPANSION"
        elif not packing["used_graph"]:
            status = "NOT_READY_GRAPH_PATH_NOT_PACKED"
    trace |= {"pool_ids":pool, "bundles":bundles, "pool_fingerprint":fingerprint,
        "query_order":order, "balanced_allocator_used":balanced, "packing":packing, "mechanism_status":status,
        "context_chunk_ids":context.chunk_ids, "context_fingerprint":context.fingerprint,
        "context_proxy_tokens":context.proxy_tokens, "profile":profile.model_dump(mode="json")}
    request_id = stable_id("scoped-flow-v1", collection=collection.manifest_sha256, pool=fingerprint,
        context=context.fingerprint, recipe=recipe.fingerprint, method=method, owner=owner_id, run=run_id)
    binding = AnswerBinding(request_id=request_id, tenant_id=collection.manifest.tenant_id, owner_id=owner_id,
        mode_requested=method, mode_used=method, used_graph=packing["used_graph"],
        warnings=("EXPERIMENTAL_SCOPED_METHOD_FLOW_V1_NOT_SERVING_APPROVED",))
    call = CallContext(tenant_id=collection.manifest.tenant_id, run_id=run_id, request_id=request_id, stage="GENERATE")
    return PreparedScopedFlow(method, status, context, binding, call, recipe, docs, chunks, trace)


def answer_scoped_flow(flow, cache, *, gateway=None):
    if type(flow) is not PreparedScopedFlow or not flow.ready:
        raise GenerationError("SCOPED_FLOW_NOT_READY_NO_FALLBACK")
    return cache.resolve_profiled(flow.call_context, source_context=flow.context, documents=flow.documents,
        chunks=flow.chunks, binding=flow.binding, recipe=flow.recipe, gateway=gateway)
