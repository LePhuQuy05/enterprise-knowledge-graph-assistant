"""Bounded local method comparison using the existing packer, generator and ledger.

This runner does not publish a serving release or replace authenticated cloud admission.
"""

import sqlite3
from datetime import UTC, datetime
from time import perf_counter

import numpy as np

from app.collection_answering import (
    Q_FAMILY,
    prepare_collection_answer,
    validate_collection_lexical_binding,
    validate_collection_reranker,
    validate_context_selection_recipe,
)
from app.contracts import LocalBgeRecipe, canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError, GenerationRecipeV9
from app.model_gateway import CallContext, safe_failure_code
from app.multi_query import PlanningTraceV1
from app.query_language import answer_language
from app.query_planning import PlannerRecipe, QueryPlanCache
from app.rag_methods import (
    BalancedMultiQueryMethodConfigV1,
    MethodConfigV1,
    MethodResultV1,
    MultiQueryMethodConfigV1,
    RerankedMethodConfigV1,
    RunInputV1,
    StageEventV1,
)


def _hash(value):
    return sha256_bytes(canonical_json(value))


def _event(stage, start, status, input_sha, count=0, error=None):
    return StageEventV1(stage=stage, duration_ms=round((perf_counter() - start) * 1000, 3),
                        status=status, input_sha256=input_sha, output_count=count, error_category=error)


def _bind_run(cache, identity, payload):
    """Pin the caller's idempotency key before any model work; failed receipts stay blocked."""
    db = sqlite3.connect(cache.path.with_name("method_runs.sqlite3"), timeout=5)
    try:
        with db:
            db.execute("CREATE TABLE IF NOT EXISTS runs (namespace TEXT,key TEXT,payload_sha TEXT,"
                       "PRIMARY KEY(namespace,key))")
            db.execute("INSERT OR IGNORE INTO runs VALUES (?,?,?)", (cache.namespace, identity, _hash(payload)))
            row = db.execute("SELECT payload_sha FROM runs WHERE namespace=? AND key=?",
                             (cache.namespace, identity)).fetchone()
            if row[0] != _hash(payload):
                raise GenerationError("LOCAL_RUN_IDEMPOTENCY_CONFLICT")
    finally:
        db.close()


def _validate_encoder(collection, encoder):
    document, query = encoder.recipe("document"), encoder.recipe("query")
    if document != collection.manifest.embedding_recipe:
        raise GenerationError("COLLECTION_DOCUMENT_ENCODER_MISMATCH")
    if isinstance(document, LocalBgeRecipe):
        excluded = {"input_role", "input_recipe_version"}
        if (not isinstance(query, LocalBgeRecipe) or document.input_role != "document"
                or query.input_role != "query"
                or document.model_dump(exclude=excluded) != query.model_dump(exclude=excluded)):
            raise GenerationError("COLLECTION_QUERY_ENCODER_MISMATCH")
    elif collection.manifest.purpose != "FIXTURE" or query != document:
        raise GenerationError("COLLECTION_QUERY_ENCODER_MISMATCH")
    return query


def run_methods(collection, lexical, request: RunInputV1, *, tenant_id, owner_id, run_id,
                recipe, cache, routes, encoder=None, embedding_cache=None,
                rewrite_cache=None, gateway=None, reranker=None, query_plan_cache=None,
                context_selection_recipe=None):
    """Compare L/D/H or explicitly configured R locally, with no implicit paid retries.

    Shared rewrite and query encoding are charged/measured once at report level.
    Per-method timings include their actual retrieval/packing work and cache lookup.
    """
    if not isinstance(request, RunInputV1) or not isinstance(recipe, GenerationRecipeV9):
        raise GenerationError("LOCAL_RUN_CONTRACT_INVALID")
    if tenant_id != collection.manifest.tenant_id:
        raise GenerationError("ANSWER_TENANT_MISMATCH")
    validate_collection_lexical_binding(collection, lexical)
    if recipe.model_metadata_sha256 != routes.account_model_metadata_sha256:
        raise GenerationError("LOCAL_RUN_MODEL_METADATA_MISMATCH")
    if request.run_kind == "retrieval" and gateway is not None:
        raise GenerationError("RETRIEVAL_ONLY_DISALLOWS_GATEWAY")
    if (collection.manifest.purpose == "FIXTURE") != (cache.namespace == "FIXTURE"):
        raise GenerationError("LOCAL_RUN_NAMESPACE_MISMATCH")
    with_reranking = "hybrid_reranked" in request.method_ids
    with_planning = any(method in Q_FAMILY for method in request.method_ids)
    with_balanced = "hybrid_multi_query_balanced" in request.method_ids
    selector = validate_context_selection_recipe(context_selection_recipe) if with_balanced else None
    if with_planning:
        if request.run_kind != "answer":
            raise GenerationError("LOCAL_Q_REQUIRES_ANSWER_RUN_CACHE_ONLY_OR_EXPLICIT_EXECUTE")
        if (type(query_plan_cache) is not QueryPlanCache or query_plan_cache.namespace != cache.namespace
                or not isinstance(query_plan_cache.recipe, PlannerRecipe)):
            raise GenerationError("LOCAL_RUN_PLANNER_NAMESPACE_OR_RECIPE_MISMATCH")
        PlannerRecipe.model_validate_json(query_plan_cache.recipe.model_dump_json())
    if with_reranking:
        # A mixed H/R request must not spend on H/rewrite before discovering that
        # its requested R method is unavailable or backed by a fixture scorer.
        validate_collection_reranker(collection, reranker)
    reranker_calls_before = getattr(reranker, "inference_calls", 0) if with_reranking else 0
    query_recipe = None
    if any(method != "bm25" for method in request.method_ids):
        if encoder is None or embedding_cache is None:
            raise GenerationError("LOCAL_RUN_ENCODER_REQUIRED")
        if embedding_cache.namespace != cache.namespace:
            raise GenerationError("LOCAL_RUN_NAMESPACE_MISMATCH")
        query_recipe = _validate_encoder(collection, encoder)
    identity = stable_id("ldh", tenant=tenant_id, owner=owner_id, key=request.idempotency_key)
    payload = {"input": request.model_dump(mode="json"), "collection_sha": collection.manifest_sha256,
               "lexical_version": lexical.manifest.index_version, "recipe": recipe.fingerprint,
               "query_recipe": query_recipe.fingerprint if query_recipe else None, "run_id": run_id}
    if with_reranking:
        payload["reranker_recipe"] = reranker.recipe.fingerprint
    if with_planning:
        payload["planner_recipe"] = query_plan_cache.recipe.fingerprint
    if with_balanced:
        payload["context_selection_recipe"] = selector.fingerprint
    _bind_run(cache, identity, payload)
    language = answer_language(request.original_question, request.language)
    question_sha = _hash(request.original_question)
    trace, warnings, search_query = [], [], request.original_question
    shared_cost, rewrite_receipt, rewrite_hit = 0, None, False
    if language == "vi" and request.run_kind == "answer":
        if rewrite_cache is None or rewrite_cache.namespace != cache.namespace:
            raise GenerationError("LOCAL_RUN_REWRITE_CACHE_REQUIRED")
        start = perf_counter()
        try:
            rewritten = rewrite_cache.resolve(CallContext(tenant_id=tenant_id, run_id=run_id,
                request_id=identity, stage="REWRITE"), question=request.original_question,
                routes=routes, gateway=gateway)
            search_query, rewrite_hit = rewritten.search_query, rewritten.cache_hit
            shared_cost, rewrite_receipt = rewritten.new_cost_micro_usd, rewritten.receipt
            trace.append(_event("query_rewrite", start, "COMPLETED", question_sha, 1))
        except Exception as exc:
            # No fallback to untranslated search while reporting a rewritten VI answer.
            error = safe_failure_code(exc)
            trace.append(_event("query_rewrite", start, "FAILED", question_sha, error=error))
            blocked = {"schema_version": 1, "status": "BLOCKED_SHARED_REWRITE", "error_category": error,
                    "run_identity": identity, "input": request.model_dump(mode="json"),
                    "shared_trace": [event.model_dump(mode="json") for event in trace],
                    "methods": [], "new_known_cost_micro_usd": 0,
                    "shared_cost_micro_usd": None if gateway else 0,
                    "cost_complete": gateway is None, "cloud_deployed": False,
                    "semantic_review": "NOT_REVIEWED"}
            if with_reranking:
                blocked.update(kind="LOCAL_COLLECTION_LDHR_COMPARISON", local_reranker_inference_calls=0,
                               reranker_recipe_sha256=reranker.recipe.fingerprint)
            if with_planning:
                blocked.update(kind="LOCAL_COLLECTION_LDHRQ_COMPARISON",
                               planner_recipe_sha256=query_plan_cache.recipe.fingerprint)
            if with_balanced:
                blocked.update(kind="LOCAL_COLLECTION_LDHRQQB_COMPARISON",
                    context_selection_recipe_sha256=selector.fingerprint,
                    planner_sharing="Q_FAMILY_V1_SINGLE_DURABLE_PLANNER_OPERATION")
            return blocked
    elif language == "vi":
        warnings.append("DIRECT_VI_RETRIEVAL_DIAGNOSTIC_ENGLISH_ENCODER_ASCII_BM25")

    vector, embedding_hit = None, None
    if query_recipe is not None:
        from app.embeddings import query_embedding_input

        start = perf_counter()
        encoded = embedding_cache.resolve_local(tenant_id, text=query_embedding_input(search_query),
                                                 recipe=query_recipe, encoder=encoder)
        vector = np.asarray(encoded.record.vector, dtype=np.float32)
        embedding_hit = encoded.cache_hit
        trace.append(_event("query_embedding", start, "COMPLETED", _hash(search_query), 1))

    results, peer_failed = [], False
    for method in request.method_ids:
        config_type = {"hybrid_reranked": RerankedMethodConfigV1,
                       "hybrid_multi_query": MultiQueryMethodConfigV1,
                       "hybrid_multi_query_balanced": BalancedMultiQueryMethodConfigV1}.get(method, MethodConfigV1)
        extra = {"reranker_recipe": reranker.recipe, "reranker_recipe_sha256": reranker.recipe.fingerprint} if method == "hybrid_reranked" else {}
        if method in Q_FAMILY:
            extra = {"planner_recipe": query_plan_cache.recipe, "planner_recipe_sha256": query_plan_cache.recipe.fingerprint}
        if method == "hybrid_multi_query_balanced":
            extra.update(context_selection_recipe=selector, context_selection_recipe_sha256=selector.fingerprint)
        config = config_type(method_id=method, fusion="RRF60" if method in {"hybrid", "hybrid_reranked", *Q_FAMILY} else "NONE",
            collection_manifest_sha256=collection.manifest_sha256,
            lexical_index_version=lexical.manifest.index_version,
            document_recipe_sha256=collection.manifest.embedding_recipe.fingerprint,
            query_recipe_sha256=query_recipe.fingerprint if method != "bm25" else None,
            generation_recipe_sha256=recipe.fingerprint, **extra)
        data = {"requested_method": method, "effective_method": method, "config": config,
                "config_sha256": config.fingerprint, "collection_version": collection.manifest.collection_version}
        events = []
        start = perf_counter()
        if (with_reranking or with_planning) and peer_failed and gateway is not None:
            events.append(_event("generation_and_validation", start, "FAILED", config.fingerprint,
                                 error="BLOCKED_AFTER_PEER_FAILURE"))
            results.append(MethodResultV1(**data, status="FAILED", trace=tuple(events),
                error_category="BLOCKED_AFTER_PEER_FAILURE", new_cost_micro_usd=0))
            continue
        rerank_traces, query_traces, selection_traces, selection_timings = [], [], [], []
        planner_cost, subqueries, subquery_vectors = 0, (), ()
        stage = "retrieval_and_pack"
        try:
            if method in Q_FAMILY:
                stage, start = "query_planning", perf_counter()
                planner_identity = ({"planner_recipe": query_plan_cache.recipe.fingerprint,
                                     "sharing": "q-family-v1"} if with_balanced else config.fingerprint)
                planned = query_plan_cache.resolve(CallContext(tenant_id=tenant_id, run_id=run_id,
                    request_id=stable_id("plan", run=identity, config=planner_identity), stage="PLAN_QUERY"),
                    question=request.original_question, search_query=search_query, language=language,
                    routes=routes, gateway=gateway)
                planner_cost, subqueries = planned.new_cost_micro_usd, planned.subqueries
                data["planning_trace"] = PlanningTraceV1(recipe_sha256=query_plan_cache.recipe.fingerprint,
                    subqueries=subqueries, cache_hit=planned.cache_hit, new_cost_micro_usd=planner_cost, receipt=planned.receipt)
                events.append(_event(stage, start, "COMPLETED", _hash([question_sha, search_query]), len(subqueries)))
                stage, start = "subquery_embeddings", perf_counter()
                encoded_queries = [embedding_cache.resolve_local(tenant_id, text=query_embedding_input(query),
                    recipe=query_recipe, encoder=encoder) for query in subqueries]
                subquery_vectors = tuple(np.asarray(value.record.vector, dtype=np.float32) for value in encoded_queries)
                events.append(_event(stage, start, "COMPLETED", _hash(subqueries), len(subqueries)))
                stage, start = "retrieval_and_pack", perf_counter()
            view, prepared = prepare_collection_answer(collection, lexical, request.original_question, vector,
                tenant_id=tenant_id, owner_id=owner_id, run_id=run_id, recipe=recipe,
                mode=method, search_query=search_query,
                request_id=stable_id("method", run=identity, config=config.fingerprint),
                reranker=reranker if method == "hybrid_reranked" else None, rerank_trace=rerank_traces,
                subqueries=subqueries, subquery_vectors=subquery_vectors, multi_query_trace=query_traces,
                context_selection_recipe=selector if method == "hybrid_multi_query_balanced" else None,
                context_selection_trace=selection_traces, context_selection_timing=selection_timings)
            if query_traces:
                data["multi_query_trace"] = query_traces[0]
            selection_ms = selection_timings[0] if selection_timings else 0
            if selection_traces:
                data["context_selection_trace"] = selection_traces[0]
                events.append(StageEventV1(stage="context_selection", duration_ms=selection_ms,
                    status="COMPLETED", input_sha256=selection_traces[0].source_trace_sha256,
                    output_count=len(selection_traces[0].ordered_chunk_ids)))
            rerank_ms = rerank_traces[0].duration_ms if rerank_traces else 0
            if rerank_traces:
                data["rerank_trace"] = rerank_traces[0]
                events.append(StageEventV1(stage="reranking", duration_ms=rerank_ms, status="COMPLETED",
                    input_sha256=_hash({"query": search_query, "candidates": rerank_traces[0].baseline_candidate_ids,
                                       "recipe": reranker.recipe.fingerprint}),
                    output_count=len(rerank_traces[0].ranked_chunk_ids)))
            events.append(_event("retrieval_and_pack", start, "COMPLETED", _hash(search_query),
                                 len(prepared.packed.chunk_ids)))
            if rerank_traces or selection_traces:
                # Keep stages additive: this event excludes the separately
                # reported reranking interval contained in the adapter call.
                events[-1] = events[-1].model_copy(update={
                    "duration_ms": round(max(0, events[-1].duration_ms - rerank_ms - selection_ms), 3)})
            data.update(ranked_chunk_ids=prepared.retrieved_chunk_ids,
                        context_chunk_ids=prepared.packed.chunk_ids, context_proxy_tokens=prepared.packed.proxy_tokens)
            sources = []
            for source_id, chunk_id in zip(prepared.packed.source_ids, prepared.packed.chunk_ids, strict=True):
                chunk = view.chunks[chunk_id]
                document = view.documents[chunk.document_version_id]
                ref = view.source_refs[chunk_id]
                sources.append({"source_id": source_id, "chunk_id": chunk_id,
                    "document_version_id": document.document_version_id, "shard_id": ref.shard_id,
                    "shard_manifest_sha256": ref.manifest_sha256, "source_uri": document.source_uri,
                    "title": document.title, "section_path": list(chunk.section_path),
                    "start_char": chunk.start_char, "end_char": chunk.end_char,
                    "text_sha256": sha256_bytes(chunk.text.encode("utf-8")), "text": chunk.text})
            data["context_sources"] = tuple(sources)
        except Exception as exc:
            error = safe_failure_code(exc)
            miss = gateway is None and "CACHE_MISS_NO_GATEWAY" in str(exc)
            events.append(_event(stage, start, "CACHE_MISS" if miss else "FAILED", _hash(search_query), error=error))
            unknown = stage == "query_planning" and gateway is not None
            results.append(MethodResultV1(**data, status="PLANNED_NO_MODEL_CALL" if miss else "FAILED",
                new_cost_micro_usd=None if unknown else planner_cost, trace=tuple(events), error_category=error))
            if not miss:
                peer_failed = True
            continue
        if request.run_kind == "retrieval":
            results.append(MethodResultV1(**data, status="RETRIEVED", trace=tuple(events)))
            continue
        start = perf_counter()
        try:
            lookup = cache.resolve(prepared.context, question=prepared.question, snapshot=view,
                packed=prepared.packed, binding=prepared.binding, recipe=recipe,
                answer_language=language, gateway=gateway)
            response = lookup.response.model_copy(update={"retrieval_backend": "collection_numpy_global_bm25"})
            events.append(_event("generation_and_validation", start, "COMPLETED", config.fingerprint,
                                 len(response.claims)))
            results.append(MethodResultV1(**data, status="COMPLETED", response=response,
                cache_hit=lookup.cache_hit, new_cost_micro_usd=lookup.new_cost_micro_usd + planner_cost,
                original_receipt_cost_micro_usd=lookup.receipt.original_cost_micro_usd if lookup.receipt else None,
                usage=lookup.receipt.usage.model_dump(mode="json") if lookup.receipt else None,
                trace=tuple(events)))
        except Exception as exc:
            error = safe_failure_code(exc)
            miss = str(exc) == "ANSWER_CACHE_MISS_NO_GATEWAY"
            events.append(_event("generation_and_validation", start, "CACHE_MISS" if miss else "FAILED",
                                 config.fingerprint, error=error))
            results.append(MethodResultV1(**data, status="PLANNED_NO_MODEL_CALL" if miss else "FAILED",
                new_cost_micro_usd=None if gateway else planner_cost, trace=tuple(events), error_category=error))
            if not miss:
                peer_failed = True
    report = {
        "schema_version": 1, "kind": ("LOCAL_COLLECTION_LDHRQ_COMPARISON" if with_planning else
            "LOCAL_COLLECTION_LDHR_COMPARISON" if with_reranking else "LOCAL_COLLECTION_LDH_COMPARISON"),
        "status": "COMPLETED" if all(r.status in {"RETRIEVED", "COMPLETED"} for r in results) else "PARTIAL",
        "created_at": datetime.now(UTC).isoformat(), "run_identity": identity,
        "input": request.model_dump(mode="json"), "answer_language": language, "search_query": search_query,
        "query_transform": "CACHED_EN_REWRITE_V1" if rewrite_receipt else "NONE",
        "collection_version": collection.manifest.collection_version,
        "collection_manifest_sha256": collection.manifest_sha256,
        "documents": collection.manifest.counts.documents, "chunks": collection.manifest.counts.chunks,
        "shared_trace": [event.model_dump(mode="json") for event in trace],
        "rewrite_cache_hit": rewrite_hit, "rewrite_receipt": rewrite_receipt,
        "query_embedding_cache_hit": embedding_hit, "shared_cost_micro_usd": shared_cost,
        "methods": [r.model_dump(mode="json") for r in results], "warnings": warnings,
        "new_known_cost_micro_usd": shared_cost + sum((r.new_cost_micro_usd if r.new_cost_micro_usd is not None else
            r.planning_trace.new_cost_micro_usd if r.planning_trace is not None else 0) for r in results),
        "cost_complete": all(r.new_cost_micro_usd is not None for r in results),
        "cost_basis": "PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND_NOT_INVOICE",
        "semantic_review": "NOT_REVIEWED", "quality_metrics": None, "cloud_deployed": False,
        "limitations": ["Execution success is not answer correctness or human review.",
            "Local preparation timings exclude AWS cold starts and network; model calls are bounded by the existing ledger.",
            "Shared rewrite/embedding occur once; their time and cost are not duplicated in each method.",
            ("Reranking is local and experimental; graph, multi-query and authenticated cloud collection serving are not enabled."
             if with_reranking else "Graph, reranking, multi-query and authenticated cloud collection serving are not enabled by this runner.")],
    }
    if with_reranking:
        report["local_reranker_inference_calls"] = getattr(reranker, "inference_calls", 0) - reranker_calls_before
        report["reranker_recipe_sha256"] = reranker.recipe.fingerprint
    if with_planning:
        report["planner_recipe_sha256"] = query_plan_cache.recipe.fingerprint
        report["limitations"][-1] = (
            "Multi-query is bounded and experimental; planner output is never answer evidence or a proof of semantic preservation. "
            "Graph and authenticated cloud collection serving are not enabled.")
    if with_balanced:
        report.update(kind="LOCAL_COLLECTION_LDHRQQB_COMPARISON",
            context_selection_recipe_sha256=selector.fingerprint,
            planner_sharing="Q_FAMILY_V1_SINGLE_DURABLE_PLANNER_OPERATION")
        report["limitations"].append(
            "Balanced context only reorders the frozen Q pool with RR_ID_ONLY; it is not semantic coverage validation.")
    return report
