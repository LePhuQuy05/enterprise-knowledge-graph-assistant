"""Local graph method execution: explicit input, bounded retrieval, cited answer and replay.

Cloud admission/capabilities remain a separate boundary. This runner never
initializes funding or chooses a provider implicitly.
"""

from time import perf_counter

import numpy as np
from pydantic import TypeAdapter

from app.collection_graph_runtime import prepare_collection_graph_answer
from app.contracts import QueryRequest, canonical_json, sha256_bytes, stable_id
from app.embeddings import query_embedding_input
from app.generation import GenerationError, GenerationRecipeV9
from app.ledger import Identifier
from app.model_gateway import CallContext
from app.query_language import answer_language
from app.rag_runner import _bind_run, _validate_encoder


def run_graph_method(collection, lexical, graph, question, *, tenant_id, owner_id, run_id,
    idempotency_key, recipe, routes, encoder, embedding_cache, answer_cache, rewrite_cache=None,
    gateway=None, hops=1, language="auto", run_kind="retrieval", allow_experimental=False):
    if type(hops) is not int or hops not in (1, 2) or run_kind not in {"retrieval", "answer"}:
        raise GenerationError("COLLECTION_GRAPH_RUN_INPUT")
    request = QueryRequest(question=question, mode=f"graph_{hops}hop")
    TypeAdapter(Identifier).validate_python(idempotency_key, strict=True)
    if language not in {"auto", "en", "vi"} or allow_experimental is not True:
        raise GenerationError("COLLECTION_GRAPH_EXPERIMENTAL_OPT_IN_REQUIRED")
    if run_kind == "retrieval" and gateway is not None:
        raise GenerationError("RETRIEVAL_ONLY_DISALLOWS_GATEWAY")
    if tenant_id != collection.manifest.tenant_id or graph.overlay.tenant_id != tenant_id:
        raise GenerationError("COLLECTION_GRAPH_RELEASE_BINDING")
    if not isinstance(recipe, GenerationRecipeV9) or recipe.model_metadata_sha256 != routes.account_model_metadata_sha256:
        raise GenerationError("COLLECTION_GRAPH_GENERATION_RECIPE")
    expected_namespace = "FIXTURE" if collection.manifest.purpose == "FIXTURE" else "LIVE"
    if answer_cache.namespace != expected_namespace or embedding_cache.namespace != expected_namespace:
        raise GenerationError("COLLECTION_GRAPH_NAMESPACE_MISMATCH")
    query_recipe = _validate_encoder(collection, encoder)
    identity = stable_id("gparent", tenant=tenant_id, owner=owner_id, key=idempotency_key)
    _bind_run(answer_cache, identity, {"question": request.question, "hops": hops, "language": language,
        "run_kind": run_kind, "run_id": run_id, "graph_file_sha256": graph.file_sha256,
        "collection_manifest_sha256": collection.manifest_sha256, "lexical_index": lexical.manifest.index_version,
        "query_recipe_sha256": query_recipe.fingerprint, "generation_recipe_sha256": recipe.fingerprint})
    selected_language = answer_language(request.question, language)
    search, rewrite, new_cost, events, warnings = request.question, None, 0, [], []
    if selected_language == "vi" and run_kind == "answer":
        if rewrite_cache is None or rewrite_cache.namespace != expected_namespace:
            raise GenerationError("COLLECTION_GRAPH_REWRITE_CACHE_REQUIRED")
        started = perf_counter()
        rewrite = rewrite_cache.resolve(CallContext(tenant_id=tenant_id, run_id=run_id, request_id=identity, stage="REWRITE"),
            question=request.question, routes=routes, gateway=gateway)
        search, new_cost = rewrite.search_query, rewrite.new_cost_micro_usd
        events.append({"stage": "query_rewrite", "duration_ms": (perf_counter() - started) * 1000,
            "cache_hit": rewrite.cache_hit, "output_count": 1})
    elif selected_language == "vi":
        warnings.append("DIRECT_VI_RETRIEVAL_DIAGNOSTIC_ENGLISH_ENCODER_ASCII_BM25")
    started = perf_counter()
    embedded = embedding_cache.resolve_local(tenant_id, text=query_embedding_input(search), recipe=query_recipe, encoder=encoder)
    vector = np.asarray(embedded.record.vector, dtype=np.float32)
    events.append({"stage": "query_embedding", "duration_ms": (perf_counter() - started) * 1000, "output_count": 1,
        "local_only": True, "recipe_sha256": query_recipe.fingerprint})
    started = perf_counter()
    view, prepared, trace = prepare_collection_graph_answer(collection, lexical, graph, request.question, vector,
        tenant_id=tenant_id, owner_id=owner_id, run_id=run_id, recipe=recipe, hops=hops, search_query=search,
        allow_experimental=True)
    events.append({"stage": "graph_retrieval_and_pack", "duration_ms": (perf_counter() - started) * 1000,
        "output_count": len(prepared.packed.chunk_ids)})
    report = {"schema_version": 1, "kind": "LOCAL_COLLECTION_GRAPH_EXPERIMENTAL_EXECUTION",
        "status": "RETRIEVED", "question": request.question, "search_query": search, "answer_language": selected_language,
        "requested_method": prepared.binding.mode_requested, "effective_method": prepared.binding.mode_used,
        "used_graph": prepared.binding.used_graph, "run_id": run_id, "request_id": prepared.binding.request_id,
        "collection_version": collection.manifest.collection_version, "collection_manifest_sha256": collection.manifest_sha256,
        "documents": collection.manifest.counts.documents, "chunks": collection.manifest.counts.chunks,
        "graph_file_sha256": graph.file_sha256, "graph_sha256": graph.overlay.graph_sha256,
        "curation_sha256": graph.overlay.curation_sha256, "human_reviewed": False, "full_relation_extraction": False,
        "trace": trace, "events": events, "response": None, "context_sources": [], "candidate_sources": [],
        "warnings": [*warnings, *prepared.binding.warnings], "rewrite_receipt": rewrite.receipt if rewrite else None,
        "new_cost_micro_usd": new_cost, "original_receipt_cost_micro_usd": None, "usage": None,
        "answer_cache_hit": False, "answer_operation_id": None, "cloud_deployed": False}
    for key in prepared.retrieved_chunk_ids:
        chunk, ref = view.chunks[key], view.source_refs[key]
        doc = view.documents[chunk.document_version_id]
        source = {"chunk_id": key, "document_version_id": doc.document_version_id, "shard_id": ref.shard_id,
            "shard_manifest_sha256": ref.manifest_sha256, "source_uri": doc.source_uri, "title": doc.title,
            "section_path": list(chunk.section_path), "start_char": chunk.start_char, "end_char": chunk.end_char,
            "text_sha256": chunk.text_sha256, "text": chunk.text}
        report["candidate_sources"].append(source)
        if key in prepared.packed.chunk_ids:
            report["context_sources"].append(source | {"source_id": "s" + str(prepared.packed.chunk_ids.index(key) + 1)})
    report["context_sources"].sort(key=lambda s: prepared.packed.chunk_ids.index(s["chunk_id"]))
    if run_kind == "answer":
        started = perf_counter()
        try:
            result = answer_cache.resolve(prepared.context, question=prepared.question, snapshot=view,
                packed=prepared.packed, binding=prepared.binding, recipe=recipe, answer_language=selected_language, gateway=gateway)
        except GenerationError as exc:
            if gateway is None and str(exc) == "ANSWER_CACHE_MISS_NO_GATEWAY":
                report["status"] = "PLANNED_NO_MODEL_CALL"
                return report
            raise
        response = result.response.model_copy(update={"retrieval_backend": "collection_numpy_global_bm25"})
        report.update(status="COMPLETED", response=response.model_dump(mode="json"),
            new_cost_micro_usd=new_cost + result.new_cost_micro_usd, answer_cache_hit=result.cache_hit,
            original_receipt_cost_micro_usd=result.receipt.original_cost_micro_usd if result.receipt else None,
            usage=result.receipt.usage.model_dump(mode="json") if result.receipt else None,
            answer_operation_id=result.receipt.operation_id if result.receipt else None)
        events.append({"stage": "generation_and_source_validation", "duration_ms": (perf_counter() - started) * 1000,
            "output_count": len(response.claims), "cache_hit": result.cache_hit})
    report["execution_identity_sha256"] = sha256_bytes(canonical_json({"request_id": prepared.binding.request_id,
        "graph_file_sha256": graph.file_sha256, "generation_recipe_sha256": recipe.fingerprint,
        "context_chunk_ids": prepared.packed.chunk_ids, "context_assertion_ids": prepared.packed.assertion_ids}))
    return report
