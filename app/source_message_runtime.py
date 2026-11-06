"""Opt-in packing adapter for common methods and typed authorization G1/G2.

The same source-message budget is required for every compared method. Graph
traversal/rank pools and native Q/QB planner/selector remain unchanged. Prior
prepared answers and paid caches are not mutated or relabelled.
"""

import json
from dataclasses import replace

from app.authorization_retrieval import AuthorizationEvidenceGraph, prepare_authorization_answer
from app.collection_answering import _evidence_view
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.graph_retrieval import GraphCandidate, GraphRetrievalResult
from app.method_execution import ReviewedGraphExecutionV3, StrictPreparedMethod
from app.retrieval import reciprocal_rank_fusion
from app.source_message_packing import POLICY, pack_source_message


def _rebind(prepared, packing, *, used_graph, warnings, ranking):
    request_id = stable_id("source-message-answer-v1", predecessor=prepared.binding.request_id,
        packing=packing.recipe_sha256, user_message=packing.user_message_sha256, audit=packing.audit_sha256)
    binding = prepared.binding.model_copy(update={"request_id": request_id, "used_graph": used_graph,
        "warnings": tuple(dict.fromkeys((*warnings, *packing.packed.warnings)))})
    return replace(prepared, packed=packing.packed, binding=binding,
        context=prepared.context.model_copy(update={"request_id": request_id}), retrieved_chunk_ids=tuple(ranking))


def packing_trace(packing):
    return {"policy": POLICY, "packing_recipe_sha256": packing.recipe_sha256,
        "audit_sha256": packing.audit_sha256, "user_message_sha256": packing.user_message_sha256,
        "system_sha256": packing.system_sha256, "user_message_proxy_tokens": packing.packed.proxy_tokens,
        "system_proxy_tokens": packing.system_proxy_tokens, "audit_bytes": len(packing.audit_json.encode()),
        "audit": json.loads(packing.audit_json), "decisions": json.loads(packing.decisions_json),
        "provider_exact_token_count": False}


def repack_source_answer(view, prepared, *, answer_language="en"):
    if prepared.binding.mode_used.startswith("graph_") or prepared.packed.assertion_ids:
        raise GenerationError("SOURCE_MESSAGE_GRAPH_REQUIRES_PATH_ADAPTER")
    ranking = prepared.retrieved_chunk_ids
    result = GraphRetrievalResult(ranking, ranking, (), (), prepared.binding.mode_used, False, (), 0, 0)
    packing = pack_source_message(view, result, question=prepared.question, recipe=prepared.recipe,
        tenant_id=prepared.binding.tenant_id, answer_language=answer_language)
    warnings = tuple(w for w in prepared.binding.warnings if w != "CONTEXT_BUNDLE_SKIPPED")
    return _rebind(prepared, packing, used_graph=False, warnings=warnings, ranking=ranking), packing_trace(packing)


def repack_strict_query_answer(result):
    if type(result) is not StrictPreparedMethod or result.prepared is None:
        raise GenerationError("SOURCE_MESSAGE_READY_STRICT_QUERY_REQUIRED")
    if result.outcome.requested_method not in {"hybrid_multi_query", "hybrid_multi_query_balanced"}:
        raise GenerationError("SOURCE_MESSAGE_STRICT_QUERY_METHOD")
    prepared, trace = repack_source_answer(result.view, result.prepared, answer_language=result.answer_language)
    outcome_data = result.outcome.model_dump(mode="json") | {"context_chunk_ids": prepared.packed.chunk_ids}
    outcome = type(result.outcome).model_validate_json(canonical_json(outcome_data))
    represented = sorted({i for i, query in enumerate(result.trace["multi_query"]["queries"])
        if set(query["ranked_chunk_ids"]) & set(prepared.packed.chunk_ids)})
    return StrictPreparedMethod(outcome, result.view, prepared,
        result.trace | {"source_message_packing": trace, "context_chunk_ids": prepared.packed.chunk_ids,
            "predecessor_context_chunk_ids": result.outcome.context_chunk_ids,
            "predecessor_context_proxy_tokens": result.trace["context_proxy_tokens"],
            "packed_chunk_ids": prepared.packed.chunk_ids, "context_proxy_tokens": prepared.packed.proxy_tokens,
            "packed_queries_represented": represented}, result.answer_language)


def prepare_source_message_graph(collection, lexical, graph, question, query_vector, **options):
    if type(graph) is not AuthorizationEvidenceGraph:
        raise GenerationError("SOURCE_MESSAGE_AUTHORIZATION_GRAPH_REQUIRED")
    predecessor = prepare_authorization_answer(collection, lexical, graph, question, query_vector, **options)
    if predecessor.prepared is None:
        return predecessor  # Timeout/failure cannot silently become a base answer.
    question = predecessor.prepared.question
    old = predecessor.trace
    by_id = {fact.assertion_id: fact for fact in graph.facts}
    eligible = sorted(old["paths"], key=lambda p: (-p["query_entities_covered"], -len(p["facet_terms"]),
        -p["hop"], p["assertion_ids"], p["entity_ids"]))
    candidates, representatives = [], set()
    for path in eligible:
        key = by_id[path["assertion_ids"][-1]].chunk_id
        if key not in representatives:
            representatives.add(key)
            candidates.append(GraphCandidate(key, path["hop"], tuple(path["entity_ids"]),
                tuple(path["assertion_ids"]), tuple(path["supporting_chunk_ids"])))
    graph_ranking = tuple(c.chunk_id for c in candidates)
    if graph_ranking != tuple(old["graph_ranking"]):
        raise GenerationError("SOURCE_MESSAGE_GRAPH_RANKING_CHANGED")
    fused = reciprocal_rank_fusion([old["vector_ranking"], old["lexical_ranking"], graph_ranking])
    prioritized = tuple(dict.fromkeys(k for c in candidates[:2] for k in (c.chunk_id, *c.supporting_chunk_ids)))
    ranking = tuple(dict.fromkeys((*prioritized, *fused)))[:40]
    candidates = tuple(c for c in candidates if set(c.supporting_chunk_ids) <= set(ranking))
    base = tuple(old["baseline_ranking"])
    method, seeds = predecessor.outcome.requested_method, tuple(old["seed_entity_ids"])
    records = {fact.assertion_id: json.loads(fact.context_json) for fact in graph.facts}
    language = predecessor.answer_language

    def pack(order, paths):
        view = _evidence_view(collection, lexical, order)
        result = GraphRetrievalResult(order, base, paths, seeds, method, bool(paths), (), old["assertion_visits"], 0)
        packing = pack_source_message(view, result, question=question, recipe=predecessor.prepared.recipe,
            tenant_id=predecessor.prepared.binding.tenant_id, answer_language=language, graph_records=records)
        return view, packing

    view, packing = pack(ranking, candidates)
    packed_paths = [p for p in eligible if set(p["assertion_ids"]) <= set(packing.packed.assertion_ids)
        and set(p["supporting_chunk_ids"]) <= set(packing.packed.chunk_ids)]
    used = bool(packed_paths)
    reason = None if used else "NO_LINKED_ENTITY" if not seeds else "NO_RELEVANT_PATH" if not eligible else "NO_PATH_FITS_CONTEXT"
    if not used:
        ranking = base
        view, packing = pack(ranking, ())
    _, base_packing = pack(base, ())
    warnings = tuple(w for w in predecessor.prepared.binding.warnings if
        not w.startswith("GRAPH_ZERO_EXPANSION_") and w != "CONTEXT_BUNDLE_SKIPPED")
    if reason:
        warnings += ("GRAPH_ZERO_EXPANSION_" + reason,)
    prepared = _rebind(predecessor.prepared, packing, used_graph=used, warnings=warnings, ranking=ranking)
    trace = old | {"source_message_policy": POLICY, "predecessor_packing_status": predecessor.outcome.mechanism_status,
        "predecessor_context_chunk_ids": predecessor.outcome.context_chunk_ids,
        "predecessor_context_proxy_tokens": old["context_proxy_tokens"], "selection_order": ranking,
        "context_chunk_ids": packing.packed.chunk_ids, "context_assertion_ids": packing.packed.assertion_ids,
        "context_proxy_tokens": packing.packed.proxy_tokens, "packed_paths": packed_paths,
        "runtime_status": "USED_GRAPH" if used else "GRAPH_ZERO_EXPANSION", "zero_expansion_reason": reason,
        "baseline_context_chunk_ids": base_packing.packed.chunk_ids,
        "graph_added_context_chunk_ids": tuple(k for k in packing.packed.chunk_ids if k not in base_packing.packed.chunk_ids),
        "source_message_packing": packing_trace(packing),
        "source_records": [records[k] for k in packing.packed.assertion_ids],
        "method_config_sha256": sha256_bytes(canonical_json({"graph_release": graph.release_sha256,
            "source_packing_recipe": packing.recipe_sha256})), "packing_warnings": packing.packed.warnings}
    outcome = ReviewedGraphExecutionV3(requested_method=method, effective_method=method, mechanism_status=trace["runtime_status"],
        executed_stages=(*predecessor.outcome.executed_stages, "PACK_EXACT_USER_MESSAGE"), graph_path_used=used,
        context_chunk_ids=packing.packed.chunk_ids, context_assertion_ids=packing.packed.assertion_ids,
        zero_expansion_reason=reason, release_sha256=graph.release_sha256)
    return StrictPreparedMethod(outcome, view, prepared, trace, language)
