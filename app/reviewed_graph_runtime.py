"""Local strict G1/G2 over reviewed typed evidence; no hidden method substitution."""

import time
from collections import defaultdict, deque
from dataclasses import replace
from types import MappingProxyType

from app.answer_pipeline import PreparedAnswer
from app.collection_answering import _evidence_view, validate_collection_lexical_binding
from app.context import pack_context
from app.contracts import QueryRequest, sha256_bytes, stable_id
from app.generation import AnswerBinding, GenerationError
from app.graph_retrieval import GraphCandidate, GraphRetrievalResult
from app.method_execution import MethodExecutionV2, ReviewedGraphExecutionV3, StrictPreparedMethod
from app.model_gateway import CallContext
from app.retrieval import reciprocal_rank_fusion
from app.reviewed_source_graph import (
    INTERPRETATION,
    ValidatedReviewedSourceGraph,
    evidence_adjacency,
)
from app.snapshot import SnapshotContent
from app.strict_collection_methods import GENERIC_ALIASES, _tokens, link_query_entities

POLICY = "SOURCE_REVIEWED_TYPED_DIRECTED_ACTIONS_ATOMIC_PATHS_V3"
# Shared verbs and generic AWS nouns are too weak to establish the question's
# operation/target: encrypt(event) must not match a question about event metadata.
WEAK_FACETS = frozenset("use used using send sent deliver publish invoke read write encrypt decrypt "
    "encrypted decrypted function event object data key bucket target work happen automatically "
    "some all each also continue can cannot".split())


def prepare_reviewed_graph_answer(collection, lexical, graph, question, query_vector, *, tenant_id, owner_id,
    run_id, recipe, hops=1, search_query=None, allow_experimental=False, time_budget_ms=300,
    clock=None, traversal_clock=None, answer_language="en"):
    if allow_experimental is not True or type(graph) is not ValidatedReviewedSourceGraph:
        raise GenerationError("REVIEWED_GRAPH_VALIDATED_RELEASE_REQUIRED")
    if type(hops) is not int or hops not in (1, 2):
        raise GenerationError("REVIEWED_GRAPH_HOP_BOUND")
    if type(time_budget_ms) is not int or not 1 <= time_budget_ms <= 300:
        raise GenerationError("REVIEWED_GRAPH_TIME_BOUND")
    if answer_language not in {"en", "vi"}:
        raise GenerationError("REVIEWED_GRAPH_ANSWER_LANGUAGE")
    method, overlay = f"graph_{hops}hop", graph.overlay
    question = QueryRequest(question=question, mode=method).question
    search = QueryRequest(question=question if search_query is None else search_query, mode=method).question
    if (tenant_id != collection.manifest.tenant_id or tenant_id != overlay.tenant_id
            or overlay.collection_manifest_sha256 != collection.manifest_sha256
            or overlay.collection_version != collection.manifest.collection_version):
        raise GenerationError("REVIEWED_GRAPH_COLLECTION_BINDING")
    validate_collection_lexical_binding(collection, lexical)
    stages = ["VALIDATE_REVIEWED_RELEASE", "RETRIEVE_BASE", "LINK_ENTITIES", "TRAVERSE_GRAPH"]
    vector_ids = tuple(h.chunk_id for h in collection.search_vector(query_vector, k=20))
    lexical_ids = tuple(k for k, _ in lexical.search(search, k=20))
    baseline = tuple(reciprocal_rank_fusion([vector_ids, lexical_ids]))[:40]
    scores = {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        scores.update((key, float(value)) for key, value in zip(shard.chunk_ids, shard.vectors @ query_vector, strict=True))
    entities = {e.entity_id: e for e in overlay.entities}
    by_id = {e.assertion_id: e for e in graph.edges}
    adjacent = evidence_adjacency(graph)
    # Only catalogue names can seed across sources. A generic/source-local name
    # in the question must never be resolved into an arbitrary document's node.
    catalogue_ids = {n["node_id"] for e in graph.edges for n in e.metadata()["nodes"] if n["identity_scope"] == "CATALOG_TYPE"}
    links = link_query_entities(tuple(e for e in overlay.entities if e.entity_id in catalogue_ids), search, question)
    direct = tuple(dict.fromkeys(link["entity_id"] for link in links if link["selected"]))
    mentions = defaultdict(set)
    for m in overlay.mentions:
        mentions[m.chunk_id].add(m.entity_id)
    derived_ranks = defaultdict(list)
    for rank, key in enumerate(baseline[:20], 1):
        for entity in mentions[key]:
            if entity in catalogue_ids and entities[entity].canonical_name.casefold() not in GENERIC_ALIASES and adjacent[entity]:
                derived_ranks[entity].append(rank)
    derived = sorted(derived_ranks, key=lambda e: (min(derived_ranks[e]), -len(derived_ranks[e]), e))
    seeds = tuple(dict.fromkeys((*direct, *derived)))[:8]
    trace = {"schema_version": 3, "policy": POLICY, "requested_method": method, "question": question,
        "search_query": search, "source_manifest_sha256": collection.manifest_sha256,
        "reviewed_release_sha256": graph.release_sha256, "links": links, "seed_entity_ids": seeds,
        "retrieval_seed_ranks": dict(derived_ranks), "relation_assertions": len(overlay.assertions),
        "action_assertions": sum(e.kind == "ACTION" for e in graph.edges),
        "negative_assertions_excluded": sum(e.polarity != "POSITIVE" for e in graph.edges),
        "vector_ranking": vector_ids, "lexical_ranking": lexical_ids, "baseline_ranking": baseline,
        "paths": [], "packed_paths": [], "assertion_visits": 0, "truncated": False,
        "interpretation": INTERPRETATION, "action_traversal": "DIRECTED_ACTOR_TO_TARGET",
        "relevance_policy": "EACH_EDGE_NEEDS_DISTINCTIVE_QUERY_FACET_OR_TWO_NAMED_CONNECTION_ENDPOINTS",
        "cross_source_condition_compatibility": "NOT_INFERRED_READ_EACH_FULL_SOURCE",
        "context_limits": {"max_chunks": 8, "proxy_tokens": 4096}, "hidden_fallbacks": 0,
        "human_reviewed": 0, "full_relation_extraction": False}
    timer = time.monotonic if clock is None else clock
    tick = timer if traversal_clock is None else traversal_clock
    started = timer()
    queue = deque((seed, (seed,), (), ()) for seed in seeds)
    visited, seen_paths, paths = set(seeds), set(), []
    linked_terms = set().union(*(_tokens(entities[e].canonical_name) for e in direct)) if direct else set()
    facets = _tokens(search) - linked_terms
    connection_intent = bool(facets & {"connection", "relationship", "related", "connect", "relation"})
    while queue:
        if (tick() - started) * 1000 >= time_budget_ms:
            trace.update(visited_entities=len(visited), paths=paths, runtime_status="BUDGET_EXCEEDED")
            outcome = MethodExecutionV2(requested_method=method, mechanism_status="BUDGET_EXCEEDED",
                executed_stages=tuple(stages), failure_reason="BUDGET_EXCEEDED")
            return StrictPreparedMethod(outcome, None, None, trace, answer_language)
        entity, entity_path, edge_path, directions = queue.popleft()
        if len(edge_path) == hops:
            continue
        neighbors = sorted(adjacent[entity], key=lambda row: (-scores[row[0].chunk_id], row[0].assertion_id))
        trace["truncated"] |= len(neighbors) > 20
        for edge, other, direction in neighbors[:20]:
            if other in entity_path:
                continue
            if trace["assertion_visits"] >= 200 or other not in visited and len(visited) >= 80:
                trace["truncated"] = True
                continue
            trace["assertion_visits"] += 1
            ids, nodes, travel = (*edge_path, edge.assertion_id), (*entity_path, other), (*directions, direction)
            if ids in seen_paths:
                continue
            seen_paths.add(ids)
            visited.add(other)
            supports = tuple(dict.fromkeys(by_id[key].chunk_id for key in ids))
            terms = sorted(facets & _tokens(edge.metadata()["record"]["clause_quote"]) - WEAK_FACETS)
            covered = len(set(nodes) & set(direct))
            named_connection = connection_intent and {edge.subject_id, edge.object_id} <= set(direct)
            # Every extension must contribute its own query facet. Terms from a
            # relevant prefix cannot justify an unrelated second edge.
            relevant = bool((covered and (terms or named_connection) or not direct and len(terms) >= 2)
                and (not edge_path or all(p["accepted"] for p in paths if tuple(p["assertion_ids"]) == edge_path)))
            paths.append({"entity_ids": nodes, "assertion_ids": ids, "traversal_directions": travel,
                "supporting_chunk_ids": supports, "hop": len(ids), "accepted": relevant,
                "reason": "RELEVANCE_HEURISTIC_ACCEPTED" if relevant else "NO_QUERY_FACET_SUPPORT",
                "facet_terms": terms, "query_entities_covered": covered,
                "source_vector_score": sum(scores[k] for k in supports) / len(supports),
                "interpretation": INTERPRETATION})
            queue.append((other, nodes, ids, travel))
    trace.update(paths=paths, visited_entities=len(visited))
    eligible = sorted((p for p in paths if p["accepted"]), key=lambda p: (
        -p["query_entities_covered"], -len(p["facet_terms"]), -p["source_vector_score"], p["hop"], p["assertion_ids"]))
    candidates, representatives = [], set()
    for path in eligible:
        key = by_id[path["assertion_ids"][-1]].chunk_id
        if key not in representatives:
            representatives.add(key)
            candidates.append(GraphCandidate(key, path["hop"], tuple(path["entity_ids"]),
                tuple(path["assertion_ids"]), tuple(path["supporting_chunk_ids"])))
    graph_ranking = tuple(c.chunk_id for c in candidates)
    fused = tuple(reciprocal_rank_fusion([vector_ids, lexical_ids, graph_ranking]))
    prioritized = tuple(dict.fromkeys(k for c in candidates[:2] for k in (c.chunk_id, *c.supporting_chunk_ids)))
    ranked = tuple(dict.fromkeys((*prioritized, *fused)))[:40]
    candidates = tuple(c for c in candidates if set(c.supporting_chunk_ids) <= set(ranked))
    view = _evidence_view(collection, lexical, ranked)
    view = replace(view, entities=MappingProxyType(entities), content=SnapshotContent((), (), overlay.entities, overlay.mentions, ()))
    records = {key: edge.context_record() for key, edge in by_id.items()}
    retrieval = GraphRetrievalResult(ranked, baseline, candidates, seeds, method, bool(candidates), (), trace["assertion_visits"], len(visited))
    stages.extend(("FUSE_RANKS", "SELECT_CONTEXT", "PACK_CONTEXT"))
    packed = pack_context(view, retrieval, graph_records=records)
    attempted_warnings = packed.warnings
    packed_paths = [p for p in eligible if set(p["assertion_ids"]) <= set(packed.assertion_ids)
        and set(p["supporting_chunk_ids"]) <= set(packed.chunk_ids)]
    used = bool(packed_paths)
    reason = None
    if not used:
        reason = "NO_LINKED_ENTITY" if not seeds else "NO_RELEVANT_PATH" if not eligible else "NO_PATH_FITS_CONTEXT"
        # Completed zero expansion is explicit, never relabelled USED_GRAPH or
        # effective_method=hybrid. Repack the same base lists without partial paths.
        ranked = baseline
        view = _evidence_view(collection, lexical, ranked)
        retrieval = GraphRetrievalResult(ranked, baseline, (), seeds, method, False, (), trace["assertion_visits"], len(visited))
        packed = pack_context(view, retrieval)
    base_view = _evidence_view(collection, lexical, baseline)
    base_result = GraphRetrievalResult(baseline, baseline, (), (), method, False, (), 0, 0)
    base_packed = pack_context(base_view, base_result)
    touched = {key for p in paths for key in p["assertion_ids"]}
    trace.update(graph_ranking=graph_ranking, fused_ranking=fused[:40], selection_order=ranked,
        prioritized_bundle_chunk_ids=prioritized, packed_paths=packed_paths,
        context_chunk_ids=packed.chunk_ids, context_assertion_ids=packed.assertion_ids,
        context_proxy_tokens=packed.proxy_tokens, packing_warnings=packed.warnings,
        attempted_path_packing_warnings=attempted_warnings,
        baseline_context_chunk_ids=base_packed.chunk_ids,
        graph_added_context_chunk_ids=tuple(k for k in packed.chunk_ids if k not in base_packed.chunk_ids),
        source_records=[{"source": by_id[key].source_record(), "metadata": by_id[key].metadata()} for key in sorted(touched)],
        runtime_status="USED_GRAPH" if used else "GRAPH_ZERO_EXPANSION", zero_expansion_reason=reason)
    identity = stable_id("reviewed-graph-v3", tenant=tenant_id, owner=owner_id, run=run_id, question=question,
        search=search, method=method, policy=POLICY, release=graph.release_sha256,
        lexical=lexical.manifest.index_version, vector=sha256_bytes(query_vector.tobytes(order="C")),
        recipe=recipe.fingerprint, answer_language=answer_language, time_budget_ms=time_budget_ms)
    binding = AnswerBinding(request_id=identity, tenant_id=tenant_id, owner_id=owner_id, mode_requested=method,
        mode_used=method, used_graph=used, warnings=("GRAPH_AI_REVIEWED_EXPERIMENTAL", "GRAPH_INCOMPLETE_RELATION_COVERAGE",
        *(("GRAPH_ZERO_EXPANSION_" + reason,) if reason else ()), *packed.warnings))
    prepared = PreparedAnswer(CallContext(tenant_id=tenant_id, run_id=run_id, request_id=identity, stage="GENERATE"),
        binding, packed, question, recipe, ranked)
    outcome = ReviewedGraphExecutionV3(requested_method=method, effective_method=method,
        mechanism_status=trace["runtime_status"], executed_stages=tuple(stages), graph_path_used=used,
        context_chunk_ids=packed.chunk_ids, context_assertion_ids=packed.assertion_ids,
        zero_expansion_reason=reason, release_sha256=graph.release_sha256)
    return StrictPreparedMethod(outcome, view, prepared, trace, answer_language)
