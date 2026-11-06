"""Strict, source-bound Graph/Q/QB preparation without hidden method substitution.

Graph v2 links query entities independently of the first retrieved chunks,
filters paths with an explicit lexical relevance heuristic and packs complete
path bundles first. It is experimental graph-assisted retrieval, not a proof
of relation relevance or Microsoft GraphRAG community/global search.
"""

import re
import time
from collections import defaultdict, deque
from dataclasses import replace
from types import MappingProxyType

from app.answer_pipeline import PreparedAnswer
from app.collection_answering import (
    _evidence_view,
    prepare_collection_answer,
    validate_collection_lexical_binding,
)
from app.collection_graph_runtime import ValidatedCollectionGraph
from app.collection_graph_v2 import ValidatedCollectionGraphV2
from app.context import pack_context
from app.context_selection import ContextSelectionRecipe
from app.contracts import QueryRequest, sha256_bytes, stable_id
from app.generation import AnswerBinding, GenerationError
from app.graph_retrieval import GraphCandidate, GraphRetrievalResult
from app.method_execution import MethodExecutionV2, StrictPreparedMethod
from app.model_gateway import CallContext
from app.query_planning_v2 import ValidatedQueryPlanV2, approved_planner_recipe
from app.retrieval import reciprocal_rank_fusion
from app.snapshot import SnapshotContent

GRAPH_POLICY_V2 = "QUERY_LINKED_RELEVANCE_GATED_ATOMIC_PATHS_STRICT_V2"
GENERIC_ALIASES = frozenset({"aws", "amazon", "object", "bucket", "function", "role", "key", "rules", "listener",
    "listeners", "service", "resource", "policy", "account", "permissions"})
STOPWORDS = frozenset("a an the what which how why when where does do is are can could should would to for of in on and or "
    "with without from by as at it its this that these those before after i my we our they their me please explain "
    "show tell need needs aws amazon service services resource resources".split())


def _tokens(text):
    return {word[:-1] if len(word) > 3 and word.endswith("s") else word
        for word in re.findall(r"[a-z][a-z0-9_:-]*", text.casefold()) if word not in STOPWORDS}


def link_query_entities(entities, search_query, question):
    """Literal, boundary-aware links; ambiguous aliases are reported, never merged."""
    aliases = defaultdict(dict)
    for entity in entities:
        for alias in (*entity.aliases, entity.canonical_name):
            normalized = " ".join(alias.split()).casefold()
            if len(normalized) >= 3 and normalized not in GENERIC_ALIASES:
                aliases[normalized][entity.entity_id] = entity
    matches = []
    for source, text in (("SEARCH_QUERY", search_query), ("ORIGINAL_QUESTION", question)):
        own = []
        for alias, candidates in aliases.items():
            pattern = r"(?<!\w)" + r"\s+".join(re.escape(w) for w in alias.split()) + r"(?!\w)"
            for match in re.finditer(pattern, text, re.I):
                own.append((match.start(), match.end(), alias, candidates, match.group()))
        # A longer literal name suppresses a nested shorter alias from another identity.
        kept = []
        for start, end, alias, candidates, quote in sorted(own, key=lambda m: (-(m[1] - m[0]), m[0], m[2])):
            if any(left <= start and end <= right for left, right in kept):
                continue
            kept.append((start, end))
            ambiguous = len(candidates) != 1
            for entity_id, entity in sorted(candidates.items()):
                matches.append({"entity_id": entity_id, "canonical_name": entity.canonical_name,
                    "source": source, "start_char": start, "end_char": end, "quote": quote,
                    "alias": alias, "method": "EXACT_BOUNDARY_ALIAS", "ambiguous": ambiguous,
                    "selected": not ambiguous})
    return tuple(matches)


def _unsupported(method, status, stages, trace):
    return StrictPreparedMethod(MethodExecutionV2(requested_method=method, mechanism_status=status,
        executed_stages=tuple(stages), failure_reason=status), None, None, trace)


def prepare_strict_graph_answer(collection, lexical, graph, question, query_vector, *, tenant_id, owner_id, run_id,
    recipe, hops=1, search_query=None, allow_experimental=False, time_budget_ms=300, clock=None, traversal_clock=None,
    answer_language="en"):
    from app.reviewed_source_graph import ValidatedReviewedSourceGraph

    if type(graph) is ValidatedReviewedSourceGraph:
        from app.reviewed_graph_runtime import prepare_reviewed_graph_answer

        return prepare_reviewed_graph_answer(collection, lexical, graph, question, query_vector,
            tenant_id=tenant_id, owner_id=owner_id, run_id=run_id, recipe=recipe, hops=hops,
            search_query=search_query, allow_experimental=allow_experimental, time_budget_ms=time_budget_ms,
            clock=clock, traversal_clock=traversal_clock, answer_language=answer_language)
    if not allow_experimental or type(graph) not in (ValidatedCollectionGraph, ValidatedCollectionGraphV2):
        raise GenerationError("STRICT_GRAPH_EXPERIMENTAL_OPT_IN_REQUIRED")
    if type(hops) is not int or hops not in (1, 2):
        raise GenerationError("STRICT_GRAPH_HOP_BOUND")
    if type(time_budget_ms) is not int or not 1 <= time_budget_ms <= 300:
        raise GenerationError("STRICT_GRAPH_TIME_BOUND")
    if answer_language not in {"en", "vi"}:
        raise GenerationError("STRICT_ANSWER_LANGUAGE")
    method = f"graph_{hops}hop"
    question = QueryRequest(question=question, mode=method).question
    search = QueryRequest(question=question if search_query is None else search_query, mode=method).question
    overlay = graph.overlay
    if (tenant_id != collection.manifest.tenant_id or tenant_id != overlay.tenant_id
            or overlay.collection_manifest_sha256 != collection.manifest_sha256
            or overlay.collection_version != collection.manifest.collection_version):
        raise GenerationError("STRICT_GRAPH_RELEASE_BINDING")
    validate_collection_lexical_binding(collection, lexical)
    stages = ["VALIDATE_INPUT", "RETRIEVE_BASE", "LINK_ENTITIES"]
    vector_ids = tuple(h.chunk_id for h in collection.search_vector(query_vector, k=20))
    lexical_ids = tuple(key for key, _ in lexical.search(search, k=20))
    baseline = tuple(reciprocal_rank_fusion([vector_ids, lexical_ids]))
    scores = {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        scores.update((key, float(value)) for key, value in zip(shard.chunk_ids, shard.vectors @ query_vector, strict=True))
    links = link_query_entities(overlay.entities, search, question)
    direct = tuple(dict.fromkeys(m["entity_id"] for m in links if m["selected"]))
    mentions, adjacent = defaultdict(set), defaultdict(list)
    entities = {e.entity_id: e for e in overlay.entities}
    for mention in overlay.mentions:
        mentions[mention.chunk_id].add(mention.entity_id)
    for assertion in overlay.assertions:
        if assertion.polarity == "POSITIVE":
            adjacent[assertion.subject_id].append(assertion)
            adjacent[assertion.object_id].append(assertion)
    rank_seeds = defaultdict(list)
    for rank, key in enumerate(baseline[:20], 1):
        for entity_id in mentions[key]:
            if entities[entity_id].canonical_name.casefold() not in GENERIC_ALIASES and adjacent[entity_id]:
                rank_seeds[entity_id].append(rank)
    derived = sorted(rank_seeds, key=lambda e: (min(rank_seeds[e]), -len(rank_seeds[e]), e))
    seeds = tuple(dict.fromkeys((*direct, *derived)))[:8]
    trace = {"schema_version": 2, "policy": GRAPH_POLICY_V2, "question": question, "search_query": search,
        "requested_method": method, "collection_manifest_sha256": collection.manifest_sha256,
        "graph_file_sha256": graph.file_sha256, "graph_assertions": len(overlay.assertions),
        "links": links, "seed_entity_ids": seeds, "retrieval_seed_ranks": dict(rank_seeds),
        "vector_ranking": vector_ids, "lexical_ranking": lexical_ids, "baseline_ranking": baseline,
        "paths": [], "packed_paths": [], "truncated": False, "assertion_visits": 0,
        "relevance_policy": "LITERAL_QUERY_ENTITY_PLUS_NONENTITY_TERM_OR_EXPLICIT_CONNECTION_INTENT",
        "context_limits": {"max_chunks": 8, "proxy_tokens": 4096}}
    if not seeds:
        return _unsupported(method, "NO_LINKED_ENTITY", stages, trace)
    stages.append("TRAVERSE_GRAPH")
    timer = time.monotonic if clock is None else clock
    tick = timer if traversal_clock is None else traversal_clock
    started = timer()
    queue = deque((seed, (seed,), ()) for seed in seeds)
    visited, seen_paths, paths = set(seeds), set(), []
    by_assertion = {a.assertion_id: a for a in overlay.assertions}
    linked_tokens = set().union(*(_tokens(entities[e].canonical_name) for e in direct)) if direct else set()
    facets = _tokens(search) - linked_tokens
    connection_intent = bool(facets & {"connection", "relationship", "related", "connect", "relation"})
    while queue:
        if (tick() - started) * 1000 >= time_budget_ms:
            trace["visited_entities"] = len(visited)
            return _unsupported(method, "BUDGET_EXCEEDED", stages, trace)
        entity, entity_path, assertion_path = queue.popleft()
        if len(assertion_path) == hops:
            continue
        neighbors = sorted(adjacent[entity], key=lambda a: (-scores[a.chunk_id], a.assertion_id))
        if len(neighbors) > 20:
            trace["truncated"] = True
        for assertion in neighbors[:20]:
            other = assertion.object_id if assertion.subject_id == entity else assertion.subject_id
            if other in entity_path:
                continue
            if trace["assertion_visits"] >= 200 or other not in visited and len(visited) >= 80:
                trace["truncated"] = True
                continue
            trace["assertion_visits"] += 1
            new_ids, new_entities = (*assertion_path, assertion.assertion_id), (*entity_path, other)
            if new_ids in seen_paths:
                continue
            seen_paths.add(new_ids)
            visited.add(other)
            supports = tuple(dict.fromkeys(by_assertion[a].chunk_id for a in new_ids))
            content = " ".join(by_assertion[a].evidence.quote + " " + by_assertion[a].predicate for a in new_ids)
            terms = sorted(facets & _tokens(content))
            direct_count = len(set(new_entities) & set(direct))
            relevant = (direct_count > 0 and (bool(terms) or connection_intent)) or (not direct and len(terms) >= 2)
            path = {"entity_ids": new_entities, "assertion_ids": new_ids, "supporting_chunk_ids": supports,
                "hop": len(new_ids), "query_entities_covered": direct_count, "facet_terms": terms,
                "accepted": relevant, "reason": "RELEVANCE_HEURISTIC_ACCEPTED" if relevant else "NO_QUERY_FACET_SUPPORT",
                "source_vector_score": sum(scores[s] for s in supports) / len(supports)}
            paths.append(path)
            queue.append((other, new_entities, new_ids))
    trace["visited_entities"], trace["paths"] = len(visited), paths
    eligible = sorted((p for p in paths if p["accepted"]), key=lambda p: (
        -p["query_entities_covered"], -len(p["facet_terms"]), -p["source_vector_score"], p["hop"], p["assertion_ids"]))
    if not eligible:
        return _unsupported(method, "NO_SUPPORTED_RELATION", stages, trace)
    stages.extend(("SELECT_CONTEXT", "PACK_CONTEXT"))
    # One provenance-carrying path per representative chunk, selected before the
    # common packer maps chunk IDs to paths. Avoid erasing a two-hop support chain.
    candidates, by_chunk = [], set()
    for path in eligible:
        representative = by_assertion[path["assertion_ids"][-1]].chunk_id
        if representative not in by_chunk:
            candidates.append(GraphCandidate(representative, path["hop"], tuple(path["entity_ids"]),
                tuple(path["assertion_ids"]), tuple(path["supporting_chunk_ids"])))
            by_chunk.add(representative)
    graph_ranking = tuple(dict.fromkeys(c.chunk_id for c in candidates))
    fused = tuple(reciprocal_rank_fusion([vector_ids, lexical_ids, graph_ranking]))
    prioritized = tuple(dict.fromkeys(k for c in candidates[:2] for k in (c.chunk_id, *c.supporting_chunk_ids)))
    ranked = tuple(dict.fromkeys((*prioritized, *fused)))[:40]
    candidates = tuple(c for c in candidates if c.chunk_id in ranked and set(c.supporting_chunk_ids) <= set(ranked))
    trace.update(graph_ranking=graph_ranking, fused_ranking=fused[:40], selection_order=ranked,
        prioritized_bundle_chunk_ids=prioritized)
    view = _evidence_view(collection, lexical, ranked)
    view = replace(view, entities=MappingProxyType(entities),
        content=SnapshotContent((), (), overlay.entities, overlay.mentions, overlay.assertions))
    retrieval = GraphRetrievalResult(ranked, baseline, candidates, seeds, method, True, (), trace["assertion_visits"], len(visited))
    packed = pack_context(view, retrieval)
    trace.update(context_chunk_ids=packed.chunk_ids, context_assertion_ids=packed.assertion_ids,
        context_proxy_tokens=packed.proxy_tokens, packing_warnings=packed.warnings)
    trace["packed_paths"] = [p for p in eligible if set(p["assertion_ids"]) <= set(packed.assertion_ids)
        and set(p["supporting_chunk_ids"]) <= set(packed.chunk_ids)]
    if not packed.assertion_ids or not trace["packed_paths"]:
        return _unsupported(method, "PATH_INSUFFICIENT", stages, trace)
    identity = stable_id("strictgraph", tenant=tenant_id, owner=owner_id, run=run_id, question=question, search=search,
        method=method, policy=GRAPH_POLICY_V2, graph=graph.file_sha256, lexical=lexical.manifest.index_version,
        vector=sha256_bytes(query_vector.tobytes(order="C")), recipe=recipe.fingerprint, time_budget_ms=time_budget_ms,
        answer_language=answer_language)
    binding = AnswerBinding(request_id=identity, tenant_id=tenant_id, owner_id=owner_id,
        mode_requested=method, mode_used=method, used_graph=True,
        warnings=("GRAPH_AI_REVIEWED_EXPERIMENTAL", "GRAPH_INCOMPLETE_RELATION_COVERAGE", *packed.warnings))
    prepared = PreparedAnswer(CallContext(tenant_id=tenant_id, run_id=run_id, request_id=identity, stage="GENERATE"),
        binding, packed, question, recipe, ranked)
    outcome = MethodExecutionV2(requested_method=method, effective_method=method, mechanism_status="USED_GRAPH",
        executed_stages=tuple(stages), graph_path_used=True, context_chunk_ids=packed.chunk_ids,
        context_assertion_ids=packed.assertion_ids, answer_status="READY")
    return StrictPreparedMethod(outcome, view, prepared, trace, answer_language)


def prepare_strict_multi_query_answer(collection, lexical, *, mode, question, search_query, query_vector,
    subquery_vectors, plan, tenant_id, owner_id, run_id, recipe, language="en", allow_legacy_diagnostic=False):
    if mode not in {"hybrid_multi_query", "hybrid_multi_query_balanced"} or type(plan) is not ValidatedQueryPlanV2:
        raise GenerationError("STRICT_QUERY_VALIDATED_PLAN_REQUIRED")
    plan = ValidatedQueryPlanV2.model_validate_json(plan.model_dump_json())
    if (plan.question_sha256 != sha256_bytes(question.encode()) or plan.search_query_sha256 != sha256_bytes(search_query.encode())
            or plan.language != language or not approved_planner_recipe(plan.recipe_sha256)):
        raise GenerationError("STRICT_QUERY_QUESTION_BINDING")
    if plan.origin == "NATIVE_V3":
        from app.query_preparation_v3 import validate_no_introduced_names_v3

        validate_no_introduced_names_v3(plan.subqueries, question + " " + search_query)
    if plan.origin == "FIXTURE" and collection.manifest.purpose != "FIXTURE":
        raise GenerationError("STRICT_QUERY_FIXTURE_PLAN_NAMESPACE")
    if plan.origin == "V1_REVALIDATED_DIAGNOSTIC" and allow_legacy_diagnostic is not True:
        raise GenerationError("STRICT_QUERY_LEGACY_DIAGNOSTIC_OPT_IN")
    queries, selection = [], []
    identity = stable_id("strictquery", tenant=tenant_id, owner=owner_id, run=run_id,
        question=question, search=search_query, method=mode, plan=plan.fingerprint, recipe=recipe.fingerprint,
        collection=collection.manifest_sha256, lexical=lexical.manifest.index_version,
        vector=sha256_bytes(query_vector.tobytes(order="C")),
        subquery_vectors=[sha256_bytes(v.tobytes(order="C")) for v in subquery_vectors],
        selector=ContextSelectionRecipe().fingerprint if mode.endswith("balanced") else None)
    view, prepared = prepare_collection_answer(collection, lexical, question, query_vector, tenant_id=tenant_id,
        owner_id=owner_id, run_id=run_id, recipe=recipe, mode=mode, request_id=identity, search_query=search_query,
        subqueries=plan.subqueries, subquery_vectors=subquery_vectors, multi_query_trace=queries,
        context_selection_recipe=ContextSelectionRecipe() if mode.endswith("balanced") else None,
        context_selection_trace=selection)
    status = "USED_MULTI_QUERY" if plan.subqueries else "NO_DECOMPOSITION_NEEDED"
    stages = ["VALIDATE_SHARED_PLAN", "RETRIEVE_QUERIES", "FUSE_RANKS"]
    if selection:
        stages.append("ALLOCATE_CONTEXT")
    stages.append("PACK_CONTEXT")
    outcome = MethodExecutionV2(requested_method=mode, effective_method=mode, mechanism_status=status,
        executed_stages=tuple(stages), planning_status="VALID", subquery_count=len(plan.subqueries),
        context_selection_used=bool(selection), context_chunk_ids=prepared.packed.chunk_ids, answer_status="READY")
    trace = {"schema_version": 2, "plan": plan.model_dump(mode="json"),
        "multi_query": queries[0].model_dump(mode="json"),
        "context_selection": selection[0].model_dump(mode="json") if selection else None,
        "packed_chunk_ids": prepared.packed.chunk_ids, "context_proxy_tokens": prepared.packed.proxy_tokens,
        "selection_order_changed": bool(selection and selection[0].ordered_chunk_ids != queries[0].ranked_chunk_ids),
        "packed_queries_represented": sorted({i for i, q in enumerate(queries[0].queries)
            if set(q.ranked_chunk_ids) & set(prepared.packed.chunk_ids)})}
    return StrictPreparedMethod(outcome, view, prepared, trace, language)
