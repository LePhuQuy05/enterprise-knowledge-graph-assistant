"""Local G1/G2 over typed evidence, including n-ary permission requirements.

A hop crosses one reviewed fact, not a principal-to-resource permission grant.
Context-service associations are labelled explicitly. No resource coreference,
condition compatibility, policy simulation, or transitive workflow is inferred.
Frozen source-runtime-v3 and its paid answer receipts are not changed.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

from app.answer_pipeline import PreparedAnswer
from app.authorization_evidence import ValidatedAuthorizationFacts
from app.collection_answering import _evidence_view, validate_collection_lexical_binding
from app.context import pack_context
from app.contracts import QueryRequest, canonical_json, sha256_bytes, stable_id
from app.external_http import strict_json
from app.generation import AnswerBinding, GenerationError
from app.graph_retrieval import GraphCandidate, GraphRetrievalResult
from app.method_execution import MethodExecutionV2, ReviewedGraphExecutionV3, StrictPreparedMethod
from app.model_gateway import CallContext
from app.retrieval import reciprocal_rank_fusion
from app.reviewed_graph_runtime import WEAK_FACETS
from app.reviewed_source_graph import INTERPRETATION, ValidatedReviewedSourceGraph
from app.strict_collection_methods import _tokens, link_query_entities

POLICY = "AUTHORIZATION_TYPED_EVIDENCE_PATHS_V1"


@dataclass(frozen=True)
class EvidenceFact:
    assertion_id: str
    chunk_id: str
    kind: str
    clause: str
    links: tuple[tuple[str, str, str], ...]
    context_json: str


@dataclass(frozen=True)
class CatalogNode:
    entity_id: str
    canonical_name: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class AuthorizationEvidenceGraph:
    collection_manifest_sha256: str
    release_sha256: str
    facts: tuple[EvidenceFact, ...]
    catalog_nodes: tuple[CatalogNode, ...]


def combine_evidence_graph(collection, reviewed, authorization):
    if (type(reviewed) is not ValidatedReviewedSourceGraph or type(authorization) is not ValidatedAuthorizationFacts
            or authorization.collection_manifest_sha256 != collection.manifest_sha256
            or reviewed.overlay.collection_manifest_sha256 != collection.manifest_sha256):
        raise GenerationError("AUTH_RUNTIME_VALIDATED_RELEASES_REQUIRED")
    nodes, facts = {}, []
    entities = {e.entity_id: e for e in reviewed.overlay.entities}
    for edge in reviewed.edges:
        if edge.polarity != "POSITIVE":
            continue
        context = edge.context_record()
        mapped = {}
        for node in context["nodes"]:
            key = node["node_id"]
            if node["identity_scope"] == "CATALOG_TYPE":
                entity = entities[key]
                key = stable_id("evidence-catalog", tenant=collection.manifest.tenant_id, key=entity.canonical_key)
                nodes[key] = CatalogNode(key, entity.canonical_name, entity.aliases)
            mapped[node["node_id"]] = key
        left, right = mapped[edge.subject_id], mapped[edge.object_id]
        links = ((left, right, "DIRECTED_DOCUMENTED_ACTION" if edge.kind == "ACTION" else "ORIGINAL_RELATION_EVIDENCE"),)
        if edge.kind != "ACTION":
            links += ((right, left, "REVERSE_SOURCE_LOOKUP_NOT_INVERSE_FACT"),)
        context["node_id_projection"] = mapped
        facts.append(EvidenceFact(edge.assertion_id, edge.chunk_id, edge.kind,
            edge.metadata()["record"]["clause_quote"], links, canonical_json(context).decode()))
    for row in authorization.records():
        candidate = row["candidate"]
        participants = row["nodes"]
        for node in participants:
            if node["identity_scope"] == "CATALOG_TYPE":
                nodes[node["node_id"]] = CatalogNode(node["node_id"], node["name"], tuple(node["aliases"]))
        if candidate["kind"] == "CONDITIONAL_ACTION":
            links = tuple((a["node_id"], b["node_id"], "DIRECTED_DOCUMENTED_CONDITIONAL_ACTION")
                for a in participants if a["role"] == "ACTOR" and a["node_id"]
                for b in participants if b["role"] == "RESOURCE" and b["node_id"])
        else:
            links = tuple((a["node_id"], b["node_id"], "CONTEXT_SERVICE_EVIDENCE_ASSOCIATION" if
                "CONTEXT_SERVICE" in {a["role"], b["role"]} else "REQUIREMENT_EVIDENCE_ASSOCIATION_NOT_GRANT")
                for a in participants if a["node_id"] for b in participants
                if b["node_id"] and a["node_id"] != b["node_id"])
        context = {"assertion_id": row["assertion_id"], "chunk_id": candidate["chunk_id"], "kind": candidate["kind"],
            "record": candidate, "nodes": participants, "origin": row["origin"], "human_reviewed": False,
            "untrusted_evidence_quote": row["evidence"]["quote"], "interpretation": INTERPRETATION,
            "permission_granted": False, "configuration_verified": False,
            "requirement_semantics": candidate["conclusion"], "source_review": row["review"]}
        facts.append(EvidenceFact(row["assertion_id"], candidate["chunk_id"], candidate["kind"], candidate["clause"]["quote"],
            tuple(dict.fromkeys(links)), canonical_json(context).decode()))
    if len(facts) > 20000 or len({f.assertion_id for f in facts}) != len(facts):
        raise GenerationError("AUTH_RUNTIME_FACT_BOUND_OR_DUPLICATE")
    release = stable_id("authorization-combined-release-v1", source=collection.manifest_sha256,
        reviewed=reviewed.release_sha256, authorization=authorization.release_sha256, policy=POLICY,
        implementation=sha256_bytes(Path(__file__).read_bytes())).split("_")[-1]
    return AuthorizationEvidenceGraph(collection.manifest_sha256, release, tuple(facts), tuple(nodes[k] for k in sorted(nodes)))


def prepare_authorization_answer(collection, lexical, graph, question, query_vector, *, tenant_id, owner_id,
    run_id, recipe, hops=1, search_query=None, allow_experimental=False, time_budget_ms=300,
    clock=None, traversal_clock=None, answer_language="en"):
    if type(graph) is not AuthorizationEvidenceGraph or allow_experimental is not True:
        raise GenerationError("AUTH_RUNTIME_LOCAL_OPT_IN_REQUIRED")
    if (graph.collection_manifest_sha256 != collection.manifest_sha256 or tenant_id != collection.manifest.tenant_id
            or type(hops) is not int or hops not in (1, 2) or answer_language not in {"en", "vi"}
            or type(time_budget_ms) is not int or not 1 <= time_budget_ms <= 300):
        raise GenerationError("AUTH_RUNTIME_BINDING_OR_LIMITS")
    method = f"graph_{hops}hop"
    question = QueryRequest(question=question, mode=method).question
    search = QueryRequest(question=question if search_query is None else search_query, mode=method).question
    validate_collection_lexical_binding(collection, lexical)
    vector = tuple(h.chunk_id for h in collection.search_vector(query_vector, k=20))
    bm25 = tuple(k for k, _ in lexical.search(search, k=20))
    baseline = tuple(reciprocal_rank_fusion([vector, bm25]))[:40]
    links = link_query_entities(graph.catalog_nodes, search, question)
    seeds = tuple(dict.fromkeys(link["entity_id"] for link in links if link["selected"]))[:8]
    adjacent = defaultdict(list)
    by_id = {fact.assertion_id: fact for fact in graph.facts}
    for fact in graph.facts:
        for left, right, label in fact.links:
            adjacent[left].append((fact, right, label))
    # Direct catalogue seeds only. No automatic local-resource or pronoun seed.
    linked_terms = set().union(*(_tokens(n.canonical_name) | set().union(*(_tokens(a) for a in n.aliases))
        for n in graph.catalog_nodes if n.entity_id in seeds)) if seeds else set()
    facets = _tokens(search) - linked_terms - WEAK_FACETS
    trace = {"schema_version": 1, "policy": POLICY, "requested_method": method, "question": question,
        "search_query": search, "collection_manifest_sha256": collection.manifest_sha256,
        "release_sha256": graph.release_sha256, "catalog_links": links, "seed_entity_ids": seeds,
        "vector_ranking": vector, "lexical_ranking": bm25, "baseline_ranking": baseline,
        "facts": len(graph.facts), "paths": [], "packed_paths": [], "assertion_visits": 0, "truncated": False,
        "hop_definition": "ONE_REVIEWED_FACT_WITH_LABELLED_PROJECTION_NOT_PERMISSION_GRANT",
        "condition_compatibility": "NOT_INFERRED", "interpretation": INTERPRETATION,
        "hidden_fallbacks": 0, "human_reviewed": 0, "answer_generation_executed": False}
    stages = ("VALIDATE_AUTHORIZATION_RELEASE", "RETRIEVE_BASE", "LINK_ENTITIES", "TRAVERSE_GRAPH")
    tick = time.monotonic if clock is None else clock
    started = tick()
    timer = tick if traversal_clock is None else traversal_clock
    queue = deque((seed, (seed,), (), ()) for seed in seeds)
    seen, visited, paths = set(), set(seeds), []
    while queue:
        if (timer() - started) * 1000 >= time_budget_ms:
            trace.update(runtime_status="BUDGET_EXCEEDED", paths=paths)
            outcome = MethodExecutionV2(requested_method=method, mechanism_status="BUDGET_EXCEEDED",
                executed_stages=stages, failure_reason="BUDGET_EXCEEDED")
            return StrictPreparedMethod(outcome, None, None, trace, answer_language)
        node, node_path, fact_path, labels = queue.popleft()
        if len(fact_path) >= hops:
            continue
        neighbors = sorted(adjacent[node], key=lambda row: (row[0].assertion_id, row[1], row[2]))
        trace["truncated"] |= len(neighbors) > 20
        for fact, other, label in neighbors[:20]:
            if fact.assertion_id in fact_path or other in node_path:
                continue
            if trace["assertion_visits"] >= 200 or other not in visited and len(visited) >= 80:
                trace["truncated"] = True
                continue
            trace["assertion_visits"] += 1
            ids = (*fact_path, fact.assertion_id)
            identity = (ids, (*node_path, other))
            if identity in seen:
                continue
            seen.add(identity)
            visited.add(other)
            terms = sorted(facets & _tokens(fact.clause))
            # Every fact must match a query facet. A relevant prefix may not
            # lend its relevance to an unrelated continuation.
            if not terms:
                continue
            support = tuple(dict.fromkeys(by_id[k].chunk_id for k in ids))
            path = {"assertion_ids": ids, "entity_ids": (*node_path, other), "hop": len(ids),
                "supporting_chunk_ids": support, "projection_labels": (*labels, label), "facet_terms": terms,
                "query_entities_covered": len(set((*node_path, other)) & set(seeds)), "accepted": True,
                "interpretation": INTERPRETATION, "condition_compatibility": "NOT_INFERRED"}
            paths.append(path)
            queue.append((other, path["entity_ids"], ids, path["projection_labels"]))
    eligible = sorted(paths, key=lambda p: (-p["query_entities_covered"], -len(p["facet_terms"]), -p["hop"], p["assertion_ids"], p["entity_ids"]))
    candidates, reps = [], set()
    for p in eligible:
        key = by_id[p["assertion_ids"][-1]].chunk_id
        if key not in reps:
            reps.add(key)
            candidates.append(GraphCandidate(key, p["hop"], p["entity_ids"], p["assertion_ids"], p["supporting_chunk_ids"]))
    graph_ranking = tuple(c.chunk_id for c in candidates)
    fused = tuple(reciprocal_rank_fusion([vector, bm25, graph_ranking]))
    prioritized = tuple(dict.fromkeys(k for c in candidates[:2] for k in (c.chunk_id, *c.supporting_chunk_ids)))
    ranked = tuple(dict.fromkeys((*prioritized, *fused)))[:40]
    view = _evidence_view(collection, lexical, ranked)
    result = GraphRetrievalResult(ranked, baseline, tuple(candidates), seeds, method, bool(candidates), (), trace["assertion_visits"], len(visited))
    packed = pack_context(view, result, graph_records={f.assertion_id: strict_json(f.context_json.encode()) for f in graph.facts})
    attempted_warnings = packed.warnings
    packed_paths = [p for p in eligible if set(p["assertion_ids"]) <= set(packed.assertion_ids)
        and set(p["supporting_chunk_ids"]) <= set(packed.chunk_ids)]
    used = bool(packed_paths)
    reason = None if used else "NO_LINKED_ENTITY" if not seeds else "NO_RELEVANT_PATH" if not paths else "NO_PATH_FITS_CONTEXT"
    if not used:
        ranked = baseline
        view = _evidence_view(collection, lexical, ranked)
        result = GraphRetrievalResult(ranked, baseline, (), seeds, method, False, (), trace["assertion_visits"], len(visited))
        packed = pack_context(view, result)
    base_view = _evidence_view(collection, lexical, baseline)
    base = pack_context(base_view, GraphRetrievalResult(baseline, baseline, (), (), method, False, (), 0, 0))
    trace.update(paths=paths, packed_paths=packed_paths, runtime_status="USED_GRAPH" if used else "GRAPH_ZERO_EXPANSION",
        zero_expansion_reason=reason, graph_ranking=graph_ranking, selection_order=ranked,
        context_chunk_ids=packed.chunk_ids, context_assertion_ids=packed.assertion_ids, context_proxy_tokens=packed.proxy_tokens,
        baseline_context_chunk_ids=base.chunk_ids, graph_added_context_chunk_ids=tuple(k for k in packed.chunk_ids if k not in base.chunk_ids),
        attempted_path_packing_warnings=attempted_warnings, packing_warnings=packed.warnings,
        source_records=[strict_json(by_id[k].context_json.encode()) for k in packed.assertion_ids])
    identity = stable_id("authorization-answer-v1", tenant=tenant_id, owner=owner_id, run=run_id, question=question,
        search=search, method=method, policy=POLICY, release=graph.release_sha256, recipe=recipe.fingerprint,
        vector=sha256_bytes(query_vector.tobytes(order="C")), lexical=lexical.manifest.index_version,
        language=answer_language, time_budget_ms=time_budget_ms)
    binding = AnswerBinding(request_id=identity, tenant_id=tenant_id, owner_id=owner_id, mode_requested=method,
        mode_used=method, used_graph=used, warnings=("GRAPH_AI_REVIEWED_EXPERIMENTAL", "AUTHORIZATION_REQUIREMENTS_NOT_GRANTS",
            "GRAPH_INCOMPLETE_RELATION_COVERAGE", *(("GRAPH_ZERO_EXPANSION_" + reason,) if reason else ()), *packed.warnings))
    prepared = PreparedAnswer(CallContext(tenant_id=tenant_id, run_id=run_id, request_id=identity, stage="GENERATE"),
        binding, packed, question, recipe, ranked)
    outcome = ReviewedGraphExecutionV3(requested_method=method, effective_method=method, mechanism_status=trace["runtime_status"],
        executed_stages=(*stages, "FUSE_RANKS", "SELECT_CONTEXT", "PACK_CONTEXT"), graph_path_used=used,
        context_chunk_ids=packed.chunk_ids, context_assertion_ids=packed.assertion_ids,
        zero_expansion_reason=reason, release_sha256=graph.release_sha256)
    return StrictPreparedMethod(outcome, view, prepared, trace, answer_language)
