"""Local, versioned source allocation within an already retrieved graph pool.

This is an internal adapter for trusted StrictPreparedMethod results, not an API
for accepting client-supplied paths. It does not search, rewrite, traverse, infer
permissions, or inspect benchmark labels. A priority quota counts unique source
chunks in complete graph bundles, rather than counting candidate paths. The
ordinary ranked tail remains available; the quota is not a final graph-source
count limit or a promise that token-heavy priority bundles will fit.
"""

import json
from dataclasses import replace
from pathlib import Path

from app.authorization_retrieval import AuthorizationEvidenceGraph
from app.collection_answering import _evidence_view
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.graph_retrieval import GraphCandidate, GraphRetrievalResult
from app.method_execution import ReviewedGraphExecutionV3, StrictPreparedMethod
from app.source_message_packing import pack_source_message
from app.source_message_runtime import packing_trace

POLICY = "SAME_POOL_ATOMIC_GRAPH_SOURCE_PRIORITY_V1"


def allocate_source_priority(pool, candidates, *, priority_source_cap=4):
    """Return a stable permutation of pool and auditable bundle decisions.

    No query text, service names, target IDs, answers or labels are inputs. At
    most half of the shared eight chunk slots can be explicitly prioritized.
    Other candidates retain predecessor order, with no candidate pool expansion.
    """
    if (type(priority_source_cap) is not int or not 0 <= priority_source_cap <= 4
            or len(pool) > 40 or len(set(pool)) != len(pool) or len(candidates) > 40
            or len({c.chunk_id for c in candidates}) != len(candidates)):
        raise GenerationError("GRAPH_ALLOCATION_BOUNDS_OR_DUPLICATE")
    pool = tuple(pool)
    available, priority, decisions = set(pool), [], []
    for candidate in candidates:
        support = candidate.supporting_chunk_ids
        if (candidate.hop not in {1, 2} or len(candidate.assertion_path) != candidate.hop
                or len(set(candidate.assertion_path)) != candidate.hop or not support
                or len(set(support)) != len(support) or candidate.chunk_id not in support):
            raise GenerationError("GRAPH_ALLOCATION_INVALID_ATOMIC_BUNDLE")
        bundle = tuple(dict.fromkeys((candidate.chunk_id, *support)))
        proposed = tuple(dict.fromkeys((*priority, *bundle)))
        in_pool = set(bundle) <= available
        accepted = in_pool and len(proposed) <= priority_source_cap
        decisions.append({"candidate_chunk_id": candidate.chunk_id,
            "assertion_ids": candidate.assertion_path, "bundle_chunk_ids": bundle,
            "incremental_source_chunks": len(proposed) - len(priority),
            "selected_for_priority": accepted,
            "reason": "OUTSIDE_FROZEN_POOL" if not in_pool else "PRIORITY_SOURCE_QUOTA" if not accepted else "PRIORITIZED"})
        if accepted:
            priority = list(proposed)
    order = tuple(dict.fromkeys((*priority, *pool)))
    if set(order) != available or len(order) != len(pool):
        raise GenerationError("GRAPH_ALLOCATION_POOL_CHANGED")
    return order, {"policy": POLICY, "priority_source_cap": priority_source_cap,
        "priority_chunk_ids": priority, "decisions": decisions,
        "pool_sha256": sha256_bytes(canonical_json(sorted(pool))),
        "predecessor_order_sha256": sha256_bytes(canonical_json(pool)),
        "allocated_order_sha256": sha256_bytes(canonical_json(order)),
        "same_pool": True, "max_chunks": 8, "user_message_proxy_token_budget": 4096,
        "quota_is_priority_only_not_final_graph_count": True}


def reallocate_graph_sources(collection, lexical, graph, predecessor, *, priority_source_cap=4):
    """Change only graph source order; preserve traversal, ranks and semantics."""
    if (type(graph) is not AuthorizationEvidenceGraph or type(predecessor) is not StrictPreparedMethod
            or predecessor.prepared is None or predecessor.outcome.requested_method not in {"graph_1hop", "graph_2hop"}):
        raise GenerationError("GRAPH_ALLOCATION_READY_TYPED_GRAPH_REQUIRED")
    old, prepared = predecessor.trace, predecessor.prepared
    pool = tuple(old["selection_order"])
    if (old["release_sha256"] != graph.release_sha256
            or old["collection_manifest_sha256"] != collection.manifest_sha256
            or pool != tuple(prepared.retrieved_chunk_ids)
            or tuple(old["context_chunk_ids"]) != prepared.packed.chunk_ids
            or old["question"] != prepared.question):
        raise GenerationError("GRAPH_ALLOCATION_PREDECESSOR_BINDING")
    by_id = {fact.assertion_id: fact for fact in graph.facts}
    eligible = sorted(old["paths"], key=lambda p: (-p["query_entities_covered"], -len(p["facet_terms"]),
        -p["hop"], p["assertion_ids"], p["entity_ids"]))
    candidates, seen = [], set()
    for path in eligible:
        try:
            support = tuple(dict.fromkeys(by_id[key].chunk_id for key in path["assertion_ids"]))
            key = by_id[path["assertion_ids"][-1]].chunk_id
        except (KeyError, IndexError) as exc:
            raise GenerationError("GRAPH_ALLOCATION_UNKNOWN_PATH_FACT") from exc
        if support != tuple(path["supporting_chunk_ids"]):
            raise GenerationError("GRAPH_ALLOCATION_PATH_SOURCE_BINDING")
        if key not in seen:
            seen.add(key)
            candidates.append(GraphCandidate(key, path["hop"], tuple(path["entity_ids"]),
                tuple(path["assertion_ids"]), support))
    if tuple(c.chunk_id for c in candidates) != tuple(old["graph_ranking"]):
        raise GenerationError("GRAPH_ALLOCATION_GRAPH_RANK_CHANGED")
    order, allocation = allocate_source_priority(pool, candidates, priority_source_cap=priority_source_cap)
    paths = tuple(c for c in candidates if set(c.supporting_chunk_ids) <= set(pool))
    records = {fact.assertion_id: json.loads(fact.context_json) for fact in graph.facts}
    base, seeds = tuple(old["baseline_ranking"]), tuple(old["seed_entity_ids"])
    method = predecessor.outcome.requested_method
    view = _evidence_view(collection, lexical, order)
    retrieval = GraphRetrievalResult(order, base, paths, seeds, method, bool(paths), (), old["assertion_visits"], 0)
    packing = pack_source_message(view, retrieval, question=prepared.question, recipe=prepared.recipe,
        tenant_id=prepared.binding.tenant_id, answer_language=predecessor.answer_language, graph_records=records)
    packed_paths = [p for p in eligible if set(p["assertion_ids"]) <= set(packing.packed.assertion_ids)
        and set(p["supporting_chunk_ids"]) <= set(packing.packed.chunk_ids)]
    used = bool(packed_paths)
    reason = None if used else "NO_LINKED_ENTITY" if not seeds else "NO_RELEVANT_PATH" if not eligible else "NO_PATH_FITS_CONTEXT"
    # Do not switch back to another pool when a priority path cannot fit. The
    # same source pool is an invariant of this allocation-only experiment.
    warnings = tuple(w for w in prepared.binding.warnings if not w.startswith("GRAPH_ZERO_EXPANSION_")
        and w != "SOURCE_MESSAGE_BUNDLE_SKIPPED")
    if reason:
        warnings += ("GRAPH_ZERO_EXPANSION_" + reason,)
    config = {"policy": POLICY, "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "predecessor_method_config_sha256": old["method_config_sha256"],
        "packing_recipe_sha256": packing.recipe_sha256, "priority_source_cap": priority_source_cap,
        "max_chunks": 8, "user_message_proxy_token_budget": 4096}
    config_sha = sha256_bytes(canonical_json(config))
    request_id = stable_id("graph-source-allocation-v1", predecessor=prepared.binding.request_id,
        config=config_sha, user_message=packing.user_message_sha256, audit=packing.audit_sha256)
    updated = replace(prepared, packed=packing.packed, retrieved_chunk_ids=order,
        binding=prepared.binding.model_copy(update={"request_id": request_id, "used_graph": used,
            "warnings": tuple(dict.fromkeys((*warnings, *packing.packed.warnings)))}),
        context=prepared.context.model_copy(update={"request_id": request_id}))
    allocation.update(recipe=config, recipe_sha256=config_sha,
        priority_chunks_actually_packed=[k for k in allocation["priority_chunk_ids"] if k in packing.packed.chunk_ids])
    trace = old | {"source_allocation": allocation, "selection_order": order,
        "allocation_predecessor_context_chunk_ids": prepared.packed.chunk_ids,
        "context_chunk_ids": packing.packed.chunk_ids, "context_assertion_ids": packing.packed.assertion_ids,
        "context_proxy_tokens": packing.packed.proxy_tokens, "packed_paths": packed_paths,
        "runtime_status": "USED_GRAPH" if used else "GRAPH_ZERO_EXPANSION", "zero_expansion_reason": reason,
        "source_message_packing": packing_trace(packing), "source_records": [records[k] for k in packing.packed.assertion_ids],
        "graph_added_context_chunk_ids": tuple(k for k in packing.packed.chunk_ids if k not in old["baseline_context_chunk_ids"]),
        "method_config_sha256": config_sha, "packing_warnings": packing.packed.warnings}
    outcome = ReviewedGraphExecutionV3(requested_method=method, effective_method=method,
        mechanism_status=trace["runtime_status"], executed_stages=(*predecessor.outcome.executed_stages, "ALLOCATE_SAME_POOL_GRAPH_SOURCES"),
        graph_path_used=used, context_chunk_ids=packing.packed.chunk_ids, context_assertion_ids=packing.packed.assertion_ids,
        zero_expansion_reason=reason, release_sha256=graph.release_sha256)
    return StrictPreparedMethod(outcome, view, updated, trace, predecessor.answer_language)
