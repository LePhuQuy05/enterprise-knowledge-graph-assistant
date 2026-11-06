"""Experimental graph priority plus a document-diverse ranked tail.

Two stable passes: first unseen document versions, then deferred same-document
chunks. Nothing is removed from the pool. Full path support still goes through
the atomic source-message packer. Document diversity is a source heuristic, not
an assertion that different documents provide different or correct facts.
"""

from dataclasses import replace
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.graph_source_allocation import reallocate_graph_sources

POLICY = "SAME_POOL_GRAPH_PRIORITY_DOCUMENT_PASSES_V2"


def document_diverse_tail(pool, priority, document_ids):
    pool, priority = tuple(pool), tuple(priority)
    if (len(set(pool)) != len(pool) or len(pool) > 40 or len(set(priority)) != len(priority)
            or not set(priority) <= set(pool) or set(document_ids) != set(pool)
            or any(not isinstance(v, str) or not v for v in document_ids.values())):
        raise GenerationError("DOCUMENT_ALLOCATION_POOL_BINDING")
    represented = {document_ids[k] for k in priority}
    diverse, deferred, decisions = [], [], []
    for key in pool:
        if key in priority:
            continue
        document = document_ids[key]
        first = document not in represented
        decisions.append({"chunk_id": key, "document_version_id": document,
            "pass": "FIRST_UNSEEN_DOCUMENT" if first else "DEFERRED_SAME_DOCUMENT"})
        (diverse if first else deferred).append(key)
        represented.add(document)
    order = (*priority, *diverse, *deferred)
    if set(order) != set(pool) or len(order) != len(pool):
        raise GenerationError("DOCUMENT_ALLOCATION_POOL_CHANGED")
    return order, decisions


def reallocate_graph_documents(collection, lexical, graph, predecessor, *, priority_source_cap=4):
    priority_result = reallocate_graph_sources(collection, lexical, graph, predecessor,
        priority_source_cap=priority_source_cap)
    pool = tuple(priority_result.prepared.retrieved_chunk_ids)
    priority = tuple(priority_result.trace["source_allocation"]["priority_chunk_ids"])
    document_ids = {k: priority_result.view.chunks[k].document_version_id for k in pool}
    # Completed no-path execution keeps the same baseline context. This policy
    # neither pretends graph was used nor silently becomes a different method.
    no_path = not predecessor.trace["paths"]
    order, decisions = (pool, []) if no_path else document_diverse_tail(pool, priority, document_ids)
    staged = replace(priority_result,
        prepared=replace(priority_result.prepared, retrieved_chunk_ids=order),
        trace=priority_result.trace | {"selection_order": order})
    final = reallocate_graph_sources(collection, lexical, graph, staged, priority_source_cap=0)
    config = {"policy": POLICY, "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "source_priority_recipe_sha256": priority_result.trace["method_config_sha256"],
        "final_atomic_packing_recipe_sha256": final.trace["method_config_sha256"],
        "priority_source_cap": priority_source_cap, "max_chunks": 8, "user_message_proxy_token_budget": 4096,
        "document_id_basis": "FROZEN_DOCUMENT_VERSION_ID", "same_pool": True,
        "same_document_chunks_deferred_not_discarded": True}
    config_sha = sha256_bytes(canonical_json(config))
    request_id = stable_id("graph-document-allocation-v2", predecessor=predecessor.prepared.binding.request_id,
        config=config_sha, user_message=final.trace["source_message_packing"]["user_message_sha256"])
    prepared = replace(final.prepared, binding=final.prepared.binding.model_copy(update={"request_id": request_id}),
        context=final.prepared.context.model_copy(update={"request_id": request_id}))
    trace = final.trace | {"method_config_sha256": config_sha, "allocation_predecessor_context_chunk_ids": predecessor.prepared.packed.chunk_ids,
        "source_allocation": priority_result.trace["source_allocation"] | {
            "priority_chunks_actually_packed": [k for k in priority if k in prepared.packed.chunk_ids]},
        "document_allocation": {"policy": POLICY, "recipe": config, "recipe_sha256": config_sha,
            "skipped_no_eligible_path": no_path, "decisions": decisions,
            "predecessor_order": tuple(predecessor.prepared.retrieved_chunk_ids),
            "priority_order": pool, "final_order": order,
            "priority_chunks_actually_packed": [k for k in priority if k in prepared.packed.chunk_ids]}}
    outcome = final.outcome.model_copy(update={"executed_stages": (*predecessor.outcome.executed_stages,
        "ALLOCATE_SOURCE_PRIORITY_AND_DOCUMENT_PASSES")})
    return replace(final, prepared=prepared, outcome=outcome, trace=trace)
