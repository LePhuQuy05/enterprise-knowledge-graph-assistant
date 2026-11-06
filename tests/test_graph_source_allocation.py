"""Synthetic allocation invariants, not AWS answer-quality labels."""

from copy import deepcopy
from dataclasses import replace

import pytest

from app.fixtures import fixture_vector
from app.generation import GenerationRecipeV9, generation_request
from app.graph_retrieval import GraphCandidate
from app.graph_source_allocation import allocate_source_priority, reallocate_graph_sources
from app.ingestion.text import ProxyTokenizer
from app.source_message_runtime import prepare_source_message_graph
from tests.test_authorization_evidence import auth_case as auth_case


def candidate(key, support=None, hop=1, suffix=""):
    support = (key,) if support is None else support
    return GraphCandidate(key, hop, (), tuple(f"a{suffix}{i}" for i in range(hop)), support)


def test_source_quota_counts_union_of_complete_bundles_instead_of_candidates():
    paths = (candidate("a", ("a", "b"), 2, "first"), candidate("c"), candidate("d"), candidate("e"))
    order, audit = allocate_source_priority(("x", "y", "a", "b", "c", "d", "e"), paths, priority_source_cap=4)
    assert order == ("a", "b", "c", "d", "x", "y", "e")
    assert audit["priority_chunk_ids"] == ["a", "b", "c", "d"]
    assert not audit["decisions"][-1]["selected_for_priority"]


def test_overlapping_bundles_count_only_incremental_unique_sources():
    paths = (candidate("a", ("a", "b"), 2, "first"), candidate("b", ("b", "c"), 2, "second"))
    _, audit = allocate_source_priority(("x", "a", "b", "c"), paths, priority_source_cap=3)
    assert audit["priority_chunk_ids"] == ["a", "b", "c"]
    assert [d["incremental_source_chunks"] for d in audit["decisions"]] == [2, 1]


def test_oversized_priority_bundle_does_not_block_smaller_complete_bundle():
    paths = (candidate("a", ("a", "b"), 2), candidate("c"))
    order, audit = allocate_source_priority(("x", "a", "b", "c"), paths, priority_source_cap=1)
    assert order == ("c", "x", "a", "b")
    assert audit["decisions"][0]["reason"] == "PRIORITY_SOURCE_QUOTA"


def test_outside_pool_source_is_never_injected_or_partly_prioritized():
    order, audit = allocate_source_priority(("x", "a"), (candidate("a", ("a", "b"), 2),))
    assert order == ("x", "a") and audit["priority_chunk_ids"] == []
    assert audit["decisions"][0]["reason"] == "OUTSIDE_FROZEN_POOL"


def test_zero_quota_preserves_order_and_pool_hash():
    pool = ("b", "c", "a")
    order, audit = allocate_source_priority(pool, (candidate("a"),), priority_source_cap=0)
    assert order == pool and audit["priority_chunk_ids"] == []
    assert audit["predecessor_order_sha256"] == audit["allocated_order_sha256"]


@pytest.mark.parametrize("cap", [-1, 5, True, 1.0, None])
def test_priority_quota_bounds(cap):
    with pytest.raises(ValueError, match="BOUNDS"):
        allocate_source_priority(("a",), (), priority_source_cap=cap)


@pytest.mark.parametrize("pool,paths", [(("a", "a"), ()), (tuple(str(i) for i in range(41)), ()),
    (("a",), (candidate("a"), candidate("a")))])
def test_duplicate_or_oversized_pool_and_candidate_input_rejected(pool, paths):
    with pytest.raises(ValueError, match="BOUNDS"):
        allocate_source_priority(pool, paths)


@pytest.mark.parametrize("path", [candidate("a", ("b",)), candidate("a", ("a", "a")),
    candidate("a", (), 1), candidate("a", ("a",), 0),
    replace(candidate("a", ("a", "b"), 2), assertion_path=("same", "same"))])
def test_atomic_bundle_invariants(path):
    with pytest.raises(ValueError, match="ATOMIC_BUNDLE"):
        allocate_source_priority(("a", "b"), (path,))


def prepare(case, question="Which Lambda execution role permissions are needed for Amazon S3?"):
    return prepare_source_message_graph(case.collection, case.lexical, case.graph, question,
        fixture_vector(question), tenant_id="demo", owner_id="fixture-allocation", run_id="fixture-allocation",
        recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64), hops=2, allow_experimental=True)


def test_adapter_keeps_pool_rankings_full_sources_budget_and_method(auth_case):
    before = prepare(auth_case)
    after = reallocate_graph_sources(auth_case.collection, auth_case.lexical, auth_case.graph, before)
    assert set(after.prepared.retrieved_chunk_ids) == set(before.prepared.retrieved_chunk_ids)
    for field in ("paths", "graph_ranking", "vector_ranking", "lexical_ranking", "seed_entity_ids", "baseline_ranking"):
        assert after.trace[field] == before.trace[field]
    assert after.outcome.requested_method == after.outcome.effective_method == "graph_2hop"
    assert after.prepared.binding.request_id != before.prepared.binding.request_id
    assert len(after.prepared.packed.chunk_ids) <= 8
    _, prompt = generation_request(after.prepared.question, after.view, after.prepared.packed, "demo", after.prepared.recipe)
    assert prompt == after.prepared.packed.serialized
    assert ProxyTokenizer().count(prompt) == after.prepared.packed.proxy_tokens <= 4096
    for path in after.trace["packed_paths"]:
        assert set(path["supporting_chunk_ids"]) <= set(after.prepared.packed.chunk_ids)
    assert all(r.get("permission_granted") is not True for r in after.trace["source_records"])


def test_no_entity_zero_expansion_is_explicit_and_preserved(auth_case):
    before = prepare(auth_case, "How do tomatoes grow in winter?")
    after = reallocate_graph_sources(auth_case.collection, auth_case.lexical, auth_case.graph, before)
    assert after.outcome.mechanism_status == "GRAPH_ZERO_EXPANSION"
    assert not after.outcome.graph_path_used and after.outcome.zero_expansion_reason == "NO_LINKED_ENTITY"
    assert after.outcome.effective_method == "graph_2hop"
    assert after.prepared.packed.chunk_ids == before.prepared.packed.chunk_ids


@pytest.mark.parametrize("field", ["release_sha256", "collection_manifest_sha256", "selection_order", "context_chunk_ids", "question"])
def test_predecessor_binding_tampering_fails_closed(auth_case, field):
    before = prepare(auth_case)
    trace = deepcopy(before.trace)
    trace[field] = [] if field.endswith("order") or field.endswith("ids") else "changed"
    with pytest.raises(ValueError, match="PREDECESSOR_BINDING"):
        reallocate_graph_sources(auth_case.collection, auth_case.lexical, auth_case.graph, replace(before, trace=trace))


def test_quota_versions_have_distinct_identity_even_if_same_context(auth_case):
    before = prepare(auth_case)
    two = reallocate_graph_sources(auth_case.collection, auth_case.lexical, auth_case.graph, before, priority_source_cap=2)
    four = reallocate_graph_sources(auth_case.collection, auth_case.lexical, auth_case.graph, before, priority_source_cap=4)
    assert two.prepared.binding.request_id != four.prepared.binding.request_id
    assert two.trace["method_config_sha256"] != four.trace["method_config_sha256"]
