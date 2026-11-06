"""Document-tail counterexamples and actual adapter budget/source integrity."""

import pytest

from app.graph_document_allocation import document_diverse_tail, reallocate_graph_documents
from tests.test_authorization_evidence import auth_case as auth_case
from tests.test_graph_source_allocation import prepare


def test_diverse_first_pass_retains_all_deferred_same_document_sources():
    pool = ("a", "a2", "b", "b2", "c", "d", "c2")
    docs = dict(a="doc-a", a2="doc-a", b="doc-b", b2="doc-b", c="doc-c", d="doc-d", c2="doc-c")
    order, decisions = document_diverse_tail(pool, ("a", "b"), docs)
    assert order == ("a", "b", "c", "d", "a2", "b2", "c2")
    assert decisions[0]["pass"] == "DEFERRED_SAME_DOCUMENT"
    assert set(order) == set(pool)


def test_two_sources_in_same_document_can_remain_in_atomic_priority_prefix():
    order, _ = document_diverse_tail(("a", "x", "a2", "b"), ("a", "a2"),
        {"a": "one", "a2": "one", "x": "two", "b": "three"})
    assert order == ("a", "a2", "x", "b")


@pytest.mark.parametrize("pool,priority,docs", [(("a", "a"), (), {"a": "one"}),
    (("a",), ("b",), {"a": "one"}), (("a",), (), {}), (("a",), (), {"a": ""}),
    (("a",), ("a", "a"), {"a": "one"}), (("a",), (), {"a": "one", "b": "two"})])
def test_wrong_document_pool_bindings_rejected(pool, priority, docs):
    with pytest.raises(ValueError, match="DOCUMENT_ALLOCATION_POOL_BINDING"):
        document_diverse_tail(pool, priority, docs)


def test_adapter_remains_same_pool_atomic_source_bound_and_has_new_identity(auth_case):
    before = prepare(auth_case)
    after = reallocate_graph_documents(auth_case.collection, auth_case.lexical, auth_case.graph, before)
    assert set(after.prepared.retrieved_chunk_ids) == set(before.prepared.retrieved_chunk_ids)
    assert len(after.prepared.packed.chunk_ids) <= 8 and after.prepared.packed.proxy_tokens <= 4096
    assert before.prepared.binding.request_id != after.prepared.binding.request_id
    assert after.outcome.requested_method == after.outcome.effective_method == before.outcome.requested_method
    for path in after.trace["packed_paths"]:
        assert set(path["supporting_chunk_ids"]) <= set(after.prepared.packed.chunk_ids)
    assert after.trace["paths"] == before.trace["paths"]


def test_no_graph_path_keeps_baseline_sources_and_explicit_zero(auth_case):
    before = prepare(auth_case, "How do tomatoes grow in winter?")
    after = reallocate_graph_documents(auth_case.collection, auth_case.lexical, auth_case.graph, before)
    assert after.prepared.packed.chunk_ids == before.prepared.packed.chunk_ids
    assert after.trace["document_allocation"]["skipped_no_eligible_path"]
    assert after.outcome.mechanism_status == "GRAPH_ZERO_EXPANSION" and not after.outcome.graph_path_used
