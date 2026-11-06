"""Bounded multi-query retrieval invariants, with synthetic IDs and source records."""

from fractions import Fraction

import numpy as np
import pytest
from pydantic import ValidationError

import app.collection_answering as adapter
import app.multi_query as module
from app.collection_answering import prepare_collection_answer
from app.contracts import canonical_json
from app.fixtures import FIXTURE_TENANT, fixture_vector
from app.generation import GenerationError, GenerationRecipeV9
from app.multi_query import MultiQueryTraceV1, QueryRankingV1, retrieve_multi_query
from tests.test_collection_answering import sources


def independent_rrf(rankings):
    scores = {}
    for ranking in rankings:
        for rank, key in enumerate(dict.fromkeys(ranking), 1):
            scores[key] = scores.get(key, Fraction()) + Fraction(1, 60 + rank)
    return tuple(sorted(scores, key=lambda key: (-scores[key], key)))


def test_three_bounded_hybrid_lists_fuse_then_cap_120_transient_ids_to_40(monkeypatch):
    branches = {query: tuple(f"{prefix}{index:02}" for index in range(40))
                for query, prefix in (("base", "a"), ("sub one", "b"), ("sub two", "c"))}
    seen, vectors = [], tuple(np.asarray([index]) for index in range(3))

    def search(collection, lexical, query, vector, *, vector_k, lexical_k):
        seen.append((query, vector, vector_k, lexical_k))
        return branches[query]

    monkeypatch.setattr(module, "retrieve_collection_hybrid", search)
    trace = retrieve_multi_query(None, None, "base", vectors[0], subqueries=("sub one", "sub two"),
                                 subquery_vectors=vectors[1:])
    expected = independent_rrf(branches.values())
    assert trace.ranked_chunk_ids == expected[:40]
    assert trace.unique_union_count == 120 and trace.truncated_count == 80
    assert [row.query for row in trace.queries] == list(branches)
    assert [row.kind for row in trace.queries] == ["BASE", "SUBQUERY", "SUBQUERY"]
    assert all(actual[1] is vector and actual[2:] == (20, 20) for actual, vector in zip(seen, vectors, strict=True))


@pytest.mark.parametrize("queries,vectors,reason", [
    (("one", "two", "three"), (None, None, None), "SUBQUERY_CAP"),
    (("one",), (), "VECTOR_PAIR_MISMATCH"),
    ((" BASE  ",), (None,), "INPUT_INVALID"),
    (("one", "ONE"), (None, None), "INPUT_INVALID"),
    (("x" * 241,), (None,), "INPUT_INVALID"),
    ((" ",), (None,), "INPUT_INVALID"),
])
def test_invalid_query_bundle_never_performs_any_retrieval(monkeypatch, queries, vectors, reason):
    monkeypatch.setattr(module, "retrieve_collection_hybrid", lambda *args, **kwargs: pytest.fail("Must reject before search"))
    with pytest.raises(ValueError, match=reason):
        retrieve_multi_query(None, None, "base", None, subqueries=queries, subquery_vectors=vectors)


def test_zero_extra_queries_matches_exact_base_ranking_and_empty_results_are_valid(monkeypatch):
    monkeypatch.setattr(module, "retrieve_collection_hybrid", lambda *args, **kwargs: ["chunk-b", "chunk-a"])
    trace = retrieve_multi_query(None, None, "base", None, subqueries=(), subquery_vectors=())
    assert trace.ranked_chunk_ids == ("chunk-b", "chunk-a") and trace.unique_union_count == 2
    assert len(trace.queries) == 1 and trace.truncated_count == 0
    monkeypatch.setattr(module, "retrieve_collection_hybrid", lambda *args, **kwargs: [])
    empty = retrieve_multi_query(None, None, "base", None, subqueries=(), subquery_vectors=())
    assert empty.ranked_chunk_ids == () and empty.unique_union_count == empty.truncated_count == 0


def test_trace_cannot_forge_anchor_ranking_counts_or_foreign_ids():
    payload = {"base_search_query": "base", "queries": [
        {"query": "base", "kind": "BASE", "ranked_chunk_ids": ["a", "b"]},
        {"query": "extra", "kind": "SUBQUERY", "ranked_chunk_ids": ["b", "c"]}],
        "ranked_chunk_ids": ["b", "a", "c"], "unique_union_count": 3, "truncated_count": 0}
    trace = MultiQueryTraceV1.model_validate_json(canonical_json(payload))
    for change in ({"base_search_query": "changed"}, {"ranked_chunk_ids": ("foreign",)},
                   {"ranked_chunk_ids": ("a", "b", "c")}, {"unique_union_count": 4}, {"truncated_count": 1}):
        with pytest.raises(ValidationError):
            MultiQueryTraceV1.model_validate(trace.model_dump() | change)
    with pytest.raises(ValidationError):
        QueryRankingV1(query="base", kind="BASE", ranked_chunk_ids=("a", "a"))
    with pytest.raises(ValidationError):
        QueryRankingV1(query="base", kind="BASE", ranked_chunk_ids=tuple(str(index) for index in range(41)))


def test_materialization_receives_only_final_40_and_keeps_original_source_spans(tmp_path, monkeypatch):
    collection, lexical = sources(tmp_path / "large-q", documents=60)
    try:
        keys = sorted(key for ref in collection.manifest.shards for key in collection.get_shard(ref.shard_id).chunks)
        assert len(keys) == 120
        queries = ("base", "sub one", "sub two")
        rankings = {query: keys[index * 40:(index + 1) * 40] for index, query in enumerate(queries)}
        monkeypatch.setattr(module, "retrieve_collection_hybrid", lambda collection, lexical, query, vector, **kwargs: rankings[query])
        original, materialized = adapter._evidence_view, []

        def bounded_view(collection, lexical, ranked):
            materialized.append(tuple(ranked))
            return original(collection, lexical, ranked)

        monkeypatch.setattr(adapter, "_evidence_view", bounded_view)
        vector = fixture_vector("synthetic service")
        traces = []
        view, prepared = prepare_collection_answer(collection, lexical, "Original question stays unchanged", vector,
            tenant_id=FIXTURE_TENANT, owner_id="fixture-owner", run_id="fixture-q", recipe=GenerationRecipeV9(
                model_metadata_sha256="0" * 64), mode="hybrid_multi_query", search_query="base",
            subqueries=queries[1:], subquery_vectors=(vector, vector), multi_query_trace=traces)
        assert materialized == [independent_rrf(rankings.values())[:40]]
        assert len(view.chunks) == len(prepared.retrieved_chunk_ids) == 40 and len(view.documents) <= 40
        assert len(prepared.packed.chunk_ids) <= 8 and prepared.packed.proxy_tokens <= 4096
        assert prepared.question == "Original question stays unchanged" and not view.entities
        assert traces[0].unique_union_count == 120 and traces[0].truncated_count == 80
        for key, chunk in view.chunks.items():
            document = view.documents[chunk.document_version_id]
            assert document.normalized_text[chunk.start_char:chunk.end_char] == chunk.text
            assert view.source_refs[key].manifest_sha256 == collection.get_shard(view.source_refs[key].shard_id).manifest_sha256
    finally:
        lexical.close()


def test_non_q_mode_rejects_hidden_extra_queries_before_any_search(tmp_path, monkeypatch):
    collection, lexical = sources(tmp_path / "plain-h", documents=2)
    try:
        monkeypatch.setattr(adapter, "retrieve_collection_hybrid", lambda *args, **kwargs: pytest.fail("Unexpected search"))
        with pytest.raises(GenerationError, match="UNEXPECTED_MULTI_QUERY_INPUT"):
            prepare_collection_answer(collection, lexical, "Service", fixture_vector("Service"),
                tenant_id=FIXTURE_TENANT, owner_id="owner", run_id="q", recipe=GenerationRecipeV9(
                    model_metadata_sha256="0" * 64), mode="hybrid", subqueries=("other",),
                subquery_vectors=(fixture_vector("other"),))
    finally:
        lexical.close()


def test_q_request_identity_binds_extra_queries_and_vectors_without_affecting_h_identity(tmp_path):
    collection, lexical = sources(tmp_path / "identity", documents=2)
    try:
        vector = fixture_vector("service")
        kwargs = dict(tenant_id=FIXTURE_TENANT, owner_id="owner", run_id="q", recipe=GenerationRecipeV9(
            model_metadata_sha256="0" * 64), search_query="service")
        _, base = prepare_collection_answer(collection, lexical, "Question", vector, mode="hybrid", **kwargs)
        _, first = prepare_collection_answer(collection, lexical, "Question", vector, mode="hybrid_multi_query",
            subqueries=("option",), subquery_vectors=(vector,), **kwargs)
        _, changed_query = prepare_collection_answer(collection, lexical, "Question", vector, mode="hybrid_multi_query",
            subqueries=("enabled",), subquery_vectors=(vector,), **kwargs)
        other = fixture_vector("unrelated")
        _, changed_vector = prepare_collection_answer(collection, lexical, "Question", vector, mode="hybrid_multi_query",
            subqueries=("option",), subquery_vectors=(other,), **kwargs)
        assert len({first.context.request_id, changed_query.context.request_id, changed_vector.context.request_id}) == 3
        _, repeated = prepare_collection_answer(collection, lexical, "Question", vector, mode="hybrid", **kwargs)
        assert base.context.request_id == repeated.context.request_id
    finally:
        lexical.close()
