import dataclasses

import pytest

from app.fixtures import FIXTURE_TENANT, fixture_vector
from app.generation import GenerationRecipeV9, generation_request
from app.method_execution import answer_strict_prepared
from app.query_planning_v2 import make_validated_plan_v2
from app.strict_collection_methods import (
    link_query_entities,
    prepare_strict_graph_answer,
    prepare_strict_multi_query_answer,
)
from tests.test_collection_graph_runtime import graph_case as graph_case


def graph_prepare(case, question="How does Service A use Resource C?", **overrides):
    collection, lexical, graph, _ = case
    params = dict(tenant_id=FIXTURE_TENANT, owner_id="reviewer", run_id="strict-fixture",
        recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64), allow_experimental=True, clock=lambda: 0.0)
    return prepare_strict_graph_answer(collection, lexical, graph, question,
        fixture_vector(question), **(params | overrides))


def test_query_linking_works_without_any_retrieved_mentions_and_respects_boundaries(graph_case):
    graph = graph_case[2]
    links = link_query_entities(graph.overlay.entities, "Service A uses a resource", "Dịch vụ Service A hoạt động ra sao?")
    assert any(link["canonical_name"] == "Service A" and link["source"] == "SEARCH_QUERY" for link in links)
    assert not link_query_entities(graph.overlay.entities, "Service ABC unrelated", "Unrelated question")


def test_real_two_hop_support_reaches_the_prompt_and_never_becomes_hybrid(graph_case):
    result = graph_prepare(graph_case, hops=2)
    assert result.outcome.mechanism_status == "USED_GRAPH" and result.prepared is not None
    assert result.outcome.effective_method == "graph_2hop" and result.prepared.binding.used_graph
    assert result.prepared.packed.assertion_ids
    paths = result.trace["packed_paths"]
    assert any(len(p["assertion_ids"]) == 2 for p in paths)
    assert all(set(p["supporting_chunk_ids"]) <= set(result.prepared.packed.chunk_ids) for p in paths)
    assert all("NEGATIVE" != a.polarity for a in graph_case[2].overlay.assertions
               if a.assertion_id in result.prepared.packed.assertion_ids)


def test_no_relation_returns_no_prepared_generation_not_hybrid(graph_case):
    graph = graph_case[2]
    empty = dataclasses.replace(graph, overlay=graph.overlay.model_copy(update={"assertions": ()}))
    case = (graph_case[0], graph_case[1], empty, graph_case[3])
    result = graph_prepare(case)
    assert result.outcome.mechanism_status == "NO_SUPPORTED_RELATION"
    assert result.prepared is None and result.outcome.effective_method is None
    assert not result.outcome.context_chunk_ids and not result.outcome.graph_path_used


def test_timeout_is_explicit_and_cross_tenant_never_retrieves(graph_case):
    result = graph_prepare(graph_case, clock=lambda: 0.0, time_budget_ms=1, traversal_clock=lambda: 1.0)
    assert result.outcome.mechanism_status == "BUDGET_EXCEEDED" and result.prepared is None
    with pytest.raises(ValueError, match="TENANT|BINDING"):
        graph_prepare(graph_case, tenant_id="other")


def test_same_shared_plan_drives_q_qb_pool_and_actual_allocation(graph_case):
    collection, lexical, _, _ = graph_case
    question = "How does Service A use Resource C?"
    subqueries = ("Resource C requires Permission D",)
    plan = make_validated_plan_v2(question, question, "en", subqueries, operation_id="fixture-plan",
        receipt_sha256="1" * 64, origin="FIXTURE")
    kwargs = dict(question=question, search_query=question, query_vector=fixture_vector(question),
        subquery_vectors=tuple(fixture_vector(s) for s in subqueries), plan=plan, tenant_id=FIXTURE_TENANT,
        owner_id="reviewer", run_id="strict-fixture", recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64))
    q = prepare_strict_multi_query_answer(collection, lexical, mode="hybrid_multi_query", **kwargs)
    qb = prepare_strict_multi_query_answer(collection, lexical, mode="hybrid_multi_query_balanced", **kwargs)
    assert q.trace["multi_query"] == qb.trace["multi_query"]
    assert set(q.prepared.retrieved_chunk_ids) == set(qb.prepared.retrieved_chunk_ids)
    assert qb.trace["context_selection"]["decisions"] and qb.outcome.context_selection_used
    assert q.outcome.mechanism_status == qb.outcome.mechanism_status == "USED_MULTI_QUERY"
    with pytest.raises(ValueError, match="QUESTION_BINDING"):
        prepare_strict_multi_query_answer(collection, lexical, mode="hybrid_multi_query", **(kwargs | {"question": "Different user task"}))


def test_answer_language_is_bound_into_graph_identity_and_actual_generation_call(graph_case):
    en = graph_prepare(graph_case)
    vi = graph_prepare(graph_case, answer_language="vi")
    assert en.prepared.context.request_id != vi.prepared.context.request_id
    class Cache:
        def resolve(self, *args, **kwargs):
            return kwargs["answer_language"]
    assert answer_strict_prepared(vi, Cache()) == "vi"
    assert answer_strict_prepared(en, Cache()) == "en"


def test_missing_path_support_cannot_enter_generation_even_if_baseline_has_sources(graph_case, monkeypatch):
    from app import strict_collection_methods as runtime

    real = runtime.pack_context
    monkeypatch.setattr(runtime, "pack_context", lambda *args, **kwargs:
        dataclasses.replace(real(*args, **kwargs), assertion_ids=()))
    result = graph_prepare(graph_case, hops=2)
    assert result.outcome.mechanism_status == "PATH_INSUFFICIENT" and result.prepared is None

    class ForbiddenCache:
        def resolve(self, *args, **kwargs):
            pytest.fail("unsupported strict graph must not reach model/cache")

    with pytest.raises(ValueError, match="NOT_READY_FOR_GENERATION"):
        answer_strict_prepared(result, ForbiddenCache())


def test_real_qualifier_and_source_passages_remain_in_the_generation_prompt(graph_case):
    result = graph_prepare(graph_case, question="Which Permission D does Resource C require in region demo-1?")
    assert result.prepared is not None
    _, prompt = generation_request(result.prepared.question, result.view, result.prepared.packed,
        FIXTURE_TENANT, result.prepared.recipe)
    assert "region demo-1" in prompt and "Resource C requires Permission D" in prompt


def test_ambiguous_alias_does_not_merge_two_resource_identities(graph_case):
    from app.contracts import Entity

    entities = tuple(Entity.create(tenant_id=FIXTURE_TENANT, canonical_key="fixture/ambiguous/" + str(i),
        canonical_name="Resource " + str(i), aliases=("Shared resource",), entity_type="RESOURCE",
        canonicalization_version="fixture-v2") for i in range(2))
    links = link_query_entities(entities, "How does Shared resource work?", "Shared resource")
    assert links and all(link["ambiguous"] and not link["selected"] for link in links)
