"""Shared native-Q/QB and strict graph mechanism/answer integration."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import app.scoped_method_flow as module
from app.contracts import sha256_bytes
from app.profiled_answer_runtime import ProfiledAnswerCache
from app.profiled_generation import ProfiledGenerationRecipe
from app.profiled_source_context import SourceContextProfile
from app.scoped_query_v6 import ScopedPlanV6, ScopeRecipeV6, validate_source_facets
from tests.test_profiled_answer_runtime import Gateway
from tests.test_profiled_source_context import fixture_sources

QUESTION = "Explain permission to invoke a function and permission to read an object."


def fixture(monkeypatch):
    docs, chunks = fixture_sources()
    collection = SimpleNamespace(manifest_sha256="a"*64,
        manifest=SimpleNamespace(tenant_id="demo", purpose="FIXTURE"))
    monkeypatch.setattr(module, "collection_records", lambda c:(docs, chunks, {}))
    monkeypatch.setattr(module, "validate_collection_lexical_binding", lambda collection,index:None)
    keys = tuple(chunks)
    rankings = {QUESTION:keys[:8], "permission to invoke a function":keys[4:12],
        "permission to read an object":(*keys[10:12], *keys[:6])}
    monkeypatch.setattr(module, "_ranking", lambda collection,index,runtime,question,search,method:(rankings[search], []))
    plan = ScopedPlanV6(origin="FIXTURE", collection_manifest_sha256="a"*64,
        question_sha256=sha256_bytes(QUESTION.encode()), search_query_sha256=sha256_bytes(QUESTION.encode()),
        language="en", facets=validate_source_facets(
            '{"facet_1":"permission to invoke a function","facet_2":"permission to read an object","facet_3":null}',
            QUESTION, QUESTION), recipe_sha256=ScopeRecipeV6().fingerprint,
        operation_id="fixture-plan", receipt_sha256="b"*64)
    return collection, docs, chunks, keys, plan


def prepare(collection, method, plan=None, cap=12):
    return module.prepare_scoped_flow(collection, None, None, None,
        question=QUESTION, search=QUESTION, language="en", method=method, plan=plan,
        run_id="fixture-flow", owner_id="fixture-owner", recipe=ProfiledGenerationRecipe(model_metadata_sha256="b"*64),
        profile=SourceContextProfile(max_chunks=cap))


def test_q_qb_share_native_plan_pool_and_original_task_but_execute_different_schedulers(monkeypatch):
    collection, _, _, _, plan = fixture(monkeypatch)
    q, qb = [prepare(collection, method, plan) for method in module.Q_METHODS]
    assert q.ready and qb.ready
    assert q.trace["pool_ids"] == qb.trace["pool_ids"]
    assert q.trace["pool_fingerprint"] == qb.trace["pool_fingerprint"]
    assert q.trace["bundles"] == qb.trace["bundles"]
    assert not q.trace["balanced_allocator_used"] and qb.trace["balanced_allocator_used"]
    assert q.trace["packing"]["bundle_order"] != qb.trace["packing"]["bundle_order"]
    assert q.context.question == qb.context.question == QUESTION
    assert q.binding.request_id != qb.binding.request_id


def test_shared_query_admission_preserves_actual_facet_prefix_inside_cap40():
    rankings = [tuple(f"q{q}-{i}" for i in range(40)) for q in range(4)]
    pool, bundles = module.shared_query_pool(rankings)
    assert len(pool) == 40 and {f"q{q}-{i}" for q in range(4) for i in range(2)} <= set(pool)
    assert {b["query_index"] for b in bundles} == {0,1,2,3}
    assert all(set(b["chunk_ids"]) <= set(pool) for b in bundles)


def test_q_cannot_substitute_hybrid_when_plan_missing_or_empty(monkeypatch):
    collection, _, _, _, plan = fixture(monkeypatch)
    with pytest.raises(ValueError, match="TYPED_PLAN"):
        prepare(collection, "hybrid_multi_query")
    empty = prepare(collection, "hybrid_multi_query", plan.model_copy(update={"facets":()}))
    assert not empty.ready and empty.status == "NOT_READY_NO_DISTINCT_NATIVE_FACETS"
    with pytest.raises(ValueError, match="NOT_READY_NO_FALLBACK"):
        module.answer_scoped_flow(empty, None)


def test_graph_zero_does_not_dispatch_a_hybrid_answer(monkeypatch):
    collection, _, _, keys, _ = fixture(monkeypatch)
    monkeypatch.setattr(module, "_graph_pool", lambda *a:(keys[:8],
        [{"bundle_id":str(i), "query_index":0, "chunk_ids":[k]} for i,k in enumerate(keys[:8])],
        {"path_count":0, "complete_path_bundles_in_pool":0}))
    graph = SimpleNamespace(facts=[])
    flow = module.prepare_scoped_flow(collection, None, graph, None, question=QUESTION, search=QUESTION,
        language="en", method="graph_2hop", run_id="fixture", owner_id="owner",
        recipe=ProfiledGenerationRecipe(model_metadata_sha256="b"*64))
    assert not flow.ready and not flow.binding.used_graph
    assert flow.binding.mode_requested == flow.binding.mode_used == "graph_2hop"
    with pytest.raises(ValueError, match="NOT_READY_NO_FALLBACK"):
        module.answer_scoped_flow(flow, None)


def test_v12_answer_cache_is_reached_end_to_end_for_native_q_and_replay(tmp_path, monkeypatch):
    collection, _, _, _, plan = fixture(monkeypatch)
    flow = prepare(collection, "hybrid_multi_query_balanced", plan)
    cache, gateway = ProfiledAnswerCache(tmp_path, namespace="FIXTURE"), Gateway(flow.recipe)
    first = module.answer_scoped_flow(flow, cache, gateway=gateway)
    second = module.answer_scoped_flow(flow, cache)
    assert gateway.calls == 1 and first.response == second.response and second.new_cost_micro_usd == 0
    assert first.response.mode_used == "hybrid_multi_query_balanced"
    assert first.response.evidence_bundle_sha256 == flow.context.fingerprint


def test_graph_source_or_traversal_errors_propagate_without_replacing_method(monkeypatch):
    collection, *_ = fixture(monkeypatch)

    def fail(*args):
        raise RuntimeError("SYNTHETIC_GRAPH_FAILURE")

    monkeypatch.setattr(module, "_graph_pool", fail)
    with pytest.raises(RuntimeError, match="SYNTHETIC_GRAPH_FAILURE"):
        prepare(collection, "graph_2hop")


def test_v12_cache_rejects_source_sidecar_tamper_before_dispatch(tmp_path, monkeypatch):
    collection, _, _, _, plan = fixture(monkeypatch)
    flow = prepare(collection, "hybrid_multi_query", plan)
    tampered = replace(flow, context=replace(flow.context, span_lookup_json="[]"))
    gateway = Gateway(flow.recipe)
    with pytest.raises(ValueError):
        module.answer_scoped_flow(tampered, ProfiledAnswerCache(tmp_path, namespace="FIXTURE"), gateway=gateway)
    assert gateway.calls == 0
