"""Synthetic QB integration checks; success is not semantic answer quality."""

import copy
import json
import sqlite3

import pytest
from pydantic import ValidationError

from app.context_selection import ContextSelectionRecipe
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.rag_methods import BalancedMultiQueryMethodConfigV1, MethodResultV1, RunInputV1
from app.rag_runner import run_methods
from scripts.replay_collection_run import assert_replay
from tests.test_collection_review import review_context as review_context
from tests.test_multi_query_runner import operations, provider
from tests.test_multi_query_runner import q_setup as q_setup
from tests.test_query_api import call, event
from tests.test_query_api import setup_api as setup_api
from tests.test_rag_runner import fixture_gateway, request
from tests.test_rag_runner import setup_runner as setup_runner

Q = "hybrid_multi_query"
QB = "hybrid_multi_query_balanced"


@pytest.mark.parametrize("methods", [(Q, QB), (QB, Q), (QB,)])
def test_shared_planner_charged_once_same_frozen_pool_and_zero_call_replay(q_setup, methods):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    observed = provider(gateway)
    item = request(run_kind="answer", method_ids=methods)
    fresh = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert fresh["status"] == "COMPLETED"
    assert [row["kind"] for row in observed] == ["PLAN_QUERY", *["GENERATE"] * len(methods)]
    assert fresh["new_known_cost_micro_usd"] == 42 * (1 + len(methods))
    assert fresh["planner_sharing"] == "Q_FAMILY_V1_SINGLE_DURABLE_PLANNER_OPERATION"
    assert kwargs["encoder"].calls == 3
    assert len([op for op in operations(gateway) if op["spec"]["stage"] == "PLAN_QUERY"]) == 1
    operation = next(op for op in operations(gateway) if op["spec"]["stage"] == "PLAN_QUERY")
    assert operation["spec"]["request_id"] == stable_id("plan", run=fresh["run_identity"],
        config={"planner_recipe": kwargs["query_plan_cache"].recipe.fingerprint, "sharing": "q-family-v1"})
    assert fresh["methods"][0]["planning_trace"]["new_cost_micro_usd"] == 42
    assert not fresh["methods"][0]["planning_trace"]["cache_hit"]
    if len(methods) == 2:
        assert fresh["methods"][1]["planning_trace"]["cache_hit"]
        assert fresh["methods"][1]["planning_trace"]["new_cost_micro_usd"] == 0
        assert fresh["methods"][0]["multi_query_trace"] == fresh["methods"][1]["multi_query_trace"]
    balanced = next(row for row in fresh["methods"] if row["effective_method"] == QB)
    trace = balanced["context_selection_trace"]
    assert balanced["config"]["schema_version"] == 4
    assert balanced["config"]["revision"] == "collection-bounded-multi-query-balanced-v1"
    assert balanced["config"]["context_selection_recipe"] == ContextSelectionRecipe().model_dump(mode="json")
    assert trace["original_pool_ids"] == balanced["multi_query_trace"]["ranked_chunk_ids"]
    assert trace["ordered_chunk_ids"] == balanced["ranked_chunk_ids"]
    assert set(trace["original_pool_ids"]) == set(balanced["ranked_chunk_ids"])
    assert len(trace["ordered_chunk_ids"]) <= 40
    assert len(balanced["context_chunk_ids"]) <= 8 and balanced["context_proxy_tokens"] <= 4096
    assert balanced["response"]["mode_used"] == balanced["response"]["mode_requested"] == QB
    assert not balanced["response"]["used_graph"]
    assert [row["stage"] for row in balanced["trace"]].count("context_selection") == 1
    assert all(row["duration_ms"] >= 0 for row in balanced["trace"])
    assert isinstance(MethodResultV1.model_validate_json(canonical_json(balanced)).config,
                      BalancedMultiQueryMethodConfigV1)
    for source in balanced["context_sources"]:
        doc = collection.get_shard(source["shard_id"]).documents[source["document_version_id"]]
        assert doc.normalized_text[source["start_char"]:source["end_char"]] == source["text"]
    for citation in balanced["response"]["citations"]:
        assert citation["chunk_id"] in balanced["context_chunk_ids"]
    replay = run_methods(collection, lexical, item, **kwargs)
    assert_replay(fresh, replay)
    assert gateway.provider.calls == len(methods) + 1
    assert kwargs["encoder"].calls == 3
    assert replay["new_known_cost_micro_usd"] == 0
    assert all(row["planning_trace"]["cache_hit"] for row in replay["methods"])
    assert balanced["context_selection_trace"] == next(
        row for row in replay["methods"] if row["effective_method"] == QB)["context_selection_trace"]


def test_vi_shares_rewrite_plan_and_subquery_encoding_once(q_setup):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    observed = provider(gateway)
    item = RunInputV1(original_question="Dịch vụ B cần quyền gì?", language="vi", run_kind="answer",
                      method_ids=(Q, QB), idempotency_key="vi-q-qb")
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert report["status"] == "COMPLETED" and gateway.provider.calls == 4
    assert [row["kind"] for row in observed] == ["REWRITE", "PLAN_QUERY", "GENERATE", "GENERATE"]
    assert report["shared_cost_micro_usd"] == 42 and report["new_known_cost_micro_usd"] == 168
    assert kwargs["encoder"].calls == 3
    assert all(row["response"]["answer_language"] == "vi" for row in report["methods"])
    for row in observed[2:]:
        assert row["prompt"]["untrusted_question"] == item.original_question
        assert "subqueries" not in row["prompt"] and "planner" not in row["prompt"]
    assert_replay(report, run_methods(collection, lexical, item, **kwargs))
    assert gateway.provider.calls == 4


def test_q_alone_keeps_legacy_config_run_binding_and_planner_id(q_setup):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    provider(gateway)
    item = request(run_kind="answer", method_ids=(Q,))
    # A diagnostic selector supplied to a non-QB run has no effect or identity footprint.
    report = run_methods(collection, lexical, item, gateway=gateway,
        context_selection_recipe=ContextSelectionRecipe(policy="RR_SCOPED_EXACT_DEDUP"), **kwargs)
    row = report["methods"][0]
    assert report["status"] == "COMPLETED" and row["config"]["schema_version"] == 3
    assert "context_selection_trace" not in row
    assert "context_selection_recipe_sha256" not in report and "planner_sharing" not in report
    expected = {"input": item.model_dump(mode="json"), "collection_sha": collection.manifest_sha256,
        "lexical_version": lexical.manifest.index_version, "recipe": kwargs["recipe"].fingerprint,
        "query_recipe": kwargs["encoder"].query.fingerprint, "run_id": kwargs["run_id"],
        "planner_recipe": kwargs["query_plan_cache"].recipe.fingerprint}
    with sqlite3.connect(kwargs["cache"].path.with_name("method_runs.sqlite3")) as db:
        assert db.execute("SELECT payload_sha FROM runs").fetchone()[0] == sha256_bytes(canonical_json(expected))
    operation = next(op for op in operations(gateway) if op["spec"]["stage"] == "PLAN_QUERY")
    assert operation["spec"]["request_id"] == stable_id("plan", run=report["run_identity"], config=row["config_sha256"])


@pytest.mark.parametrize("selector", [ContextSelectionRecipe(policy="RR_SCOPED_EXACT_DEDUP"),
    ContextSelectionRecipe().model_copy(update={"max_pool_candidates": 41}), {}])
def test_invalid_or_diagnostic_selector_rejected_before_any_work(q_setup, selector):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    provider(gateway)
    item = RunInputV1(original_question="Dịch vụ B là gì?", language="vi", run_kind="answer",
                      method_ids=("hybrid", QB), idempotency_key="bad-selector")
    with pytest.raises(GenerationError, match="CONTEXT_SELECTION"):
        run_methods(collection, lexical, item, context_selection_recipe=selector, gateway=gateway, **kwargs)
    assert gateway.provider.calls == kwargs["encoder"].calls == 0
    assert not kwargs["cache"].path.with_name("method_runs.sqlite3").exists()


def test_selection_failure_keeps_planner_cost_and_blocks_next_paid_method(q_setup, monkeypatch):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    observed = provider(gateway)

    def fail(*args, **kw):
        raise GenerationError("SYNTHETIC_SELECTOR_FAILURE")

    monkeypatch.setattr("app.collection_answering.select_context_order", fail)
    item = request(run_kind="answer", method_ids=(QB, Q))
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert report["status"] == "PARTIAL" and gateway.provider.calls == 1
    assert [row["kind"] for row in observed] == ["PLAN_QUERY"]
    assert report["methods"][0]["status"] == "FAILED"
    assert report["methods"][0]["new_cost_micro_usd"] == report["new_known_cost_micro_usd"] == 42
    assert report["methods"][1]["error_category"] == "BLOCKED_AFTER_PEER_FAILURE"
    run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert gateway.provider.calls == 1


@pytest.mark.parametrize("tamper", ["missing", "rank", "trace_hash", "recipe", "original_pool", "foreign_context",
                                  "source_provenance", "query_provenance"])
def test_result_rejects_unbound_selector_trace(q_setup, tamper):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    provider(gateway)
    row = run_methods(collection, lexical, request(run_kind="answer", method_ids=(QB,)),
                      gateway=gateway, **kwargs)["methods"][0]
    changed = copy.deepcopy(row)
    if tamper == "missing":
        changed.pop("context_selection_trace")
    elif tamper == "rank":
        changed["ranked_chunk_ids"] = changed["ranked_chunk_ids"][::-1]
    elif tamper == "trace_hash":
        changed["context_selection_trace"]["source_trace_sha256"] = "f" * 64
    elif tamper == "recipe":
        changed["config"]["context_selection_recipe_sha256"] = "f" * 64
        changed["config_sha256"] = sha256_bytes(canonical_json(changed["config"]))
    elif tamper == "original_pool":
        changed["multi_query_trace"]["ranked_chunk_ids"] = changed["multi_query_trace"]["ranked_chunk_ids"][::-1]
    elif tamper == "foreign_context":
        changed["context_chunk_ids"] = ["foreign-chunk"]
    elif tamper == "source_provenance":
        changed["context_selection_trace"]["candidates"][0]["text_sha256"] = "f" * 64
    else:
        # Preserve the result's internal decision/occurrence shape while forging the query kind.
        changed["context_selection_trace"]["candidates"][0]["query_occurrences"][0]["kind"] = "SUBQUERY"
    with pytest.raises(ValidationError):
        MethodResultV1.model_validate_json(canonical_json(changed))


def test_replay_rejects_selector_trace_change(q_setup):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    provider(gateway)
    item = request(run_kind="answer", method_ids=(QB,))
    original = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    replay = run_methods(collection, lexical, item, **kwargs)
    replay["methods"][0]["context_selection_trace"]["source_trace_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="REPLAY_METHOD"):
        assert_replay(original, replay)


def test_qb_retrieval_only_and_cloud_rejected_before_any_admission(q_setup, setup_api):
    collection, lexical, kwargs = q_setup
    with pytest.raises(GenerationError, match="LOCAL_Q_REQUIRES_ANSWER_RUN"):
        run_methods(collection, lexical, request(method_ids=(QB,)), **kwargs)
    assert kwargs["encoder"].calls == 0
    status, payload = call(setup_api, event(body={"question": "Service B", "mode": QB}))
    assert status == 400 and "MODE_UNAVAILABLE" in json.dumps(payload)
    assert setup_api.gateway.calls == 0 and setup_api.repo.quota_remaining() == 300
