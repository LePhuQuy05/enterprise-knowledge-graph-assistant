"""Synthetic integration checks, never evidence of AWS answer quality."""

import json
import sqlite3

import pytest
from pydantic import ValidationError

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.query_planning import QueryPlanCache
from app.rag_methods import MethodResultV1, MultiQueryMethodConfigV1, RunInputV1
from app.rag_runner import run_methods
from tests.test_collection_answering import FixtureReranker
from tests.test_collection_review import review_context as review_context
from tests.test_external_pilot import configured
from tests.test_query_api import call, event
from tests.test_query_api import setup_api as setup_api
from tests.test_rag_runner import fixture_gateway, request
from tests.test_rag_runner import setup_runner as setup_runner

Q = "hybrid_multi_query"
SUBQUERIES = ["Service B configuration requirements", "Service A integration with Service B"]


@pytest.fixture
def q_setup(setup_runner, tmp_path):
    collection, lexical, kwargs = setup_runner
    kwargs["query_plan_cache"] = QueryPlanCache(tmp_path / "plans", namespace="FIXTURE")
    return collection, lexical, kwargs


def provider(gateway, *, planner_raw=None, planner_error=False, invalid_generation=False):
    """A fixture provider retaining real gateway admission, settlement and cache paths."""
    observed = []

    def respond(name, method, path, body):
        assert (name, method, path) == ("DEEPSEEK", "POST", "/chat/completions")
        gateway.provider.calls += 1
        payload = json.loads(body)
        prompt = json.loads(payload["messages"][1]["content"])
        if "untrusted_original_question" in prompt:
            kind = "PLAN_QUERY"
            assert payload["max_tokens"] == 256
            raw = json.dumps({"subqueries": SUBQUERIES}) if planner_raw is None else planner_raw
        elif "untrusted_sources" in prompt:
            kind = "GENERATE"
            first = prompt["untrusted_sources"][0]
            raw = json.dumps({"answer_status": "ANSWERED", "reason": "NONE", "claims": [{
                "text": "Dịch vụ B." if "Vietnamese" in payload["messages"][0]["content"] else "Synthetic Service B.",
                "evidence": [{"source_id": first["source_id"],
                              "span_id": "s8p99" if invalid_generation else first["spans"][0]["span_id"]}]}]})
        else:
            kind = "REWRITE"
            raw = json.dumps({"search_query": "Service B"})
        observed.append({"kind": kind, "prompt": prompt, "payload": payload})
        if kind == "PLAN_QUERY" and planner_error:
            raise TimeoutError("SYNTHETIC_TIMEOUT_MUST_NOT_ESCAPE")
        return {"model": "deepseek-flash", "usage": {
            "prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": raw}}]}

    gateway.provider.request = respond
    return observed


def operations(gateway):
    return [row for row in gateway.ledger.store._items.values() if row.get("record_type") == "MODEL_OPERATION"]


def test_h_q_share_base_sources_original_question_and_controls_with_zero_call_replay(q_setup):
    collection, lexical, kwargs = q_setup
    item = request(run_kind="answer", method_ids=("hybrid", Q))
    gateway = fixture_gateway(kwargs)
    observed = provider(gateway)
    fresh = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert fresh["status"] == "COMPLETED" and gateway.provider.calls == 3
    assert [row["kind"] for row in observed] == ["GENERATE", "PLAN_QUERY", "GENERATE"]
    assert kwargs["encoder"].calls == 3  # Shared anchor plus two distinct local subquery encodings.
    baseline, multi = fresh["methods"]
    assert baseline["config"]["schema_version"] == 1 and multi["config"]["schema_version"] == 3
    assert multi["config"]["revision"] == "collection-bounded-multi-query-v1"
    assert multi["multi_query_trace"]["queries"][0]["ranked_chunk_ids"] == baseline["ranked_chunk_ids"]
    assert multi["multi_query_trace"]["base_search_query"] == fresh["search_query"] == item.original_question
    assert multi["planning_trace"]["subqueries"] == SUBQUERIES
    assert all(row["prompt"].get("untrusted_question", item.original_question) == item.original_question
               for row in observed)
    assert observed[0]["payload"]["messages"][0] == observed[-1]["payload"]["messages"][0]
    assert observed[1]["prompt"] == {"untrusted_original_question": item.original_question,
        "untrusted_english_search_query": item.original_question, "original_language": "en"}
    assert multi["response"]["mode_used"] == multi["response"]["mode_requested"] == Q
    assert not multi["response"]["used_graph"]
    assert multi["new_cost_micro_usd"] == 84 and baseline["new_cost_micro_usd"] == 42
    assert fresh["new_known_cost_micro_usd"] == 126 and fresh["shared_cost_micro_usd"] == 0
    assert fresh["cost_complete"] and fresh["quality_metrics"] is None
    for row in fresh["methods"]:
        assert len(row["context_chunk_ids"]) <= 8 and row["context_proxy_tokens"] <= 4096
        for source in row["context_sources"]:
            snapshot = collection.get_shard(source["shard_id"])
            document = snapshot.documents[source["document_version_id"]]
            assert document.normalized_text[source["start_char"]:source["end_char"]] == source["text"]
        assert all(citation["chunk_id"] in row["context_chunk_ids"] for citation in row["response"]["citations"])
        assert MethodResultV1.model_validate_json(json.dumps(row)).model_dump(mode="json") == row
    assert isinstance(MethodResultV1.model_validate_json(json.dumps(multi)).config, MultiQueryMethodConfigV1)
    replay = run_methods(collection, lexical, item, **kwargs)
    assert replay["status"] == "COMPLETED" and gateway.provider.calls == 3 and kwargs["encoder"].calls == 3
    assert replay["new_known_cost_micro_usd"] == replay["shared_cost_micro_usd"] == 0
    assert all(row["cache_hit"] for row in replay["methods"])
    assert replay["methods"][1]["planning_trace"]["cache_hit"]
    assert replay["methods"][1]["planning_trace"]["new_cost_micro_usd"] == 0
    assert replay["methods"][1]["multi_query_trace"] == multi["multi_query_trace"]
    assert [row["response"] for row in replay["methods"]] == [row["response"] for row in fresh["methods"]]
    assert sorted(row["spec"]["stage"] for row in operations(gateway)) == ["GENERATE", "GENERATE", "PLAN_QUERY"]
    assert all(row["status"] == "SETTLED" for row in operations(gateway))


def test_vi_h_q_use_one_shared_rewrite_and_one_planner_without_leaking_plan_as_evidence(q_setup):
    collection, lexical, kwargs = q_setup
    item = RunInputV1(original_question="Dịch vụ B cần quyền gì?", language="vi", run_kind="answer",
                      method_ids=(Q, "hybrid"), idempotency_key="vi-h-q")
    gateway = fixture_gateway(kwargs)
    observed = provider(gateway)
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert report["status"] == "COMPLETED" and gateway.provider.calls == 4
    assert [row["kind"] for row in observed] == ["REWRITE", "PLAN_QUERY", "GENERATE", "GENERATE"]
    assert report["shared_cost_micro_usd"] == 42 and report["new_known_cost_micro_usd"] == 168
    assert report["search_query"] == "Service B" and report["answer_language"] == "vi"
    assert observed[1]["prompt"] == {"untrusted_original_question": item.original_question,
        "untrusted_english_search_query": "Service B", "original_language": "vi"}
    for row in observed[2:]:
        assert row["prompt"]["untrusted_question"] == item.original_question
        assert "subqueries" not in row["prompt"] and "planner" not in row["prompt"]
    assert all(row["response"]["answer_language"] == "vi" for row in report["methods"])
    replay = run_methods(collection, lexical, item, **kwargs)
    assert replay["status"] == "COMPLETED" and replay["rewrite_cache_hit"]
    assert replay["new_known_cost_micro_usd"] == 0 and gateway.provider.calls == 4


@pytest.mark.parametrize("methods", [("bm25", "vector", "hybrid"), ("hybrid", "hybrid_reranked")])
def test_no_q_keeps_exact_historical_config_shape_and_run_binding(q_setup, methods):
    collection, lexical, kwargs = q_setup
    scorer = FixtureReranker()
    item = request(method_ids=methods)
    report = run_methods(collection, lexical, item, reranker=scorer, **kwargs)
    assert "planner_recipe_sha256" not in report
    expected_payload = {"input": item.model_dump(mode="json"), "collection_sha": collection.manifest_sha256,
        "lexical_version": lexical.manifest.index_version, "recipe": kwargs["recipe"].fingerprint,
        "query_recipe": kwargs["encoder"].query.fingerprint, "run_id": kwargs["run_id"]}
    if "hybrid_reranked" in methods:
        expected_payload["reranker_recipe"] = scorer.recipe.fingerprint
    identity = stable_id("ldh", tenant=kwargs["tenant_id"], owner=kwargs["owner_id"], key=item.idempotency_key)
    with sqlite3.connect(kwargs["cache"].path.with_name("method_runs.sqlite3")) as db:
        assert db.execute("SELECT payload_sha FROM runs WHERE key=?", (identity,)).fetchone()[0] == sha256_bytes(canonical_json(expected_payload))
    for row in report["methods"]:
        method = row["effective_method"]
        expected = {"schema_version": 1, "method_id": method, "revision": "collection-ldh-v1",
            "supported_scopes": ["local_collection"], "feature_state": "experimental", "candidates_per_branch": 20,
            "fusion": "RRF60" if method in {"hybrid", "hybrid_reranked"} else "NONE",
            "max_context_chunks": 8, "context_proxy_token_budget": 4096, "max_output_tokens": 800,
            "collection_manifest_sha256": collection.manifest_sha256, "lexical_index_version": lexical.manifest.index_version,
            "document_recipe_sha256": collection.manifest.embedding_recipe.fingerprint,
            "query_recipe_sha256": None if method == "bm25" else kwargs["encoder"].query.fingerprint,
            "generation_recipe_sha256": kwargs["recipe"].fingerprint}
        if method == "hybrid_reranked":
            expected.update(schema_version=2, revision="collection-hybrid-reranked-v1", max_candidates=40,
                reranker_recipe=scorer.recipe.model_dump(mode="json"), reranker_recipe_sha256=scorer.recipe.fingerprint)
        assert row["config"] == expected and row["config_sha256"] == sha256_bytes(canonical_json(expected))
        assert "planning_trace" not in row and "multi_query_trace" not in row
        if method != "hybrid_reranked":
            assert "rerank_trace" not in row


@pytest.mark.parametrize("language", ["en", "vi"])
def test_q_retrieval_only_rejected_without_planning_rewrite_or_local_encoding(q_setup, language):
    collection, lexical, kwargs = q_setup
    with pytest.raises(GenerationError, match="LOCAL_Q_REQUIRES_ANSWER_RUN"):
        run_methods(collection, lexical, request(method_ids=(Q,), language=language), **kwargs)
    assert kwargs["encoder"].calls == 0
    assert not kwargs["cache"].path.with_name("method_runs.sqlite3").exists()
    with sqlite3.connect(kwargs["query_plan_cache"].path) as db:
        assert db.execute("SELECT count(*) FROM query_plans").fetchone()[0] == 0


@pytest.mark.parametrize("mismatch", ["missing", "namespace", "invalid_recipe"])
def test_q_invalid_planner_is_rejected_before_paid_shared_vi_rewrite(q_setup, tmp_path, mismatch):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    if mismatch == "missing":
        kwargs["query_plan_cache"] = None
    elif mismatch == "namespace":
        kwargs["query_plan_cache"] = QueryPlanCache(tmp_path / "live-plans", namespace="LIVE")
    else:
        planner = kwargs["query_plan_cache"]
        planner.recipe = planner.recipe.model_copy(update={"max_subqueries": 3})
    item = RunInputV1(original_question="Dịch vụ B làm gì?", language="vi", run_kind="answer",
                      method_ids=("hybrid", Q), idempotency_key="invalid-planner")
    with pytest.raises(ValueError):
        run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert gateway.provider.calls == kwargs["encoder"].calls == 0
    assert not kwargs["cache"].path.with_name("method_runs.sqlite3").exists()


@pytest.mark.parametrize("raw", ['{"subqueries":["one","two","three"]}', 'not JSON',
                                 '{"subqueries":["https://example.invalid/injected"]}'])
def test_invalid_settled_planner_is_durable_failed_and_blocks_later_h_without_retry(q_setup, raw):
    collection, lexical, kwargs = q_setup
    item = request(run_kind="answer", method_ids=(Q, "hybrid"))
    gateway = fixture_gateway(kwargs)
    observed = provider(gateway, planner_raw=raw)
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert report["status"] == "PARTIAL" and gateway.provider.calls == 1
    assert [row["kind"] for row in observed] == ["PLAN_QUERY"]
    assert report["methods"][0]["status"] == "FAILED" and report["methods"][0]["new_cost_micro_usd"] is None
    assert report["methods"][1]["error_category"] == "BLOCKED_AFTER_PEER_FAILURE"
    assert not report["cost_complete"]  # The failed branch requires receipt reconciliation, never a free-call claim.
    with sqlite3.connect(kwargs["query_plan_cache"].path) as db:
        state, payload = db.execute("SELECT state,payload FROM query_plans").fetchone()
    assert state == "FAILED" and json.loads(payload)["provider_receipt"]["raw_output"] == raw
    assert json.loads(payload)["provider_receipt"]["cost_micro_usd"] == 42
    assert operations(gateway)[0]["status"] == "SETTLED"
    repeated = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert repeated["status"] == "PARTIAL" and gateway.provider.calls == 1


def test_uncertain_planner_never_refunds_or_retries_and_blocks_remaining_methods(q_setup):
    collection, lexical, kwargs = q_setup
    item = request(run_kind="answer", method_ids=(Q, "hybrid"))
    gateway = fixture_gateway(kwargs)
    provider(gateway, planner_error=True)
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert report["status"] == "PARTIAL" and gateway.provider.calls == 1 and not report["cost_complete"]
    assert report["methods"][1]["error_category"] == "BLOCKED_AFTER_PEER_FAILURE"
    operation = operations(gateway)[0]
    assert operation["status"] == "UNKNOWN_SPEND" and operation["spec"]["stage"] == "PLAN_QUERY"
    run = gateway.ledger.store.get(gateway.ledger.run_key(kwargs["run_id"]))
    assert run["reserved_micro_usd"] == operation["spec"]["max_cost_micro_usd"] > 0
    with sqlite3.connect(kwargs["query_plan_cache"].path) as db:
        assert db.execute("SELECT state,payload FROM query_plans").fetchone() == ("FAILED", None)
    run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert gateway.provider.calls == 1


def test_known_planner_cost_survives_invalid_answer_and_peer_h_is_blocked(q_setup):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    provider(gateway, invalid_generation=True)
    item = request(run_kind="answer", method_ids=(Q, "hybrid"))
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert gateway.provider.calls == 2 and report["status"] == "PARTIAL" and not report["cost_complete"]
    failed, blocked = report["methods"]
    assert failed["planning_trace"]["new_cost_micro_usd"] == report["new_known_cost_micro_usd"] == 42
    assert failed["new_cost_micro_usd"] is None and blocked["error_category"] == "BLOCKED_AFTER_PEER_FAILURE"
    with sqlite3.connect(kwargs["cache"].path) as db:
        state, payload = db.execute("SELECT state,payload FROM answers").fetchone()
    assert state == "FAILED" and json.loads(payload)["original_cost_micro_usd"] == 42
    repeated = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert repeated["methods"][0]["status"] == "FAILED" and gateway.provider.calls == 2


@pytest.mark.parametrize("emitted,expected", [([], []), ([" SERVICE  B ", "service b"], []),
                                              ([" Service  A ", "service a"], ["Service A"])])
def test_empty_or_repeated_plans_keep_one_original_vote_without_an_extra_planner_attempt(q_setup, emitted, expected):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    provider(gateway, planner_raw=json.dumps({"subqueries": emitted}))
    report = run_methods(collection, lexical, request(run_kind="answer", method_ids=(Q,)), gateway=gateway, **kwargs)
    result = report["methods"][0]
    assert report["status"] == "COMPLETED" and gateway.provider.calls == 2
    assert result["planning_trace"]["subqueries"] == expected
    assert len(result["multi_query_trace"]["queries"]) == len(expected) + 1
    assert result["multi_query_trace"]["queries"][0]["query"] == "Service B"
    assert kwargs["encoder"].calls == len(expected) + 1 and result["new_cost_micro_usd"] == 84


def test_failed_h_answer_blocks_later_q_before_planner_or_subquery_work(q_setup):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    observed = provider(gateway, invalid_generation=True)
    item = request(run_kind="answer", method_ids=("hybrid", Q))
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert report["status"] == "PARTIAL" and gateway.provider.calls == 1
    assert [row["kind"] for row in observed] == ["GENERATE"]
    assert report["methods"][0]["status"] == "FAILED"
    assert report["methods"][1]["error_category"] == "BLOCKED_AFTER_PEER_FAILURE"
    assert kwargs["encoder"].calls == 1
    with sqlite3.connect(kwargs["query_plan_cache"].path) as db:
        assert db.execute("SELECT count(*) FROM query_plans").fetchone()[0] == 0


def test_paid_planning_cost_is_preserved_when_local_subquery_encoding_fails(q_setup, monkeypatch):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    observed = provider(gateway)
    original = kwargs["encoder"].encode

    def fail_subquery(texts, *, role):
        if "configuration requirements" in texts[0]:
            raise RuntimeError("SYNTHETIC_LOCAL_ENCODER_FAILURE")
        return original(texts, role=role)

    monkeypatch.setattr(kwargs["encoder"], "encode", fail_subquery)
    item = request(run_kind="answer", method_ids=(Q, "hybrid"))
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert gateway.provider.calls == 1 and [row["kind"] for row in observed] == ["PLAN_QUERY"]
    assert report["methods"][0]["status"] == "FAILED"
    assert report["methods"][0]["trace"][-1]["stage"] == "subquery_embeddings"
    assert report["methods"][0]["new_cost_micro_usd"] == report["new_known_cost_micro_usd"] == 42
    assert report["cost_complete"] and report["methods"][1]["error_category"] == "BLOCKED_AFTER_PEER_FAILURE"
    with sqlite3.connect(kwargs["query_plan_cache"].path) as db:
        assert db.execute("SELECT state FROM query_plans").fetchone()[0] == "READY"
    run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert gateway.provider.calls == 1  # Local failure never dispatches another planner call.


def test_existing_budget_denial_blocks_planner_transport_without_initializing_another_run(q_setup):
    collection, lexical, kwargs = q_setup
    gateway = configured()
    gateway.ledger.initialize_run("kg-r72", kwargs["tenant_id"], kwargs["run_id"], cap_micro_usd=100)
    provider(gateway)
    item = request(run_kind="answer", method_ids=(Q, "hybrid"))
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert report["status"] == "PARTIAL" and gateway.provider.calls == 0
    assert report["methods"][0]["status"] == "FAILED"
    assert report["methods"][1]["error_category"] == "BLOCKED_AFTER_PEER_FAILURE"
    run = gateway.ledger.store.get(gateway.ledger.run_key(kwargs["run_id"]))
    assert run["cap_micro_usd"] == 100 and run["settled_micro_usd"] == run["reserved_micro_usd"] == 0
    assert operations(gateway) == []
    run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert gateway.provider.calls == 0


def test_cache_only_q_miss_plans_no_call_and_does_not_claim_an_answer(q_setup):
    collection, lexical, kwargs = q_setup
    result = run_methods(collection, lexical, request(run_kind="answer", method_ids=(Q,)), **kwargs)
    assert result["status"] == "PARTIAL" and result["new_known_cost_micro_usd"] == 0
    assert result["methods"][0]["status"] == "PLANNED_NO_MODEL_CALL" and result["methods"][0]["response"] is None
    assert result["cost_complete"]
    with sqlite3.connect(kwargs["query_plan_cache"].path) as db:
        assert db.execute("SELECT count(*) FROM query_plans").fetchone()[0] == 0


def test_q_is_not_available_in_cloud_before_admission_or_model_use(setup_api):
    status, payload = call(setup_api, event(body={"question": "Service B", "mode": Q}))
    assert status == 400 and "MODE_UNAVAILABLE" in json.dumps(payload)
    assert setup_api.gateway.calls == 0 and setup_api.repo.quota_remaining() == 300


@pytest.mark.parametrize("extra", [{"subqueries": ["injected"]}, {"gold_answer": "leak"}, {"planner_recipe": {}}])
def test_public_run_input_cannot_inject_queries_answers_or_planner_configuration(extra):
    with pytest.raises(ValidationError):
        RunInputV1.model_validate_json(canonical_json({"original_question": "Service B", "idempotency_key": "x",
            "method_ids": [Q], "run_kind": "answer", **extra}))
