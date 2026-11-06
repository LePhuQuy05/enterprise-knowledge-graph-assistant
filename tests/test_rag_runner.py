import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from app.answer_cache import AnswerCache
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalRoutes
from app.generation import GenerationError, GenerationRecipeV9
from app.rag_methods import MethodConfigV1, MethodResultV1, RerankedMethodConfigV1, RunInputV1
from app.rag_runner import run_methods
from app.rewrite_cache import RewriteCache
from tests.test_collection_answering import FixtureReranker
from tests.test_collection_review import review_context as review_context
from tests.test_external_pilot import configured


class Encoder:
    fixture_only = True

    def __init__(self, document, query):
        self.document, self.query, self.calls = document, query, 0

    def recipe(self, role):
        return self.document if role == "document" else self.query

    def encode(self, texts, *, role):
        self.calls += 1
        assert role == "query" and len(texts) == 1
        vector = np.zeros((1, 384), dtype=np.float32)
        vector[0, 0] = 1
        return vector

    def count(self, text, *, role):
        return 10


@pytest.fixture
def setup_runner(review_context, tmp_path):
    collection, lexical, _, _, params = review_context
    kwargs = dict(tenant_id=collection.manifest.tenant_id, owner_id="fixture-owner", run_id="fixture-collection",
        recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64),
        cache=AnswerCache(tmp_path / "answers", namespace="FIXTURE"),
        routes=ExternalRoutes(account_model_metadata_sha256="0" * 64),
        encoder=Encoder(params["document_recipe"], params["query_recipe"]),
        embedding_cache=EmbeddingCache(tmp_path / "embeddings", namespace="FIXTURE"),
        rewrite_cache=RewriteCache(tmp_path / "rewrite", namespace="FIXTURE"))
    return collection, lexical, kwargs


def fixture_gateway(kwargs):
    gateway = configured()
    gateway.ledger.initialize_run("kg-r72", kwargs["tenant_id"], kwargs["run_id"], cap_micro_usd=500_000)
    return gateway


def request(**changes):
    return RunInputV1(original_question="Service B", idempotency_key="fixture-run-1", **changes)


def install_provider(gateway, observed, *, invalid=False):
    def respond(provider, method, path, body):
        gateway.provider.calls += 1
        value = json.loads(body)
        system, prompt = value["messages"]
        prompt = json.loads(prompt["content"])
        observed.append((system["content"], prompt))
        if "untrusted_sources" not in prompt:
            raw = {"search_query": "Service B"}
        else:
            first = prompt["untrusted_sources"][0]
            raw = {"answer_status": "ANSWERED", "reason": "NONE", "claims": [{
                "text": "Dịch vụ B." if "Vietnamese" in system["content"] else "Synthetic Service B.",
                "evidence": [{"source_id": first["source_id"],
                              "span_id": "s8p99" if invalid else first["spans"][0]["span_id"]}]}]}
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": json.dumps(raw)}}]}
    gateway.provider.request = respond


def test_retrieval_compare_encodes_once_and_never_generates(setup_runner):
    collection, lexical, kwargs = setup_runner
    result = run_methods(collection, lexical, request(), **kwargs)
    assert result["status"] == "COMPLETED" and result["quality_metrics"] is None
    assert kwargs["encoder"].calls == 1
    assert result["new_known_cost_micro_usd"] == 0
    assert [row["effective_method"] for row in result["methods"]] == ["bm25", "vector", "hybrid"]
    assert all(row["status"] == "RETRIEVED" and row["response"] is None for row in result["methods"])
    assert all(len(row["context_chunk_ids"]) <= 8 and row["context_proxy_tokens"] <= 4096 for row in result["methods"])
    for row in result["methods"]:
        restored = MethodResultV1.model_validate_json(json.dumps(row))
        assert restored.model_dump(mode="json") == row
        assert [source["chunk_id"] for source in row["context_sources"]] == row["context_chunk_ids"]
        for source in row["context_sources"]:
            shard = collection.get_shard(source["shard_id"])
            doc = shard.documents[source["document_version_id"]]
            assert doc.normalized_text[source["start_char"]:source["end_char"]] == source["text"]
    replay = run_methods(collection, lexical, request(), **kwargs)
    assert kwargs["encoder"].calls == 1 and replay["query_embedding_cache_hit"]


def test_bm25_only_needs_no_encoder_or_embedding_cache(setup_runner):
    collection, lexical, kwargs = setup_runner
    kwargs.update(encoder=None, embedding_cache=None)
    report = run_methods(collection, lexical, request(method_ids=("bm25",)), **kwargs)
    assert report["status"] == "COMPLETED" and report["query_embedding_cache_hit"] is None
    assert report["shared_trace"] == []
    assert report["methods"][0]["config"]["query_recipe_sha256"] is None


def test_retrieval_only_rejects_gateway_before_dispatch(setup_runner):
    collection, lexical, kwargs = setup_runner
    gateway = fixture_gateway(kwargs)
    with pytest.raises(GenerationError, match="DISALLOWS_GATEWAY"):
        run_methods(collection, lexical, request(), gateway=gateway, **kwargs)
    assert gateway.provider.calls == 0


def test_answer_cache_only_plans_all_methods_and_paid_results_replay_offline(setup_runner):
    collection, lexical, kwargs = setup_runner
    item = request(run_kind="answer")
    planned = run_methods(collection, lexical, item, **kwargs)
    assert {row["status"] for row in planned["methods"]} == {"PLANNED_NO_MODEL_CALL"}
    gateway, observed = fixture_gateway(kwargs), []
    install_provider(gateway, observed)
    fresh = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert fresh["status"] == "COMPLETED" and gateway.provider.calls == 3
    assert fresh["new_known_cost_micro_usd"] > 0 and fresh["cost_complete"]
    for row in fresh["methods"]:
        assert row["response"]["mode_used"] == row["effective_method"]
        assert row["response"]["retrieval_backend"] == "collection_numpy_global_bm25"
        assert not row["response"]["used_graph"] and row["semantic_review"] == "NOT_REVIEWED"
        assert row["response"]["citations"]
    replay = run_methods(collection, lexical, item, **kwargs)
    assert replay["new_known_cost_micro_usd"] == 0 and gateway.provider.calls == 3
    assert all(row["cache_hit"] for row in replay["methods"])
    assert [r["response"] for r in fresh["methods"]] == [r["response"] for r in replay["methods"]]
    from scripts.replay_collection_run import assert_replay

    assert_replay(fresh, replay)
    changed = json.loads(json.dumps(replay))
    changed["methods"][0]["response"]["answer"] = "Tampered reply"
    with pytest.raises(ValueError, match="REPLAY_METHOD_MISMATCH"):
        assert_replay(fresh, changed)


def test_vi_rewrite_shared_once_original_question_kept_and_replay_zero_calls(setup_runner):
    collection, lexical, kwargs = setup_runner
    item = RunInputV1(original_question="Dịch vụ B làm gì?", language="vi", run_kind="answer",
                      idempotency_key="fixture-vi")
    gateway, observed = fixture_gateway(kwargs), []
    install_provider(gateway, observed)
    report = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert report["status"] == "COMPLETED" and gateway.provider.calls == 4
    assert report["search_query"] == "Service B" and report["query_transform"] == "CACHED_EN_REWRITE_V1"
    assert kwargs["encoder"].calls == 1
    assert len([p for _, p in observed if "untrusted_sources" not in p]) == 1
    assert all(p["untrusted_question"] == item.original_question for _, p in observed)
    assert all(row["response"]["answer_language"] == "vi" for row in report["methods"])
    assert report["new_known_cost_micro_usd"] == report["shared_cost_micro_usd"] + sum(
        row["new_cost_micro_usd"] for row in report["methods"])
    replay = run_methods(collection, lexical, item, **kwargs)
    assert replay["status"] == "COMPLETED" and replay["rewrite_cache_hit"]
    assert replay["new_known_cost_micro_usd"] == 0 and kwargs["encoder"].calls == 1


def test_vi_retrieval_is_labeled_direct_diagnostic_not_multilingual_claim(setup_runner):
    collection, lexical, kwargs = setup_runner
    item = RunInputV1(original_question="Dịch vụ B", idempotency_key="direct-vi")
    report = run_methods(collection, lexical, item, **kwargs)
    assert report["query_transform"] == "NONE" and report["warnings"]
    assert report["rewrite_receipt"] is None and report["new_known_cost_micro_usd"] == 0


def test_idempotency_rejects_changed_question_before_provider(setup_runner):
    collection, lexical, kwargs = setup_runner
    run_methods(collection, lexical, request(), **kwargs)
    gateway = fixture_gateway(kwargs)
    with pytest.raises(GenerationError, match="IDEMPOTENCY_CONFLICT"):
        run_methods(collection, lexical, RunInputV1(original_question="changed", idempotency_key="fixture-run-1"),
                    **kwargs)
    assert gateway.provider.calls == 0


def test_invalid_generation_stays_failed_without_retry_even_with_gateway(setup_runner):
    collection, lexical, kwargs = setup_runner
    item = request(run_kind="answer", method_ids=("bm25",))
    gateway, observed = fixture_gateway(kwargs), []
    install_provider(gateway, observed, invalid=True)
    first = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert first["methods"][0]["status"] == "FAILED" and not first["cost_complete"]
    second = run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert second["methods"][0]["status"] == "FAILED" and gateway.provider.calls == 1


def test_invalid_encoder_rejected_before_any_embedding_or_generation(setup_runner):
    collection, lexical, kwargs = setup_runner
    kwargs["encoder"].document = kwargs["encoder"].document.model_copy(update={"artifact_manifest_sha256": "b" * 64})
    with pytest.raises(GenerationError, match="DOCUMENT_ENCODER_MISMATCH"):
        run_methods(collection, lexical, request(), **kwargs)
    assert kwargs["encoder"].calls == 0


def test_wrong_index_is_rejected_before_shared_vi_rewrite(setup_runner):
    from types import SimpleNamespace

    collection, lexical, kwargs = setup_runner
    wrong = SimpleNamespace(manifest=lexical.manifest.model_copy(update={"collection_manifest_sha256": "f" * 64}))
    item = RunInputV1(original_question="Dịch vụ B", run_kind="answer", idempotency_key="wrong-index")
    gateway = fixture_gateway(kwargs)
    with pytest.raises(GenerationError, match="LEXICAL_BINDING_MISMATCH"):
        run_methods(collection, wrong, item, gateway=gateway, **kwargs)
    assert gateway.provider.calls == 0 and kwargs["encoder"].calls == 0


@pytest.mark.parametrize("change", [{"method_ids": ["graph_1hop"]}, {"method_ids": ["bm25", "bm25"]},
                                    {"original_question": " "}, {"gold_answer": "leak"}])
def test_run_input_rejects_unavailable_methods_duplicates_empty_and_gold(change):
    payload = {"original_question": "Service B", "idempotency_key": "fixture"} | change
    with pytest.raises(ValidationError):
        RunInputV1.model_validate_json(json.dumps(payload))


def test_r_method_has_distinct_config_response_same_pool_and_replay_does_not_dispatch(setup_runner):
    collection, lexical, kwargs = setup_runner
    scorer = FixtureReranker()
    item = request(run_kind="answer", method_ids=("hybrid", "hybrid_reranked"))
    gateway, observed = fixture_gateway(kwargs), []
    install_provider(gateway, observed)
    report = run_methods(collection, lexical, item, reranker=scorer, gateway=gateway, **kwargs)
    assert report["status"] == "COMPLETED" and gateway.provider.calls == 2 and kwargs["encoder"].calls == 1
    baseline, reranked = report["methods"]
    assert baseline["config"]["schema_version"] == 1 and "reranker_recipe" not in baseline["config"]
    assert "rerank_trace" not in baseline
    assert reranked["config"]["schema_version"] == 2
    assert reranked["config"]["reranker_recipe_sha256"] == scorer.recipe.fingerprint
    assert reranked["response"]["mode_used"] == reranked["response"]["mode_requested"] == "hybrid_reranked"
    assert not reranked["response"]["used_graph"]
    assert reranked["rerank_trace"]["baseline_candidate_ids"] == baseline["ranked_chunk_ids"]
    assert set(reranked["ranked_chunk_ids"]) == set(baseline["ranked_chunk_ids"])
    assert report["local_reranker_inference_calls"] == 1
    assert sum(event["stage"] == "reranking" for event in reranked["trace"]) == 1
    parsed = MethodResultV1.model_validate_json(json.dumps(reranked))
    assert isinstance(parsed.config, RerankedMethodConfigV1) and parsed.model_dump(mode="json") == reranked
    assert observed[0][0] == observed[1][0]  # Same V9 system prompt, different ranked source data.
    assert all(citation["chunk_id"] in reranked["context_chunk_ids"] for citation in reranked["response"]["citations"])
    replay = run_methods(collection, lexical, item, reranker=scorer, **kwargs)
    assert gateway.provider.calls == 2 and replay["new_known_cost_micro_usd"] == 0
    assert all(row["cache_hit"] for row in replay["methods"])
    assert [row["response"] for row in report["methods"]] == [row["response"] for row in replay["methods"]]
    assert replay["local_reranker_inference_calls"] == 1  # Honest local recomputation, not a score-cache claim.


def test_baseline_config_and_idempotency_payload_remain_legacy_bytes_when_no_r_is_requested(setup_runner):
    collection, lexical, kwargs = setup_runner
    item = request()
    report = run_methods(collection, lexical, item, reranker=FixtureReranker(), **kwargs)
    assert report["kind"] == "LOCAL_COLLECTION_LDH_COMPARISON" and "reranker_recipe_sha256" not in report
    for row in report["methods"]:
        config = row["config"]
        assert config["schema_version"] == 1 and config["revision"] == "collection-ldh-v1"
        assert MethodConfigV1.model_validate_json(json.dumps(config)).fingerprint == row["config_sha256"]
        assert row["config_sha256"] == sha256_bytes(canonical_json(config))
        assert "rerank_trace" not in row
    legacy_payload = {"input": item.model_dump(mode="json"), "collection_sha": collection.manifest_sha256,
        "lexical_version": lexical.manifest.index_version, "recipe": kwargs["recipe"].fingerprint,
        "query_recipe": kwargs["encoder"].query.fingerprint, "run_id": kwargs["run_id"]}
    expected_identity = stable_id("ldh", tenant=kwargs["tenant_id"], owner=kwargs["owner_id"], key=item.idempotency_key)
    assert report["run_identity"] == expected_identity
    with sqlite3.connect(kwargs["cache"].path.with_name("method_runs.sqlite3")) as db:
        assert db.execute("SELECT payload_sha FROM runs WHERE key=?", (expected_identity,)).fetchone()[0] == sha256_bytes(canonical_json(legacy_payload))


@pytest.mark.parametrize("language", ["en", "vi"])
def test_missing_reranker_blocks_mixed_answer_request_before_shared_or_paid_work(setup_runner, language):
    collection, lexical, kwargs = setup_runner
    item = RunInputV1(original_question="Dịch vụ B làm gì?" if language == "vi" else "Service B",
        language=language, run_kind="answer", method_ids=("hybrid", "hybrid_reranked"), idempotency_key="missing-r")
    gateway = fixture_gateway(kwargs)
    with pytest.raises(GenerationError, match="VERIFIED_RERANKER_REQUIRED"):
        run_methods(collection, lexical, item, gateway=gateway, **kwargs)
    assert gateway.provider.calls == kwargs["encoder"].calls == 0
    assert not kwargs["cache"].path.with_name("method_runs.sqlite3").exists()


def test_synthetic_reranker_is_never_accepted_for_a_live_namespace(setup_runner, tmp_path):
    collection, lexical, kwargs = setup_runner
    current = SimpleNamespace(manifest=collection.manifest.model_copy(update={"purpose": "DEMO"}),
                              manifest_sha256=collection.manifest_sha256)
    kwargs["cache"] = AnswerCache(tmp_path / "live", namespace="LIVE")
    with pytest.raises(GenerationError, match="RERANKER_NAMESPACE_MISMATCH"):
        run_methods(current, lexical, request(method_ids=("hybrid_reranked",)), reranker=FixtureReranker(), **kwargs)
    assert kwargs["encoder"].calls == 0


def test_changed_reranker_recipe_conflicts_before_embedding_rewrite_or_dispatch(setup_runner):
    collection, lexical, kwargs = setup_runner
    scorer = FixtureReranker()
    item = request(method_ids=("hybrid_reranked",))
    run_methods(collection, lexical, item, reranker=scorer, **kwargs)
    calls = kwargs["encoder"].calls
    scorer.recipe = scorer.recipe.model_copy(update={"artifact_manifest_sha256": "b" * 64})
    with pytest.raises(GenerationError, match="IDEMPOTENCY_CONFLICT"):
        run_methods(collection, lexical, item, reranker=scorer, **kwargs)
    assert kwargs["encoder"].calls == calls and scorer.inference_calls == 1


def test_invalid_reranker_recipe_is_rejected_before_any_shared_work(setup_runner):
    collection, lexical, kwargs = setup_runner
    scorer = FixtureReranker()
    scorer.recipe = scorer.recipe.model_copy(update={"max_candidates": 41})
    gateway = fixture_gateway(kwargs)
    with pytest.raises(GenerationError, match="RERANKER_RECIPE_INVALID"):
        run_methods(collection, lexical, request(run_kind="answer", method_ids=("hybrid", "hybrid_reranked")),
                    reranker=scorer, gateway=gateway, **kwargs)
    assert scorer.inference_calls == kwargs["encoder"].calls == gateway.provider.calls == 0


def test_reranker_runtime_error_does_not_generate_or_fall_back_to_hybrid(setup_runner):
    collection, lexical, kwargs = setup_runner
    scorer = FixtureReranker()
    scorer.fail = True
    gateway = fixture_gateway(kwargs)
    report = run_methods(collection, lexical, request(run_kind="answer", method_ids=("hybrid_reranked", "hybrid")),
                         reranker=scorer, gateway=gateway, **kwargs)
    assert report["status"] == "PARTIAL" and gateway.provider.calls == 0
    failed, blocked = report["methods"]
    assert failed["status"] == "FAILED" and failed["effective_method"] == "hybrid_reranked"
    assert failed["response"] is None and blocked["error_category"] == "BLOCKED_AFTER_PEER_FAILURE"
    assert not any(event["stage"] == "generation_and_validation" for event in failed["trace"])


def test_hybrid_generation_failure_in_h_r_compare_blocks_later_r_dispatch(setup_runner):
    collection, lexical, kwargs = setup_runner
    scorer, observed = FixtureReranker(), []
    gateway = fixture_gateway(kwargs)
    install_provider(gateway, observed, invalid=True)
    report = run_methods(collection, lexical, request(run_kind="answer", method_ids=("hybrid", "hybrid_reranked")),
                         reranker=scorer, gateway=gateway, **kwargs)
    assert gateway.provider.calls == 1 and not report["cost_complete"]
    assert report["methods"][0]["status"] == "FAILED"
    assert report["methods"][1]["error_category"] == "BLOCKED_AFTER_PEER_FAILURE"
    assert scorer.inference_calls == 0 and report["methods"][1]["new_cost_micro_usd"] == 0


def test_r_bad_generation_receipt_stays_failed_without_retry(setup_runner):
    collection, lexical, kwargs = setup_runner
    scorer, observed = FixtureReranker(), []
    item = request(run_kind="answer", method_ids=("hybrid_reranked",))
    gateway = fixture_gateway(kwargs)
    install_provider(gateway, observed, invalid=True)
    first = run_methods(collection, lexical, item, reranker=scorer, gateway=gateway, **kwargs)
    second = run_methods(collection, lexical, item, reranker=scorer, gateway=gateway, **kwargs)
    assert first["methods"][0]["status"] == second["methods"][0]["status"] == "FAILED"
    assert gateway.provider.calls == 1


def test_r_retrieval_only_rejects_gateway_and_keeps_default_method_count_bound(setup_runner):
    collection, lexical, kwargs = setup_runner
    gateway = fixture_gateway(kwargs)
    with pytest.raises(GenerationError, match="DISALLOWS_GATEWAY"):
        run_methods(collection, lexical, request(method_ids=("hybrid_reranked",)), gateway=gateway,
                    reranker=FixtureReranker(), **kwargs)
    with pytest.raises(ValidationError):
        request(method_ids=("bm25", "vector", "hybrid", "hybrid_reranked"))
    assert gateway.provider.calls == kwargs["encoder"].calls == 0
