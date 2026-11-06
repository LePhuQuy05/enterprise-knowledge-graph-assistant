"""Workflow guards and wiring use mocks; answer/replay traces use local fixtures."""

import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalRoutes
from app.generation import GenerationRecipeV9
from app.rag_methods import RunInputV1
from app.rag_runner import run_methods
from scripts import replay_collection_run, run_collection_methods
from tests.test_collection_answering import FixtureReranker
from tests.test_collection_review import review_context as review_context
from tests.test_rag_runner import (
    fixture_gateway,
    install_provider,
)
from tests.test_rag_runner import (
    setup_runner as setup_runner,
)


def forbid_loaders(monkeypatch, module):
    def forbidden(*args, **kwargs):
        pytest.fail("Guard must run before loading a model, collection or gateway")

    for name in ("BgeEncoder", "OnnxCrossEncoder", "load_collection", "run_methods"):
        monkeypatch.setattr(module, name, forbidden)
    if hasattr(module, "managed_gateway"):
        monkeypatch.setattr(module, "managed_gateway", forbidden)


def test_execute_cannot_overwrite_live_receipt_before_any_load_or_dispatch(tmp_path, monkeypatch):
    output = tmp_path / "live.json"
    original = b'{"fixture":"original immutable live receipt"}\n'
    output.write_bytes(original)
    forbid_loaders(monkeypatch, run_collection_methods)

    with pytest.raises(ValueError, match="METHOD_LIVE_RECEIPT_IMMUTABLE_USE_REPLAY"):
        run_collection_methods.main([
            "Fixture question", "--execute", "--run-kind", "answer",
            "--idempotency-key", "immutable-fixture", "--methods", "hybrid_reranked",
            "--pipeline", str(tmp_path / "missing-pipeline.json"), "--output", str(output),
        ])

    assert output.read_bytes() == original


def test_execute_retrieval_rejected_before_pipeline_model_or_gateway(tmp_path, monkeypatch):
    forbid_loaders(monkeypatch, run_collection_methods)
    output = tmp_path / "must-not-exist.json"
    with pytest.raises(SystemExit) as error:
        run_collection_methods.main([
            "Fixture question", "--execute", "--idempotency-key", "retrieval-fixture",
            "--pipeline", str(tmp_path / "missing.json"), "--output", str(output),
        ])
    assert error.value.code == 2
    assert not output.exists()


@pytest.mark.parametrize("alias", [False, True])
def test_replay_cannot_replace_original_even_through_resolved_alias(tmp_path, monkeypatch, alias):
    original = tmp_path / "receipt.json"
    raw = b'{"fixture":"original cached answer receipt"}\n'
    original.write_bytes(raw)
    output = tmp_path / "child" / ".." / original.name if alias else original
    forbid_loaders(monkeypatch, replay_collection_run)

    with pytest.raises(ValueError, match="REPLAY_CANNOT_OVERWRITE_ORIGINAL_RECEIPT"):
        replay_collection_run.main([
            str(original), "--pipeline", str(tmp_path / "missing.json"), "--output", str(output),
        ])

    assert original.read_bytes() == raw


def wire_cli_mocks(monkeypatch, tmp_path, module, *, collection=None):
    """Replace every expensive/external boundary; no model or AWS can be reached."""
    root = tmp_path / "mock-project"
    routes_path = root / "configs/models/external-direct-v1.json"
    routes_path.parent.mkdir(parents=True)
    routes = ExternalRoutes(account_model_metadata_sha256="0" * 64)
    routes_path.write_bytes(canonical_json(routes.model_dump(mode="json")))
    pipeline = tmp_path / "pipeline.json"
    pipeline.write_bytes(canonical_json({
        "collection_path": "fixture-collection.json",
        "lexical_manifest_path": "fixture-lexical.json",
    }))
    collection = collection or SimpleNamespace(
        manifest_sha256="a" * 64,
        manifest=SimpleNamespace(collection_version="fixture-collection"),
    )
    encoder = SimpleNamespace(
        inference_calls=4,
        recipe=lambda role: SimpleNamespace(model_id="BAAI/bge-small-en-v1.5"),
    )
    reranker = SimpleNamespace(inference_calls=7)
    gateway = SimpleNamespace(provider=SimpleNamespace(dispatch_attempts=11))
    lexical = object()
    mocks = {
        "BgeEncoder": Mock(return_value=encoder),
        "OnnxCrossEncoder": Mock(return_value=reranker),
        "load_collection": Mock(return_value=collection),
        "load_global_lexical_index": Mock(return_value=nullcontext(lexical)),
        "EmbeddingCache": Mock(return_value=object()),
        "AnswerCache": Mock(return_value=object()),
        "RewriteCache": Mock(return_value=object()),
    }
    if hasattr(module, "managed_gateway"):
        mocks["managed_gateway"] = Mock(return_value=gateway)
    for name, mock in mocks.items():
        monkeypatch.setattr(module, name, mock)
    monkeypatch.setattr(module, "ROOT", root)
    return SimpleNamespace(
        root=root, pipeline=pipeline, collection=collection, encoder=encoder,
        reranker=reranker, gateway=gateway, lexical=lexical, mocks=mocks,
    )


def completed_cli_report(request):
    return {
        "status": "COMPLETED", "new_known_cost_micro_usd": 0,
        "methods": [{
            "requested_method": method, "effective_method": method,
            "status": "COMPLETED" if request.run_kind == "answer" else "RETRIEVED",
            "cache_hit": False,
        } for method in request.method_ids],
    }


@pytest.mark.parametrize("execute", [False, True])
def test_default_cli_keeps_legacy_methods_and_does_not_load_reranker(tmp_path, monkeypatch, execute):
    env = wire_cli_mocks(monkeypatch, tmp_path, run_collection_methods)
    runner = Mock(side_effect=lambda collection, lexical, request, **kw: completed_cli_report(request))
    monkeypatch.setattr(run_collection_methods, "run_methods", runner)
    output = tmp_path / "default.json"
    argv = [
        "Fixture question", "--idempotency-key", "default-fixture",
        "--pipeline", str(env.pipeline), "--output", str(output),
    ]
    if execute:
        argv += ["--execute", "--run-kind", "answer"]

    assert run_collection_methods.main(argv) == 0

    call = runner.call_args
    assert call.args[:2] == (env.collection, env.lexical)
    assert call.args[2].method_ids == ("bm25", "vector", "hybrid")
    assert call.kwargs["reranker"] is None
    assert call.kwargs["encoder"] is env.encoder
    assert isinstance(call.kwargs["recipe"], GenerationRecipeV9)
    env.mocks["OnnxCrossEncoder"].assert_not_called()
    if execute:
        env.mocks["managed_gateway"].assert_called_once_with(prices_path=None)
        assert call.kwargs["gateway"] is env.gateway
    else:
        env.mocks["managed_gateway"].assert_not_called()
        assert call.kwargs["gateway"] is None
    stored = json.loads(output.read_bytes())
    assert stored["local_query_onnx_calls_this_invocation"] == 4
    assert "local_reranker_onnx_calls_this_invocation" not in stored
    assert stored["model_dispatch_attempts_this_invocation"] == 0


def test_explicit_r_cli_passes_verified_reranker_and_selected_price_sheet(tmp_path, monkeypatch):
    env = wire_cli_mocks(monkeypatch, tmp_path, run_collection_methods)

    def fake_runner(collection, lexical, request, **kwargs):
        assert kwargs["gateway"] is env.gateway
        env.gateway.provider.dispatch_attempts += 2  # Synthetic counter, never a real dispatch.
        return completed_cli_report(request)

    runner = Mock(side_effect=fake_runner)
    monkeypatch.setattr(run_collection_methods, "run_methods", runner)
    output, prices = tmp_path / "r-live.json", tmp_path / "reviewed-prices.json"
    assert run_collection_methods.main([
        "Fixture question", "--methods", "hybrid", "hybrid_reranked",
        "--run-kind", "answer", "--execute", "--price-sheet", str(prices),
        "--idempotency-key", "explicit-r-fixture", "--pipeline", str(env.pipeline),
        "--output", str(output),
    ]) == 0

    env.mocks["OnnxCrossEncoder"].assert_called_once_with(
        env.root / "data/models/ms-marco-MiniLM-L2-v2",
        env.root / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json",
    )
    env.mocks["managed_gateway"].assert_called_once_with(prices_path=prices)
    assert runner.call_args.args[2].method_ids == ("hybrid", "hybrid_reranked")
    assert runner.call_args.kwargs["reranker"] is env.reranker
    assert runner.call_args.kwargs["encoder"] is env.encoder
    stored = json.loads(output.read_bytes())
    assert stored["local_reranker_onnx_calls_this_invocation"] == 7
    assert stored["local_query_onnx_calls_this_invocation"] == 4
    assert stored["model_dispatch_attempts_this_invocation"] == 2


def test_bm25_only_cli_loads_neither_encoder_nor_reranker(tmp_path, monkeypatch):
    env = wire_cli_mocks(monkeypatch, tmp_path, run_collection_methods)
    runner = Mock(side_effect=lambda collection, lexical, request, **kw: completed_cli_report(request))
    monkeypatch.setattr(run_collection_methods, "run_methods", runner)
    output = tmp_path / "bm25.json"
    assert run_collection_methods.main([
        "Fixture question", "--methods", "bm25", "--idempotency-key", "bm25-fixture",
        "--pipeline", str(env.pipeline), "--output", str(output),
    ]) == 0
    for name in ("BgeEncoder", "OnnxCrossEncoder", "EmbeddingCache", "managed_gateway"):
        env.mocks[name].assert_not_called()
    assert runner.call_args.kwargs["encoder"] is None
    assert runner.call_args.kwargs["embedding_cache"] is None
    assert runner.call_args.kwargs["reranker"] is None
    assert json.loads(output.read_bytes())["local_query_onnx_calls_this_invocation"] == 0


@pytest.fixture
def actual_r_reports(setup_runner):
    collection, lexical, kwargs = setup_runner
    request = RunInputV1(
        original_question="Service B", language="en", run_kind="answer",
        method_ids=("hybrid", "hybrid_reranked"), idempotency_key="cli-replay-fixture",
    )
    scorer = FixtureReranker()
    gateway, observed = fixture_gateway(kwargs), []
    install_provider(gateway, observed)
    original = run_methods(collection, lexical, request, reranker=scorer, gateway=gateway, **kwargs)
    replay = run_methods(collection, lexical, request, reranker=scorer, gateway=None, **kwargs)
    assert original["status"] == replay["status"] == "COMPLETED"
    assert gateway.provider.calls == 2  # Fixture provider only; no network/API.
    baseline, reranked = original["methods"]
    trace = reranked["rerank_trace"]
    assert trace["baseline_candidate_ids"] == baseline["ranked_chunk_ids"]
    assert trace["ranked_chunk_ids"] == reranked["ranked_chunk_ids"]
    assert trace["scores"] and trace["windows"]
    return collection, original, replay


def test_replay_accepts_changed_rerank_duration_without_mutating_reports(actual_r_reports):
    _, original, replay = actual_r_reports
    snapshot = copy.deepcopy(original)
    modified = copy.deepcopy(replay)
    modified["methods"][1]["rerank_trace"]["duration_ms"] += 123.456

    replay_collection_run.assert_replay(original, modified)

    assert original == snapshot
    assert replay["new_known_cost_micro_usd"] == 0
    assert all(row["cache_hit"] for row in replay["methods"])


@pytest.mark.parametrize("damage", [
    "baseline_ids", "ranked_ids", "score", "score_original_rank", "window_offset",
    "window_tokens", "recipe", "extra_field",
])
def test_replay_duration_normalization_cannot_hide_rerank_trace_changes(actual_r_reports, damage):
    _, original, replay = actual_r_reports
    modified = copy.deepcopy(replay)
    trace = modified["methods"][1]["rerank_trace"]
    trace["duration_ms"] += 123.456
    if damage == "baseline_ids":
        trace["baseline_candidate_ids"][0] = "invented-source"
    elif damage == "ranked_ids":
        trace["ranked_chunk_ids"][0] = "invented-source"
    elif damage == "score":
        trace["scores"][0]["raw_logit"] += 0.125
    elif damage == "score_original_rank":
        trace["scores"][0]["original_rank"] += 1
    elif damage == "window_offset":
        trace["windows"][0]["end_char"] -= 1
    elif damage == "window_tokens":
        trace["windows"][0]["input_tokens"] += 1
    elif damage == "recipe":
        trace["recipe_sha256"] = "f" * 64
    else:
        trace["unexpected_field"] = "must not be normalized away"

    with pytest.raises(ValueError, match="COLLECTION_REPLAY_RERANK_TRACE_MISMATCH"):
        replay_collection_run.assert_replay(original, modified)


def test_replay_cli_loads_requested_r_but_cannot_construct_paid_gateway(
    actual_r_reports, tmp_path, monkeypatch,
):
    collection, original, replay = actual_r_reports
    env = wire_cli_mocks(monkeypatch, tmp_path, replay_collection_run, collection=collection)
    runner = Mock(return_value=copy.deepcopy(replay))
    monkeypatch.setattr(replay_collection_run, "run_methods", runner)
    source, output = tmp_path / "live.json", tmp_path / "replay.json"
    raw = canonical_json(original) + b"\n"
    source.write_bytes(raw)

    assert replay_collection_run.main([
        str(source), "--pipeline", str(env.pipeline), "--output", str(output),
    ]) == 0

    env.mocks["OnnxCrossEncoder"].assert_called_once()
    assert runner.call_args.kwargs["reranker"] is env.reranker
    assert runner.call_args.kwargs["gateway"] is None
    assert source.read_bytes() == raw
    stored = json.loads(output.read_bytes())
    assert stored["replay_verified_against_sha256"] == sha256_bytes(raw)
    assert stored["model_dispatch_attempts_this_invocation"] == 0
    assert stored["aws_api_calls_this_invocation"] == 0
    assert stored["local_reranker_onnx_calls_this_invocation"] == 7
    assert stored["local_query_onnx_calls_this_invocation"] == 4
    assert stored["new_known_cost_micro_usd"] == 0
