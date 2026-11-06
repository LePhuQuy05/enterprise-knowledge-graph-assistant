"""Never retry uncertain/failed cells or mask semantic changes during replay."""

import json
import sqlite3
from types import SimpleNamespace

import pytest

from app.answer_cache import AnswerCache, AnswerReceipt
from app.contracts import canonical_json, sha256_bytes
from app.cost import TokenUsage
from app.generation import GenerationRecipeV9
from scripts import continue_source_message_answers as continuation


def seed(root, *, state="FAILED", payload_present=True, corrupt=False):
    cache = AnswerCache(root)
    receipt = AnswerReceipt(namespace="LIVE", cache_key="fixture", operation_id="fixture-op", recipe_sha256="0" * 64,
        binding_sha256="1" * 64, raw_output='{"answer_status":"ANSWERED","claims":[]}', stop_reason="end_turn",
        usage=TokenUsage(input_tokens=100, output_tokens=10), original_cost_micro_usd=42)
    raw = receipt.model_dump_json().encode() if payload_present else None
    with sqlite3.connect(cache.path) as db:
        db.execute("INSERT INTO answers(namespace,cache_key,state,owner,intent,payload,payload_sha256) VALUES(?,?,?,?,?,?,?)",
            ("LIVE", "fixture", state, "fixture", canonical_json({"context": {"request_id": "fixture-request"}}),
                raw, "f" * 64 if corrupt else sha256_bytes(raw) if raw else None))
    return receipt


@pytest.mark.parametrize("state,payload,corrupt", [("IN_PROGRESS", False, False), ("FAILED", False, False),
    ("READY", True, True), ("FAILED", True, True)])
def test_uncertain_or_corrupt_cache_blocks_continuation(tmp_path, monkeypatch, state, payload, corrupt):
    seed(tmp_path, state=state, payload_present=payload, corrupt=corrupt)
    monkeypatch.setattr(continuation, "CACHE", tmp_path)
    with pytest.raises(ValueError, match="UNCERTAIN_CACHE"):
        continuation.cached_answers()


def test_durable_failed_receipt_is_read_without_changing_state_or_raw_output(tmp_path, monkeypatch):
    receipt = seed(tmp_path)
    monkeypatch.setattr(continuation, "CACHE", tmp_path)
    saved = continuation.cached_answers()
    assert saved["fixture-request"] == ("FAILED", receipt)
    assert continuation.cached_answers()["fixture-request"][0] == "FAILED"


def test_schema_error_stays_in_attempted_denominator_without_filling_reason():
    receipt = seed_receipt = AnswerReceipt(namespace="LIVE", cache_key="fixture", operation_id="fixture-op", recipe_sha256="0" * 64,
        binding_sha256="1" * 64, raw_output='{"answer_status":"ANSWERED","claims":[]}', stop_reason="end_turn",
        usage=TokenUsage(input_tokens=100, output_tokens=10), original_cost_micro_usd=42)
    case = {"id": "fixture", "intent": "fixture", "language": "en", "method": "vector", "outcome": {}, "trace": {},
        "prepared": SimpleNamespace(recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64))}
    failed = continuation.failure_record(case, receipt)
    assert failed["answer_status"] == "FAILED_VALIDATION" and failed["validation_error"] == "GENERATION_SCHEMA_INVALID"
    assert failed["included_in_attempted_method_denominator"] and not failed["repaired"] and not failed["retried"]
    assert failed["receipt"]["raw_output"] == seed_receipt.raw_output
    assert "reason" not in json.loads(failed["receipt"]["raw_output"])


def test_rerank_elapsed_time_is_only_observational_field_restored():
    old = {"reranking": [{"duration_ms": 100.0, "scores": [0.2], "ranked_chunk_ids": ["a"]}], "selection": ["a"]}
    new = {"reranking": [{"duration_ms": 90.0, "scores": [0.2], "ranked_chunk_ids": ["a"]}], "selection": ["a"]}
    assert continuation.restore_recorded_rerank_timings(new, old) == old
    assert new["reranking"][0]["duration_ms"] == 90.0
    new["reranking"][0]["scores"] = [0.3]
    assert continuation.restore_recorded_rerank_timings(new, old) != old
    new["selection"] = ["b"]
    assert continuation.restore_recorded_rerank_timings(new, old)["selection"] == ["b"]


def test_invalid_or_missing_rerank_timing_fails_closed():
    for new in ({"reranking": [{"scores": []}]}, {"reranking": [{"duration_ms": -1}]}):
        with pytest.raises(ValueError, match="INVALID_RERANK_TIMING"):
            continuation.restore_recorded_rerank_timings(new, {"reranking": [{"duration_ms": 50}]})
