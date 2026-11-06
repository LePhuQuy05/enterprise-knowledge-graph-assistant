import json

import pytest
from pydantic import ValidationError

from app.answer_cache import AnswerCache
from app.generation import generation_request
from scripts.compare_citation_recipes import (
    RUN,
    ComparisonCase,
    execute_pair,
    main,
    pin_comparison_plan,
    prepare_pair,
)
from tests.test_collection_review import review_context as review_context
from tests.test_external_pilot import configured
from tests.test_rag_runner import install_provider


def make_pair(review_context):
    collection, lexical, _, vectors, _ = review_context
    gateway = configured()
    gateway.ledger.initialize_run("kg-r72", collection.manifest.tenant_id, RUN, cap_micro_usd=500_000)
    case = ComparisonCase(id="fixture-comparison", language="en", question="Service B", search_query="Service B",
                          review_focus=("DO_NOT_LEAK_REVIEW_CRITERIA_IN_PROMPT",))
    view, pair, plan = prepare_pair(collection, lexical, case, vectors["dev-01:en"], routes=gateway.routes)
    return view, pair, plan, gateway, collection


def test_paired_comparison_has_same_data_distinct_policy_and_operation_identity(review_context):
    view, pair, plan, _, collection = make_pair(review_context)
    assert pair[0].packed is pair[1].packed and pair[0].question == pair[1].question
    assert pair[0].context.request_id != pair[1].context.request_id
    assert pair[0].recipe.model_id == pair[1].recipe.model_id
    assert pair[0].recipe.max_output_tokens == pair[1].recipe.max_output_tokens == 800
    prompts = [generation_request(item.question, view, item.packed, collection.manifest.tenant_id, item.recipe) for item in pair]
    assert prompts[0][1] == prompts[1][1] and prompts[0][0] != prompts[1][0]
    assert all("DO_NOT_LEAK_REVIEW_CRITERIA" not in item for prompt in prompts for item in prompt)
    assert plan["variants"][0]["recipe_sha256"] != plan["variants"][1]["recipe_sha256"]


def test_pair_live_fixture_and_replay_use_distinct_durable_receipts(review_context, tmp_path):
    view, pair, _, gateway, _ = make_pair(review_context)
    seen = []
    install_provider(gateway, seen)
    cache = AnswerCache(tmp_path, namespace="FIXTURE")
    fresh = execute_pair(view, pair, cache=cache, language="en", gateway=gateway)
    replay = execute_pair(view, pair, cache=cache, language="en")
    assert gateway.provider.calls == 2
    assert {row["variant"] for row in fresh} == {"v9", "v10"}
    assert all(row["cache_hit"] and row["new_cost_micro_usd"] == 0 for row in replay)
    assert [row["response"] for row in fresh] == [row["response"] for row in replay]
    assert fresh[0]["receipt"]["operation_id"] != fresh[1]["receipt"]["operation_id"]


def test_failure_does_not_lose_completed_baseline_receipt_or_retry(review_context, tmp_path):
    view, pair, _, gateway, _ = make_pair(review_context)
    observed = []
    install_provider(gateway, observed)
    original = gateway.generate

    def fail_candidate(context, **kwargs):
        if context.request_id == pair[1].context.request_id:
            raise TimeoutError("synthetic uncertain candidate")
        return original(context, **kwargs)

    gateway.generate = fail_candidate
    saved = []
    cache = AnswerCache(tmp_path, namespace="FIXTURE")
    with pytest.raises(TimeoutError):
        execute_pair(view, pair, cache=cache, language="en", gateway=gateway, checkpoint=saved.append)
    assert len(saved) == 1 and saved[0]["variant"] == "v9" and saved[0]["receipt"]
    with pytest.raises(ValueError, match="RECONCILIATION_REQUIRED"):
        execute_pair(view, pair, cache=cache, language="en", gateway=gateway)
    assert gateway.provider.calls == 1


def test_empty_evidence_pair_abstains_without_transport(review_context, tmp_path):
    collection, lexical, _, _, _ = review_context
    gateway = configured()
    case = ComparisonCase(id="empty", language="en", question="Unknown fixture", search_query="qzxvplmunknown", mode="bm25")
    view, pair, _ = prepare_pair(collection, lexical, case, None, routes=gateway.routes)
    result = execute_pair(view, pair, cache=AnswerCache(tmp_path, namespace="FIXTURE"), language="en")
    assert all(row["deterministic_no_evidence"] and row["receipt"] is None
               and row["response"]["answer_status"] == "ABSTAINED" for row in result)
    assert gateway.provider.calls == 0


def test_case_rejects_gold_fields_and_unsupported_graph_mode():
    payload = {"id": "bad", "language": "en", "question": "Question", "search_query": "query", "gold_answer": "leak"}
    with pytest.raises(ValidationError):
        ComparisonCase.model_validate_json(json.dumps(payload))
    payload.pop("gold_answer")
    payload["mode"] = "graph_1hop"
    with pytest.raises(ValidationError):
        ComparisonCase.model_validate_json(json.dumps(payload))


def test_existing_live_receipt_cannot_be_overwritten_or_start_paid_work(tmp_path, monkeypatch):
    target = tmp_path / "live.json"
    original = b'{"dispatches_this_invocation": 9, "new_known_cost_micro_usd": 12546}\n'
    target.write_bytes(original)

    def forbidden_gateway():
        pytest.fail("Existing receipt must be checked before constructing a live gateway")

    monkeypatch.setattr("scripts.compare_citation_recipes.managed_gateway", forbidden_gateway)
    with pytest.raises(ValueError, match="LIVE_RECEIPT_IMMUTABLE"):
        main(["--execute", "--cases", str(tmp_path / "does-not-exist.json"), "--output-dir", str(tmp_path)])
    assert target.read_bytes() == original


def test_pinned_plan_reuses_first_timestamp_and_rejects_tampered_body(tmp_path):
    target = tmp_path / "plan.json"
    first = pin_comparison_plan(target, {"created_at": "first", "cases": [{"question": "original"}]})
    assert pin_comparison_plan(target, {"created_at": "later", "cases": [{"question": "original"}]}) == first
    original_bytes = target.read_bytes()
    with pytest.raises(ValueError, match="PLAN_IMMUTABLE"):
        pin_comparison_plan(target, {"created_at": "later", "cases": [{"question": "different"}]})
    assert target.read_bytes() == original_bytes
    first["cases"][0]["question"] = "tampered without updating stored digest"
    target.write_text(json.dumps(first), encoding="utf-8")
    tampered_bytes = target.read_bytes()
    with pytest.raises(ValueError, match="PLAN_IMMUTABLE"):
        pin_comparison_plan(target, {"created_at": "later", "cases": [{"question": "original"}]})
    assert target.read_bytes() == tampered_bytes
