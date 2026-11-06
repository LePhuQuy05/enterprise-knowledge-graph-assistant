from types import SimpleNamespace

import pytest

from app.benchmark_queries import (
    FailedSharedQueryRow,
    QueryInput,
    SharedQueryBundle,
    query_context,
    validate_shared_bundle,
    validate_shared_row,
)
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.external_gateway import ExternalRoutes
from app.model_gateway import ModelResult
from app.query_planning import QueryPlanCache
from app.rewrite_cache import RewriteCache
from scripts.prepare_benchmark_queries import BoundedQueryGateway, prepare_row


class Gateway:
    def __init__(self, routes):
        self.routes, self.provider, self.calls = routes, SimpleNamespace(live=False), []

    def generate(self, context, **kwargs):
        self.calls.append((context, kwargs))
        raw = {"search_query": "Why can an IAM role not call kms:Decrypt?"} if context.stage == "REWRITE" else {
            "subqueries": ["AWS KMS key policy delegation to IAM policies"]}
        return ModelResult(operation_id=stable_id("op", **context.model_dump(mode="json")),
            usage=TokenUsage(input_tokens=100, output_tokens=40), actual_micro_usd=50,
            text=canonical_json(raw).decode(), stop_reason="end_turn")


@pytest.fixture
def prepared(tmp_path):
    routes = ExternalRoutes(account_model_metadata_sha256="a" * 64)
    gateway = Gateway(routes)
    task = QueryInput(scenario_id="aws100-007", language="vi", question="Vì sao IAM role không gọi được kms:Decrypt?")
    kwargs = dict(routes=routes, rewrite_cache=RewriteCache(tmp_path, namespace="FIXTURE"),
                  plan_cache=QueryPlanCache(tmp_path, namespace="FIXTURE"), gateway=gateway)
    row = prepare_row(task, "b" * 64, **kwargs)
    return task, row, gateway, kwargs


def test_shared_vi_row_replays_two_receipts_without_provider_or_expected_facts(prepared):
    task, row, gateway, kwargs = prepared
    assert len(gateway.calls) == 2
    assert prepare_row(task, "b" * 64, **(kwargs | {"gateway": None})) == row
    assert len(gateway.calls) == 2
    for _, call in gateway.calls:
        assert "reference_answer" not in call["prompt"] and "evidence_groups" not in call["prompt"]
    bundle = SharedQueryBundle(protocol_sha256="b" * 64, model_metadata_sha256="a" * 64, rows=(row,))
    assert validate_shared_bundle(bundle, (task,), protocol_sha256="b" * 64, metadata_sha256="a" * 64) == bundle


@pytest.mark.parametrize("fault", ["question", "search", "plan", "operation", "recipe"])
def test_tampered_shared_input_or_receipt_is_rejected(prepared, fault):
    task, row, _, _ = prepared
    changes = {
        "question": {"question_sha256": "c" * 64},
        "search": {"search_query": "A hand-written replacement search"},
        "plan": {"subqueries": ("A fake expected fact inserted after retrieval",)},
        "operation": {"planner_receipt": row.planner_receipt.model_copy(update={"operation_id": "foreign-operation"})},
        "recipe": {"planner_recipe_sha256": "f" * 64},
    }[fault]
    with pytest.raises(ValueError):
        validate_shared_row(row.model_copy(update=changes), task, "b" * 64)


def test_en_queries_are_not_silently_rewritten(prepared):
    task, row, _, _ = prepared
    task = task.model_copy(update={"language": "en"})
    row = row.model_copy(update={"language": "en"})
    with pytest.raises(ValueError, match="EN_MUST_USE_ORIGINAL"):
        validate_shared_row(row, task, "b" * 64)


def test_gateway_cannot_answer_retry_or_dispatch_outside_the_frozen_input_set(prepared):
    task, _, raw, _ = prepared
    bounded = BoundedQueryGateway(raw, (task,), "b" * 64)
    ctx = query_context(task, "b" * 64, "PLAN_QUERY")
    bounded.generate(ctx, system="Fixture", prompt="Question only", max_output_tokens=256)
    with pytest.raises(ValueError, match="DISPATCH_SCOPE"):
        bounded.generate(ctx, system="Fixture", prompt="Question only", max_output_tokens=256)
    ctx = ctx.model_copy(update={"stage": "GENERATE"})
    with pytest.raises(ValueError, match="DISPATCH_SCOPE"):
        bounded.generate(ctx, system="Fixture", prompt="Question only", max_output_tokens=256)


def test_shared_bundle_requires_both_the_exact_inputs_and_receipted_question_identity(prepared):
    task, row, _, _ = prepared
    bundle = SharedQueryBundle(protocol_sha256="b" * 64, model_metadata_sha256="a" * 64, rows=(row,))
    with pytest.raises(ValueError, match="INPUT_SET"):
        validate_shared_bundle(bundle, (), protocol_sha256="b" * 64, metadata_sha256="a" * 64)
    assert row.question_sha256 == sha256_bytes(task.question.encode("utf-8"))


def test_settled_invalid_plan_is_retained_as_failure_and_needs_review_binding(prepared):
    task, row, _, _ = prepared
    raw = '{"subqueries":["DynamoDB query --consistent-read"]}'
    provider = row.planner_receipt.model_copy(update={"raw_output": raw, "raw_output_sha256": sha256_bytes(raw.encode()),
                                                    "raw_output_bytes": len(raw)})
    values = row.model_dump(mode="json") | {"preparation_status": "PLAN_FAILED", "subqueries": [],
        "planner_receipt": provider.model_dump(mode="json"), "failure_code": "QUERY_PLAN_OUTPUT_INVALID",
        "failure_review_sha256": "e" * 64}
    failed = FailedSharedQueryRow.model_validate_json(canonical_json(values))
    assert validate_shared_row(failed, task, "b" * 64) == failed
    bundle = SharedQueryBundle.model_validate_json(canonical_json({"protocol_sha256": "b" * 64,
        "model_metadata_sha256": "a" * 64, "rows": [failed.model_dump(mode="json")]}))
    assert bundle.rows[0].preparation_status == "PLAN_FAILED"
    constructed = SharedQueryBundle(protocol_sha256="b" * 64, model_metadata_sha256="a" * 64, rows=(failed,))
    assert constructed.model_dump(mode="json")["rows"][0]["failure_review_sha256"] == "e" * 64
    with pytest.raises(ValueError, match="REVIEW_REQUIRED"):
        validate_shared_bundle(bundle, (task,), protocol_sha256="b" * 64, metadata_sha256="a" * 64)
    review = {"status": "SETTLED_VALIDATION_FAILURE_RETAIN_NO_RETRY", "operation_id": provider.operation_id,
        "raw_output_sha256": provider.raw_output_sha256, "ledger_state": "SETTLED", "cost_micro_usd": provider.cost_micro_usd}
    validate_shared_bundle(bundle, (task,), protocol_sha256="b" * 64, metadata_sha256="a" * 64,
                           failure_reviews={"e" * 64: review})
    with pytest.raises(ValueError, match="CANNOT_BE_RELABELLED"):
        validate_shared_row(failed.model_copy(update={"planner_receipt": row.planner_receipt}), task, "b" * 64)
