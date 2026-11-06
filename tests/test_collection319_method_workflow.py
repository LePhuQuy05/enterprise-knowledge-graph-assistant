"""Preparation dispatch restrictions, not model quality or AWS inference."""

from types import SimpleNamespace

import pytest

from app.contracts import canonical_json, sha256_bytes
from scripts.check_collection319_methods import (
    PLANNER_SYSTEM_V3,
    QUESTIONS,
    REWRITE_SYSTEM_V3,
    STOP_MICRO_USD,
    ScopedGateway,
    context_for,
    preparation_payload,
    provider_body,
)


class FakeGateway:
    provider = SimpleNamespace(live=False)
    routes = SimpleNamespace()

    def __init__(self, *, cost=100, error=False):
        self.calls, self.cost, self.error = [], cost, error

    def generate(self, context, **kwargs):
        self.calls.append((context, kwargs))
        if self.error:
            raise RuntimeError("synthetic-dispatch-failure")
        return SimpleNamespace(actual_micro_usd=self.cost)


def request(gateway, intent="s3-permission-layers", language="en", stage="PLAN_QUERY"):
    question = QUESTIONS[intent][language]
    context = gateway.allow(intent, language, stage, question)
    return context, {"system": REWRITE_SYSTEM_V3 if stage == "REWRITE" else PLANNER_SYSTEM_V3,
        "prompt": preparation_payload(question, question, language), "max_output_tokens": 256}


def test_collection_manifest_is_in_new_preparation_identity():
    old = context_for("s3-permission-layers", "vi", "PLAN_QUERY", "a" * 64)
    new = context_for("s3-permission-layers", "vi", "PLAN_QUERY", "b" * 64)
    assert old.request_id != new.request_id
    assert old.run_id == new.run_id


def test_unknown_dispatch_does_not_reach_provider():
    base = FakeGateway()
    gateway = ScopedGateway(base, "a" * 64)
    context = context_for("s3-permission-layers", "en", "PLAN_QUERY", "a" * 64)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context, system="anything", prompt="anything", max_output_tokens=256)
    assert base.calls == []


@pytest.mark.parametrize("field,value", [("system", "changed"), ("prompt", "changed"), ("max_output_tokens", 800)])
def test_allowed_request_is_exact(field, value):
    base = FakeGateway()
    gateway = ScopedGateway(base, "a" * 64)
    context, params = request(gateway)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context, **(params | {field: value}))
    assert not base.calls


def test_same_operation_cannot_be_dispatched_twice():
    base = FakeGateway()
    gateway = ScopedGateway(base, "a" * 64)
    context, params = request(gateway)
    gateway.generate(context, **params)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context, **params)
    assert len(base.calls) == 1 and gateway.cost == 100


@pytest.mark.parametrize("stage,language", [("GENERATE", "en"), ("EXTRACT", "vi"), ("REWRITE", "en")])
def test_answer_extraction_and_english_rewrite_are_not_authorized(stage, language):
    gateway = ScopedGateway(FakeGateway(), "a" * 64)
    with pytest.raises(ValueError, match="PREPARATION_STAGE"):
        gateway.allow("s3-permission-layers", language, stage, QUESTIONS["s3-permission-layers"][language])


def test_original_question_cannot_be_replaced_by_a_new_prompt():
    gateway = ScopedGateway(FakeGateway(), "a" * 64)
    with pytest.raises(ValueError, match="ORIGINAL_QUERY"):
        gateway.allow("s3-permission-layers", "en", "PLAN_QUERY", "another question")


def test_dispatch_failure_halts_the_entire_native_arm():
    base = FakeGateway(error=True)
    gateway = ScopedGateway(base, "a" * 64)
    context, params = request(gateway)
    with pytest.raises(RuntimeError):
        gateway.generate(context, **params)
    context, params = request(gateway, "kms-cross-account")
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context, **params)
    assert gateway.halted and len(base.calls) == 1


def test_observed_cost_stop_prevents_next_call():
    base = FakeGateway(cost=STOP_MICRO_USD)
    gateway = ScopedGateway(base, "a" * 64)
    context, params = request(gateway)
    gateway.generate(context, **params)
    context, params = request(gateway, "kms-cross-account")
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context, **params)
    assert len(base.calls) == 1


def test_foreign_manifest_context_is_not_registered():
    base = FakeGateway()
    gateway = ScopedGateway(base, "a" * 64)
    _, params = request(gateway)
    foreign = context_for("s3-permission-layers", "en", "PLAN_QUERY", "b" * 64)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(foreign, **params)
    assert not base.calls


def test_payload_matches_gateway_json_with_output_256_and_untrusted_fields():
    import json

    question = QUESTIONS["s3-permission-layers"]["vi"]
    prompt = preparation_payload(question, question, "vi")
    body = json.loads(provider_body(REWRITE_SYSTEM_V3, prompt))
    assert body["max_tokens"] == 256 and body["temperature"] == 0
    assert body["thinking"] == {"type": "disabled"}
    assert json.loads(body["messages"][1]["content"])["untrusted_original_question"] == question
    assert provider_body(REWRITE_SYSTEM_V3, prompt) == canonical_json(body)
    assert len(sha256_bytes(provider_body(REWRITE_SYSTEM_V3, prompt))) == 64
