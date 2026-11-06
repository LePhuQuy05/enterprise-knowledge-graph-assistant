"""The coverage contract exposes gaps; it never fabricates facts or repairs JSON."""

from types import SimpleNamespace

import pytest

from app.generation import GenerationRecipeV11, generation_request, parse_generated, resolve_answer
from app.generation_coverage_protocol import (
    PROTOCOL_SHA256,
    CoverageProtocolGateway,
    CoverageProtocolRecipe,
    coverage_contract_sha256,
    protocol_system,
)
from app.generation_output_protocol import RequiredReasonRecipe
from app.model_gateway import CallContext
from scripts.pilot_coverage_protocol import RUN, STOP, ScopedGateway
from tests.test_authorization_evidence import auth_case as auth_case
from tests.test_graph_source_allocation import prepare
from tests.test_reviewed_source_answer_canary import Base


def context(index=0):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=f"coverage-fixture-{index}", stage="GENERATE")


def test_recipe_distinctness_and_pinned_protocol(monkeypatch):
    recipe = CoverageProtocolRecipe(model_metadata_sha256="0" * 64)
    assert recipe.fingerprint != RequiredReasonRecipe(model_metadata_sha256="0" * 64).fingerprint
    assert isinstance(recipe, GenerationRecipeV11) and recipe.max_output_tokens == 800
    assert len(coverage_contract_sha256(recipe)) == 64 and len(PROTOCOL_SHA256) == 64
    monkeypatch.setattr("app.generation_coverage_protocol.PROTOCOL_SHA256", "f" * 64)
    with pytest.raises(ValueError, match="COVERAGE_OUTPUT_PROTOCOL_CHANGED"):
        coverage_contract_sha256(recipe)


@pytest.mark.parametrize("language", ["en", "vi"])
def test_source_user_message_unchanged_while_system_includes_gap_schema(auth_case, language):
    before = prepare(auth_case)
    recipe = CoverageProtocolRecipe(model_metadata_sha256="0" * 64)
    old = generation_request(before.prepared.question, before.view, before.prepared.packed, "demo",
        RequiredReasonRecipe(model_metadata_sha256="0" * 64), answer_language=language)
    new = generation_request(before.prepared.question, before.view, before.prepared.packed, "demo", recipe, answer_language=language)
    assert old[1] == new[1] and old[0] != new[0]
    assert "EXPLICIT LIMITATIONS CONTRACT" in new[0] and "CITATION ANTECEDENTS" in new[0]


def test_partial_assessment_is_separate_from_cited_facts_and_localized_by_server(auth_case):
    before = prepare(auth_case)
    p, recipe = before.prepared, CoverageProtocolRecipe(model_metadata_sha256="0" * 64)
    raw = '{"reason":"INSUFFICIENT_EVIDENCE","answer_status":"PARTIAL","claims":[{"text":"Synthetic source claim.","evidence":[{"source_id":"s1","span_id":"s1p1"}]}],"limitations":[{"code":"MISSING_SUPPLIED_EVIDENCE","question_part":"Phần còn thiếu của câu hỏi"}]}'
    parsed = parse_generated(raw, "end_turn", recipe)
    response = resolve_answer(parsed, snapshot=before.view, packed=p.packed, binding=p.binding, recipe=recipe, answer_language="vi")
    assert len(response.claims) == 1 and response.assessment.limitations[0].reported_by == "MODEL_REPORTED"
    assert response.assessment.limitations[0].question_part == "Phần còn thiếu của câu hỏi"
    assert response.assessment.limitations[0].message.startswith("Các đoạn tài liệu")
    assert "Phần còn thiếu" not in response.answer and len(response.citations) == 1


@pytest.mark.parametrize("raw", [
    '{"reason":"NONE","answer_status":"ANSWERED","claims":[{"text":"X","evidence":[{"source_id":"s1","span_id":"s1p1"}]}]}',
    '{"reason":"NONE","answer_status":"ANSWERED","claims":[{"text":"X","evidence":[{"source_id":"s1","span_id":"s1p1"}]}],"limitations":[{"code":"MISSING_SUPPLIED_EVIDENCE","question_part":"missing"}]}',
    '{"reason":"INSUFFICIENT_EVIDENCE","answer_status":"PARTIAL","claims":[{"text":"X","evidence":[{"source_id":"s1","span_id":"s1p1"}]}],"limitations":[]}',
])
def test_missing_or_status_inconsistent_gap_contract_is_rejected(raw):
    with pytest.raises(ValueError, match="GENERATION_SCHEMA_INVALID"):
        parse_generated(raw, "end_turn", CoverageProtocolRecipe(model_metadata_sha256="0" * 64))


def test_protocol_gateway_keeps_raw_output_unchanged():
    reply = SimpleNamespace(text='{"answer_status":"PARTIAL"}')

    class Provider:
        provider, routes = object(), object()

        def generate(self, ctx, **kwargs):
            assert kwargs["system"] == protocol_system("s") and kwargs["prompt"] == "p"
            return reply

    result = CoverageProtocolGateway(Provider()).generate(context(), system="s", prompt="p", max_output_tokens=800)
    assert result is reply
    with pytest.raises(ValueError, match="GENERATION_SCHEMA_INVALID"):
        parse_generated(result.text, "end_turn", CoverageProtocolRecipe(model_metadata_sha256="0" * 64))


def test_coverage_protocol_cannot_modify_query_planning_calls():
    gateway = CoverageProtocolGateway(SimpleNamespace(provider=object(), routes=object()))
    with pytest.raises(ValueError, match="GENERATION_ONLY"):
        gateway.generate(context().model_copy(update={"stage": "PLAN_QUERY"}), system="s", prompt="p", max_output_tokens=256)


@pytest.mark.parametrize("changed", ["prompt", "tokens", "repeat", "context"])
def test_dispatch_allows_only_frozen_eight_requests(changed):
    base, gateway = Base(), None
    gateway = ScopedGateway(base)
    gateway.allow(context(), "s", "p")
    if changed == "repeat":
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(1) if changed == "context" else context(), system="s",
            prompt="changed" if changed == "prompt" else "p", max_output_tokens=801 if changed == "tokens" else 800)
    assert base.calls == (1 if changed == "repeat" else 0)


@pytest.mark.parametrize("stop", ["cost", "count"])
def test_calls_and_observed_cost_have_hard_dispatch_guards(stop):
    base = Base()
    gateway = ScopedGateway(base)
    gateway.allow(context(), "s", "p")
    if stop == "cost":
        gateway.cost = STOP
    else:
        gateway.seen = {str(i) for i in range(8)}
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    assert base.calls == 0


def test_uncertain_failure_halts_without_second_call():
    base = Base(fail=True)
    gateway = ScopedGateway(base)
    for i in (0, 1):
        gateway.allow(context(i), "s", "p")
    with pytest.raises(ValueError):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(1), system="s", prompt="p", max_output_tokens=800)
    assert base.calls == 1 and gateway.halted
