from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.generation import GenerationRecipeV9, generation_request, parse_generated
from app.generation_output_protocol import (
    PROTOCOL,
    RequiredReasonGateway,
    RequiredReasonRecipe,
    protocol_system,
)
from app.model_gateway import CallContext
from tests.test_authorization_evidence import auth_case as auth_case
from tests.test_source_message_packing import pack


def test_recipe_is_distinct_and_binds_the_exact_format_protocol():
    old = GenerationRecipeV9(model_metadata_sha256="0" * 64)
    new = RequiredReasonRecipe(model_metadata_sha256="0" * 64)
    assert old.fingerprint != new.fingerprint and new.max_output_tokens == old.max_output_tokens == 800
    assert new.output_schema_version == old.output_schema_version
    with pytest.raises(ValidationError, match="OUTPUT_PROTOCOL_RECIPE_CHANGED"):
        RequiredReasonRecipe(model_metadata_sha256="0" * 64, output_protocol_sha256="f" * 64)


@pytest.mark.parametrize("language", ["en", "vi"])
def test_user_sources_and_base_system_are_identical_before_provider_adapter(auth_case, language):
    question = "Lambda cần quyền gì cho S3?" if language == "vi" else "Which permissions does Lambda need for S3?"
    old = GenerationRecipeV9(model_metadata_sha256="0" * 64)
    new = RequiredReasonRecipe(model_metadata_sha256="0" * 64)
    view, _, packing = pack(auth_case, with_path=True, question=question, answer_language=language)
    before = generation_request(question, view, packing.packed, "demo", old, answer_language=language)
    after = generation_request(question, view, packing.packed, "demo", new, answer_language=language)
    assert before == after
    assert protocol_system(after[0]) == before[0] + PROTOCOL


def test_provider_response_is_not_repaired_when_it_omits_reason():
    raw = '{"answer_status":"ANSWERED","claims":[]}'
    reply = SimpleNamespace(text=raw)

    class Base:
        provider = SimpleNamespace(live=False)
        routes = object()

        def generate(self, context, **kwargs):
            assert kwargs["system"] == "trusted system" + PROTOCOL
            assert kwargs["prompt"] == "untrusted source message"
            return reply

    gateway = RequiredReasonGateway(Base())
    result = gateway.generate(CallContext(tenant_id="demo", run_id="fixture", request_id="fixture", stage="GENERATE"),
        system="trusted system", prompt="untrusted source message", max_output_tokens=800)
    assert result is reply and result.text == raw
    with pytest.raises(ValueError, match="GENERATION_SCHEMA_INVALID"):
        parse_generated(result.text, "end_turn", RequiredReasonRecipe(model_metadata_sha256="0" * 64))


def test_format_adapter_cannot_modify_native_planning_calls():
    base = SimpleNamespace(provider=object(), routes=object())
    gateway = RequiredReasonGateway(base)
    with pytest.raises(ValueError, match="GENERATION_ONLY"):
        gateway.generate(CallContext(tenant_id="demo", run_id="fixture", request_id="fixture", stage="PLAN_QUERY"),
            system="s", prompt="p", max_output_tokens=256)


def test_structured_reason_validates_without_changing_claim_or_citation_limits():
    raw = '{"reason":"NONE","answer_status":"ANSWERED","claims":[{"text":"A supported fact.","evidence":[{"source_id":"s1","span_id":"s1p1"}]}]}'
    recipe = RequiredReasonRecipe(model_metadata_sha256="0" * 64)
    parsed = parse_generated(raw, "end_turn", recipe)
    assert parsed.reason == "NONE" and len(parsed.claims) == 1 and recipe.max_evidence_per_claim == 2
