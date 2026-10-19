import copy
import dataclasses
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from app.answer_cache import AnswerCache, AnswerCacheBlocked
from app.answer_pipeline import prepare_answer
from app.context import pack_context
from app.fixtures import fixture_vector
from app.generation import (
    AnswerBinding,
    GenerationError,
    GenerationRecipe,
    GenerationRecipeV2,
    GenerationRecipeV6,
    GenerationRecipeV9,
    authorized_spans,
    empty_answer,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.graph_retrieval import GraphIndex
from app.model_gateway import CallContext, UncertainModelCall
from tests.test_external_pilot import configured

QUESTION = "How does Service A use Service B?"
QUOTE = "Service A uses Service B."


def setup(loaded):
    retrieval = GraphIndex(loaded).retrieve(QUESTION, fixture_vector(QUESTION), hops=0)
    chunk = next(c for c in loaded.content.chunks if c.text == QUOTE)
    packed = pack_context(loaded, dataclasses.replace(retrieval, ranked_chunk_ids=(chunk.chunk_id,)))
    binding = AnswerBinding(request_id="answer-r1", tenant_id=loaded.manifest.tenant_id, owner_id="owner-a",
                            mode_requested="hybrid", mode_used="hybrid", used_graph=False)
    recipe = GenerationRecipe(model_metadata_sha256="0" * 64)
    return packed, binding, recipe


def raw_answer():
    return {"answer_status": "ANSWERED", "reason": "NONE", "claims": [
        {"text": "Service A uses Service B.", "evidence": [{"source_id": "s1", "quote": QUOTE}]}]}


def output(data=None):
    return parse_generated(json.dumps(raw_answer() if data is None else data), "end_turn")


def test_claims_render_answer_and_citations_resolve_from_pinned_source(loaded):
    packed, binding, _ = setup(loaded)
    result = resolve_answer(output(), snapshot=loaded, packed=packed, binding=binding)
    assert result.answer == "\n".join(c.text for c in result.claims)
    citation = result.citations[0]
    assert citation.quote == QUOTE and citation.source_uri == loaded.documents[citation.document_version_id].source_uri
    assert result.claims[0].citation_ids == (citation.citation_id,)


@pytest.mark.parametrize("kind", ["unknown_source", "invented_quote", "ambiguous_quote", "model_url", "empty_claim",
                                  "bad_status", "uncited_claim", "duplicate_key", "nan", "truncated", "fenced_json"])
def test_invalid_outputs_are_errors_not_normal_abstentions(loaded, kind):
    packed, binding, _ = setup(loaded)
    data = raw_answer()
    if kind == "unknown_source":
        data["claims"][0]["evidence"][0]["source_id"] = "s2"
    elif kind == "invented_quote":
        data["claims"][0]["evidence"][0]["quote"] = "Service A can decrypt everything."
    elif kind == "ambiguous_quote":
        data["claims"][0]["evidence"][0]["quote"] = "Service"
    elif kind == "model_url":
        data["claims"][0]["evidence"][0]["source_uri"] = "https://evil.invalid/"
    elif kind == "empty_claim":
        data["claims"][0]["text"] = " "
    elif kind == "bad_status":
        data["answer_status"] = "ABSTAINED"
    elif kind == "uncited_claim":
        data["claims"][0]["evidence"] = []
    with pytest.raises(GenerationError):
        if kind == "duplicate_key":
            parsed = parse_generated('{"answer_status":"ANSWERED","answer_status":"ABSTAINED"}', "end_turn")
        elif kind == "nan":
            parsed = parse_generated('{"claims":NaN}', "end_turn")
        elif kind == "truncated":
            parsed = parse_generated(json.dumps(data), "max_tokens")
        elif kind == "fenced_json":
            parsed = parse_generated("```json\n" + json.dumps(data) + "\n```", "end_turn")
        else:
            parsed = output(data)
        resolve_answer(parsed, snapshot=loaded, packed=packed, binding=binding)


def test_empty_context_abstains_without_gateway_operation_or_cache_claim(loaded, tmp_path):
    packed, binding, recipe = setup(loaded)
    empty = dataclasses.replace(packed, chunk_ids=(), source_ids=(), assertion_ids=(), serialized="{}", proxy_tokens=2)
    result = AnswerCache(tmp_path, namespace="FIXTURE").resolve(
        CallContext(tenant_id=binding.tenant_id, run_id="answer-test", request_id=binding.request_id, stage="GENERATE"),
        question=QUESTION, snapshot=loaded, packed=empty, binding=binding, recipe=recipe)
    assert result.deterministic_no_evidence and result.receipt is None and result.new_cost_micro_usd == 0
    assert result.response.answer_status == "ABSTAINED" and not result.response.claims


def test_partial_and_model_abstention_have_explicit_reasons(loaded):
    packed, binding, _ = setup(loaded)
    data = raw_answer() | {"answer_status": "PARTIAL", "reason": "INSUFFICIENT_EVIDENCE"}
    partial = resolve_answer(output(data), snapshot=loaded, packed=packed, binding=binding)
    assert partial.claims and "GENERATION_REASON_INSUFFICIENT_EVIDENCE" in partial.warnings
    abstained = resolve_answer(output({"answer_status": "ABSTAINED", "claims": [], "reason": "OUT_OF_SCOPE"}),
                               snapshot=loaded, packed=packed, binding=binding)
    assert not abstained.citations and abstained.answer_status == "ABSTAINED"


def test_prompt_uses_trusted_snapshot_not_forged_packed_text_or_graph_summary(loaded):
    packed, binding, recipe = setup(loaded)
    forged = dataclasses.replace(packed, serialized='{"untrusted_text":"OVERRIDE_SYSTEM_FAKE"}')
    system, prompt = generation_request(QUESTION, loaded, forged, binding.tenant_id, recipe)
    assert "OVERRIDE_SYSTEM_FAKE" not in prompt and QUOTE in prompt
    assert "untrusted_graph_assertions" not in prompt and "no tools" not in prompt
    assert "untrusted" in system
    request = json.loads(prompt)
    assert request["untrusted_question"] == QUESTION
    assert request["untrusted_sources"] == [{"source_id": "s1", "untrusted_text": QUOTE}]


def test_prompt_v2_is_distinct_and_does_not_relax_required_reason_or_change_legacy_prompt(loaded):
    packed, binding, v1 = setup(loaded)
    v2 = GenerationRecipeV2(model_metadata_sha256=v1.model_metadata_sha256)
    before, prompt1 = generation_request(QUESTION, loaded, packed, binding.tenant_id, v1)
    after, prompt2 = generation_request(QUESTION, loaded, packed, binding.tenant_id, v2)
    assert v2.fingerprint != v1.fingerprint and prompt1 == prompt2
    assert "MANDATORY OUTPUT KEYS" in after and "MANDATORY OUTPUT KEYS" not in before
    missing = raw_answer()
    del missing["reason"]
    with pytest.raises(GenerationError, match="SCHEMA_INVALID"):
        parse_generated(json.dumps(missing), "end_turn")


def test_foreign_tenant_and_invalid_context_ids_rejected(loaded):
    packed, binding, recipe = setup(loaded)
    with pytest.raises(GenerationError, match="TENANT"):
        generation_request(QUESTION, loaded, packed, "other-tenant", recipe)
    with pytest.raises(GenerationError, match="CONTEXT"):
        generation_request(QUESTION, loaded, dataclasses.replace(packed, chunk_ids=("unknown",)), binding.tenant_id, recipe)
    with pytest.raises(GenerationError):
        empty_answer(snapshot=loaded, packed=packed, binding=binding)


def fixture_gateway(loaded, *, raw=None, fail=False):
    gateway = configured(fail)
    gateway.ledger.initialize_run("kg-r72", loaded.manifest.tenant_id, "answer-test", cap_micro_usd=500_000)
    original = gateway.provider.request

    def request(*args):
        result = copy.deepcopy(original(*args))
        result["choices"][0]["message"]["content"] = json.dumps(raw_answer()) if raw is None else raw
        return result

    gateway.provider.request = request
    return gateway


def parameters(loaded):
    packed, binding, recipe = setup(loaded)
    context = CallContext(tenant_id=binding.tenant_id, run_id="answer-test", request_id=binding.request_id, stage="GENERATE")
    return context, dict(question=QUESTION, snapshot=loaded, packed=packed, binding=binding, recipe=recipe)


def test_settled_receipt_reopens_and_replays_without_calls_and_is_owner_scoped(loaded, tmp_path):
    gateway = fixture_gateway(loaded)
    context, params = parameters(loaded)
    first = AnswerCache(tmp_path, namespace="FIXTURE").resolve(context, gateway=gateway, **params)
    second = AnswerCache(tmp_path, namespace="FIXTURE").resolve(context, **params)
    assert not first.cache_hit and second.cache_hit and first.response == second.response
    assert first.new_cost_micro_usd > 0 and second.new_cost_micro_usd == 0 and gateway.provider.calls == 1
    other = params | {"binding": params["binding"].model_copy(update={"owner_id": "owner-b"})}
    with pytest.raises(GenerationError, match="MISS"):
        AnswerCache(tmp_path, namespace="FIXTURE").resolve(context, **other)
    assert gateway.provider.calls == 1


def test_invalid_settled_output_is_durable_failed_and_never_auto_repaired(loaded, tmp_path):
    gateway = fixture_gateway(loaded, raw="not valid JSON")
    context, params = parameters(loaded)
    cache = AnswerCache(tmp_path, namespace="FIXTURE")
    with pytest.raises(GenerationError):
        cache.resolve(context, gateway=gateway, **params)
    with sqlite3.connect(cache.path) as db:
        state, size = db.execute("SELECT state,length(payload) FROM answers").fetchone()
    assert state == "FAILED" and size > 0
    with pytest.raises(AnswerCacheBlocked):
        cache.resolve(context, gateway=gateway, **params)
    assert gateway.provider.calls == 1


def test_uncertain_call_keeps_reservation_and_failed_claim_no_retry(loaded, tmp_path):
    gateway = fixture_gateway(loaded, fail=True)
    context, params = parameters(loaded)
    cache = AnswerCache(tmp_path, namespace="FIXTURE")
    with pytest.raises(UncertainModelCall):
        cache.resolve(context, gateway=gateway, **params)
    counter = gateway.ledger.store.get(gateway.ledger.run_key("answer-test"))
    assert counter["reserved_micro_usd"] > 0
    with pytest.raises(AnswerCacheBlocked):
        cache.resolve(context, gateway=gateway, **params)
    assert gateway.provider.calls == 1


def test_corrupt_cache_and_live_fixture_namespace_mismatch_fail_before_dispatch(loaded, tmp_path):
    gateway = fixture_gateway(loaded)
    context, params = parameters(loaded)
    with pytest.raises(GenerationError, match="GATEWAY"):
        AnswerCache(tmp_path / "live", namespace="LIVE").resolve(context, gateway=gateway, **params)
    assert gateway.provider.calls == 0
    cache = AnswerCache(tmp_path / "fixture", namespace="FIXTURE")
    cache.resolve(context, gateway=gateway, **params)
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE answers SET payload_sha256=?", ("0" * 64,))
    with pytest.raises(GenerationError, match="CHECKSUM"):
        cache.resolve(context, gateway=gateway, **params)
    assert gateway.provider.calls == 1


def test_request_identity_binds_owner_question_snapshot_mode_and_recipe(loaded):
    recipe = GenerationRecipe(model_metadata_sha256="0" * 64)
    kwargs = dict(tenant_id=loaded.manifest.tenant_id, owner_id="owner-a", run_id="answer-test", recipe=recipe)
    first = prepare_answer(loaded, QUESTION, fixture_vector(QUESTION), **kwargs)
    same = prepare_answer(loaded, "  " + QUESTION + "  ", fixture_vector(QUESTION), **kwargs)
    other_owner = prepare_answer(loaded, QUESTION, fixture_vector(QUESTION), **(kwargs | {"owner_id": "owner-b"}))
    other_mode = prepare_answer(loaded, QUESTION, fixture_vector(QUESTION), mode="vector", **kwargs)
    assert first.context.request_id == same.context.request_id
    assert len({first.context.request_id, other_owner.context.request_id, other_mode.context.request_id}) == 3


def test_unreviewed_graph_not_exposed_as_default_answer_backend(loaded):
    with pytest.raises(GenerationError, match="QUALITY_APPROVED"):
        prepare_answer(loaded, QUESTION, fixture_vector(QUESTION), tenant_id=loaded.manifest.tenant_id,
            owner_id="owner-a", run_id="answer-test", recipe=GenerationRecipe(model_metadata_sha256="0" * 64), mode="graph_1hop")


def test_concurrent_request_is_claimed_once_before_provider_dispatch(loaded, tmp_path):
    gateway = fixture_gateway(loaded)
    context, params = parameters(loaded)
    entered, release = threading.Event(), threading.Event()
    original = gateway.provider.request

    def held_request(*args):
        entered.set()
        if not release.wait(5):
            raise RuntimeError("FIXTURE_GATE_TIMEOUT")
        return original(*args)

    gateway.provider.request = held_request
    cache = AnswerCache(tmp_path, namespace="FIXTURE")
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(cache.resolve, context, gateway=gateway, **params)
        try:
            assert entered.wait(5)
            with pytest.raises(AnswerCacheBlocked):
                AnswerCache(tmp_path, namespace="FIXTURE").resolve(context, gateway=gateway, **params)
        finally:
            release.set()
        assert pending.result(timeout=5).response.answer_status == "ANSWERED"
    assert gateway.provider.calls == 1


def test_exact_citation_resolution_alone_is_not_a_semantic_support_proof(loaded):
    packed, binding, _ = setup(loaded)
    bad_meaning = raw_answer()
    bad_meaning["claims"][0]["text"] = "Service A can decrypt all resources without permissions."
    response = resolve_answer(output(bad_meaning), snapshot=loaded, packed=packed, binding=binding)
    # Intentional demonstration of the structural validator's boundary. Human/AI
    # semantic source review must reject this claim despite a real source quote.
    assert response.citations[0].quote == QUOTE
    assert response.claims[0].text == bad_meaning["claims"][0]["text"]


def test_selected_spans_resolve_trusted_original_text_and_reject_invented_ids(loaded):
    packed, binding, _ = setup(loaded)
    recipe = GenerationRecipeV6(model_metadata_sha256="0" * 64)
    groups, spans = authorized_spans(loaded, packed, binding.tenant_id)
    assert groups[0]["spans"][0]["text"] == QUOTE
    data = {"answer_status": "ANSWERED", "reason": "NONE", "claims": [{"text": "Service A uses Service B.",
        "evidence": [{"source_id": "s1", "span_id": "s1p1"}]}]}
    candidate = parse_generated(json.dumps(data), "end_turn", recipe)
    answer = resolve_answer(candidate, snapshot=loaded, packed=packed, binding=binding)
    assert answer.citations[0].quote == QUOTE
    bad = copy.deepcopy(data)
    bad["claims"][0]["evidence"][0]["span_id"] = "s2p1"
    with pytest.raises(GenerationError, match="SPAN_NOT_IN_REQUEST"):
        resolve_answer(parse_generated(json.dumps(bad), "end_turn", recipe), snapshot=loaded, packed=packed, binding=binding)
    bad["claims"][0]["evidence"][0]["quote"] = "Invented model quote"
    with pytest.raises(GenerationError, match="SCHEMA_INVALID"):
        parse_generated(json.dumps(bad), "end_turn", recipe)
    _, prompt = generation_request(QUESTION, loaded, packed, binding.tenant_id, recipe, answer_language="vi")
    assert json.loads(prompt)["untrusted_sources"] == groups


def test_span_v3_schema_enforces_at_most_eight_evidence_references_by_construction():
    recipe = GenerationRecipeV9(model_metadata_sha256="0" * 64)
    pair = [{"source_id": "s1", "span_id": "s1p1"}, {"source_id": "s1", "span_id": "s1p2"}]
    data = {"answer_status": "ANSWERED", "reason": "NONE", "claims": [
        {"text": "Atomic supported claim", "evidence": pair} for _ in range(4)]}
    value = parse_generated(json.dumps(data), "end_turn", recipe)
    assert sum(len(c.evidence) for c in value.claims) == 8
    data["claims"][0]["evidence"] = [*pair, {"source_id": "s1", "span_id": "s1p3"}]
    with pytest.raises(GenerationError, match="SCHEMA_INVALID"):
        parse_generated(json.dumps(data), "end_turn", recipe)
