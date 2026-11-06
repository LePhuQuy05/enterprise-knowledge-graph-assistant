"""Real V12 source resolution/cache path without dispatching a model."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.answer_cache import AnswerCacheBlocked
from app.contracts import stable_id
from app.cost import TokenUsage
from app.external_gateway import ExternalRoutes
from app.generation import AnswerBinding
from app.model_gateway import CallContext, ModelResult, UncertainModelCall
from app.profiled_answer_runtime import ProfiledAnswerCache, ProfiledModelGateway
from app.profiled_generation import ProfiledGenerationRecipe
from app.profiled_source_context import render_profiled_context
from tests.test_profiled_source_context import render


class Gateway:
    def __init__(self, recipe, *, raw=None, fail=None):
        self.routes = ExternalRoutes(account_model_metadata_sha256=recipe.model_metadata_sha256)
        self.provider = SimpleNamespace(live=False)
        self.calls, self.fail = 0, fail
        self.raw = raw or json.dumps({"answer_status":"ANSWERED", "reason":"NONE", "limitations":[],
            "claims":[{"text":"Synthetic fixture claim.", "evidence":[{"source_id":"s12", "span_id":"s12p1"}]}]})

    def generate_profiled(self, context, **kwargs):
        self.calls += 1
        if self.fail:
            raise self.fail
        return ModelResult(operation_id=stable_id("op", **context.model_dump(mode="json")),
            usage=TokenUsage(input_tokens=200, output_tokens=60), actual_micro_usd=123,
            text=self.raw, stop_reason="end_turn")


def setup(tmp_path, *, language="en", cap=12):
    docs, chunks, source = render(12 if cap == 12 else 8, cap)
    source = replace(source, language=language)
    recipe = ProfiledGenerationRecipe(model_metadata_sha256="b"*64)
    context = CallContext(tenant_id="demo", run_id="fixture-v12", request_id="fixture-v12", stage="GENERATE")
    binding = AnswerBinding(request_id=context.request_id, tenant_id="demo", owner_id="fixture-owner",
        mode_requested="hybrid_multi_query", mode_used="hybrid_multi_query", used_graph=False)
    cache = ProfiledAnswerCache(tmp_path, namespace="FIXTURE")
    return docs, chunks, source, recipe, context, binding, cache


def resolve(parts, **kwargs):
    docs, chunks, source, recipe, context, binding, cache = parts
    return cache.resolve_profiled(context, source_context=source, documents=docs, chunks=chunks,
        binding=binding, recipe=recipe, **kwargs)


def test_twelfth_source_answer_is_durable_replay_has_zero_dispatch_and_original_owner(tmp_path):
    p = setup(tmp_path)
    gateway = Gateway(p[3])
    first = resolve(p, gateway=gateway)
    reopened = ProfiledAnswerCache(tmp_path, namespace="FIXTURE")
    second = resolve((*p[:-1], reopened))
    assert gateway.calls == 1 and second.cache_hit and second.new_cost_micro_usd == 0
    assert first.response == second.response and first.new_cost_micro_usd == 123
    assert first.response.citations[0].chunk_id == p[2].chunk_ids[11]
    assert p[-1].path.name == "profiled_answers_v12.sqlite3"
    assert not (tmp_path / "answers.sqlite3").exists()


@pytest.mark.parametrize("change", ["owner", "language", "profile", "question"])
def test_same_operation_cannot_reuse_a_different_owner_or_task(tmp_path, change):
    p = list(setup(tmp_path))
    gateway = Gateway(p[3])
    resolve(p, gateway=gateway)
    if change == "owner":
        p[5] = p[5].model_copy(update={"owner_id":"other-owner"})
    else:
        source = p[2]
        opts = {"question":source.question, "language":source.language, "tenant_id":source.tenant_id,
            "collection_sha256":source.collection_sha256, "profile":source.profile}
        keys = source.chunk_ids
        if change == "language":
            opts["language"] = "vi"
        elif change == "question":
            opts["question"] = "Different original question?"
        else:
            keys = keys[:8]
        p[2] = render_profiled_context(p[0], p[1], keys, **opts)
    with pytest.raises(ValueError, match="TASK_OWNER_OR_RECIPE"):
        resolve(p, gateway=gateway)
    assert gateway.calls == 1


@pytest.mark.parametrize("raw", ['{"answer_status":"ANSWERED"}',
    '{"answer_status":"ANSWERED","reason":"NONE","limitations":[],"claims":[{"text":"X","evidence":[{"source_id":"s1","span_id":"s1p99"}]}]}'])
def test_invalid_or_unsupplied_citation_output_remains_raw_failed_without_retry(tmp_path, raw):
    p = setup(tmp_path)
    gateway = Gateway(p[3], raw=raw)
    with pytest.raises(ValueError):
        resolve(p, gateway=gateway)
    with p[-1]._connect() as db:
        state, payload = db.execute("SELECT state,payload FROM answers").fetchone()
    assert state == "FAILED" and json.loads(payload)["raw_output"] == raw
    with pytest.raises(AnswerCacheBlocked):
        resolve(p, gateway=gateway)
    assert gateway.calls == 1


def test_unknown_provider_call_is_not_retried(tmp_path):
    p = setup(tmp_path)
    gateway = Gateway(p[3], fail=UncertainModelCall("fixture-unknown"))
    with pytest.raises(UncertainModelCall):
        resolve(p, gateway=gateway)
    with pytest.raises(AnswerCacheBlocked):
        resolve(p, gateway=gateway)
    assert gateway.calls == 1


def test_empty_sources_abstain_without_provider_call(tmp_path):
    p = list(setup(tmp_path, language="vi"))
    p[2] = render_profiled_context(p[0], p[1], (), question="Câu hỏi thiếu nguồn?", language="vi",
        tenant_id="demo", collection_sha256="a"*64, profile=p[2].profile)
    gateway = Gateway(p[3])
    result = resolve(p, gateway=gateway)
    assert result.response.answer_status == "ABSTAINED" and not result.response.claims
    assert result.deterministic_no_evidence and result.receipt is None and gateway.calls == 0
    assert result.response.answer_language == "vi"


def test_live_namespace_cannot_dispatch_a_fixture_gateway(tmp_path):
    p = list(setup(tmp_path))
    p[-1] = ProfiledAnswerCache(tmp_path, namespace="LIVE")
    gateway = Gateway(p[3])
    with pytest.raises(ValueError, match="NAMESPACE_OR_MODEL"):
        resolve(p, gateway=gateway)
    assert gateway.calls == 0


def test_profile_gateway_higher_cap_is_only_reachable_with_validated_v12_context(tmp_path):
    docs, chunks, source, recipe, context, _, _ = setup(tmp_path)
    calls = []

    def dispatch(ctx, provider, body, input_bound, output_bound, role):
        calls.append((ctx, provider, json.loads(body), input_bound, output_bound, role))
        return "stub"

    routes = ExternalRoutes(account_model_metadata_sha256=recipe.model_metadata_sha256)
    gateway = ProfiledModelGateway(SimpleNamespace(provider=SimpleNamespace(live=False), routes=routes, _call=dispatch))
    assert gateway.generate_profiled(context, source_context=source, documents=docs, chunks=chunks, recipe=recipe) == "stub"
    assert calls[0][2]["max_tokens"] == calls[0][4] == 1600
    assert calls[0][2]["thinking"] == {"type":"disabled"}
    with pytest.raises(ValueError):
        gateway.generate_profiled(context, source_context=replace(source, user_message="tampered"),
            documents=docs, chunks=chunks, recipe=recipe)
    assert len(calls) == 1
