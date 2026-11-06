import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.answer_cache import AnswerCacheBlocked
from app.contracts import stable_id
from app.cost import TokenUsage
from app.model_gateway import CallContext, ModelResult, UncertainModelCall
from app.profiled_source_context import render_profiled_context
from app.qualified_answer_service import resolve_qualified_answer, resolve_qualified_flow
from app.routed_source_reader import ReaderStageCache
from app.task_scoped_generation_v14 import TaskScopedRecipeV14
from tests.test_answer_qualification import setup
from tests.test_scoped_method_flow import fixture, prepare


class BaseGateway:
    provider = SimpleNamespace(live=False)
    routes = SimpleNamespace(account_model_metadata_sha256="b" * 64, generation_context_ceiling=1048576)

    def __init__(self, raw=None, failure=None):
        self.calls, self.failure = 0, failure
        self.raw = raw or {"answer_status": "ANSWERED", "reason": "NONE", "limitations": [],
            "claims": [{"text": "For uploads, the policy requires the configured key.",
                "evidence": [{"source_id": "s1", "span_id": "s1p1"}, {"source_id": "s2", "span_id": "s2p1"}]}]}

    def _call(self, call, provider, body, input_bound, output_bound, input_role):
        self.calls += 1
        assert json.loads(body)["max_tokens"] == output_bound == 1600
        if self.failure:
            raise self.failure
        return ModelResult(operation_id=stable_id("op", **call.model_dump(mode="json")),
            text=json.dumps(self.raw), stop_reason="end_turn", actual_micro_usd=123,
            usage=TokenUsage(input_tokens=200, output_tokens=52))


def params(tmp_path):
    docs, chunks, context, binding, _ = setup()
    call = CallContext(tenant_id="demo", run_id="qualified-fixture", request_id=binding.request_id, stage="GENERATE")
    return {"call": call, "source_context": context, "documents": docs, "chunks": chunks, "binding": binding,
        "recipe": TaskScopedRecipeV14(model_metadata_sha256="b" * 64), "search": context.question,
        "raw_cache": ReaderStageCache(tmp_path, kind="answer-v13", namespace="FIXTURE")}


def test_common_service_has_one_native_call_and_replays_display_eligible_result_without_gateway(tmp_path):
    p, gateway = params(tmp_path), BaseGateway()
    first = resolve_qualified_answer(**p, gateway=gateway)
    second = resolve_qualified_answer(**p)
    assert first.qualification.status == second.qualification.status == "NO_MECHANICAL_FLAGS"
    assert first.new_cost_micro_usd == 123 and second.new_cost_micro_usd == 0
    assert second.cache_hit and gateway.calls == 1


def test_review_required_raw_receipt_is_ready_for_replay_but_never_publicly_exposed(tmp_path):
    raw = {"answer_status": "ANSWERED", "reason": "NONE", "limitations": [],
        "claims": [{"text": "Potentially unsafe request assertion.", "evidence": [{"source_id": "s2", "span_id": "s2p1"}]}]}
    p, gateway = params(tmp_path), BaseGateway(raw)
    first = resolve_qualified_answer(**p, gateway=gateway)
    second = resolve_qualified_answer(**p)
    with p["raw_cache"]._connect() as db:
        state = db.execute("SELECT state FROM answers").fetchone()[0]
    assert state == "READY" and first.qualification.status == second.qualification.status == "REVIEW_REQUIRED"
    assert first.public_payload()["response"] is None and second.cache_hit and gateway.calls == 1
    assert "Potentially unsafe" not in json.dumps(second.public_payload())
    assert first.receipt.raw_output == second.receipt.raw_output == json.dumps(raw)


def test_postprocessing_policy_change_requalifies_raw_cache_without_paying_again(tmp_path, monkeypatch):
    import app.answer_qualification as policy

    p, gateway = params(tmp_path), BaseGateway()
    first = resolve_qualified_answer(**p, gateway=gateway)
    monkeypatch.setattr(policy, "POLICY_VERSION", "qualification-test-v2")
    second = resolve_qualified_answer(**p)
    assert second.qualification.policy_sha256 != first.qualification.policy_sha256
    assert second.receipt == first.receipt and second.cache_hit and gateway.calls == 1


def test_owner_change_cannot_reuse_another_owners_cached_response(tmp_path):
    p, gateway = params(tmp_path), BaseGateway()
    resolve_qualified_answer(**p, gateway=gateway)
    p["binding"] = p["binding"].model_copy(update={"owner_id": "owner-b"})
    with pytest.raises(ValueError, match="TASK_OR_RECIPE"):
        resolve_qualified_answer(**p, gateway=gateway)
    assert gateway.calls == 1


def test_unknown_call_does_not_become_a_review_result_or_get_automatically_retried(tmp_path):
    p, gateway = params(tmp_path), BaseGateway(failure=UncertainModelCall("fixture-unknown"))
    with pytest.raises(UncertainModelCall):
        resolve_qualified_answer(**p, gateway=gateway)
    with pytest.raises(AnswerCacheBlocked):
        resolve_qualified_answer(**p, gateway=gateway)
    assert gateway.calls == 1


def test_empty_sources_produce_explicit_server_abstention_without_model_or_cache(tmp_path):
    p, gateway = params(tmp_path), BaseGateway()
    source = p["source_context"]
    p["source_context"] = render_profiled_context(p["documents"], p["chunks"], (), question=source.question,
        language=source.language, tenant_id=source.tenant_id, collection_sha256=source.collection_sha256, profile=source.profile)
    result = resolve_qualified_answer(**p, gateway=gateway)
    assert result.receipt is None and result.deterministic_no_evidence and gateway.calls == 0
    public = result.public_payload()
    assert public["answer_origin"] == "SERVER_DETERMINISTIC_ABSTENTION"
    assert public["response"]["answer_status"] == "ABSTAINED"


def test_q_not_ready_keeps_requested_method_and_never_dispatches_hybrid(monkeypatch, tmp_path):
    collection, _, _, _, plan = fixture(monkeypatch)
    flow = prepare(collection, "hybrid_multi_query", plan.model_copy(update={"facets": ()}))
    gateway = BaseGateway()
    result = resolve_qualified_flow(flow, raw_cache=None, gateway=gateway)
    assert result.public_payload()["qualification"] == "NOT_READY" and gateway.calls == 0
    assert result.public_payload()["mode_used"] == "hybrid_multi_query"


@pytest.mark.parametrize("method", ["graph_1hop", "graph_2hop", "hybrid_multi_query_balanced"])
def test_other_unready_methods_do_not_reach_storage_or_provider(monkeypatch, method):
    import app.scoped_method_flow as module

    collection, _, _, keys, plan = fixture(monkeypatch)
    if method.startswith("graph_"):
        base = prepare(collection, "hybrid")
        monkeypatch.setattr(module, "_graph_pool", lambda *args: (keys[:8],
            [{"bundle_id": str(i), "query_index": 0, "chunk_ids": [key]} for i, key in enumerate(keys[:8])],
            {"path_count": 0, "complete_path_bundles_in_pool": 0}))
        flow = module.prepare_scoped_flow(collection, None, SimpleNamespace(facts=[]), None,
            question=base.context.question, search=base.context.question, language="en",
            method=method, run_id="unready-graph", owner_id="owner", recipe=base.recipe)
    else:
        flow = prepare(collection, method, plan.model_copy(update={"facets": ()}))

    class NoStorage:
        def resolve_stage(self, *args, **kwargs):
            raise AssertionError("NOT_READY must not consult the raw answer cache")

    gateway = BaseGateway()
    result = resolve_qualified_flow(flow, raw_cache=NoStorage(), gateway=gateway).public_payload()
    assert result["qualification"] == "NOT_READY" and result["response"] is None
    assert result["mode_requested"] == result["mode_used"] == method
    assert result["hidden_fallbacks"] == 0 and not result["model_dispatched"] and gateway.calls == 0


def test_common_flow_derives_new_generation_identity_instead_of_reusing_v12_operation(monkeypatch, tmp_path):
    collection, _, _, _, _ = fixture(monkeypatch)
    flow = prepare(collection, "hybrid")
    gateway = BaseGateway(raw={"answer_status": "ANSWERED", "reason": "NONE", "limitations": [],
        "claims": [{"text": "Synthetic fixture source only.", "evidence": [{"source_id": "s1", "span_id": "s1p1"}]}]})
    cache = ReaderStageCache(tmp_path, kind="answer-v13", namespace="FIXTURE")
    result = resolve_qualified_flow(flow, raw_cache=cache, gateway=gateway)
    assert result.qualification.native_response.request_id != flow.binding.request_id
    assert result.public_payload()["mode_used"] == "hybrid" and gateway.calls == 1
    changed = replace(flow, trace=flow.trace | {"hidden_fallbacks": 1})
    with pytest.raises(ValueError, match="METHOD_CHANGED"):
        resolve_qualified_flow(changed, raw_cache=cache, gateway=gateway)
    assert gateway.calls == 1
