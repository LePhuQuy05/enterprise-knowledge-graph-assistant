import json
from types import SimpleNamespace

import pytest

from app.answer_cache import AnswerCacheBlocked
from app.cost import TokenUsage
from app.external_gateway import ExternalModelGateway
from app.ledger import BudgetDenied
from app.model_gateway import UncertainModelCall
from app.qualified_answer_service import resolve_qualified_answer
from app.query_auth import AuthContext
from app.run_budget import EnvelopeLedger, ReceiptedStageGateway, RunBudgetController
from app.scoped_query_v6 import SYSTEM
from app.scoped_run_budget import PLAN_SLOT, ScopedRunBudgetController
from app.transactional_answer_cache import TransactionalAnswerCache
from tests.test_external_pilot import configured
from tests.test_qualified_answer_service import BaseGateway, params


def configured_run(tmp_path, methods=("hybrid",), language="en"):
    original = configured()
    prices = original.prices.model_copy(update={"rates": tuple(rate.model_copy(update={
        "input_micro_usd_per_unit": 10000, "output_micro_usd_per_unit": 10000}) for rate in original.prices.rates)})
    gateway = ExternalModelGateway(original.ledger, original.provider,
        original.routes.model_copy(update={"account_model_metadata_sha256": "b"*64}), prices)
    ledger = gateway.ledger
    ledger.initialize_run("kg-r72", "demo", "scoped-owned-fixture", cap_micro_usd=500000)
    controller = ScopedRunBudgetController(gateway, budget_run_id="scoped-owned-fixture")
    auth = AuthContext("demo", "11111111-1111-1111-1111-111111111111")
    p = params(tmp_path)
    parent = {"run_id": "fixture-owned-run", "tenant_id": "demo", "owner_id": auth.user_id,
        "admission_token": "a"*32, "expires_at": int(ledger._now().timestamp())+600,
        "request": {"method_ids": list(methods), "original_question": p["source_context"].question,
            "language": language, "run_kind": "answer"}}
    return SimpleNamespace(gateway=gateway, ledger=ledger, controller=controller, auth=auth, parent=parent, params=p)


def admit(s):
    s.controller.store.commit(s.controller.admission_changes(s.parent))
    return s.controller.envelope(s.auth, s.parent)


def native_gateway(s, slot):
    return ExternalModelGateway(EnvelopeLedger(s.controller, s.auth, s.parent, slot, lambda: None),
        s.gateway.provider, s.gateway.routes, s.gateway.prices)


def provider_output(s, *, failure=False):
    seen = []

    def request(provider, method, path, body):
        s.gateway.provider.calls += 1
        value = json.loads(body)
        seen.append(value)
        if failure:
            raise TimeoutError("SYNTHETIC scoped transport uncertainty")
        content = json.dumps(BaseGateway().raw) if value["max_tokens"] == 1600 else json.dumps({
            "facet_1": None, "facet_2": None, "facet_3": None})
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}

    s.gateway.provider.request = request
    return seen


def answer_params(s):
    p = s.params
    call = s.controller.context(s.parent, s.controller.slot_for("hybrid"))
    return p | {"call": call, "binding": p["binding"].model_copy(update={
        "request_id": call.request_id, "owner_id": s.auth.user_id}),
        "raw_cache": TransactionalAnswerCache(s.controller.store, tenant_id=s.auth.tenant_id,
            owner_id=s.auth.user_id, namespace="FIXTURE")}


def test_v14_quotes_1600_generation_and_one_shared_plan_and_optional_rewrite(tmp_path):
    s = configured_run(tmp_path, methods=("hybrid", "hybrid_multi_query", "hybrid_multi_query_balanced"), language="vi")
    envelope = admit(s)
    slots = envelope["slots"]
    assert len(slots) == 5 and PLAN_SLOT in slots and "REWRITE:shared" in slots
    assert [s.controller.output_tokens(key) for key in slots].count(1600) == 3
    assert [s.controller.output_tokens(key) for key in slots].count(256) == 2
    assert envelope["max_cost_micro_usd"] == sum(row["max_cost_micro_usd"] for row in slots.values())
    assert s.ledger.store.get(s.ledger.run_key(s.controller.budget_run_id))["reserved_micro_usd"] == envelope["max_cost_micro_usd"]
    assert s.gateway.provider.calls == 0


def test_legacy_controller_keeps_800_output_and_separate_operation_identity(tmp_path):
    s = configured_run(tmp_path)
    legacy = RunBudgetController(s.gateway, budget_run_id=s.controller.budget_run_id)
    slot = s.controller.slot_for("hybrid")
    rate = s.gateway.prices.rate_for(s.gateway.routes.generation_model_id, s.gateway.routes.generation_invocation_id,
        "external-default", now=s.ledger._now(), live=False)
    assert legacy._spec(s.parent, slot, "0"*64).max_cost_micro_usd == rate.quote(TokenUsage(
        input_tokens=s.gateway.routes.generation_context_ceiling, output_tokens=800))
    assert s.controller.output_tokens(slot) == 1600
    assert legacy.context(s.parent, slot).request_id != s.controller.context(s.parent, slot).request_id
    assert list(legacy.admission_changes(s.parent)[-1].item["slots"]) == [slot]
    assert PLAN_SLOT not in s.controller.admission_slots(s.parent)


def test_actual_v14_gateway_envelope_cache_qualification_composition_never_double_reserves(tmp_path):
    s = configured_run(tmp_path)
    envelope = admit(s)
    seen = provider_output(s)
    p = answer_params(s)
    result = resolve_qualified_answer(**p, gateway=native_gateway(s, "GENERATE:hybrid"))
    replay = resolve_qualified_answer(**p)
    counter = s.ledger.store.get(s.ledger.run_key(s.controller.budget_run_id))
    operation = s.ledger.operation(result.receipt.operation_id)
    assert result.qualification.status == "NO_MECHANICAL_FLAGS" and replay.cache_hit
    assert len(seen) == s.gateway.provider.calls == 1 and seen[0]["max_tokens"] == 1600
    assert operation.status == "SETTLED" and operation.spec.max_cost_micro_usd == envelope["max_cost_micro_usd"]
    assert counter["reserved_micro_usd"] == 0 and counter["settled_micro_usd"] == result.receipt.original_cost_micro_usd
    assert replay.receipt == result.receipt and replay.new_cost_micro_usd == 0
    assert result.qualification.native_response.request_id == s.controller.context(s.parent, "GENERATE:hybrid").request_id


def test_shared_native_plan_uses_256_quote_and_durable_receipt_in_envelope(tmp_path):
    s = configured_run(tmp_path, methods=("hybrid_multi_query", "hybrid_multi_query_balanced"))
    admit(s)
    seen = provider_output(s)
    gateway = ReceiptedStageGateway(s.controller, s.auth, s.parent, PLAN_SLOT, lambda: None)
    context = s.controller.context(s.parent, PLAN_SLOT)
    request = {"system": SYSTEM, "prompt": json.dumps({"untrusted_english_search_query": "A fixture query"}),
        "max_output_tokens": 256}
    first = gateway.generate(context, **request)
    second = gateway.generate(context, **request)
    # Legacy StageReceipt reconstructs ModelResult, not its transport subclass.
    fields = ("operation_id", "text", "stop_reason", "usage", "actual_micro_usd")
    assert all(getattr(first, name) == getattr(second, name) for name in fields)
    assert gateway.cache_hit and s.gateway.provider.calls == 1
    assert seen[0]["max_tokens"] == 256 and context.stage == "PLAN_QUERY"
    assert s.ledger.operation(first.operation_id).status == "SETTLED"
    s.controller.release_unused(s.auth, s.parent, all_slots=True)
    counter = s.ledger.store.get(s.ledger.run_key(s.controller.budget_run_id))
    assert counter["reserved_micro_usd"] == 0 and counter["settled_micro_usd"] == first.actual_micro_usd


@pytest.mark.parametrize("shared_plan", [False, True])
def test_unknown_generation_or_plan_never_refunds_started_operation_or_retries(tmp_path, shared_plan):
    s = configured_run(tmp_path, methods=("hybrid", "hybrid_multi_query") if shared_plan else ("hybrid",))
    admit(s)
    provider_output(s, failure=True)
    slot = PLAN_SLOT if shared_plan else "GENERATE:hybrid"
    if shared_plan:
        gateway = ReceiptedStageGateway(s.controller, s.auth, s.parent, slot, lambda: None)
        with pytest.raises(UncertainModelCall):
            gateway.generate(s.controller.context(s.parent, slot), system=SYSTEM, prompt="JSON fixture", max_output_tokens=256)
    else:
        p = answer_params(s)
        with pytest.raises(UncertainModelCall):
            resolve_qualified_answer(**p, gateway=native_gateway(s, slot))
        with pytest.raises(AnswerCacheBlocked):
            resolve_qualified_answer(**p, gateway=native_gateway(s, slot))
    assert not s.controller.safe_failure(s.auth, s.parent, "hybrid")
    s.controller.release_unused(s.auth, s.parent, all_slots=True)
    own = s.controller.envelope(s.auth, s.parent)["slots"][slot]
    operation = s.ledger.operation(own["operation_id"])
    counter = s.ledger.store.get(s.ledger.run_key(s.controller.budget_run_id))
    assert operation.status == "UNKNOWN_SPEND" and operation.call_started
    assert counter["reserved_micro_usd"] == operation.spec.max_cost_micro_usd
    assert s.gateway.provider.calls == 1


def test_scoped_budget_rejects_unknown_or_duplicate_methods_before_any_hold(tmp_path):
    s = configured_run(tmp_path, methods=("hybrid", "hybrid"))
    with pytest.raises(BudgetDenied, match="METHODS_NOT_ALLOWED"):
        admit(s)
    with pytest.raises(BudgetDenied, match="STAGE_NOT_ALLOWED"):
        s.controller.context(s.parent, "GENERATE:invented_method")
    assert s.gateway.provider.calls == 0
    assert s.ledger.store.get(s.ledger.run_key(s.controller.budget_run_id))["reserved_micro_usd"] == 0


def test_quote_matches_exact_v14_output_bound(tmp_path):
    s = configured_run(tmp_path)
    spec = s.controller._spec(s.parent, "GENERATE:hybrid", "0"*64)
    rate = s.gateway.prices.rate_for(s.gateway.routes.generation_model_id, s.gateway.routes.generation_invocation_id,
        "external-default", now=s.ledger._now(), live=False)
    assert spec.max_cost_micro_usd == rate.quote(TokenUsage(
        input_tokens=s.gateway.routes.generation_context_ceiling, output_tokens=1600))


def test_scoped_controller_cannot_consume_legacy_envelope_even_with_same_run_id(tmp_path):
    s = configured_run(tmp_path)
    legacy = RunBudgetController(s.gateway, budget_run_id=s.controller.budget_run_id)
    s.controller.store.commit(legacy.admission_changes(s.parent))
    with pytest.raises(BudgetDenied, match="ENVELOPE_VERSION"):
        s.controller.envelope(s.auth, s.parent)
    assert s.gateway.provider.calls == 0
