import copy
import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

from app.cost import TokenUsage
from app.external_gateway import (
    COST_BASIS,
    ExternalCallUncertain,
    ExternalModelGateway,
    ExternalPriceSheet,
    ExternalRoutes,
)
from app.external_http import ExternalHttpError
from app.ledger import AlreadyDispatched, BudgetDenied, Ledger, MemoryLedgerStore
from app.model_gateway import CallContext, GatewayError

ROOT = Path(__file__).resolve().parents[1]


def generation():
    return {"model": "deepseek-flash", "choices": [{"index": 0, "finish_reason": "stop",
            "message": {"role": "assistant", "content": '{"ok":true}'}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14,
                      "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 8}}


class Stub:
    live = False

    def __init__(self, payload=None):
        self.payload = generation() if payload is None else payload
        self.calls = []

    def request(self, provider, method, path, body):
        self.calls.append((provider, method, path, json.loads(body)))
        if isinstance(self.payload, Exception):
            raise self.payload
        return copy.deepcopy(self.payload)


@pytest.fixture
def gateway():
    prices = ExternalPriceSheet.model_validate_json((ROOT / 'configs/prices/external-direct-2026-10-17.json').read_bytes())
    now = prices.checked_at + timedelta(seconds=1)
    ledger = Ledger(MemoryLedgerStore(), clock=lambda: now)
    ledger.initialize_month("kg-r72", now.strftime("%Y-%m"))
    ledger.initialize_run("kg-r72", "demo", "external-test", cap_micro_usd=500_000)
    ledger.initialize_external_funding("kg-r72", cap_micro_usd=2_000_000)
    return ExternalModelGateway(ledger, Stub(), ExternalRoutes(account_model_metadata_sha256="0" * 64),
                                prices.model_copy(update={"purpose": "FIXTURE"}))


def context(**changes):
    return CallContext(**(dict(tenant_id="demo", run_id="external-test", request_id="req1", stage="SMOKE") | changes))


def generate(gateway, **kwargs):
    return gateway.generate(context(), system="Return JSON only.", prompt="Return {\"ok\":true}.",
                            max_output_tokens=64, **kwargs)


def test_request_parameters_settlement_basis_and_replay_are_explicit(gateway):
    result = generate(gateway)
    assert result.text == '{"ok":true}' and result.stop_reason == "end_turn"
    assert result.usage == TokenUsage(input_tokens=10, output_tokens=4)
    assert result.actual_micro_usd == 8 and result.cost_basis == COST_BASIS
    payload = gateway.provider.calls[0][3]
    assert payload["thinking"] == {"type": "disabled"}
    assert payload["response_format"] == {"type": "json_object"}
    assert "tools" not in payload and payload["stream"] is False
    operation = gateway.ledger.operation(result.operation_id)
    assert operation.spec.provider == "DEEPSEEK" and operation.spec.max_cost_micro_usd == 314_650
    assert operation.status == "SETTLED"
    with pytest.raises(AlreadyDispatched):
        generate(gateway)
    assert len(gateway.provider.calls) == 1


@pytest.mark.parametrize("usage", [
    {}, {"prompt_tokens": True, "completion_tokens": 1, "total_tokens": 2},
    {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 99},
    {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11, "prompt_cache_hit_tokens": 2},
    {"prompt_tokens": 10, "completion_tokens": 1, "total_tokens": 11,
     "prompt_cache_hit_tokens": 2, "prompt_cache_miss_tokens": 9},
])
def test_missing_or_inconsistent_usage_keeps_all_reservations_without_retry(gateway, usage):
    gateway.provider.payload["usage"] = usage
    with pytest.raises(ExternalCallUncertain) as error:
        generate(gateway)
    operation = gateway.ledger.operation(error.value.operation_id)
    assert operation.status == "UNKNOWN_SPEND"
    assert gateway.ledger.store.get(gateway.ledger.external_funding_key("kg-r72"))["reserved_micro_usd"] == 314_650
    with pytest.raises(AlreadyDispatched):
        generate(gateway)
    assert len(gateway.provider.calls) == 1


@pytest.mark.parametrize("category", ["AUTHENTICATION", "BALANCE", "RATE_LIMIT", "TIMEOUT"])
def test_http_failure_is_safe_unknown_not_hidden_retry(gateway, category):
    gateway.provider.payload = ExternalHttpError(category)
    with pytest.raises(ExternalCallUncertain) as error:
        generate(gateway)
    assert error.value.category == category
    assert gateway.ledger.operation(error.value.operation_id).status == "UNKNOWN_SPEND"
    assert len(gateway.provider.calls) == 1


@pytest.mark.parametrize("change", ["reasoning", "tools", "empty", "wrong_model", "wrong_choice"])
def test_bad_outputs_are_rejected_after_usage_is_settled(gateway, change):
    payload = gateway.provider.payload
    if change == "reasoning":
        payload["choices"][0]["message"]["reasoning_content"] = "DO_NOT_EXPOSE_PRIVATE_REASONING"
    elif change == "tools":
        payload["choices"][0]["message"]["tool_calls"] = [{"function": {"name": "unsafe"}}]
    elif change == "empty":
        payload["choices"][0]["message"]["content"] = ""
    elif change == "wrong_model":
        payload["model"] = "not-approved"
    else:
        payload["choices"][0]["index"] = 1
    with pytest.raises(GatewayError, match="already settled") as error:
        generate(gateway)
    assert "DO_NOT_EXPOSE" not in str(error.value)
    operation = next(item for key, item in gateway.ledger.store._items.items() if key[0].startswith("OP#"))
    assert operation["status"] == "SETTLED" and operation["actual_micro_usd"] == 8


def test_length_is_not_complete_and_token_overrun_halts_every_counter(gateway):
    gateway.provider.payload["choices"][0]["finish_reason"] = "length"
    assert generate(gateway).stop_reason == "max_tokens"
    gateway.provider.payload["usage"].update(completion_tokens=65, total_tokens=75)
    with pytest.raises(GatewayError, match="budgets halted"):
        gateway.generate(context(request_id="req2"), system="JSON", prompt="JSON", max_output_tokens=64)
    assert gateway.ledger.store.get(gateway.ledger.external_funding_key("kg-r72"))["halted"]


def test_stale_prices_missing_funding_or_exhausted_run_prevent_dispatch(gateway):
    gateway.ledger.clock = lambda: gateway.prices.expires_at
    with pytest.raises(ValueError, match="stale"):
        generate(gateway)
    assert not gateway.provider.calls
    gateway.ledger.clock = lambda: gateway.prices.checked_at
    gateway.ledger.initialize_run("kg-r72", "demo", "small", cap_micro_usd=1)
    with pytest.raises(BudgetDenied):
        gateway.generate(context(run_id="small"), system="JSON", prompt="JSON", max_output_tokens=64)
    assert not gateway.provider.calls


def test_fixture_cannot_use_live_prices_or_enable_live_on_memory_ledger(gateway):
    with pytest.raises(GatewayError, match="fixture prices"):
        ExternalModelGateway(gateway.ledger, Stub(), gateway.routes,
                             gateway.prices.model_copy(update={"purpose": "LIVE"}))
    live = Stub()
    live.live = True
    with pytest.raises(GatewayError, match="disabled or transport"):
        ExternalModelGateway(gateway.ledger, live, gateway.routes, gateway.prices)


def vector_payload(**changes):
    return {"model": "voyage-4-lite", "usage": {"total_tokens": 12},
            "data": [{"index": 0, "embedding": [2.0] + [0.0] * 1023}]} | changes


def test_voyage_roles_are_in_actual_payload_and_operation_fingerprint(gateway):
    gateway.provider.payload = vector_payload()
    document = gateway.embed(context(stage="EMBED"), text="Lambda invokes.")
    query = gateway.embed(context(stage="EMBED", request_id="query1"), text="Lambda invokes.", input_role="query")
    assert document.input_role == "document" and query.input_role == "query"
    assert len(document.embedding) == 1024 and np.linalg.norm(document.embedding) == 1
    assert document.actual_micro_usd == 1
    assert gateway.provider.calls[0][3]["input_type"] == "document"
    assert gateway.provider.calls[1][3]["input_type"] == "query"
    assert gateway.provider.calls[0][3]["truncation"] is False
    assert gateway.ledger.operation(document.operation_id).spec.input_sha256 != gateway.ledger.operation(query.operation_id).spec.input_sha256


@pytest.mark.parametrize("data", [
    [], [{"index": 1, "embedding": [1.0] * 1024}], [{"index": 0, "embedding": [1.0]}],
    [{"index": 0, "embedding": [0.0] * 1024}], [{"index": 0, "embedding": [True] * 1024}],
    [{"index": 0, "embedding": [float("nan")] * 1024}],
])
def test_invalid_voyage_vectors_settle_usage_then_fail(gateway, data):
    gateway.provider.payload = vector_payload(data=data)
    with pytest.raises(GatewayError, match="already settled"):
        gateway.embed(context(stage="EMBED"), text="Public fixture.")
    operation = next(item for key, item in gateway.ledger.store._items.items() if key[0].startswith("OP#"))
    assert operation["status"] == "SETTLED"


def test_no_implicit_embedding_usage_when_provider_response_is_empty(gateway):
    gateway.provider.payload = {}
    with pytest.raises(ExternalCallUncertain):
        gateway.embed(context(stage="EMBED"), text="Fixture.")
    assert len(gateway.provider.calls) == 1
