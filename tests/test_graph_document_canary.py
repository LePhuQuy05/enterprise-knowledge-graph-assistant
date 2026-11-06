"""Paid-arm dispatch guards using a synthetic provider, never real API calls."""

import pytest

from app.model_gateway import CallContext
from scripts.pilot_graph_document_answers import RUN, STOP_MICRO_USD, ScopedGateway
from tests.test_reviewed_source_answer_canary import Base


def context(index=0):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=f"fixture-{index}", stage="GENERATE")


@pytest.mark.parametrize("change", ["prompt", "tokens", "context", "repeat"])
def test_no_modified_or_repeated_paid_dispatch(change):
    base = Base()
    gateway = ScopedGateway(base)
    gateway.allow(context(), "system", "prompt")
    if change == "repeat":
        gateway.generate(context(), system="system", prompt="prompt", max_output_tokens=800)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(1) if change == "context" else context(), system="system",
            prompt="changed" if change == "prompt" else "prompt", max_output_tokens=801 if change == "tokens" else 800)
    assert base.calls == (1 if change == "repeat" else 0)


@pytest.mark.parametrize("change", [{"run_id": "other"}, {"stage": "PLAN_QUERY"}, {"tenant_id": "other"}, {"attempt": 2}])
def test_only_this_run_tenant_stage_attempt_may_register(change):
    gateway = ScopedGateway(Base())
    with pytest.raises(ValueError, match="DISPATCH_SCOPE"):
        gateway.allow(context().model_copy(update=change), "s", "p")


def test_unknown_provider_exception_prevents_later_calls():
    base, gateway = Base(fail=True), None
    gateway = ScopedGateway(base)
    for i in (0, 1):
        gateway.allow(context(i), "s", "p")
    with pytest.raises(ValueError):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(1), system="s", prompt="p", max_output_tokens=800)
    assert base.calls == 1 and gateway.halted


@pytest.mark.parametrize("stop", ["cost", "count"])
def test_observed_spend_and_eight_call_caps(stop):
    base = Base()
    gateway = ScopedGateway(base)
    gateway.allow(context(), "s", "p")
    if stop == "cost":
        gateway.cost = STOP_MICRO_USD
    else:
        gateway.seen = {f"old-{i}" for i in range(8)}
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    assert base.calls == 0
