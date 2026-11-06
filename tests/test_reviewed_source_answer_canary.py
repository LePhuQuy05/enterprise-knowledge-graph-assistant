from types import SimpleNamespace

import pytest

from app.model_gateway import CallContext
from scripts.pilot_reviewed_source_answers import METHODS, QUESTIONS, RUN, ScopedGateway


class Base:
    provider = SimpleNamespace(live=True)
    routes = object()

    def __init__(self, fail=False):
        self.calls, self.fail = 0, fail

    def generate(self, *args, **kwargs):
        self.calls += 1
        if self.fail:
            raise ValueError("fixture provider failure")
        return SimpleNamespace(actual_micro_usd=1000)


def context(index=0):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=f"fixture-{index}", stage="GENERATE")


def test_scope_has_eight_methods_two_languages_not_sixteen_independent_questions():
    assert len(set(METHODS)) == 8 and set(QUESTIONS) == {"en", "vi"}
    assert {"graph_1hop", "graph_2hop", "hybrid_multi_query", "hybrid_multi_query_balanced"} <= set(METHODS)


@pytest.mark.parametrize("change", ["prompt", "tokens", "context", "repeat"])
def test_gateway_exact_frozen_dispatch_and_no_duplicate_provider_calls(change):
    base = Base()
    gateway = ScopedGateway(base)
    gateway.allow(context(), "system", "prompt", 800)
    if change == "repeat":
        gateway.generate(context(), system="system", prompt="prompt", max_output_tokens=800)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(1) if change == "context" else context(), system="system",
            prompt="changed" if change == "prompt" else "prompt", max_output_tokens=801 if change == "tokens" else 800)
    assert base.calls == (1 if change == "repeat" else 0)


def test_gateway_halts_after_uncertain_failure_and_does_not_retry():
    base = Base(fail=True)
    gateway = ScopedGateway(base)
    gateway.allow(context(), "s", "p", 800)
    gateway.allow(context(1), "s", "p", 800)
    with pytest.raises(ValueError, match="fixture provider"):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(1), system="s", prompt="p", max_output_tokens=800)
    assert base.calls == 1


def test_gateway_observed_stop_and_call_count_are_separate_from_ledger_admission():
    gateway = ScopedGateway(Base())
    gateway.allow(context(), "s", "p", 800)
    gateway.cost = 50000
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    gateway.cost = 0
    gateway.seen = set(str(i) for i in range(19))
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
