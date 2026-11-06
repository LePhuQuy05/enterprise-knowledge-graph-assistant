from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.source_graph_jobs import SourceBoundedGateway, context_for


def setup_gateway(*, fail=False):
    unit = {"v7_unit_id": "one-source", "occurrences": [{"chunk_id": "chunk"}]}
    plan = {"run_id": "bounded-fixture", "selection": [unit], "maximum_new_calls": 1,
            "actual_cost_stop_between_calls_micro_usd": 100}
    class Provider:
        routes = object()
        provider = SimpleNamespace(live=False)
        calls = 0
        def generate(self, *args, **kwargs):
            self.calls += 1
            if fail:
                raise RuntimeError("UNKNOWN_FIXTURE")
            return SimpleNamespace(actual_micro_usd=1)
    base = Provider()
    scoped = SourceBoundedGateway(base, plan, {"chunk": SimpleNamespace(text="Pinned source.")})
    context = context_for(unit, plan["run_id"])
    system, prompt = scoped.allowed[context.model_dump_json()]
    return base, scoped, context, system, prompt


def test_concurrent_duplicate_source_has_only_one_dispatch():
    base, scoped, context, system, prompt = setup_gateway()
    def call(_):
        try:
            scoped.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
            return "DISPATCHED"
        except ValueError:
            return "DENIED"
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(call, range(4)))
    assert results.count("DISPATCHED") == 1 and base.calls == 1 and scoped.cost == 1


def test_unknown_dispatch_stops_without_retry_or_reset():
    base, scoped, context, system, prompt = setup_gateway(fail=True)
    with pytest.raises(RuntimeError):
        scoped.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
    with pytest.raises(ValueError, match="DISPATCH_SCOPE_OR_COST"):
        scoped.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
    assert scoped.halted and base.calls == 1


@pytest.mark.parametrize("mutation", ["context", "prompt", "system", "tokens", "cost"])
def test_scope_or_cost_error_never_dispatches(mutation):
    base, scoped, context, system, prompt = setup_gateway()
    tokens = 2048
    if mutation == "context":
        context = context.model_copy(update={"request_id": "forged"})
    elif mutation == "prompt":
        prompt += "different source"
    elif mutation == "system":
        system += "different system"
    elif mutation == "tokens":
        tokens = 4096
    else:
        scoped.cost = 100
    with pytest.raises(ValueError, match="DISPATCH_SCOPE_OR_COST"):
        scoped.generate(context, system=system, prompt=prompt, max_output_tokens=tokens)
    assert base.calls == 0
