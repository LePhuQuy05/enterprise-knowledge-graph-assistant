from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.source_graph_jobs_v10 import BoundedGatewayV10, context_for


def fixture(*, unknown=False):
    unit = {"v10_unit_id": "source", "occurrences": [{"chunk_id": "chunk"}]}
    plan = {"run_id": "fixture-v10", "selection": [unit], "maximum_new_calls": 1,
        "actual_cost_stop_between_calls_micro_usd": 100}
    class Base:
        provider = SimpleNamespace(live=False)
        routes = object()
        calls = 0
        def generate(self, *args, **kwargs):
            self.calls += 1
            if unknown:
                raise RuntimeError("UNCERTAIN_FIXTURE")
            return SimpleNamespace(actual_micro_usd=1)
    base = Base()
    scoped = BoundedGatewayV10(base, plan, {"chunk": SimpleNamespace(text="Pinned source.")})
    context = context_for(unit, plan["run_id"])
    system, prompt = scoped.allowed[context.model_dump_json()]
    return base, scoped, context, system, prompt


@pytest.mark.parametrize("change", ["context", "system", "prompt", "tokens", "cost"])
def test_unpinned_or_over_budget_input_never_reaches_provider(change):
    base, scoped, context, system, prompt = fixture()
    tokens = 2048
    if change == "context":
        context = context.model_copy(update={"request_id": "forged"})
    elif change == "system":
        system += "changed"
    elif change == "prompt":
        prompt += "changed"
    elif change == "tokens":
        tokens = 4096
    else:
        scoped.cost = 100
    with pytest.raises(ValueError, match="DISPATCH_SCOPE_OR_COST"):
        scoped.generate(context, system=system, prompt=prompt, max_output_tokens=tokens)
    assert base.calls == 0


def test_unknown_failure_halts_native_v10_without_retry():
    base, scoped, context, system, prompt = fixture(unknown=True)
    with pytest.raises(RuntimeError):
        scoped.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
    with pytest.raises(ValueError):
        scoped.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
    assert base.calls == 1 and scoped.halted


def test_duplicate_parallel_calls_have_one_dispatch_owner():
    base, scoped, context, system, prompt = fixture()
    def run(_):
        try:
            scoped.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
            return True
        except ValueError:
            return False
    with ThreadPoolExecutor(max_workers=3) as pool:
        assert sum(pool.map(run, range(3))) == 1
    assert base.calls == 1
