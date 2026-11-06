"""Four-call native scope; all gateways here are synthetic and offline."""


import pytest

from app.faceted_query_v4 import SYSTEM, preparation_context
from scripts.check_collection319_methods import preparation_payload
from scripts.run_facet_recall_v4 import RUN, STOP, ScopedGateway
from tests.test_collection319_method_workflow import FakeGateway

TASK = {"question": "Compare S3 invocation and Lambda execution role.",
    "search": "Compare S3 invocation and Lambda execution role.", "language": "en"}


def setup(base=None):
    base = base or FakeGateway()
    gateway = ScopedGateway(base, "a" * 64, [TASK])
    context = preparation_context("a" * 64, TASK["question"], TASK["search"], "en", RUN)
    request = dict(system=SYSTEM, prompt=preparation_payload(TASK["question"], TASK["search"], "en"), max_output_tokens=256)
    return gateway, base, context, request


@pytest.mark.parametrize("field,value", [("max_output_tokens", 800), ("prompt", "new question"), ("system", "new system")])
def test_bound_native_payload_only(field, value):
    gateway, base, context, request = setup()
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context, **(request | {field: value}))
    assert not base.calls


@pytest.mark.parametrize("change", [{"stage": "GENERATE"}, {"stage": "EXTRACT"}, {"run_id": "other"}, {"attempt": 2}])
def test_no_generation_extraction_foreign_run_or_retry(change):
    gateway, base, context, request = setup()
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context.model_copy(update=change), **request)
    assert not base.calls


def test_repeat_native_call_is_blocked():
    gateway, base, context, request = setup()
    gateway.generate(context, **request)
    with pytest.raises(ValueError):
        gateway.generate(context, **request)
    assert len(base.calls) == 1


def test_provider_failure_stops_arm_without_retry():
    gateway, base, context, request = setup(FakeGateway(error=True))
    with pytest.raises(RuntimeError):
        gateway.generate(context, **request)
    assert gateway.halted
    with pytest.raises(ValueError):
        gateway.generate(context, **request)
    assert len(base.calls) == 1


def test_cost_stop_blocks_another_distinct_task():
    base = FakeGateway(cost=STOP)
    tasks = [TASK, TASK | {"question": "Which policies allow KMS?", "search": "Which policies allow KMS?"}]
    gateway = ScopedGateway(base, "a" * 64, tasks)
    for i, task in enumerate(tasks):
        context = preparation_context("a" * 64, task["question"], task["search"], "en", RUN)
        request = dict(system=SYSTEM, prompt=preparation_payload(task["question"], task["search"], "en"), max_output_tokens=256)
        if i == 0:
            gateway.generate(context, **request)
        else:
            with pytest.raises(ValueError):
                gateway.generate(context, **request)
    assert len(base.calls) == 1
