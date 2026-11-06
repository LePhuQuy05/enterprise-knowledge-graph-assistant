from types import SimpleNamespace

import pytest

from app.full_graph_workflow import ScopedExtractionGateway, qualification_units
from app.model_gateway import CallContext


def units():
    return [{"unit_id": f"u-{g}-{i}", "text_sha256": str(i) * 64,
        "occurrences": [{"service_group": g, "document_version_id": f"d-{g}-{i}"}]}
        for g in ("s3", "lambda", "kms") for i in range(3)]


def test_source_only_stratification_deterministic_distinct_and_covers_groups():
    expected = qualification_units(units())
    assert qualification_units(list(reversed(units()))) == expected
    assert len(expected) == 6 and len({u["unit_id"] for u in expected}) == 6
    assert {u["selected_group"] for u in expected} == {"s3", "lambda", "kms"}
    with pytest.raises(ValueError, match="BOUND"):
        qualification_units(units(), per_group=0)
    with pytest.raises(ValueError, match="INSUFFICIENT"):
        qualification_units(units(), per_group=4)


def test_gateway_binds_actual_prompt_stage_output_and_single_dispatch(monkeypatch):
    from app import full_graph_workflow as workflow

    monkeypatch.setattr(workflow, "extraction_request", lambda text, recipe: ("system", "prompt " + text))
    class Gateway:
        provider, routes = SimpleNamespace(live=True), None
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            return SimpleNamespace(actual_micro_usd=9)
    underlying = Gateway()
    context = CallContext(tenant_id="demo", run_id="fixture", request_id="unit", stage="EXTRACT")
    guarded = ScopedExtractionGateway(underlying, ((context, "source"),), None)
    for changed in ({"prompt": "edited"}, {"max_output_tokens": 4096}):
        with pytest.raises(ValueError, match="SCOPE"):
            guarded.generate(context, **(dict(system="system", prompt="prompt source", max_output_tokens=2048) | changed))
    guarded.generate(context, system="system", prompt="prompt source", max_output_tokens=2048)
    with pytest.raises(ValueError, match="SCOPE"):
        guarded.generate(context, system="system", prompt="prompt source", max_output_tokens=2048)
    assert underlying.calls == 1 and guarded.cost == 9
