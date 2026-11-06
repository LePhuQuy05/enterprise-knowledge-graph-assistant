from types import SimpleNamespace

import pytest

from app.model_gateway import CallContext
from scripts.reader_canary_runtime import ControlledJob


def test_job_does_not_let_arbitrary_protocol_or_1600_token_text_reach_base_gateway(tmp_path):
    base = SimpleNamespace(provider=SimpleNamespace(live=False), routes=SimpleNamespace())
    job = ControlledJob(base, None, tmp_path, "fixture-job", {"max_calls": 4,
        "allowed_protocols": [("Return JSON", "EXTRACT", 768)]})
    call = CallContext(tenant_id="demo", run_id="fixture-job", request_id="fixture", stage="GENERATE")
    with pytest.raises(ValueError, match="STAGE_OR_BODY"):
        job.allow(call, "Return JSON", "fixture", 1600)
    extract = call.model_copy(update={"stage": "EXTRACT"})
    with pytest.raises(ValueError, match="NOT_FROZEN"):
        job.allow(extract, "Return another JSON format", "fixture", 768)
    assert job.calls == 0


@pytest.mark.parametrize("exhaustion", ["count", "cost", "halt"])
def test_job_admission_stops_without_dispatch_after_any_frozen_limit(tmp_path, exhaustion):
    base = SimpleNamespace(provider=SimpleNamespace(live=False), routes=SimpleNamespace())
    job = ControlledJob(base, None, tmp_path, "fixture-job", {"max_calls": 4,
        "allowed_protocols": [("Return JSON", "EXTRACT", 768)]})
    call = CallContext(tenant_id="demo", run_id="fixture-job", request_id="fixture", stage="EXTRACT")
    job.allow(call, "Return JSON", "fixture", 768)
    if exhaustion == "count":
        job.calls = 4
    elif exhaustion == "cost":
        job.cost = 100000
    else:
        job.halted = True
    with pytest.raises(ValueError, match="NOT_ALLOWED"):
        job.generate(call, system="Return JSON", prompt="fixture", max_output_tokens=768)
    assert not (tmp_path / "dispatch-intents").exists()
