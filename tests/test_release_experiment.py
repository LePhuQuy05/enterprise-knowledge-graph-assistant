from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.model_gateway import CallContext
from app.release_experiment import ExperimentStopped, ReleaseExperiment


def gateway(*, actual=1000, status="SETTLED", fail=False):
    calls = []
    def generate(context, **kwargs):
        calls.append(context.request_id)
        if fail:
            raise TimeoutError("synthetic")
        return "result"
    return SimpleNamespace(calls=calls, generate=generate, provider=SimpleNamespace(live=False),
        ledger=SimpleNamespace(_now=lambda: datetime.now(UTC), operation=lambda _: SimpleNamespace(status=status, actual_micro_usd=actual)),
        routes=SimpleNamespace(generation_context_ceiling=1048576),
        prices=SimpleNamespace(rate_for=lambda *a, **k: SimpleNamespace(quote=lambda _: 315533)))


def context(name):
    return CallContext(tenant_id="demo", run_id="external-pilot10-r72-v1", request_id=name, stage="GENERATE")


def test_single_writer_and_existing_intents_cannot_dispatch_twice(tmp_path):
    g = gateway()
    with ReleaseExperiment(tmp_path, g) as e:
        with pytest.raises(ExperimentStopped):
            with ReleaseExperiment(tmp_path, g):
                pass
        assert e.generate(context("first"), system="a", prompt="b") == "result"
        with pytest.raises(ExperimentStopped, match="EXISTING_INTENT"):
            e.generate(context("first"), system="a", prompt="b")
    with ReleaseExperiment(tmp_path, g) as e:
        assert e.generate(context("second"), system="a", prompt="b") == "result"
    assert g.calls == ["first", "second"]


def test_unknown_spend_is_retained_and_stops_later_operations(tmp_path):
    g = gateway(status="UNKNOWN_SPEND", actual=None, fail=True)
    with ReleaseExperiment(tmp_path, g) as e:
        with pytest.raises(TimeoutError):
            e.generate(context("first"), system="a", prompt="b")
        with pytest.raises(ExperimentStopped, match="UNRESOLVED"):
            e.generate(context("second"), system="a", prompt="b")
    assert len(g.calls) == 1


def test_bound_and_settled_count_are_gates_before_dispatch(tmp_path):
    g = gateway(actual=60000)
    with ReleaseExperiment(tmp_path, g) as e:
        e.generate(context("first"), system="a", prompt="b")
        e.OBLIGATION_CAP = 370000
        with pytest.raises(ExperimentStopped, match="OBLIGATION"):
            e.generate(context("second"), system="a", prompt="b")
        e.OBLIGATION_CAP = 400000
        e.MAX_OPERATIONS = 1
        with pytest.raises(ExperimentStopped, match="COUNT"):
            e.generate(context("second"), system="a", prompt="b")
    assert len(g.calls) == 1
