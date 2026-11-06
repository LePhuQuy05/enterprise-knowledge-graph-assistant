"""Local dispatch guards; no real credentials, funding, network or inference."""

from types import SimpleNamespace

import pytest

from app.model_gateway import CallContext
from scripts.pilot_source_message_answers_v2 import (
    METHODS,
    QUESTIONS,
    RUN,
    ScopedGateway,
    preparation_context,
    validate_run_capacity,
)
from tests.test_reviewed_source_answer_canary import Base


def context(index=0):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=f"fixture-{index}", stage="GENERATE")


def test_two_intents_four_language_cases_are_not_thirty_two_independent_questions():
    assert len(QUESTIONS) == 2 and all(set(q) == {"en", "vi"} for q in QUESTIONS.values())
    assert len(set(METHODS)) == 8 and len(QUESTIONS) * 2 * len(METHODS) == 32
    ids = {preparation_context(intent, language, "PLAN_QUERY").request_id for intent in QUESTIONS for language in ("en", "vi")}
    assert len(ids) == 4


@pytest.mark.parametrize("change", ["prompt", "tokens", "context", "repeat"])
def test_frozen_dispatch_rejects_changed_or_repeated_request(change):
    base = Base()
    gateway = ScopedGateway(base)
    gateway.allow(context(), "system", "prompt", 800)
    if change == "repeat":
        gateway.generate(context(), system="system", prompt="prompt", max_output_tokens=800)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(1) if change == "context" else context(), system="system",
            prompt="changed" if change == "prompt" else "prompt", max_output_tokens=801 if change == "tokens" else 800)
    assert base.calls == (1 if change == "repeat" else 0)


def test_uncertain_failure_halts_the_run_without_a_second_call():
    base = Base(fail=True)
    gateway = ScopedGateway(base)
    gateway.allow(context(), "s", "p", 800)
    gateway.allow(context(1), "s", "p", 800)
    with pytest.raises(ValueError):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(1), system="s", prompt="p", max_output_tokens=800)
    assert base.calls == 1


def test_run_count_and_observed_spend_stop_are_both_enforced():
    base = Base()
    gateway = ScopedGateway(base)
    gateway.allow(context(), "s", "p", 800)
    gateway.cost = 150000
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    gateway.cost = 0
    gateway.seen = set(str(i) for i in range(38))
    with pytest.raises(ValueError, match="DISPATCH_BOUND"):
        gateway.generate(context(), system="s", prompt="p", max_output_tokens=800)
    assert base.calls == 0


def test_foreign_run_cannot_register_a_paid_operation():
    gateway = ScopedGateway(Base())
    foreign = context().model_copy(update={"run_id": "other"})
    with pytest.raises(ValueError, match="NATIVE_SCOPE"):
        gateway.allow(foreign, "s", "p", 800)
    assert gateway.provider == SimpleNamespace(live=True)


def test_real_accepted_context_ceiling_fits_run_without_reducing_reservation():
    from pathlib import Path

    from app.external_gateway import ExternalPriceSheet, ExternalRoutes

    root = Path(__file__).resolve().parents[1]
    routes = ExternalRoutes.model_validate_json((root / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((root / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    rate = next(r for r in prices.rates if r.model_id == routes.generation_model_id)
    bound = validate_run_capacity(rate, routes)
    assert bound == 315533 and bound + 150000 <= 500000
    with pytest.raises(ValueError, match="SINGLE_CALL_RESERVATION"):
        validate_run_capacity(rate, routes, run_cap=300000)
