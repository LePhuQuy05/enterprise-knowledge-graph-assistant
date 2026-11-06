"""Exercise the actual paid-run admission seam without API calls."""

from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.contracts import canonical_json
from app.external_http import OfficialHttpsTransport
from app.scoped_query_v6 import SYSTEM, preparation_context
from scripts import run_scoped_flow_canary as canary


def test_canary_transport_settings_obey_actual_transport_contract():
    transport = OfficialHttpsTransport({"DEEPSEEK": SecretStr("fixture-key")},
        timeout_seconds=canary.LIVE_TIMEOUT_SECONDS)
    assert transport.dispatch_attempts == 0


def bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(canary, "OUT", tmp_path)
    (tmp_path / "scope.json").write_bytes(b"{}")
    scope = {"inputs": [{"question": "What can S3 invoke?", "search": "What can S3 invoke?", "language": "en"}],
        "collection_sha256": "a" * 64, "max_calls": 24, "published_usage_stop_micro_usd": 100000}
    base = SimpleNamespace(provider=SimpleNamespace(live=False), routes=SimpleNamespace())
    gateway = canary.BoundedGateway(base, scope)
    task = scope["inputs"][0]
    context = preparation_context("a" * 64, task["question"], task["search"], "en", canary.RUN)
    prompt = canonical_json({"untrusted_original_question": task["question"],
        "untrusted_english_search_query": task["search"], "original_language": "en"}).decode()
    return gateway, context, ("PLAN_QUERY", SYSTEM, prompt, 256)


def test_actual_dispatch_admission_is_single_use_and_journals_intent(tmp_path, monkeypatch):
    gateway, context, spec = bounded(tmp_path, monkeypatch)
    gateway._before(context, spec)
    assert gateway.calls == 1
    assert len(list((tmp_path / "dispatch-intents").glob("*.json"))) == 1
    with pytest.raises(ValueError, match="DISPATCH_NOT_ALLOWED"):
        gateway._before(context, spec)
    assert gateway.calls == 1


@pytest.mark.parametrize("change", ["prompt", "run", "attempt", "cap", "cost", "halt"])
def test_mutated_or_exhausted_dispatch_is_rejected_before_intent(tmp_path, monkeypatch, change):
    gateway, context, spec = bounded(tmp_path, monkeypatch)
    if change == "prompt":
        spec = (spec[0], spec[1], "different task", spec[3])
    elif change in {"run", "attempt"}:
        context = context.model_copy(update={"run_id": "different-run"} if change == "run" else {"attempt": 2})
    elif change == "cap":
        gateway.calls = 24
    elif change == "cost":
        gateway.cost = 100000
    else:
        gateway.halted = True
    with pytest.raises(ValueError, match="DISPATCH_NOT_ALLOWED"):
        gateway._before(context, spec)
    assert not (tmp_path / "dispatch-intents").exists()
