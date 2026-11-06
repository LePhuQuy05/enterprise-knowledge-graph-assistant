import json
import sys
from datetime import timedelta

import pytest

from app.contracts import sha256_bytes
from app.cost import TokenUsage
from app.external_gateway import ExternalPriceSheet
from app.extraction_cache import ExtractionAttempt
from app.ledger import ExternalOperationSpec, Ledger, MemoryLedgerStore
from app.model_gateway import CallContext
from scripts import continue_collection_graph_pilot, reconcile_collection_graph_pilot


@pytest.fixture
def settled():
    prices = ExternalPriceSheet.model_validate_json((reconcile_collection_graph_pilot.ROOT /
        "configs/prices/external-direct-2026-10-24.json").read_bytes())
    now = prices.checked_at + timedelta(seconds=1)
    ledger = Ledger(MemoryLedgerStore(), clock=lambda: now)
    ledger.initialize_month("kg-r72", now.strftime("%Y-%m"))
    ledger.initialize_external_funding("kg-r72", cap_micro_usd=2000000)
    context = CallContext(tenant_id="demo", run_id="collection316-graph-pilot-v1", request_id="graph316-chunk", stage="EXTRACT")
    ledger.initialize_run("kg-r72", "demo", context.run_id, cap_micro_usd=500000)
    digest = sha256_bytes(b"fixture exact body")
    spec = ExternalOperationSpec(**context.model_dump(), provider="DEEPSEEK", input_sha256=digest,
        model_id="deepseek-flash", invocation_id="deepseek-flash", source_region="external-default",
        price_sheet_version=prices.version, max_cost_micro_usd=317031)
    ledger.reserve(spec)
    ledger.begin_call(spec.operation_id)
    usage = TokenUsage(input_tokens=100, output_tokens=50)
    ledger.settle(spec.operation_id, 90, usage)
    receipt = ExtractionAttempt(operation_id=spec.operation_id, attempt=1, text="{}", stop_reason="end_turn",
        usage=usage, original_cost_micro_usd=90)
    return ledger.operation(spec.operation_id), context, receipt, digest, prices


def test_recovery_recomputes_exact_receipt_cost_usage_and_input(settled):
    operation, context, receipt, digest, prices = settled
    reconcile_collection_graph_pilot.verify_operation(operation, context, receipt, digest, prices)
    for bad_receipt, bad_digest, code in [
        (receipt.model_copy(update={"original_cost_micro_usd": 89}), digest, "COST_MISMATCH"),
        (receipt.model_copy(update={"usage": TokenUsage(input_tokens=101, output_tokens=50)}), digest, "COST_MISMATCH"),
        (receipt, "0" * 64, "BINDING"),
    ]:
        with pytest.raises(ValueError, match=code):
            reconcile_collection_graph_pilot.verify_operation(operation, context, bad_receipt, bad_digest, prices)


def test_recovery_never_treats_unknown_spend_as_safe_to_continue(settled):
    operation, context, receipt, digest, prices = settled
    with pytest.raises(ValueError, match="NOT_SETTLED"):
        reconcile_collection_graph_pilot.verify_operation(operation.model_copy(update={"status": "UNKNOWN_SPEND"}),
            context, receipt, digest, prices)


@pytest.mark.parametrize("module", [continue_collection_graph_pilot, reconcile_collection_graph_pilot])
def test_default_commands_remain_offline(module, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [module.__name__])
    monkeypatch.setattr(module, "managed_ledger", lambda: (_ for _ in ()).throw(AssertionError("NO_AWS")))
    module.main()
    assert json.loads(capsys.readouterr().out)["status"] in {
        "PLANNED_NO_MODEL_OR_AWS_CALL", "NO_AWS_OR_MODEL_CALLS_USE_READ_MANAGED"}


def test_continuation_refuses_a_started_checkpoint_before_any_managed_read(monkeypatch):
    path = continue_collection_graph_pilot.ROOT / "results/collection-graph/pilot-continuation-v1.json"
    # Simulate a prior process marker without touching the real experiment outputs.
    original = type(path).exists
    monkeypatch.setattr(type(path), "exists", lambda p: True if p == path else original(p))
    monkeypatch.setattr(sys, "argv", ["continue_collection_graph_pilot", "--execute"])
    monkeypatch.setattr(continue_collection_graph_pilot, "managed_ledger", lambda: (_ for _ in ()).throw(AssertionError("NO_AWS")))
    with pytest.raises(ValueError, match="ALREADY_STARTED"):
        continue_collection_graph_pilot.main()
