"""Probe four-item transactions on existing DynamoDB using a separate fixture project.

No model call, new table, IAM change, production budget reset or provider charge.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import boto3
from botocore.config import Config

from app.contracts import text_hash
from app.cost import TokenUsage
from app.ledger import (
    AlreadyDispatched,
    BudgetDenied,
    DynamoLedgerStore,
    ExternalOperationSpec,
    Ledger,
    LedgerError,
)

ROOT = Path(__file__).resolve().parents[1]
PROJECT = "external-protocol-fixture-v1"


def verify(ledger: Ledger) -> dict:
    run = "external-probe-" + uuid4().hex[:16]
    period = ledger._now().strftime("%Y-%m")
    ledger.initialize_month(PROJECT, period, cap_micro_usd=2_000,
                            other_cost_estimate_micro_usd=0, gross_stop_micro_usd=2_000)
    ledger.initialize_run(PROJECT, "fixture", run, cap_micro_usd=2_000)
    ledger.initialize_external_funding(PROJECT, cap_micro_usd=1_000)

    def spec(request, provider="DEEPSEEK", bound=600):
        return ExternalOperationSpec(
            project_id=PROJECT, tenant_id="fixture", run_id=run, request_id=request,
            stage="SMOKE", attempt=1, input_sha256=text_hash("FIXTURE_NO_PROVIDER_CALL"),
            provider=provider, model_id="fixture-only", invocation_id="fixture-only",
            source_region="external-default", price_sheet_version="FIXTURE_NO_PROVIDER_CALL",
            max_cost_micro_usd=bound,
        )

    first = ledger.reserve(spec("first"))
    assert ledger.reserve(first.spec) == first
    second = ledger.reserve(spec("second", "VOYAGE", 400))
    try:
        ledger.reserve(spec("denied", bound=1))
        raise AssertionError("Shared funding cap was bypassed")
    except BudgetDenied:
        pass
    ledger.cancel_before_call(second.operation_id)
    ledger.cancel_before_call(second.operation_id)

    def begin(_):
        try:
            ledger.begin_call(first.operation_id)
            return True
        except AlreadyDispatched:
            return False

    with ThreadPoolExecutor(max_workers=4) as pool:
        winners = sum(pool.map(begin, range(4)))
    assert winners == 1
    ledger.mark_unknown(first.operation_id)
    try:
        ledger.cancel_before_call(first.operation_id)
        raise AssertionError("Unknown reservation was refunded")
    except LedgerError:
        pass
    counter = ledger.store.get(ledger.external_funding_key(PROJECT))
    assert counter["reserved_micro_usd"] == 600
    # Explicitly synthetic usage/zero charge in a separate fixture project, never inference.
    usage = TokenUsage(input_tokens=1, output_tokens=0)
    ledger.settle(first.operation_id, 0, usage)
    ledger.settle(first.operation_id, 0, usage)
    fresh = Ledger(ledger.store)
    for key in (ledger.month_key(PROJECT, period), ledger.run_key(run), ledger.external_funding_key(PROJECT)):
        counter = fresh.store.get(key)
        assert counter["reserved_micro_usd"] == counter["settled_micro_usd"] == 0
    assert fresh.operation(first.operation_id).spec.provider == "DEEPSEEK"
    return {"status": "PASS", "fixture_project": PROJECT, "run_id": run,
            "four_item_reserve_settle_cancel": True, "cross_provider_cap": True,
            "competing_dispatches": 4, "authorized_dispatches": winners,
            "unknown_not_refunded": True, "replay_no_double_charge": True,
            "fresh_reader": True, "model_calls": 0, "production_budget_mutations": 0}


def main():
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "BLOCKED", "model_calls": 0}
    try:
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=10)
        identity = session.client("sts", config=cfg).get_caller_identity()
        if not identity["Arn"].endswith(":user/kg-dev"):
            raise ValueError("Known developer identity required")
        client = session.client("dynamodb", config=cfg)
        if client.meta.endpoint_url != "https://dynamodb.us-east-1.amazonaws.com":
            raise ValueError("Official endpoint required")
        report.update(verify(Ledger(DynamoLedgerStore(client, "kg-r72-query-state"))))
        report["backend"] = "MANAGED_DYNAMODB"
    except Exception as exc:
        report["error_category"] = type(exc).__name__
    (ROOT / "results/external_ledger_managed.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
