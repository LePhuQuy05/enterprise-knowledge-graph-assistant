"""Opt-in DynamoDB Local transaction/concurrency smoke, never managed AWS endpoints."""

import json
import subprocess
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

from app.contracts import text_hash
from app.cost import TokenUsage
from app.ledger import (
    AlreadyDispatched,
    BudgetDenied,
    DynamoLedgerStore,
    Ledger,
    LedgerError,
    OperationSpec,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    image = json.loads((ROOT / "configs/runtime_images.json").read_text())["dynamodb_local"]
    name = "kg-r72-ledger-smoke-" + uuid.uuid4().hex[:12]
    table = "LedgerSmoke" + uuid.uuid4().hex[:12]
    report = dict(status="FAIL", checked_at=datetime.now(UTC).isoformat(), image=image,
                  managed_aws_calls=0, inference_calls=0, synthetic_money=True,
                  limitation="DynamoDB Local, not AWS IAM/billing/managed service validation")
    started = False

    def docker(arguments, timeout=30):
        return subprocess.run(["docker", *arguments], capture_output=True, text=True, timeout=timeout)

    try:
        result = docker(["run", "--detach", "--rm", "--name", name, "--memory", "768m",
                         "--cpus", "1", "--publish", "127.0.0.1::8000", image,
                         "-jar", "DynamoDBLocal.jar", "-inMemory", "-sharedDb"])
        if result.returncode:
            raise RuntimeError("Container start failed")
        started = True
        port_output = docker(["port", name, "8000/tcp"])
        address, port = port_output.stdout.strip().rsplit(":", 1)
        if port_output.returncode or address != "127.0.0.1" or not 1 <= int(port) <= 65535:
            raise RuntimeError("Container must bind only an ephemeral loopback port")
        client = boto3.client(
            "dynamodb", region_name="us-east-1", endpoint_url=f"http://127.0.0.1:{int(port)}",
            aws_access_key_id="fixture", aws_secret_access_key="fixture",
            config=Config(connect_timeout=2, read_timeout=10, retries={"total_max_attempts": 1},
                          max_pool_connections=16),
        )
        deadline = time.monotonic() + 90
        while True:
            try:
                client.list_tables(Limit=1)
                break
            except (BotoCoreError, ClientError):
                if time.monotonic() >= deadline:
                    raise RuntimeError("DynamoDB Local readiness timeout") from None
                time.sleep(1)
        client.create_table(
            TableName=table, BillingMode="PAY_PER_REQUEST",
            KeySchema=[{"AttributeName": "PK", "KeyType": "HASH"},
                       {"AttributeName": "SK", "KeyType": "RANGE"}],
            AttributeDefinitions=[{"AttributeName": "PK", "AttributeType": "S"},
                                  {"AttributeName": "SK", "AttributeType": "S"}],
        )
        now = datetime(2026, 10, 16, tzinfo=UTC)
        ledger = Ledger(DynamoLedgerStore(client, table), clock=lambda: now)
        ledger.initialize_month("local", "2026-10", cap_micro_usd=100,
                                other_cost_estimate_micro_usd=0, gross_stop_micro_usd=100)
        ledger.initialize_run("local", "tenant_fixture", "localrun", cap_micro_usd=100)
        report["stage"] = "reserve_race"

        def spec(request):
            return OperationSpec(
                project_id="local", tenant_id="tenant_fixture", run_id="localrun", request_id=request,
                stage="SMOKE", attempt=1, input_sha256=text_hash("synthetic ledger integration"),
                model_id="fixture", invocation_id="fixture", source_region="us-east-1",
                price_sheet_version="fixture", max_cost_micro_usd=60,
            )

        def reserve(index):
            try:
                return ledger.reserve(spec(f"req{index}"))
            except BudgetDenied:
                return None

        with ThreadPoolExecutor(max_workers=8) as executor:
            operations = [operation for operation in executor.map(reserve, range(16)) if operation]
        report["successful_reservations"] = len(operations)
        assert len(operations) == 1
        operation = operations[0]
        assert ledger.reserve(operation.spec).operation_id == operation.operation_id

        def begin(_):
            try:
                ledger.begin_call(operation.operation_id)
                return True
            except AlreadyDispatched:
                return False

        report["stage"] = "dispatch_race"
        with ThreadPoolExecutor(max_workers=8) as executor:
            dispatches = sum(executor.map(begin, range(12)))
        report["authorized_dispatches"] = dispatches
        assert dispatches == 1
        report["stage"] = "unknown_and_rollover_settlement"
        ledger.mark_unknown(operation.operation_id)
        try:
            ledger.cancel_before_call(operation.operation_id)
            raise AssertionError("Unknown operation was refunded")
        except LedgerError:
            pass
        counter = ledger.store.get(ledger.month_key("local", "2026-10"))
        assert counter["reserved_micro_usd"] == 60 and counter["settled_micro_usd"] == 0
        now = datetime(2026, 11, 1, tzinfo=UTC)
        ledger.initialize_month("local", "2026-11", cap_micro_usd=100)
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        ledger.settle(operation.operation_id, 25, usage)
        ledger.settle(operation.operation_id, 25, usage)
        fresh = Ledger(DynamoLedgerStore(client, table), clock=lambda: now)
        october = fresh.store.get(fresh.month_key("local", "2026-10"))
        november = fresh.store.get(fresh.month_key("local", "2026-11"))
        assert october["settled_micro_usd"] == 25 and october["reserved_micro_usd"] == 0
        assert november["settled_micro_usd"] == november["reserved_micro_usd"] == 0
        assert fresh.operation(operation.operation_id).actual_micro_usd == 25
        report.update(status="PASS", competing_reservations=16, successful_reservations=1,
                      competing_dispatches=12, authorized_dispatches=1,
                      replay_no_double_charge=True, unknown_not_refunded=True,
                      rollover_settled_original_month=True, fresh_reader_observes_result=True)
    except (OSError, ValueError, RuntimeError, AssertionError, BotoCoreError, ClientError,
            subprocess.TimeoutExpired) as exc:
        report["error_category"] = type(exc).__name__
    finally:
        if started:
            result = docker(["stop", "--timeout", "10", name])
            report["temporary_container_stopped"] = result.returncode == 0
        directory = ROOT / "results" / "ledger_dynamodb_local"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" and report.get("temporary_container_stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
