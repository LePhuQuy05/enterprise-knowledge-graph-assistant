"""Read-only DynamoDB reconciliation of every native-V7 exported receipt."""

import json
from collections import defaultdict
from datetime import UTC, datetime
from decimal import Decimal

from app.contracts import canonical_json, sha256_bytes
from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
from app.ledger import DynamoLedgerStore, Ledger, Operation
from scripts.audit_source_graph_tranche import completed_rows
from scripts.run_semantic_graph_canary import ROOT


def integral(value):
    if not isinstance(value, Decimal) or value != value.to_integral_value():
        raise ValueError("TYPED_RECONCILIATION_NONINTEGRAL_DYNAMODB_NUMBER")
    return int(value)


def main():
    import boto3
    from boto3.dynamodb.types import TypeDeserializer
    from botocore.config import Config

    base = ROOT / "results/semantic-graph-v7"
    rows = [json.loads(p.read_bytes()) for p in (base / "qualification-v1/units").glob("*.json")]
    rows += [json.loads(p.read_bytes()) for p in (base / "composite-qualification-v1/units").glob("*.json")]
    rows += completed_rows(include_recovery=True)[0]
    receipts = {r["receipt"]["operation_id"]: r["receipt"] for r in rows}
    if len(receipts) != len(rows) or len(rows) != 180:
        raise ValueError("TYPED_RECONCILIATION_EXPECTED_180_UNIQUE_RECEIPTS")
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=20)
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("TYPED_RECONCILIATION_OPERATOR")
    client = session.client("dynamodb", config=cfg)
    ledger = Ledger(DynamoLedgerStore(client, "kg-r72-query-state"))
    decoder = TypeDeserializer()
    operations = {}
    ids = sorted(receipts)
    for offset in range(0, len(ids), 100):
        keys = [{"PK": {"S": "OP#" + op}, "SK": {"S": "STATE"}} for op in ids[offset:offset + 100]]
        response = client.batch_get_item(RequestItems={"kg-r72-query-state": {"Keys": keys, "ConsistentRead": True}})
        if response.get("UnprocessedKeys"):
            raise ValueError("TYPED_RECONCILIATION_UNPROCESSED_READS_NO_AUTOMATIC_RETRY")
        for raw in response["Responses"]["kg-r72-query-state"]:
            # DynamoDB returns integral Decimal values; JSON conversion retains
            # strict model validation without printing account credentials.
            data = {k: decoder.deserialize(v) for k, v in raw.items()}
            operation = Operation.model_validate_json(json.dumps(data, default=integral))
            receipt = receipts[operation.operation_id]
            if (operation.status != "SETTLED" or operation.bound_exceeded or not operation.call_started
                    or operation.actual_micro_usd != receipt["cost_micro_usd"]
                    or operation.usage.model_dump(mode="json") != receipt["usage"]
                    or any(getattr(operation.spec, k) != v for k, v in receipt["context"].items())):
                raise ValueError("TYPED_RECONCILIATION_RECEIPT_OPERATION_MISMATCH")
            operations[operation.operation_id] = operation
    if operations.keys() != receipts.keys():
        raise ValueError("TYPED_RECONCILIATION_MISSING_OPERATION")
    costs = defaultdict(int)
    for op in operations.values():
        costs[op.spec.run_id] += op.actual_micro_usd
    counters = {run: ledger.store.get(ledger.run_key(run)) for run in sorted(costs)}
    if any(c["reserved_micro_usd"] or c["halted"] or c["settled_micro_usd"] != costs[run] for run, c in counters.items()):
        raise ValueError("TYPED_RECONCILIATION_RUN_COUNTER_MISMATCH")
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    report = {"status": "ALL_180_NATIVE_RECEIPTS_SETTLED_USAGE_COST_AND_CONTEXT_MATCH", "checked_at": datetime.now(UTC).isoformat(),
        "operations": len(operations), "settled_micro_usd": sum(costs.values()), "run_costs": dict(costs), "run_counters": counters,
        "operation_ids_sha256": sha256_bytes(canonical_json(sorted(operations))),
        "funding": funding.model_dump(mode="json"),
        "headroom_micro_usd": funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd,
        "provider_calls": 0, "ledger_mutations": 0}
    directory = base / "verification-2026-10-27"
    directory.mkdir(exist_ok=True)
    path = directory / "ledger-reconciliation.json"
    if path.exists():
        raise ValueError("TYPED_RECONCILIATION_REPORT_ALREADY_EXISTS")
    path.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
