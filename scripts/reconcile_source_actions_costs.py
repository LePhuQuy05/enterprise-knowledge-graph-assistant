"""Read-only ledger audit of native V10, including exact provider request hashes."""

import json
from datetime import UTC, datetime
from decimal import Decimal

from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalPriceSheet
from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
from app.ledger import DynamoLedgerStore, Ledger, Operation
from app.semantic_extraction_cache_v10 import SemanticExtractionCache
from app.semantic_extraction_v10 import semantic_request
from app.source_graph_jobs_v10 import context_for
from scripts.run_source_actions_qualification import DIRECTORY, ROOT, RUN, freeze


def integral(value):
    if not isinstance(value, Decimal) or value != value.to_integral_value():
        raise ValueError("ACTION_RECONCILIATION_NONINTEGRAL_NUMBER")
    return int(value)


def main():
    import boto3
    from boto3.dynamodb.types import TypeDeserializer
    from botocore.config import Config

    plan, chunks, recipe = freeze()
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v10", namespace="LIVE")
    receipts, hashes = {}, {}
    for unit in plan["selection"]:
        text = chunks[unit["occurrences"][0]["chunk_id"]].text
        lookup = cache.resolve(context_for(unit, RUN), text=text, recipe=recipe)
        saved = json.loads((DIRECTORY / "units" / (unit["v10_unit_id"] + ".json")).read_bytes())
        if canonical_json(lookup.receipt) != canonical_json(saved["receipt"]):
            raise ValueError("ACTION_RECONCILIATION_NATIVE_RECEIPT_CHANGED")
        system, prompt = semantic_request(text)
        payload = canonical_json({"model": recipe.model_id, "messages": [{"role": "system", "content": system},
            {"role": "user", "content": prompt}], "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": 2048, "stream": False, "response_format": {"type": "json_object"}})
        receipts[lookup.receipt["operation_id"]] = lookup.receipt
        hashes[lookup.receipt["operation_id"]] = sha256_bytes(payload)
    if len(receipts) != 32:
        raise ValueError("ACTION_RECONCILIATION_EXPECTED_32_DISTINCT_RECEIPTS")
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    rate = prices.rate_for(recipe.model_id, recipe.invocation_id, recipe.source_region, now=datetime.now(UTC), live=True)
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=20)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("ACTION_RECONCILIATION_OPERATOR")
    client = session.client("dynamodb", config=cfg)
    ledger = Ledger(DynamoLedgerStore(client, "kg-r72-query-state"))
    response = client.batch_get_item(RequestItems={"kg-r72-query-state": {"ConsistentRead": True,
        "Keys": [{"PK": {"S": "OP#" + key}, "SK": {"S": "STATE"}} for key in sorted(receipts)]}})
    if response.get("UnprocessedKeys"):
        raise ValueError("ACTION_RECONCILIATION_UNPROCESSED_READS_NO_RETRY")
    decoder, checked = TypeDeserializer(), set()
    for raw in response["Responses"]["kg-r72-query-state"]:
        op = Operation.model_validate_json(json.dumps({k: decoder.deserialize(v) for k, v in raw.items()}, default=integral))
        receipt = receipts[op.operation_id]
        if (op.status != "SETTLED" or op.bound_exceeded or not op.call_started or op.spec.input_sha256 != hashes[op.operation_id]
                or op.actual_micro_usd != receipt["cost_micro_usd"] or op.actual_micro_usd != rate.quote(op.usage)
                or op.usage.model_dump(mode="json") != receipt["usage"] or op.spec.price_sheet_version != prices.version
                or any(getattr(op.spec, key) != value for key, value in receipt["context"].items())):
            raise ValueError("ACTION_RECONCILIATION_OPERATION_OR_PAYLOAD_MISMATCH")
        checked.add(op.operation_id)
    if checked != receipts.keys():
        raise ValueError("ACTION_RECONCILIATION_MISSING_RECEIPT")
    counter = ledger.store.get(ledger.run_key(RUN))
    cost = sum(r["cost_micro_usd"] for r in receipts.values())
    if counter["reserved_micro_usd"] or counter["halted"] or counter["settled_micro_usd"] != cost:
        raise ValueError("ACTION_RECONCILIATION_RUN_COUNTER_MISMATCH")
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    report = {"status": "ALL_32_NATIVE_V10_RECEIPTS_SETTLED_CONTEXT_USAGE_COST_PAYLOAD_MATCH", "checked_at": datetime.now(UTC).isoformat(),
        "operations": len(checked), "settled_micro_usd": cost, "run_counter": counter, "funding": funding.model_dump(mode="json"),
        "headroom_micro_usd": funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd,
        "operation_ids_sha256": sha256_bytes(canonical_json(sorted(checked))), "provider_calls": 0, "ledger_mutations": 0}
    path = DIRECTORY / "ledger-reconciliation.json"
    if path.exists():
        raise ValueError("ACTION_RECONCILIATION_REPORT_ALREADY_EXISTS")
    path.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
