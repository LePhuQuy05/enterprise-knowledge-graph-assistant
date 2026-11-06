"""Reconcile a stopped V4 canary without retrying or changing its cache."""

import json
import sqlite3
from datetime import UTC, datetime

from app.contracts import canonical_json, sha256_bytes
from app.faceted_query_v4 import validate_facets
from scripts.pilot_reviewed_source_answers import immutable
from scripts.run_facet_recall_v4 import CACHE, OUTPUT, RUN, SYSTEM, inputs, provider_body


def audit():
    import boto3
    from botocore.config import Config

    from app.ledger import DynamoLedgerStore, Ledger

    tasks, _ = inputs()
    before = sha256_bytes((CACHE / "task_facets_v4.sqlite3").read_bytes())
    db = sqlite3.connect(f"file:{(CACHE / 'task_facets_v4.sqlite3').as_posix()}?mode=ro", uri=True)
    rows = db.execute("SELECT state,binding,payload,payload_sha,error FROM preparations ORDER BY context_key").fetchall()
    db.close()
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("STOP_AUDIT_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    audited = []
    for state, binding, payload, digest, error in rows:
        if sha256_bytes(payload) != digest:
            raise ValueError("STOP_AUDIT_CACHE_CHECKSUM")
        binding, receipt = json.loads(binding), json.loads(payload)
        task = next(t for t in tasks if t["language"] == binding["language"]
            and sha256_bytes(t["question"].encode()) == binding["question"])
        op = ledger.operation(receipt["operation_id"])
        expected = sha256_bytes(provider_body(SYSTEM, canonical_json({
            "untrusted_original_question": task["question"], "untrusted_english_search_query": task["search"],
            "original_language": task["language"]}).decode()))
        if (op.status != "SETTLED" or not op.call_started or op.bound_exceeded or op.spec.run_id != RUN
                or op.spec.input_sha256 != expected or op.usage.model_dump(mode="json") != receipt["usage"]
                or op.actual_micro_usd != receipt["cost_micro_usd"]
                or sha256_bytes(receipt["raw_output"].encode()) != receipt["raw_output_sha256"]):
            raise ValueError("STOP_AUDIT_LEDGER_RECEIPT")
        try:
            facets = validate_facets(receipt["raw_output"], task["question"], task["search"])
            valid, reason = True, None
        except ValueError as exc:
            facets, valid, reason = (), False, str(exc)
        if (state == "READY") != valid or state not in {"READY", "FAILED"}:
            raise ValueError("STOP_AUDIT_CACHE_STATE")
        raw = json.loads(receipt["raw_output"])
        audited.append({"intent": task["intent"], "language": task["language"], "state": state,
            "receipt": receipt, "payload_sha256": digest, "provider_payload_sha256": expected,
            "context": binding["context"], "validation_passed": valid, "validation_error": reason,
            "raw_facet_count": len(raw["facets"]), "validated_facet_count": len(facets),
            "cache_error": error, "ledger_status": op.status})
    cost = sum(r["receipt"]["cost_micro_usd"] for r in audited)
    run = ledger.store.get(ledger.run_key(RUN))
    funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
    if (len(audited) != 2 or run["reserved_micro_usd"] or run["settled_micro_usd"] != cost
            or sha256_bytes((CACHE / "task_facets_v4.sqlite3").read_bytes()) != before):
        raise ValueError("STOP_AUDIT_COUNTER_OR_CACHE_MUTATION")
    result = {"status": "TWO_SETTLED_ONE_READY_ONE_FAILED_NO_RETRY", "checked_at": datetime.now(UTC).isoformat(),
        "rows": audited, "run_counter": run, "funding_counter": funding,
        "headroom_micro_usd": funding["cap_micro_usd"] - funding["settled_micro_usd"] - funding["reserved_micro_usd"],
        "cost_micro_usd": cost, "new_model_calls": 0, "raw_cache_sha256_unchanged": before,
        "no_complete_four_input_comparison": True, "generation_calls": 0, "extraction_calls": 0,
        "audit_code_sha256": sha256_bytes(__import__('pathlib').Path(__file__).read_bytes())}
    immutable(OUTPUT / "stopped-ledger-audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"rows", "run_counter", "funding_counter"}}))


if __name__ == "__main__":
    audit()
