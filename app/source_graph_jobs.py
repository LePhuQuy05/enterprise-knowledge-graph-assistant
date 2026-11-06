"""Reusable bounded native-V7/composite-review jobs; no serving publication."""

import json
import threading
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.model_gateway import CallContext, safe_failure_code
from app.semantic_extraction_cache_v7 import SemanticExtractionCache
from app.semantic_extraction_v7 import semantic_request
from app.semantic_extraction_v8_review import review_output
from app.source_candidate_policy import retained


def context_for(unit, run_id):
    return CallContext(tenant_id="demo", run_id=run_id, request_id=unit["v7_unit_id"], stage="EXTRACT")


class SourceBoundedGateway:
    def __init__(self, gateway, plan, chunks):
        self.gateway, self.provider, self.routes = gateway, gateway.provider, gateway.routes
        self.allowed = {context_for(u, plan["run_id"]).model_dump_json(): semantic_request(chunks[u["occurrences"][0]["chunk_id"]].text)
                        for u in plan["selection"]}
        self.maximum, self.stop = plan["maximum_new_calls"], plan["actual_cost_stop_between_calls_micro_usd"]
        self.seen, self.cost, self.halted, self.lock = set(), 0, False, threading.Lock()

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        with self.lock:
            if (self.halted or self.cost >= self.stop or key in self.seen or key not in self.allowed
                    or len(self.seen) >= self.maximum or self.allowed[key] != (system, prompt) or max_output_tokens != 2048):
                raise ValueError("SOURCE_JOB_DISPATCH_SCOPE_OR_COST_BOUND")
            self.seen.add(key)
        try:
            result = self.gateway.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
        except Exception:
            with self.lock:
                self.halted = True
            raise
        with self.lock:
            self.cost += result.actual_micro_usd
        return result


def managed_source_gateway(root, plan, chunks):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=20)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("SOURCE_JOB_OPERATOR_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    if funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < plan["run_cap_micro_usd"]:
        raise ValueError("SOURCE_JOB_AUDITED_FUNDING_HEADROOM_REQUIRED")
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    if not month or month["halted"] or month["cap_micro_usd"] != 12000000:
        raise ValueError("SOURCE_JOB_EXISTING_MONTH_REQUIRED")
    prices_raw = (root / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(prices_raw) != plan["price_sheet_sha256"]:
        raise ValueError("SOURCE_JOB_PRICE_SHEET_CHANGED")
    prices = ExternalPriceSheet.model_validate_json(prices_raw)
    routes = ExternalRoutes.model_validate_json((root / "configs/models/external-direct-v1.json").read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    ledger.initialize_run("kg-r72", "demo", plan["run_id"], cap_micro_usd=plan["run_cap_micro_usd"])
    run = ledger.store.get(ledger.run_key(plan["run_id"]))
    if run["halted"] or run["reserved_micro_usd"] or run["cap_micro_usd"] != plan["run_cap_micro_usd"]:
        raise ValueError("SOURCE_JOB_RUN_RECONCILIATION_REQUIRED")
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(root, "DEEPSEEK")}, timeout_seconds=25)
    account = deepseek_preflight(transport)
    if not account["available"] or account["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("SOURCE_JOB_PROVIDER_CHANGED")
    base = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "PINNED_NATIVE_V7_AND_SOURCE_REVIEW_JOB"}), prices)
    return SourceBoundedGateway(base, plan, chunks), ledger


def execute_source_plan(root: Path, directory: Path, plan, chunks, recipe, *, replay=False):
    gateway, ledger = (None, None) if replay else managed_source_gateway(root, plan, chunks)
    cache = SemanticExtractionCache(root / "data/cache/semantic-extraction-v7", namespace="LIVE")
    output = directory / "units"
    output.mkdir(exist_ok=True)
    rows, status, error = [], "REPLAY_MATCH" if replay else "NATIVE_V7_COMPOSITE_QUALIFICATION_COMPLETE_REVIEW_PENDING", None
    try:
        for unit in plan["selection"]:
            text = chunks[unit["occurrences"][0]["chunk_id"]].text
            if sha256_bytes(text.encode()) != unit["text_sha256"]:
                raise ValueError("SOURCE_JOB_TEXT_CHANGED")
            lookup = cache.resolve(context_for(unit, plan["run_id"]), text=text, recipe=recipe, gateway=gateway)
            decisions = []
            for decision in review_output(lookup.validation.output, text):
                allowed, reason = retained(decision)
                decisions.append(dict(decision, retention_status="RETAINED_PENDING_SOURCE_REVIEW" if allowed else "QUARANTINED",
                    retention_reason=reason))
            row = {"unit_id": unit["v7_unit_id"], "text_sha256": unit["text_sha256"], "occurrences": unit["occurrences"],
                "receipt": lookup.receipt, "coverage_note": lookup.validation.output.coverage_note, "decisions": decisions,
                "composite_recipe_sha256": sha256_bytes(canonical_json(plan["composite_recipe"])), "semantic_review": "PENDING_NOT_SERVING"}
            raw, path = canonical_json(row) + b"\n", output / (unit["v7_unit_id"] + ".json")
            if path.exists() and path.read_bytes() != raw:
                raise ValueError("SOURCE_JOB_REPLAY_CHANGED")
            path.write_bytes(raw)
            rows.append(row)
            print(json.dumps({"processed": len(rows), "planned": len(plan["selection"]),
                "retained": sum(d["retention_status"] == "RETAINED_PENDING_SOURCE_REVIEW" for d in decisions),
                "quarantined": sum(d["retention_status"] == "QUARANTINED" for d in decisions),
                "new_cost_micro_usd": gateway.cost if gateway else 0}), flush=True)
    except Exception as exc:
        status, error = "STOPPED_NO_AUTOMATIC_RETRY", safe_failure_code(exc)
    counter = ledger.store.get(ledger.run_key(plan["run_id"])) if ledger else None
    own_cost = sum(r["receipt"]["cost_micro_usd"] for r in rows if r["receipt"]["context"]["run_id"] == plan["run_id"])
    if counter and error is None and (counter["reserved_micro_usd"] or counter["settled_micro_usd"] != own_cost):
        status, error = "RECONCILIATION_REQUIRED", "SOURCE_JOB_COST_TOTAL_MISMATCH"
    all_decisions = [d for r in rows for d in r["decisions"]]
    report = {"status": status, "error": error, "processed": len(rows), "planned": len(plan["selection"]),
        "retained_candidates": sum(d["retention_status"] == "RETAINED_PENDING_SOURCE_REVIEW" for d in all_decisions),
        "quarantined_candidates": sum(d["retention_status"] == "QUARANTINED" for d in all_decisions),
        "coverage_notes": dict(Counter(r["coverage_note"] for r in rows)), "run_counter": counter,
        "provider_calls_this_invocation": len(gateway.seen) if gateway else 0, "new_cost_micro_usd": gateway.cost if gateway else 0,
        "own_run_receipt_total_micro_usd": own_cost, "native_v8_inference": False, "human_reviewed": 0,
        "full_batch_executed": False, "serving_records_added": 0, "funding_cap_changed": False,
        "finished_at": datetime.now(UTC).isoformat()}
    attempts = len(list(directory.glob("attempt-*.json"))) + 1
    (directory / ("replay.json" if replay else f"attempt-{attempts:03d}.json")).write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))
    return 0 if error is None else 2
