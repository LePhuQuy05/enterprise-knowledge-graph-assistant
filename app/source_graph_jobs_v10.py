"""Scoped native-V10 calls with the existing audited ledger and direct provider."""

import threading
from datetime import UTC, datetime

from app.contracts import sha256_bytes
from app.model_gateway import CallContext
from app.semantic_extraction_v10 import semantic_request


def context_for(unit, run_id):
    return CallContext(tenant_id="demo", run_id=run_id, request_id=unit["v10_unit_id"], stage="EXTRACT")


class BoundedGatewayV10:
    def __init__(self, base, plan, chunks):
        self.base, self.provider, self.routes = base, base.provider, base.routes
        self.allowed = {context_for(u, plan["run_id"]).model_dump_json(): semantic_request(chunks[u["occurrences"][0]["chunk_id"]].text)
            for u in plan["selection"]}
        self.maximum, self.stop = plan["maximum_new_calls"], plan["actual_cost_stop_between_calls_micro_usd"]
        self.seen, self.cost, self.halted, self.lock = set(), 0, False, threading.Lock()

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        with self.lock:
            if (self.halted or self.cost >= self.stop or key in self.seen or key not in self.allowed
                    or len(self.seen) >= self.maximum or self.allowed[key] != (system, prompt) or max_output_tokens != 2048):
                raise ValueError("V10_DISPATCH_SCOPE_OR_COST_BOUND")
            self.seen.add(key)
        try:
            response = self.base.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
        except Exception:
            with self.lock:
                self.halted = True
            raise
        with self.lock:
            self.cost += response.actual_micro_usd
        return response


def managed_gateway(root, plan, chunks):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=20)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("V10_OPERATOR_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    if (funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < plan["run_cap_micro_usd"]
            or not month or month["halted"] or month["cap_micro_usd"] != 12000000):
        raise ValueError("V10_EXISTING_FUNDING_AND_MONTH_HEADROOM_REQUIRED")
    raw = (root / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(raw) != plan["price_sheet_sha256"]:
        raise ValueError("V10_PINNED_PRICE_SHEET_CHANGED")
    prices = ExternalPriceSheet.model_validate_json(raw)
    routes = ExternalRoutes.model_validate_json((root / "configs/models/external-direct-v1.json").read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    ledger.initialize_run("kg-r72", "demo", plan["run_id"], cap_micro_usd=plan["run_cap_micro_usd"])
    run = ledger.store.get(ledger.run_key(plan["run_id"]))
    if run["halted"] or run["reserved_micro_usd"] or run["cap_micro_usd"] != plan["run_cap_micro_usd"]:
        raise ValueError("V10_RUN_RECONCILIATION_REQUIRED")
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(root, "DEEPSEEK")}, timeout_seconds=25)
    preflight = deepseek_preflight(transport)
    if not preflight["available"] or preflight["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("V10_PROVIDER_ROUTE_CHANGED")
    base = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "FROZEN_NATIVE_V10_SOURCE_QUALIFICATION"}), prices)
    return BoundedGatewayV10(base, plan, chunks), ledger
