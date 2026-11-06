"""Managed, bounded local reader jobs. No deployments or automatic retries."""

import json
from datetime import UTC, datetime

from app.contracts import canonical_json, sha256_bytes
from app.routed_reader_json_v2 import require_json_mode_instruction
from scripts.pilot_reviewed_source_answers import immutable


def managed_job(root, out, run_id, scope):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("READER_JOB_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    if funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 500000:
        raise ValueError("READER_JOB_FUNDING_HEADROOM")
    routes = ExternalRoutes.model_validate_json((root / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((root / scope["price_file"]).read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region,
        now=datetime.now(UTC), live=True)
    provider = OfficialHttpsTransport({"DEEPSEEK": load_private_key(root, "DEEPSEEK")}, timeout_seconds=25)
    metadata = deepseek_preflight(provider)
    if not metadata["available"] or metadata["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("READER_JOB_MODEL_CHANGED")
    ledger.initialize_month("kg-r72", datetime.now(UTC).strftime("%Y-%m"))
    ledger.initialize_run("kg-r72", "demo", run_id, cap_micro_usd=500000)
    counter = ledger.store.get(ledger.run_key(run_id))
    if counter["halted"] or counter["reserved_micro_usd"]:
        raise ValueError("READER_JOB_RECONCILIATION_REQUIRED")
    immutable(out / "preflight.json", {"checked_at": datetime.now(UTC).isoformat(),
        "funding": funding.model_dump(mode="json"), "model_metadata_sha256": metadata["metadata_sha256"],
        "run_cap_micro_usd": 500000, "published_usage_soft_stop_micro_usd": 100000})
    base = ExternalModelGateway(ledger, provider, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "FROZEN_READER_JOB_V3"}), prices)
    return ControlledJob(base, ledger, out, run_id, scope)


class ControlledJob:
    def __init__(self, base, ledger, out, run_id, scope):
        self.base, self.ledger, self.out, self.run_id, self.scope = base, ledger, out, run_id, scope
        self.provider, self.routes = base.provider, base.routes
        self.allowed, self.calls, self.cost, self.halted = {}, 0, 0, False

    def allow(self, context, system, prompt, cap, *, dispatcher=None):
        require_json_mode_instruction(system)
        if (context.run_id != self.run_id or context.attempt != 1
                or (context.stage, cap) not in {("PLAN_QUERY", 256), ("EXTRACT", 768), ("GENERATE", 1600)}
                or context.stage == "GENERATE" and dispatcher is None
                or len(canonical_json({"model": "deepseek-flash", "messages": [
                    {"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "max_tokens": cap, "thinking": {"type": "disabled"}, "temperature": 0,
                    "stream": False, "response_format": {"type": "json_object"}})) > 100000):
            raise ValueError("READER_JOB_STAGE_OR_BODY_BOUND")
        if (system, context.stage, cap) not in self.scope["allowed_protocols"]:
            raise ValueError("READER_JOB_PROTOCOL_NOT_FROZEN")
        self.allowed[context.request_id] = ((context.stage, system, prompt, cap), dispatcher)

    def generate(self, context, *, system, prompt, max_output_tokens):
        entry = self.allowed.pop(context.request_id, None)
        if (self.halted or self.calls >= self.scope["max_calls"] or self.cost >= 100000
                or context.run_id != self.run_id or entry is None
                or entry[0] != (context.stage, system, prompt, max_output_tokens)):
            raise ValueError("READER_JOB_DISPATCH_NOT_ALLOWED")
        immutable(self.out / "dispatch-intents" / (context.request_id + ".json"),
            {"context": context.model_dump(mode="json"), "system_sha256": sha256_bytes(system.encode()),
                "prompt_sha256": sha256_bytes(prompt.encode()), "max_output_tokens": max_output_tokens})
        self.calls += 1
        try:
            result = (self.base.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
                if entry[1] is None else entry[1]())
            self.cost += result.actual_micro_usd
            immutable(self.out / "dispatch-receipts" / (result.operation_id + ".json"),
                {"operation_id": result.operation_id, "usage": result.usage.model_dump(mode="json"),
                    "published_cost_micro_usd": result.actual_micro_usd, "raw_output": result.text,
                    "stop_reason": result.stop_reason})
            return result
        except Exception:
            self.halted = True
            raise

    def reconciled(self):
        rows = [json.loads(p.read_bytes()) for p in (self.out / "dispatch-receipts").glob("*.json")]
        states = [self.ledger.operation(r["operation_id"]) for r in rows]
        if (any(s is None or s.status != "SETTLED" for s in states)
                or sum(r["published_cost_micro_usd"] for r in rows) != self.cost):
            raise ValueError("READER_JOB_LEDGER_MISMATCH")
        return {"new_calls": self.calls, "published_cost_micro_usd": self.cost, "settled_operations": len(states)}
