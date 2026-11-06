"""Paired format-only arm over ALL 32 frozen contexts, not five cherry-picked retries."""

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime

from app.answer_cache import AnswerCache
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import generation_request
from app.generation_output_protocol import (
    PROTOCOL_SHA256,
    RequiredReasonGateway,
    RequiredReasonRecipe,
    protocol_system,
)
from app.ingestion.text import ProxyTokenizer
from scripts.continue_source_message_answers import load_cases
from scripts.pilot_reviewed_source_answers import immutable
from scripts.pilot_source_message_answers_v2 import validate_run_capacity
from scripts.prepare_authorization_evidence import ROOT

RUN = "source-message-required-reason-protocol-v1"
OUTPUT = ROOT / "results/required-reason-protocol-v1"
CACHE = ROOT / "data/cache/required-reason-protocol-v1"
PARENT = ROOT / "results/source-message-answers-v2"
CODE = ("app/generation_output_protocol.py", "scripts/compare_required_reason_protocol.py",
    "scripts/continue_source_message_answers.py")


class ScopedGateway:
    def __init__(self, base):
        self.base, self.provider, self.routes = base, base.provider, base.routes
        self.allowed, self.seen, self.cost, self.halted = {}, set(), 0, False

    def allow(self, context, system, prompt):
        if context.run_id != RUN or context.stage != "GENERATE" or context.attempt != 1:
            raise ValueError("REASON_PROTOCOL_SCOPE")
        self.allowed[context.model_dump_json()] = (system, prompt, 800)

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or key in self.seen or len(self.seen) >= 32 or self.cost >= 150000
                or self.allowed.get(key) != (system, prompt, max_output_tokens)):
            raise ValueError("REASON_PROTOCOL_DISPATCH_BOUND")
        self.seen.add(key)
        try:
            reply = self.base.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
        except Exception:
            self.halted = True
            raise
        self.cost += reply.actual_micro_usd
        return reply


def protocol_cases():
    parent_scope, original, _ = load_cases()
    cases = []
    for case in original:
        old = case["prepared"]
        recipe = RequiredReasonRecipe(model_metadata_sha256=old.recipe.model_metadata_sha256)
        request_id = stable_id("required-reason-arm-v1", baseline_request=old.binding.request_id,
            recipe=recipe.fingerprint, protocol=PROTOCOL_SHA256, run=RUN)
        prepared = replace(old, recipe=recipe, context=old.context.model_copy(update={"run_id": RUN, "request_id": request_id}),
            binding=old.binding.model_copy(update={"request_id": request_id}))
        system, prompt = generation_request(prepared.question, case["view"], prepared.packed, "demo", recipe,
            answer_language=case["language"])
        if prompt != case["prompt"] or system != case["system"]:
            raise ValueError("REASON_PROTOCOL_BASE_INPUT_CHANGED")
        cases.append(case | {"prepared": prepared, "system": system, "prompt": prompt,
            "provider_system": protocol_system(system), "baseline_request_id": old.binding.request_id})
    return parent_scope, cases


def managed_gateway(scope):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("REASON_PROTOCOL_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    if funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 500000:
        raise ValueError("REASON_PROTOCOL_FUNDING")
    raw_prices = (ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(raw_prices) != scope["price_sheet_sha256"]:
        raise ValueError("REASON_PROTOCOL_PRICE_PIN")
    prices = ExternalPriceSheet.model_validate_json(raw_prices)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    admission = validate_run_capacity(rate, routes)
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    provider = deepseek_preflight(transport)
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("REASON_PROTOCOL_PROVIDER_CHANGED")
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
    run = ledger.store.get(ledger.run_key(RUN))
    if run["halted"] or run["reserved_micro_usd"] or run["cap_micro_usd"] != 500000:
        raise ValueError("REASON_PROTOCOL_RUN_UNCERTAIN")
    immutable(OUTPUT / "preflight.json", {"checked_at": datetime.now(UTC).isoformat(), "provider": provider,
        "funding": funding.model_dump(mode="json"), "maximum_single_call_reservation_micro_usd": admission})
    external = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "PAIRED_REQUIRED_REASON_PROTOCOL_ALL_32_CONTEXTS"}), prices)
    return ScopedGateway(RequiredReasonGateway(external)), ledger


def run(*, execute=False, replay=False):
    parent_scope = json.loads((PARENT / "scope.json").read_bytes())
    scope = {"version": "paired-required-reason-format-arm-v1", "run_id": RUN,
        "parent_scope_sha256": sha256_bytes((PARENT / "scope.json").read_bytes()),
        "parent_answer_plan_sha256": sha256_bytes((PARENT / "answer-plan.json").read_bytes()),
        "parent_audit_sha256": sha256_bytes((PARENT / "audit.json").read_bytes()),
        "parent_code_sha256": parent_scope["code_sha256"], "questions": parent_scope["questions"],
        "methods": parent_scope["methods"], "format_protocol_sha256": PROTOCOL_SHA256,
        "maximum_calls": 32, "run_cap_micro_usd": 500000, "observed_stop_micro_usd": 150000,
        "fixed_user_message_source_span_ids_and_order": True, "max_output_tokens": 800,
        "only_treatment": "APPEND_STATIC_REQUIRED_REASON_FIRST_FORMAT_INSTRUCTION_TO_SYSTEM",
        "failed_baseline_cells_kept_in_denominator": True, "automatic_response_repair": False,
        "price_sheet_sha256": parent_scope["price_sheet_sha256"],
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}, "deploy": False, "human_reviewed": 0}
    immutable(OUTPUT / "scope.json", scope)
    if not (execute or replay):
        print(json.dumps({"status": "FROZEN_ALL_32_PAIRED_CONTEXTS_SCOPE_ONLY", "new_model_calls": 0}))
        return
    _, cases = protocol_cases()
    gateway, ledger = managed_gateway(scope) if execute else (None, None)
    cache, results = AnswerCache(CACHE, namespace="LIVE"), []
    t = ProxyTokenizer()
    plan = {"recipe": cases[0]["prepared"].recipe.model_dump(mode="json"), "cases": [{"id": c["id"],
        "baseline_request_id": c["baseline_request_id"], "call_context": c["prepared"].context.model_dump(mode="json"),
        "binding": c["prepared"].binding.model_dump(mode="json"), "base_system_sha256": sha256_bytes(c["system"].encode()),
        "actual_provider_system_sha256": sha256_bytes(c["provider_system"].encode()),
        "user_message_sha256": sha256_bytes(c["prompt"].encode()), "actual_system_proxy_tokens": t.count(c["provider_system"]),
        "user_message_proxy_tokens": t.count(c["prompt"])} for c in cases]}
    immutable(OUTPUT / "answer-plan.json", plan)
    if gateway:
        for case in cases:
            gateway.allow(case["prepared"].context, case["system"], case["prompt"])
    for case in cases:
        p = case["prepared"]
        try:
            lookup = cache.resolve(p.context, question=p.question, snapshot=case["view"], packed=p.packed,
                binding=p.binding, recipe=p.recipe, gateway=gateway, answer_language=case["language"])
        except Exception as exc:
            # Raw receipts, not repaired output, are the evidence for failures.
            import sqlite3

            with sqlite3.connect(f"file:{cache.path.as_posix()}?mode=ro", uri=True) as db:
                row = db.execute("SELECT state,payload FROM answers WHERE intent=?", (canonical_json({
                    "context": p.context.model_dump(mode="json"), "binding_sha256": sha256_bytes(canonical_json(p.binding.model_dump(mode="json"))),
                    "recipe_sha256": p.recipe.fingerprint}),)).fetchone()
            if row is None or row[0] != "FAILED" or row[1] is None:
                raise
            receipt = json.loads(row[1])
            if ledger is not None and ledger.operation(receipt["operation_id"]).status != "SETTLED":
                raise
            record = {"id": case["id"], "status": "FAILED_VALIDATION", "receipt": receipt,
                "error_class": type(exc).__name__, "repaired": False, "retried": False}
            immutable(OUTPUT / "failed-answers" / (case["id"] + ".json"), record)
            results.append({"id": case["id"], "status": "FAILED_VALIDATION", "cost_micro_usd": receipt["original_cost_micro_usd"]})
            continue
        immutable(OUTPUT / "answers" / (case["id"] + ".json"), {"id": case["id"], "intent": case["intent"],
            "method": case["method"], "language": case["language"], "baseline_request_id": case["baseline_request_id"],
            "response": lookup.response.model_dump(mode="json"), "receipt": lookup.receipt.model_dump(mode="json"),
            "protocol_sha256": PROTOCOL_SHA256, "repaired": False, "retried": False})
        results.append({"id": case["id"], "status": lookup.response.answer_status, "cost_micro_usd": lookup.receipt.original_cost_micro_usd,
            "cache_hit": lookup.cache_hit})
        print(json.dumps(results[-1]), flush=True)
    total = sum(r["cost_micro_usd"] for r in results)
    if ledger:
        counter = ledger.store.get(ledger.run_key(RUN))
        if counter["reserved_micro_usd"] or counter["settled_micro_usd"] != total:
            raise ValueError("REASON_PROTOCOL_RUN_RECONCILIATION")
    if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
        raise ValueError("REASON_PROTOCOL_CODE_CHANGED_DURING_RUN")
    report = {"status": "EXACT_PAIRED_ARM_REPLAY" if replay else "ALL_32_FORMAT_TREATMENT_CELLS_ATTEMPTED",
        "cases": results, "matrix_cells": len(results), "failed_validation_cells": sum(r["status"] == "FAILED_VALIDATION" for r in results),
        "new_provider_calls": len(gateway.seen) if gateway else 0, "new_cost_micro_usd": gateway.cost if gateway else 0,
        "total_receipt_cost_micro_usd": total, "context_changed": False, "response_repairs": 0,
        "human_reviewed": 0, "quality_accuracy_measured": False}
    immutable(OUTPUT / ("replay.json" if replay else "result.json"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.execute and args.replay:
        parser.error("Choose execute or replay")
    run(execute=args.execute, replay=args.replay)
