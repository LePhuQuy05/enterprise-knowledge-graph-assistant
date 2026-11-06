"""Eight fixed-context graph cells with V11 explicit gap/antecedent contract.

Not a format-only arm. No supplemental documents enter these frozen prompts;
no source-extraction, rewrite or planning calls are made. Old answers remain
unchanged. Replay reads FAILED as FAILED, never retrying/repairing a response.
"""

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import ExitStack
from dataclasses import replace
from datetime import UTC, datetime
from unittest.mock import patch

from app.answer_cache import AnswerCache, AnswerReceipt
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import generation_request, parse_generated
from app.generation_coverage_protocol import (
    PROTOCOL_SHA256,
    CoverageProtocolGateway,
    CoverageProtocolRecipe,
    coverage_contract_sha256,
    protocol_system,
)
from scripts.pilot_graph_document_answers import CODE as PARENT_CODE
from scripts.pilot_graph_document_answers import allocated_cases
from scripts.pilot_reviewed_source_answers import immutable
from scripts.pilot_source_message_answers_v2 import validate_run_capacity
from scripts.prepare_authorization_evidence import ROOT
from scripts.reconcile_reviewed_source_answers import provider_body

RUN = "graph-v11-explicit-coverage-v1"
OUTPUT = ROOT / "results/graph-coverage-protocol-v1"
CACHE = ROOT / "data/cache/graph-coverage-protocol-v1"
BASELINE = ROOT / "results/graph-document-answers-v1"
CODE = (*PARENT_CODE, "app/generation_coverage_protocol.py", "app/answer_assessment.py",
    "app/generation.py", "app/answer_cache.py", "scripts/pilot_coverage_protocol.py")
STOP = 20000


class ScopedGateway:
    def __init__(self, base):
        self.base, self.provider, self.routes = base, base.provider, base.routes
        self.allowed, self.seen, self.cost, self.halted = {}, set(), 0, False

    def allow(self, context, system, prompt):
        key, value = context.model_dump_json(), (system, prompt, 800)
        if (context.run_id != RUN or context.tenant_id != "demo" or context.stage != "GENERATE" or context.attempt != 1
                or key in self.allowed and self.allowed[key] != value):
            raise ValueError("COVERAGE_NATIVE_SCOPE")
        self.allowed[key] = value

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or len(self.seen) >= 8 or self.cost >= STOP or key in self.seen
                or self.allowed.get(key) != (system, prompt, max_output_tokens)):
            raise ValueError("COVERAGE_NATIVE_DISPATCH_BOUND")
        self.seen.add(key)
        try:
            reply = self.base.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
        except Exception:
            self.halted = True
            raise
        self.cost += reply.actual_micro_usd
        return reply


def cases():
    scope = json.loads((BASELINE / "scope.json").read_bytes())
    if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in PARENT_CODE}:
        raise ValueError("COVERAGE_PARENT_CODE_CHANGED")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=RuntimeError("COVERAGE_PREPARATION_NETWORK_BLOCKED")))
        original, _ = allocated_cases()
    prepared_cases = []
    for case in original:
        old = case["prepared"]
        recipe = CoverageProtocolRecipe(model_metadata_sha256=old.recipe.model_metadata_sha256)
        request_id = stable_id("coverage-arm-v1", baseline=old.binding.request_id, run=RUN,
            contract=coverage_contract_sha256(recipe))
        prepared = replace(old, recipe=recipe, binding=old.binding.model_copy(update={"request_id": request_id}),
            context=old.context.model_copy(update={"request_id": request_id, "run_id": RUN}))
        system, prompt = generation_request(prepared.question, case["view"], prepared.packed, "demo", recipe, answer_language=case["language"])
        if prompt != case["prompt"] or prompt != prepared.packed.serialized:
            raise ValueError("COVERAGE_FIXED_SOURCE_USER_MESSAGE_CHANGED")
        prepared_cases.append(case | {"prepared": prepared, "system": system, "prompt": prompt,
            "provider_system": protocol_system(system), "baseline_request_id": old.binding.request_id,
            "baseline_provider_system_sha256": sha256_bytes(case["provider_system"].encode())})
    if len(prepared_cases) != 8 or len({c["prepared"].binding.request_id for c in prepared_cases}) != 8:
        raise ValueError("COVERAGE_EIGHT_CELL_MATRIX_REQUIRED")
    return prepared_cases


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
        raise ValueError("COVERAGE_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    if (funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 500000
            or not month or month["cap_micro_usd"] != 12000000 or month["halted"]):
        raise ValueError("COVERAGE_FUNDING_CAP_OR_UNCERTAINTY")
    raw = (ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(raw) != scope["price_sheet_sha256"]:
        raise ValueError("COVERAGE_PRICE_CHANGED")
    prices = ExternalPriceSheet.model_validate_json(raw)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    reserve = validate_run_capacity(rate, routes, run_cap=500000, observed_stop=STOP)
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    provider = deepseek_preflight(transport)
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("COVERAGE_PROVIDER_CHANGED")
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
    counter = ledger.store.get(ledger.run_key(RUN))
    if counter["halted"] or counter["reserved_micro_usd"] or counter["settled_micro_usd"] or counter["cap_micro_usd"] != 500000:
        raise ValueError("COVERAGE_RUN_ALREADY_ATTEMPTED_OR_UNCERTAIN")
    immutable(OUTPUT / "preflight.json", {"checked_at": datetime.now(UTC).isoformat(), "funding": funding.model_dump(mode="json"),
        "provider": provider, "worst_single_call_reservation_micro_usd": reserve})
    base = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "EIGHT_FIXED_CONTEXT_V11_COVERAGE_CONTRACT_CELLS"}), prices)
    return ScopedGateway(CoverageProtocolGateway(base)), ledger


def cached_receipt(prepared):
    intent = canonical_json({"context": prepared.context.model_dump(mode="json"),
        "binding_sha256": sha256_bytes(canonical_json(prepared.binding.model_dump(mode="json"))), "recipe_sha256": prepared.recipe.fingerprint})
    with sqlite3.connect(f"file:{(CACHE / 'answers.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        row = db.execute("SELECT state,payload,payload_sha256 FROM answers WHERE namespace='LIVE' AND intent=?", (intent,)).fetchone()
    if row is None or row[0] not in {"READY", "FAILED"} or row[1] is None or sha256_bytes(row[1]) != row[2]:
        raise ValueError("COVERAGE_CACHE_UNCERTAIN_OR_CHANGED")
    return row[0], AnswerReceipt.model_validate_json(row[1])


def run(*, execute=False, replay=False, ledger_audit=False):
    scope = {"version": "graph-fixed-context-v11-coverage-arm-v1", "run_id": RUN, "maximum_calls": 8,
        "baseline_scope_sha256": sha256_bytes((BASELINE / "scope.json").read_bytes()),
        "baseline_answer_plan_sha256": sha256_bytes((BASELINE / "answer-plan.json").read_bytes()),
        "baseline_audit_sha256": sha256_bytes((BASELINE / "audit.json").read_bytes()),
        "treatment": "V11_ANTECEDENT_CITATION_COMPLETENESS_EXPLICIT_GAPS_PLUS_FOUR_KEY_FORMAT",
        "format_only_comparison": False, "fixed_user_sources_span_ids_and_order": True,
        "supplemental_sources_used": False, "max_chunks": 8, "user_message_proxy_tokens": 4096, "output_tokens": 800,
        "run_cap_micro_usd": 500000, "observed_stop_micro_usd": STOP, "new_preparation_or_extraction_calls": 0,
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}, "protocol_sha256": PROTOCOL_SHA256,
        "protocol_bound_in": "NEW_REQUEST_ID_AND_EXPERIMENT_SCOPE_WITH_EXACT_V11_RECIPE",
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "human_reviewed": 0, "independent_quality_accuracy_measured": False, "deploy": False}
    immutable(OUTPUT / "scope.json", scope)
    prepared_cases = cases()
    immutable(OUTPUT / "answer-plan.json", {"cases": [{"id": c["id"], "baseline_request_id": c["baseline_request_id"],
        "context": c["prepared"].context.model_dump(mode="json"), "binding": c["prepared"].binding.model_dump(mode="json"),
        "recipe": c["prepared"].recipe.model_dump(mode="json"), "source_chunk_ids": c["prepared"].packed.chunk_ids,
        "source_assertion_ids": c["prepared"].packed.assertion_ids, "source_audit_sha256": c["trace"]["source_message_packing"]["audit_sha256"],
        "user_message_sha256": sha256_bytes(c["prompt"].encode()), "base_system_sha256": sha256_bytes(c["system"].encode()),
        "provider_system_sha256": sha256_bytes(c["provider_system"].encode()),
        "baseline_provider_system_sha256": c["baseline_provider_system_sha256"]} for c in prepared_cases]})
    if not (execute or replay):
        print(json.dumps({"status": "FROZEN_EIGHT_FIXED_CONTEXT_COVERAGE_CELLS_NO_MODEL_CALLS"}))
        return
    gateway, ledger = managed_gateway(scope) if execute else (None, None)
    if gateway:
        for c in prepared_cases:
            gateway.allow(c["prepared"].context, c["system"], c["prompt"])
    cache, results, operations, claims_review = AnswerCache(CACHE, namespace="LIVE"), [], [], []
    with ExitStack() as stack:
        if replay:
            for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
                stack.enter_context(patch(target, side_effect=AssertionError("COVERAGE_REPLAY_NETWORK_BLOCKED")))
        for case in prepared_cases:
            p = case["prepared"]
            failed = replay and cached_receipt(p)[0] == "FAILED"
            try:
                if failed:
                    raise ValueError("COVERAGE_RECORDED_FAILED_OUTPUT")
                lookup = cache.resolve(p.context, question=p.question, snapshot=case["view"], packed=p.packed,
                    binding=p.binding, recipe=p.recipe, gateway=gateway, answer_language=case["language"])
            except Exception:
                state, receipt = cached_receipt(p)
                if state != "FAILED":
                    raise
                if ledger:
                    op = ledger.operation(receipt.operation_id)
                    if op.status != "SETTLED" or not op.call_started or op.bound_exceeded or op.actual_micro_usd != receipt.original_cost_micro_usd:
                        raise ValueError("COVERAGE_UNCERTAIN_FAILURE_STOP")
                row = {"id": case["id"], "status": "FAILED_VALIDATION", "receipt": receipt.model_dump(mode="json"),
                    "retried": False, "repaired": False}
                target = OUTPUT / "failed-answers" / (case["id"] + ".json")
                immutable(target, row)
                raw = receipt.raw_output
                try:
                    parse_generated(raw, receipt.stop_reason, p.recipe)
                except ValueError:
                    pass
                else:
                    raise ValueError("COVERAGE_FAILED_SCHEMA_NEEDS_SOURCE_VALIDATION_AUDIT")
                response, status, gap_count = None, "FAILED_VALIDATION", 0
            else:
                if lookup.receipt is None or execute and lookup.cache_hit or replay and not lookup.cache_hit:
                    raise ValueError("COVERAGE_NATIVE_OR_REPLAY_RECEIPT_REQUIRED")
                receipt, response = lookup.receipt, lookup.response.model_dump(mode="json")
                if (response["request_id"] != p.binding.request_id or response["mode_used"] != case["method"]
                        or not response["used_graph"] or response["answer_language"] != case["language"]):
                    raise ValueError("COVERAGE_RESPONSE_METHOD_OR_LANGUAGE")
                row = {"id": case["id"], "method": case["method"], "language": case["language"],
                    "response": response, "receipt": receipt.model_dump(mode="json"), "retried": False, "repaired": False}
                immutable(OUTPUT / "answers" / (case["id"] + ".json"), row)
                by_id = {c["citation_id"]: c for c in response["citations"]}
                for citation in by_id.values():
                    chunk = case["view"].chunks[citation["chunk_id"]]
                    doc = case["view"].documents[chunk.document_version_id]
                    if (chunk.chunk_id not in p.packed.chunk_ids or citation["quote"] not in chunk.text
                            or citation["source_uri"] != doc.source_uri or doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
                        raise ValueError("COVERAGE_CITATION_ORIGINAL_SOURCE")
                for claim in response["claims"]:
                    claims_review.append({"case": case["id"], "claim": claim, "citations": [by_id[k] for k in claim["citation_ids"]],
                        "status": "PENDING_INDEPENDENT_ADJUDICATION_NOT_GOLD"})
                status = response["answer_status"]
                gap_count = len(response["assessment"]["limitations"])
            operations.append({"operation_id": receipt.operation_id, "usage": receipt.usage.model_dump(mode="json"),
                "cost_micro_usd": receipt.original_cost_micro_usd,
                "payload_sha256": sha256_bytes(provider_body(case["provider_system"], case["prompt"], 800))})
            baseline = json.loads((BASELINE / "answers" / (case["id"] + ".json")).read_bytes())["response"]
            results.append({"id": case["id"], "status_self_report": status, "limitations_count": gap_count,
                "baseline_status_self_report": baseline["answer_status"], "fixed_user_message": True,
                "cost_micro_usd": receipt.original_cost_micro_usd, "claim_count": len(response["claims"]) if response else 0})
            print(json.dumps(results[-1]), flush=True)
    total = sum(o["cost_micro_usd"] for o in operations)
    report = {"status": "EIGHT_COVERAGE_CELLS_REPLAY_PASS" if replay else "EIGHT_COVERAGE_CELLS_ATTEMPTED",
        "cases": results, "operations": operations, "matrix_cells": len(results),
        "statuses_self_report": dict(Counter(r["status_self_report"] for r in results)),
        "total_receipt_cost_micro_usd": total, "new_model_calls": len(gateway.seen) if gateway else 0,
        "new_cost_micro_usd": gateway.cost if gateway else 0, "valid_claims": len(claims_review),
        "supplemental_sources_used": False, "source_contexts_fixed": True, "response_repairs": 0,
        "human_reviewed": 0, "quality_accuracy_measured": False, "serving_published": False}
    if replay and ledger_audit:
        import boto3
        from botocore.config import Config

        from app.ledger import DynamoLedgerStore, Ledger

        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
            raise ValueError("COVERAGE_AUDIT_OPERATOR")
        ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    if ledger:
        for row in operations:
            op = ledger.operation(row["operation_id"])
            if (op.status != "SETTLED" or not op.call_started or op.bound_exceeded or op.spec.run_id != RUN
                    or op.spec.input_sha256 != row["payload_sha256"] or op.spec.stage != "GENERATE"
                    or op.actual_micro_usd != row["cost_micro_usd"] or op.usage.model_dump(mode="json") != row["usage"]):
                raise ValueError("COVERAGE_LEDGER_PAYLOAD_USAGE_COST")
        counter = ledger.store.get(ledger.run_key(RUN))
        funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
        if counter["reserved_micro_usd"] or counter["halted"] or counter["settled_micro_usd"] != total:
            raise ValueError("COVERAGE_RUN_RECONCILIATION")
        report.update(ledger_status="ALL_EIGHT_SETTLED_PAYLOAD_USAGE_COST_MATCH", checked_at=datetime.now(UTC).isoformat(),
            run_counter=counter, funding_counter=funding,
            headroom_micro_usd=funding["cap_micro_usd"] - funding["settled_micro_usd"] - funding["reserved_micro_usd"])
    if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
        raise ValueError("COVERAGE_CODE_CHANGED_DURING_RUN")
    (OUTPUT / ("ledger-audit.json" if ledger_audit else "replay.json" if replay else "result.json")).write_bytes(canonical_json(report) + b"\n")
    (OUTPUT / "claim-source-review-queue.json").write_bytes(canonical_json({"claims": claims_review, "human_reviewed": 0}) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"cases", "operations", "funding_counter", "run_counter"}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--execute", action="store_true")
    choice.add_argument("--replay", action="store_true")
    parser.add_argument("--ledger-audit", action="store_true")
    args = parser.parse_args()
    if args.ledger_audit and not args.replay:
        parser.error("--ledger-audit requires --replay")
    run(execute=args.execute, replay=args.replay, ledger_audit=args.ledger_audit)
