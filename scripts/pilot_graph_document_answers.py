"""Bounded eight-cell EN/VI graph allocation arm; historical outputs stay intact.

Only G1/G2 contexts change. Use the same required-reason protocol/model/output
cap as the previous 32-cell arm. No new rewrite/planner/extraction requests and
no automatic retry/repair. Failed replies retain their raw usage/cost receipts.
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
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import generation_request
from app.generation_output_protocol import PROTOCOL_SHA256, RequiredReasonGateway, protocol_system
from app.graph_document_allocation import reallocate_graph_documents
from app.method_execution import ReviewedGraphExecutionV3, StrictPreparedMethod
from scripts.compare_graph_document_allocation import CODE as ALLOCATION_CODE
from scripts.compare_graph_source_allocation import load_graph
from scripts.compare_required_reason_protocol import CODE as BASELINE_CODE
from scripts.compare_required_reason_protocol import protocol_cases
from scripts.pilot_reviewed_source_answers import immutable
from scripts.pilot_source_message_answers_v2 import validate_run_capacity
from scripts.prepare_authorization_evidence import ROOT
from scripts.run_collection_benchmark import load_inputs

RUN = "graph-document-source-answers-v1"
OUTPUT = ROOT / "results/graph-document-answers-v1"
CACHE = ROOT / "data/cache/graph-document-answers-v1"
BASELINE = ROOT / "results/required-reason-protocol-v1"
ALLOCATION = ROOT / "results/graph-document-allocation-v2"
CODE = (*ALLOCATION_CODE, "scripts/pilot_graph_document_answers.py")
STOP_MICRO_USD = 20000


def validate_offline_gate():
    report = json.loads((ALLOCATION / "report.json").read_bytes())
    plan = json.loads((ALLOCATION / "plan.json").read_bytes())
    if (plan["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in ALLOCATION_CODE}
            or report["cases"] != 42 or report["arm_rows"] != 126 or not report["all_same_pool"]
            or sha256_bytes((ALLOCATION / "receipts.jsonl").read_bytes()) != report["receipts_sha256"]):
        raise ValueError("GRAPH_DOCUMENT_OFFLINE_SCOPE_GATE")
    cells = [r for r in report["native_graph_cell_source_presence"] if r["arm"] == "DOCUMENT_TAIL_PRIORITY_4"]
    if len(cells) != 8:
        raise ValueError("GRAPH_DOCUMENT_EIGHT_CELLS_GATE")
    for row in cells:
        presence = row["presence"]
        if row["id"].startswith("s3-") and not (presence["kms:Decrypt"] and presence["source_scoped_LambdaEdge_READ_example"]):
            raise ValueError("GRAPH_DOCUMENT_EXPOSED_READ_DECRYPT_SOURCE_GATE")
        if row["id"].startswith("kms-") and not presence["Cross-account access"]:
            raise ValueError("GRAPH_DOCUMENT_KMS_SOURCE_GATE")
    return plan, report


def allocated_cases():
    baseline_scope = json.loads((BASELINE / "scope.json").read_bytes())
    if baseline_scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in BASELINE_CODE}:
        raise ValueError("GRAPH_DOCUMENT_FROZEN_PROTOCOL_CODE_CHANGED")
    _, report = validate_offline_gate()
    rows = [json.loads(line) for line in (ALLOCATION / "receipts.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = {r["id"]: r for r in rows if r["section"] == "NATIVE_EN_VI_GRAPH_CELLS" and r["arm"] == "DOCUMENT_TAIL_PRIORITY_4"}
    _, parent_cases = protocol_cases()
    collection, graph = load_graph()
    _, _, _, _, pipeline = load_inputs()
    cases = []
    with load_global_lexical_index(ROOT / pipeline["lexical_manifest_path"], collection) as lexical:
        for case in parent_cases:
            if not case["method"].startswith("graph_"):
                continue
            predecessor = StrictPreparedMethod(ReviewedGraphExecutionV3.model_validate_json(canonical_json(case["outcome"])),
                case["view"], case["prepared"], case["trace"], case["language"])
            result = reallocate_graph_documents(collection, lexical, graph, predecessor, priority_source_cap=4)
            old, saved = result.prepared, expected[case["id"]]
            request_id = stable_id("graph-document-answer-v1", allocation_request=old.binding.request_id,
                run=RUN, protocol=PROTOCOL_SHA256, recipe=old.recipe.fingerprint)
            prepared = replace(old, binding=old.binding.model_copy(update={"request_id": request_id}),
                context=old.context.model_copy(update={"request_id": request_id, "run_id": RUN}))
            system, prompt = generation_request(prepared.question, result.view, prepared.packed, "demo", prepared.recipe,
                answer_language=case["language"])
            if (system != case["system"] or sha256_bytes(prompt.encode()) != saved["user_message_sha256"]
                    or prepared.packed.chunk_ids != tuple(saved["context_chunk_ids"])
                    or set(prepared.retrieved_chunk_ids) != set(case["prepared"].retrieved_chunk_ids)
                    or not result.outcome.graph_path_used or result.outcome.effective_method != case["method"]):
                raise ValueError("GRAPH_DOCUMENT_NATIVE_CONTEXT_BINDING")
            cases.append(case | {"prepared": prepared, "view": result.view, "trace": result.trace,
                "outcome": result.outcome.model_dump(mode="json"), "system": system, "prompt": prompt,
                "provider_system": protocol_system(system), "baseline_request_id": case["prepared"].binding.request_id,
                "baseline_prompt_sha256": sha256_bytes(case["prompt"].encode())})
    if len(cases) != 8 or len({c["prepared"].binding.request_id for c in cases}) != 8:
        raise ValueError("GRAPH_DOCUMENT_NATIVE_MATRIX_BOUND")
    return cases, report


class ScopedGateway:
    def __init__(self, base):
        self.base, self.provider, self.routes = base, base.provider, base.routes
        self.allowed, self.seen, self.cost, self.halted = {}, set(), 0, False

    def allow(self, context, system, prompt):
        key = context.model_dump_json()
        value = (system, prompt, 800)
        if (context.run_id != RUN or context.stage != "GENERATE" or context.attempt != 1
                or context.tenant_id != "demo" or key in self.allowed and self.allowed[key] != value):
            raise ValueError("GRAPH_DOCUMENT_DISPATCH_SCOPE")
        self.allowed[key] = value

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or len(self.seen) >= 8 or self.cost >= STOP_MICRO_USD or key in self.seen
                or self.allowed.get(key) != (system, prompt, max_output_tokens)):
            raise ValueError("GRAPH_DOCUMENT_DISPATCH_BOUND")
        self.seen.add(key)
        try:
            reply = self.base.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
        except Exception:
            self.halted = True
            raise
        self.cost += reply.actual_micro_usd
        return reply


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
        raise ValueError("GRAPH_DOCUMENT_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    if (funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 500000
            or not month or month["cap_micro_usd"] != 12000000 or month["halted"]):
        raise ValueError("GRAPH_DOCUMENT_FUNDING_OR_MONTH_CAP")
    raw_prices = (ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(raw_prices) != scope["price_sheet_sha256"]:
        raise ValueError("GRAPH_DOCUMENT_PRICE_PIN")
    prices = ExternalPriceSheet.model_validate_json(raw_prices)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    reservation = validate_run_capacity(rate, routes, run_cap=500000, observed_stop=STOP_MICRO_USD)
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    provider = deepseek_preflight(transport)
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("GRAPH_DOCUMENT_PROVIDER_CHANGED")
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
    counter = ledger.store.get(ledger.run_key(RUN))
    if counter["halted"] or counter["reserved_micro_usd"] or counter["settled_micro_usd"] or counter["cap_micro_usd"] != 500000:
        raise ValueError("GRAPH_DOCUMENT_NEW_RUN_MUST_BE_UNATTEMPTED")
    immutable(OUTPUT / "preflight.json", {"checked_at": datetime.now(UTC).isoformat(), "provider": provider,
        "funding": funding.model_dump(mode="json"), "worst_call_reservation_micro_usd": reservation,
        "observed_stop_micro_usd": STOP_MICRO_USD, "run_cap_micro_usd": 500000})
    external = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "GRAPH_SOURCE_ALLOCATION_EIGHT_FROZEN_EN_VI_CELLS"}), prices)
    return ScopedGateway(RequiredReasonGateway(external)), ledger


def failed_receipt(prepared):
    with sqlite3.connect(f"file:{(CACHE / 'answers.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        row = db.execute("SELECT state,payload,payload_sha256 FROM answers WHERE namespace='LIVE' AND intent=?", (canonical_json({
            "context": prepared.context.model_dump(mode="json"), "binding_sha256": sha256_bytes(canonical_json(prepared.binding.model_dump(mode="json"))),
            "recipe_sha256": prepared.recipe.fingerprint}),)).fetchone()
    if row is None or row[0] != "FAILED" or row[1] is None or sha256_bytes(row[1]) != row[2]:
        raise ValueError("GRAPH_DOCUMENT_FAILED_OR_UNCERTAIN_RECEIPT")
    return AnswerReceipt.model_validate_json(row[1])


def run(*, execute=False):
    _, offline = validate_offline_gate()
    scope = {"version": "graph-document-source-answer-arm-v1", "run_id": RUN,
        "baseline_scope_sha256": sha256_bytes((BASELINE / "scope.json").read_bytes()),
        "baseline_answer_plan_sha256": sha256_bytes((BASELINE / "answer-plan.json").read_bytes()),
        "allocation_plan_sha256": sha256_bytes((ALLOCATION / "plan.json").read_bytes()),
        "allocation_receipts_sha256": offline["receipts_sha256"], "maximum_calls": 8,
        "methods": ["graph_1hop", "graph_2hop"], "intents": ["s3-permission-layers", "kms-cross-account"],
        "languages": ["en", "vi"], "max_chunks": 8, "user_message_proxy_tokens": 4096, "max_output_tokens": 800,
        "run_cap_micro_usd": 500000, "observed_stop_micro_usd": STOP_MICRO_USD,
        "protocol_sha256": PROTOCOL_SHA256, "new_preparation_or_extraction_calls": 0,
        "only_treatment": "SAME_POOL_SOURCE_PRIORITY_FOUR_AND_DOCUMENT_DIVERSE_TAIL",
        "baseline_outputs_preserved": True, "automatic_retry_or_response_repair": False,
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "human_reviewed": 0, "quality_accuracy_measured": False, "deploy": False}
    immutable(OUTPUT / "scope.json", scope)
    if not execute:
        print(json.dumps({"status": "EIGHT_CELL_GRAPH_ALLOCATION_SCOPE_FROZEN_NO_MODEL_CALLS"}))
        return
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=RuntimeError("GRAPH_DOCUMENT_PREPARATION_NETWORK_BLOCKED")))
        cases, _ = allocated_cases()
    plan = {"recipe": cases[0]["prepared"].recipe.model_dump(mode="json"), "cases": [{"id": c["id"],
        "baseline_request_id": c["baseline_request_id"], "baseline_prompt_sha256": c["baseline_prompt_sha256"],
        "call_context": c["prepared"].context.model_dump(mode="json"), "binding": c["prepared"].binding.model_dump(mode="json"),
        "outcome": c["outcome"], "trace": c["trace"], "base_system_sha256": sha256_bytes(c["system"].encode()),
        "provider_system_sha256": sha256_bytes(c["provider_system"].encode()),
        "user_message_sha256": sha256_bytes(c["prompt"].encode())} for c in cases]}
    immutable(OUTPUT / "answer-plan.json", plan)
    gateway, ledger = managed_gateway(scope)
    for case in cases:
        gateway.allow(case["prepared"].context, case["system"], case["prompt"])
    cache, results = AnswerCache(CACHE, namespace="LIVE"), []
    for case in cases:
        p = case["prepared"]
        try:
            lookup = cache.resolve(p.context, question=p.question, snapshot=case["view"], packed=p.packed,
                binding=p.binding, recipe=p.recipe, gateway=gateway, answer_language=case["language"])
        except Exception as exc:
            receipt = failed_receipt(p)
            op = ledger.operation(receipt.operation_id)
            if (op.status != "SETTLED" or op.bound_exceeded or not op.call_started
                    or op.actual_micro_usd != receipt.original_cost_micro_usd):
                raise ValueError("GRAPH_DOCUMENT_UNCERTAIN_CALL_STOP") from exc
            immutable(OUTPUT / "failed-answers" / (case["id"] + ".json"), {"id": case["id"],
                "status": "FAILED_VALIDATION", "receipt": receipt.model_dump(mode="json"), "error_class": type(exc).__name__,
                "repaired": False, "retried": False})
            results.append({"id": case["id"], "status": "FAILED_VALIDATION", "cost_micro_usd": receipt.original_cost_micro_usd})
            continue
        if lookup.receipt is None or lookup.cache_hit:
            raise ValueError("GRAPH_DOCUMENT_NEW_NATIVE_RECEIPT_REQUIRED")
        immutable(OUTPUT / "answers" / (case["id"] + ".json"), {"id": case["id"], "method": case["method"],
            "intent": case["intent"], "language": case["language"], "response": lookup.response.model_dump(mode="json"),
            "receipt": lookup.receipt.model_dump(mode="json"), "repaired": False, "retried": False})
        results.append({"id": case["id"], "status": lookup.response.answer_status, "cost_micro_usd": lookup.receipt.original_cost_micro_usd})
        print(json.dumps(results[-1]), flush=True)
    total = sum(r["cost_micro_usd"] for r in results)
    counter = ledger.store.get(ledger.run_key(RUN))
    if counter["reserved_micro_usd"] or counter["settled_micro_usd"] != total:
        raise ValueError("GRAPH_DOCUMENT_COUNTER_RECONCILIATION")
    if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
        raise ValueError("GRAPH_DOCUMENT_CODE_CHANGED_DURING_RUN")
    report = {"status": "EIGHT_GRAPH_ALLOCATION_NATIVE_CELLS_ATTEMPTED", "cases": results,
        "statuses": dict(Counter(r["status"] for r in results)), "new_model_calls": len(gateway.seen),
        "new_cost_micro_usd": gateway.cost, "total_receipt_cost_micro_usd": total,
        "new_planner_rewrite_extraction_calls": 0, "response_repairs": 0, "human_reviewed": 0,
        "quality_accuracy_measured": False, "serving_published": False}
    immutable(OUTPUT / "result.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    run(execute=parser.parse_args().execute)
