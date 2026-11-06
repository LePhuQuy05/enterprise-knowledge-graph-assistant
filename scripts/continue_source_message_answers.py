"""Finish only never-attempted cells of the frozen V2 matrix.

Previously FAILED outputs remain FAILED and are never repaired/retried. Every
previous call must be SETTLED before continuation; UNKNOWN blocks everything.
The original scope/code/questions/model and total 38-call bound stay in force.
"""

import argparse
import json
import sqlite3
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path

from app.answer_cache import AnswerCache, AnswerReceipt
from app.authorization_evidence import load_authorization_facts
from app.authorization_retrieval import combine_evidence_graph
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.generation import parse_generated
from app.reviewed_source_graph import load_reviewed_source_graph
from scripts.pilot_reviewed_source_answers import immutable
from scripts.pilot_source_message_answers_v2 import (
    CACHE,
    OUTPUT,
    RUN,
    ScopedGateway,
    code_hashes,
    prepare_cases,
    validate_run_capacity,
)
from scripts.prepare_authorization_evidence import ROOT, load_sources
from scripts.run_collection_benchmark import OfflineRuntime, load_inputs


def cached_answers():
    with sqlite3.connect(f"file:{(CACHE / 'answers.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT state,intent,payload,payload_sha256 FROM answers WHERE namespace='LIVE'").fetchall()
    result = {}
    for state, intent_raw, payload, digest in rows:
        intent = json.loads(intent_raw)
        if state not in {"READY", "FAILED"} or payload is None or sha256_bytes(payload) != digest:
            raise ValueError("SOURCE_MESSAGE_CONTINUATION_UNCERTAIN_CACHE")
        receipt = AnswerReceipt.model_validate_json(payload)
        result[intent["context"]["request_id"]] = (state, receipt)
    return result


def restore_recorded_rerank_timings(current, recorded):
    """Only elapsed wall time is observational; ranks/scores/evidence stay exact."""
    result = deepcopy(current)
    if "reranking" in result:
        for now, then in zip(result["reranking"], recorded["reranking"], strict=True):
            if "duration_ms" not in now or type(now["duration_ms"]) not in {int, float} or now["duration_ms"] < 0:
                raise ValueError("SOURCE_MESSAGE_INVALID_RERANK_TIMING")
            now["duration_ms"] = then["duration_ms"]
    return result


def load_cases():
    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    if scope["code_sha256"] != code_hashes():
        raise ValueError("SOURCE_MESSAGE_CONTINUATION_PARENT_CODE_CHANGED")
    collection, chunks, documents = load_sources()
    _, _, _, _, pipeline = load_inputs()
    config = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    base = load_reviewed_source_graph(ROOT / config["release_path"], collection,
        trusted_sha256=config["release_sha256"], allow_experimental=True)
    config = json.loads((ROOT / "configs/graph/authorization-evidence-local-v1.json").read_bytes())
    auth = load_authorization_facts(ROOT / config["release_path"], collection, chunks, documents,
        trusted_sha256=config["release_sha256"], allow_experimental=True)
    graph = combine_evidence_graph(collection, base, auth)
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
        cases, preparations = prepare_cases(collection, lexical, graph, runtime, None)
    frozen = json.loads((OUTPUT / "answer-plan.json").read_bytes())
    for case, saved in zip(cases, frozen["cases"], strict=True):
        case["trace"] = restore_recorded_rerank_timings(case["trace"], saved["trace"])
        if (case["id"] != saved["id"] or canonical_json(case["outcome"]) != canonical_json(saved["outcome"])
                or canonical_json(case["trace"]) != canonical_json(saved["trace"])
                or sha256_bytes(case["system"].encode()) != saved["system_sha256"]
                or sha256_bytes(case["prompt"].encode()) != saved["prompt_sha256"]):
            raise ValueError("SOURCE_MESSAGE_CONTINUATION_FROZEN_CASE_CHANGED")
    return scope, cases, preparations


def continuation_gateway(scope, cases, preparations, previous):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("SOURCE_MESSAGE_CONTINUATION_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    run = ledger.store.get(ledger.run_key(RUN))
    if (funding.cap_micro_usd != 8000000 or run["cap_micro_usd"] != 500000 or run["halted"] or run["reserved_micro_usd"]):
        raise ValueError("SOURCE_MESSAGE_CONTINUATION_BUDGET_OR_UNCERTAINTY")
    old_receipts = [p["receipt"] for p in preparations] + [r.model_dump(mode="json") for _, r in previous.values()]
    for receipt in old_receipts:
        op = ledger.operation(receipt["operation_id"])
        cost = receipt.get("original_cost_micro_usd", receipt.get("cost_micro_usd"))
        if (op.status != "SETTLED" or op.bound_exceeded or not op.call_started or op.spec.run_id != RUN
                or op.actual_micro_usd != cost):
            raise ValueError("SOURCE_MESSAGE_CONTINUATION_PRIOR_CALL_NOT_SETTLED")
    prior_cost = sum(r.get("original_cost_micro_usd", r.get("cost_micro_usd")) for r in old_receipts)
    if run["settled_micro_usd"] != prior_cost:
        raise ValueError("SOURCE_MESSAGE_CONTINUATION_PRIOR_COST_MISMATCH")
    raw_prices = (ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(raw_prices) != scope["price_sheet_sha256"]:
        raise ValueError("SOURCE_MESSAGE_CONTINUATION_PRICE_CHANGED")
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json(raw_prices)
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    validate_run_capacity(rate, routes)
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    provider = deepseek_preflight(transport)
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("SOURCE_MESSAGE_CONTINUATION_PROVIDER_CHANGED")
    gateway = ScopedGateway(ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "CONTINUE_ONLY_NEVER_ATTEMPTED_FROZEN_CELLS"}), prices))
    # Preserve whole-run spend/call accounting, not merely this process's count.
    gateway.cost = prior_cost
    gateway.seen = {"prior-settled:" + r["operation_id"] for r in old_receipts}
    for case in cases:
        if case["prepared"].binding.request_id not in previous:
            gateway.allow(case["prepared"].context, case["system"], case["prompt"], 800)
    return gateway, ledger, prior_cost


def failure_record(case, receipt):
    reason = "OTHER_VALIDATION_FAILURE"
    try:
        parse_generated(receipt.raw_output, receipt.stop_reason, case["prepared"].recipe)
    except ValueError as exc:
        reason = str(exc)
    return {"id": case["id"], "intent": case["intent"], "language": case["language"], "method": case["method"],
        "outcome": case["outcome"], "trace": case["trace"], "receipt": receipt.model_dump(mode="json"),
        "answer_status": "FAILED_VALIDATION", "validation_error": reason,
        "repaired": False, "retried": False, "included_in_attempted_method_denominator": True}


def continue_run(*, execute=False, replay=False):
    scope, cases, preparations = load_cases()
    previous = cached_answers()
    path = OUTPUT / "continuation-scope.json"
    if path.exists():
        continuation = json.loads(path.read_bytes())
        if (continuation["parent_scope_sha256"] != sha256_bytes((OUTPUT / "scope.json").read_bytes())
                or continuation["implementation_sha256"] != sha256_bytes(Path(__file__).read_bytes())):
            raise ValueError("SOURCE_MESSAGE_CONTINUATION_SCOPE_CHANGED")
    else:
        continuation = {"version": "source-message-unattempted-continuation-v1", "run_id": RUN,
            "parent_scope_sha256": sha256_bytes((OUTPUT / "scope.json").read_bytes()),
            "frozen_answer_plan_sha256": sha256_bytes((OUTPUT / "answer-plan.json").read_bytes()),
            "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "previous_cells": len(previous), "previous_failed_cells": sum(state == "FAILED" for state, _ in previous.values()),
            "allowed_never_attempted_case_ids": [c["id"] for c in cases if c["prepared"].binding.request_id not in previous],
            "maximum_total_run_provider_calls": 38, "retry_or_repair_failed_cells": False}
        continuation["only_observational_field_restored_for_replay"] = "reranking[].duration_ms"
        immutable(path, continuation)
    if not (execute or replay):
        print(json.dumps(continuation))
        return
    gateway, ledger, prior_cost = (None, None, 0)
    if execute:
        gateway, ledger, prior_cost = continuation_gateway(scope, cases, preparations, previous)
    cache, results = AnswerCache(CACHE, namespace="LIVE"), []
    for case in cases:
        p = case["prepared"]
        saved = previous.get(p.binding.request_id)
        if saved and saved[0] == "FAILED":
            record = failure_record(case, saved[1])
            immutable(OUTPUT / "failed-answers" / (case["id"] + ".json"), record)
            results.append({"id": case["id"], "status": "FAILED_VALIDATION", "receipt_cost_micro_usd": saved[1].original_cost_micro_usd})
            continue
        if not saved and case["id"] not in continuation["allowed_never_attempted_case_ids"]:
            raise ValueError("SOURCE_MESSAGE_CONTINUATION_NEW_CASE_NOT_AUTHORIZED")
        try:
            lookup = cache.resolve(p.context, question=p.question, snapshot=case["view"], packed=p.packed,
                binding=p.binding, recipe=p.recipe, gateway=gateway, answer_language=case["language"])
        except Exception:
            # Only a durable, settled validation failure permits other NEW cells.
            # Unknown transport/settlement/cache states stop without retry.
            failed = cached_answers().get(p.binding.request_id)
            if (ledger is None or not failed or failed[0] != "FAILED"
                    or ledger.operation(failed[1].operation_id).status != "SETTLED"):
                raise
            immutable(OUTPUT / "failed-answers" / (case["id"] + ".json"), failure_record(case, failed[1]))
            results.append({"id": case["id"], "status": "FAILED_VALIDATION", "receipt_cost_micro_usd": failed[1].original_cost_micro_usd})
            continue
        immutable(OUTPUT / "answers" / (case["id"] + ".json"), {"id": case["id"], "intent": case["intent"],
            "language": case["language"], "method": case["method"], "outcome": case["outcome"], "trace": case["trace"],
            "response": lookup.response.model_dump(mode="json"), "receipt": lookup.receipt.model_dump(mode="json"), "semantic_review": "PENDING_NOT_GOLD"})
        results.append({"id": case["id"], "status": lookup.response.answer_status,
            "receipt_cost_micro_usd": lookup.receipt.original_cost_micro_usd, "cache_hit": lookup.cache_hit})
        print(json.dumps(results[-1]), flush=True)
    own_cost = sum(r["receipt_cost_micro_usd"] for r in results) + sum(p["receipt"]["cost_micro_usd"] for p in preparations)
    if ledger:
        counter = ledger.store.get(ledger.run_key(RUN))
        if counter["reserved_micro_usd"] or counter["settled_micro_usd"] != own_cost:
            raise ValueError("SOURCE_MESSAGE_CONTINUATION_FINAL_COUNTERS")
    report = {"status": "FROZEN_MATRIX_ALL_CELLS_REPLAYED" if replay else "FROZEN_MATRIX_ALL_CELLS_ATTEMPTED_ERRORS_RETAINED",
        "cases": results, "matrix_cells": len(results), "failed_validation_cells": sum(r["status"] == "FAILED_VALIDATION" for r in results),
        "new_provider_calls": len(gateway.seen) - len(previous) - len(preparations) if gateway else 0,
        "new_cost_micro_usd": gateway.cost - prior_cost if gateway else 0, "total_receipt_cost_micro_usd": own_cost,
        "retries_of_failed_cells": 0, "repairs": 0, "human_reviewed": 0, "quality_accuracy_measured": False}
    immutable(OUTPUT / ("continuation-replay.json" if replay else "continuation-results.json"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.execute and args.replay:
        parser.error("Choose execute or replay")
    continue_run(execute=args.execute, replay=args.replay)
