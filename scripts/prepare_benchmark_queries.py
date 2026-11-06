"""Prepare <=100 shared VI rewrites and <=200 shared EN/VI plans. No answers.

Default quotes locally. --execute uses a separate 0.50 USD run under the existing
2 USD funding and 12 USD month caps; neither global cap is raised or reset.
Failed/unknown cache claims stop the batch and are never retried automatically.
"""

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from app.benchmark_queries import (
    BATCH_RUN,
    FailedSharedQueryRow,
    QueryInput,
    SharedQueryBundle,
    SharedQueryRow,
    parse_shared_row,
    query_context,
    validate_shared_bundle,
    validate_shared_row,
)
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.model_gateway import safe_failure_code
from app.query_planning import PlannerRecipe, QueryPlanCache
from app.query_planning import _Receipt as SavedPlanReceipt
from app.rewrite_cache import RewriteCache
from scripts.run_collection_benchmark import ROOT, _write_new, load_inputs

PRICES = ROOT / "configs/prices/external-direct-2026-10-24.json"


def tasks_from(dataset):
    return tuple(QueryInput(scenario_id=s.scenario_id, language=language,
                            question=getattr(s, "question_" + language))
                 for s in dataset.scenarios for language in ("en", "vi"))


def plan_for(tasks, protocol_sha256, routes, prices):
    return {"schema_version": 1, "kind": "BENCHMARK_SHARED_QUERY_PLAN", "budget_run_id": BATCH_RUN,
        "protocol_sha256": protocol_sha256, "routes_sha256": sha256_bytes(canonical_json(routes.model_dump(mode="json"))),
        "price_sheet_sha256": sha256_bytes(PRICES.read_bytes()), "planner_recipe_sha256": PlannerRecipe().fingerprint,
        "tasks": [t.model_dump(mode="json") for t in tasks], "maximum_planner_operations": len(tasks),
        "maximum_rewrite_operations": sum(t.language == "vi" for t in tasks),
        "maximum_output_tokens_each": 256, "run_cap_micro_usd": 500_000,
        "existing_funding_cap_micro_usd": 2_000_000, "existing_month_cap_micro_usd": 12_000_000,
        "global_cap_raise": False, "automatic_retry": False, "generate_answer_calls": 0,
        "rate_input_micro_usd_per_million": prices.rates[0].input_micro_usd_per_unit,
        "rate_output_micro_usd_per_million": prices.rates[0].output_micro_usd_per_unit,
        "provider_model_metadata_sha256": routes.account_model_metadata_sha256}


def counters(ledger):
    return {name: ledger.store.get(key) for name, key in (
        ("run", ledger.run_key(BATCH_RUN)), ("funding", ledger.external_funding_key("kg-r72")),
        ("month", ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m"))))}


def gateway_for(routes, prices):
    import boto3
    from botocore.config import Config

    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.ledger import DynamoLedgerStore, Ledger

    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=30)
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("BENCHMARK_SHARED_DEVELOPER_IDENTITY_REQUIRED")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    before = counters(ledger)
    for name, cap in (("funding", 2_000_000), ("month", 12_000_000)):
        row = before[name]
        if not row or row["halted"] or row["cap_micro_usd"] != cap:
            raise ValueError("BENCHMARK_SHARED_GLOBAL_BUDGET_NOT_READY")
    if before["run"] and (before["run"]["halted"] or before["run"]["reserved_micro_usd"] != 0
                          or before["run"]["cap_micro_usd"] != 500_000):
        raise ValueError("BENCHMARK_SHARED_RUN_NEEDS_RECONCILIATION")
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")})
    account = deepseek_preflight(transport)
    if not account["available"] or account["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("BENCHMARK_SHARED_PROVIDER_MODEL_MISMATCH")
    # A named experiment run is created idempotently; shared funding/month counters
    # are neither initialized nor raised, and existing reservations are untouched.
    ledger.initialize_run("kg-r72", "demo", BATCH_RUN, cap_micro_usd=500_000)
    gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={
        "live_inference_enabled": True, "live_inference_gate": "EXPLICIT_BENCHMARK_QUERY_PREPARATION_ONLY"}), prices)
    return gateway, before, account


class BoundedQueryGateway:
    def __init__(self, gateway, tasks, protocol_sha256):
        self.gateway, self.provider, self.routes = gateway, gateway.provider, gateway.routes
        self.allowed = {query_context(t, protocol_sha256, stage).model_dump_json()
            for t in tasks for stage in (("REWRITE", "PLAN_QUERY") if t.language == "vi" else ("PLAN_QUERY",))}
        self.seen, self.halted = set(), False

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or key not in self.allowed or key in self.seen or max_output_tokens != 256
                or len(system.encode()) + len(prompt.encode()) > 12_000):
            raise ValueError("BENCHMARK_SHARED_DISPATCH_SCOPE_OR_BOUND")
        self.seen.add(key)
        try:
            return self.gateway.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
        except Exception:
            self.halted = True
            raise


def prepare_row(task, protocol_sha256, *, routes, rewrite_cache, plan_cache, gateway=None):
    rewrite, search = None, task.question
    if task.language == "vi":
        result = rewrite_cache.resolve(query_context(task, protocol_sha256, "REWRITE"),
            question=task.question, routes=routes, gateway=gateway)
        rewrite, search = result.receipt, result.search_query
    planned = plan_cache.resolve(query_context(task, protocol_sha256, "PLAN_QUERY"),
        question=task.question, search_query=search, routes=routes, gateway=gateway, language=task.language)
    row = SharedQueryRow.model_validate_json(canonical_json({"scenario_id": task.scenario_id,
        "language": task.language, "question_sha256": sha256_bytes(task.question.encode("utf-8")),
        "search_query": search, "subqueries": planned.subqueries, "planner_recipe_sha256": plan_cache.recipe.fingerprint,
        "planner_receipt": planned.receipt, "rewrite_receipt": rewrite}))
    return validate_shared_row(row, task, protocol_sha256)


def failure_reviews(output):
    return {sha256_bytes(p.read_bytes()): json.loads(p.read_bytes()) for p in output.glob("failure-review-*.json")}


def retain_settled_plan_failure(task, protocol_sha, *, output, ledger, routes, rewrite_cache, plan_cache):
    """Explicit continuation policy: record the failure, never repair or retry it."""
    from app.ledger import Operation

    ctx = query_context(task, protocol_sha, "PLAN_QUERY")
    key = stable_id("query-plan-context", **ctx.model_dump(mode="json"))
    with sqlite3.connect(plan_cache.path) as db:
        saved = db.execute("SELECT state,payload,payload_sha256,intent FROM query_plans WHERE namespace='LIVE' AND context_key=?", (key,)).fetchone()
    if not saved or saved[0] != "FAILED" or not saved[1] or len(saved[1]) > 64 * 1024 or sha256_bytes(saved[1]) != saved[2]:
        raise ValueError("BENCHMARK_FAILURE_NOT_COMPLETE_RECONCILIATION_REQUIRED")
    payload = SavedPlanReceipt.model_validate_json(saved[1])
    if payload.namespace != "LIVE" or payload.binding_sha256 != sha256_bytes(saved[3]):
        raise ValueError("BENCHMARK_FAILURE_CACHE_BINDING_MISMATCH")
    receipt = payload.provider_receipt
    intent = json.loads(saved[3])
    if intent["context"] != ctx.model_dump(mode="json") or intent["question_sha256"] != sha256_bytes(task.question.encode()):
        raise ValueError("BENCHMARK_FAILURE_CONTEXT_MISMATCH")
    operation = ledger.store.get(("OP#" + receipt.operation_id, "STATE"))
    if operation is None:
        raise ValueError("BENCHMARK_FAILURE_LEDGER_OPERATION_MISSING")
    op = Operation.model_validate_json(canonical_json(operation))
    if op.status != "SETTLED" or op.actual_micro_usd != receipt.cost_micro_usd or op.bound_exceeded:
        raise ValueError("BENCHMARK_FAILURE_REQUIRES_LEDGER_RECONCILIATION")
    rewrite = rewrite_cache.resolve(query_context(task, protocol_sha, "REWRITE"), question=task.question,
        routes=routes) if task.language == "vi" else None
    review = {"status": "SETTLED_VALIDATION_FAILURE_RETAIN_NO_RETRY", "scenario_id": task.scenario_id,
        "language": task.language, "question_sha256": sha256_bytes(task.question.encode()),
        "operation_id": receipt.operation_id, "raw_output_sha256": receipt.raw_output_sha256,
        "ledger_state": "SETTLED", "cost_micro_usd": receipt.cost_micro_usd,
        "policy": "Complete receipted plan rejected by frozen parser; dependent Q/QB fail; no output repair, fallback or redispatch"}
    raw = canonical_json(review) + b"\n"
    row = FailedSharedQueryRow.model_validate_json(canonical_json({"preparation_status": "PLAN_FAILED",
        "scenario_id": task.scenario_id, "language": task.language,
        "question_sha256": sha256_bytes(task.question.encode()),
        "search_query": rewrite.search_query if rewrite else task.question, "subqueries": [],
        "planner_recipe_sha256": plan_cache.recipe.fingerprint,
        "planner_receipt": receipt.model_dump(mode="json"), "rewrite_receipt": rewrite.receipt if rewrite else None,
        "failure_code": "QUERY_PLAN_OUTPUT_INVALID", "failure_review_sha256": sha256_bytes(raw)}))
    validate_shared_row(row, task, protocol_sha)
    path = output / ("failure-review-" + receipt.operation_id + ".json")
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("BENCHMARK_FAILURE_REVIEW_CHANGED")
    if not path.exists():
        _write_new(path, review)
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/benchmark316/shared-queries-v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--continue-settled-validation-failures", action="store_true",
                        help="Retain fully settled invalid plans as failed cells; no retries or output edits")
    args = parser.parse_args(argv)
    protocol_raw, dataset, _, _, _ = load_inputs()
    protocol_sha = sha256_bytes(protocol_raw)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json(PRICES.read_bytes())
    tasks = tasks_from(dataset)
    output = args.output_dir.resolve()
    if output != (ROOT / "results/benchmark316/shared-queries-v1").resolve():
        raise ValueError("BENCHMARK_SHARED_V1_OUTPUT_PINNED")
    output.mkdir(parents=True, exist_ok=True)
    intent = plan_for(tasks, protocol_sha, routes, prices)
    raw_intent = canonical_json(intent) + b"\n"
    plan_path = output / "plan.json"
    if plan_path.exists() and plan_path.read_bytes() != raw_intent:
        raise ValueError("BENCHMARK_SHARED_PLAN_CHANGED")
    if not plan_path.exists():
        _write_new(plan_path, intent)
    if not args.execute:
        print(json.dumps({"status": "PLANNED_ONLY", "max_calls": 300, "cap_usd": 0.5,
                          "generate_answer_calls": 0, "ledger_writes": 0}))
        return 0
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id,
                    routes.source_region, now=datetime.now(UTC), live=True)
    selected = tasks
    if args.case_ids:
        if not set(args.case_ids) <= {s.scenario_id for s in dataset.scenarios}:
            parser.error("Unknown scenario IDs")
        selected = tuple(t for t in tasks if t.scenario_id in args.case_ids)
    raw_gateway, before, account = gateway_for(routes, prices)
    gateway = BoundedQueryGateway(raw_gateway, selected, protocol_sha)
    cache_path = ROOT / "data/cache/benchmark316-shared-v1"
    rewrite_cache, plan_cache = RewriteCache(cache_path, namespace="LIVE"), QueryPlanCache(cache_path, namespace="LIVE")
    (output / "rows").mkdir(exist_ok=True)
    try:
        for task in selected:
            path = output / "rows" / f"{task.scenario_id}.{task.language}.json"
            if path.exists():
                validate_shared_row(parse_shared_row(path.read_bytes()), task, protocol_sha)
                continue
            try:
                row = prepare_row(task, protocol_sha, routes=routes, rewrite_cache=rewrite_cache,
                                  plan_cache=plan_cache, gateway=gateway)
            except Exception:
                if not args.continue_settled_validation_failures:
                    raise
                row = retain_settled_plan_failure(task, protocol_sha, output=output, ledger=raw_gateway.ledger,
                    routes=routes, rewrite_cache=rewrite_cache, plan_cache=plan_cache)
            _write_new(path, row.model_dump(mode="json"))
            print(json.dumps({"case": task.scenario_id, "language": task.language,
                "subqueries": len(row.subqueries), "dispatches": len(gateway.seen)}), flush=True)
        complete = all((output / "rows" / f"{t.scenario_id}.{t.language}.json").exists() for t in tasks)
        if complete:
            rows = tuple(parse_shared_row((output / "rows" / f"{t.scenario_id}.{t.language}.json").read_bytes()) for t in tasks)
            bundle = SharedQueryBundle(protocol_sha256=protocol_sha, model_metadata_sha256=routes.account_model_metadata_sha256, rows=rows)
            validate_shared_bundle(bundle, tasks, protocol_sha256=protocol_sha, metadata_sha256=routes.account_model_metadata_sha256,
                                   failure_reviews=failure_reviews(output))
            path = output / "bundle.json"
            if not path.exists():
                _write_new(path, bundle.model_dump(mode="json"))
        after = counters(raw_gateway.ledger)
        record = {"status": "COMPLETE" if complete else "CANARY_COMPLETE", "account": account,
            "dispatches_this_invocation": len(gateway.seen), "before": before, "after": after,
            "bundle_sha256": sha256_bytes((output / "bundle.json").read_bytes()) if complete else None,
            "generate_answer_calls": 0, "global_budget_caps_changed": False}
        if complete:
            receipts = [r.planner_receipt for r in rows] + [r.rewrite_receipt for r in rows if r.rewrite_receipt]
            total = sum(r.cost_micro_usd for r in receipts)
            if after["run"]["reserved_micro_usd"] != 0 or after["run"]["settled_micro_usd"] != total:
                raise ValueError("BENCHMARK_SHARED_LEDGER_RECEIPT_RECONCILIATION_REQUIRED")
            completion = {"status": "COMPLETE", "protocol_sha256": protocol_sha,
                "bundle_sha256": record["bundle_sha256"], "plan_sha256": sha256_bytes(plan_path.read_bytes()),
                "provider_operations": len(receipts), "preparation_cost_micro_usd": total,
                "failed_preparation_inputs": sum(r.preparation_status == "PLAN_FAILED" for r in rows),
                "cost_basis": "PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND", "run_reserved_micro_usd": 0,
                "run_settled_micro_usd": total, "run_cap_micro_usd": 500_000,
                "global_caps_changed": False, "generate_answer_calls": 0}
            path = output / "completion.json"
            if path.exists() and path.read_bytes() != canonical_json(completion) + b"\n":
                raise ValueError("BENCHMARK_SHARED_COMPLETION_IMMUTABLE")
            if not path.exists():
                _write_new(path, completion)
    except Exception as exc:
        record = {"status": "STOPPED_NO_AUTOMATIC_RETRY", "error_category": safe_failure_code(exc),
            "dispatches_this_invocation": len(gateway.seen), "generate_answer_calls": 0}
        try:
            record["after"] = counters(raw_gateway.ledger)
        except Exception:
            record["after"] = "RECONCILIATION_REQUIRED"
        path = output / ("attempt-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
        _write_new(path, record)
        print(json.dumps(record, default=str))
        return 2
    path = output / ("attempt-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".json")
    _write_new(path, record)
    print(json.dumps({"status": record["status"], "dispatches": len(gateway.seen), "output": str(path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
