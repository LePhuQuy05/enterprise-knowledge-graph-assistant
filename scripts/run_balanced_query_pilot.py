"""Preregister three Q/QB answer pairs; explicit bounded execution, never resume.

Planning reads local metadata only. --preflight also reads provider availability
and the existing managed counters; neither mode dispatches inference. Historical
Q/R receipts are hashed without interpreting their outcomes.
"""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.answer_cache import AnswerCache
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context_selection import ContextSelectionRecipe
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
from app.generation import GenerationRecipeV9
from app.local_bge import BgeEncoder
from app.model_gateway import safe_failure_code
from app.query_planning import PlannerRecipe, QueryPlanCache
from app.rag_methods import MethodResultV1, RunInputV1
from app.rag_runner import run_methods
from app.rewrite_cache import RewriteCache
from scripts.answer_local import ROOT, RUN, managed_gateway
from scripts.run_reranker_answer_pilot import write_once

METHODS = ("hybrid_multi_query", "hybrid_multi_query_balanced")
SOURCE_CASES_SHA256 = "42d3eb6d9f569eb5b7c2d85d9928f2f47c22f24ad1cd1569fcaacaa4b44d2f5b"
CASE_IDS = ("citation-dev-ecs-en", "citation-dev-lambda-partial-vi", "rerank-dev-s3-consistency-en")
STAGE_LIMITS = {"GENERATE": 6, "PLAN_QUERY": 3, "REWRITE": 1}
OUTPUT_LIMITS = {"GENERATE": 800, "PLAN_QUERY": 256, "REWRITE": 256}
# Previously reconciled legacy reservations remain charged against the shared
# budgets. They are neither new pilot capacity nor permission to refund/reset.
PRESERVED_RESERVATIONS = {"run": 640, "funding": 640, "month": 1250}
COMMON_FIELDS = ("collection_manifest_sha256", "lexical_index_version", "document_recipe_sha256",
    "query_recipe_sha256", "generation_recipe_sha256", "max_context_chunks",
    "context_proxy_token_budget", "max_output_tokens", "candidates_per_branch", "fusion",
    "max_queries", "max_candidates", "max_transient_candidates", "query_fusion",
    "planner_recipe", "planner_recipe_sha256")
HISTORICAL_DIRECTORIES = ("results/multi-query/pilot-v1", "results/reranker-answering/pilot-v1")
HISTORICAL_NAMES = ("case-1.json", "case-2.json", "case-3.json", "case-1-replay.json",
    "case-2-replay.json", "case-3-replay.json", "plan.json", "summary.json", "reconciliation.json")
IMPLEMENTATIONS = ("app/context_selection.py", "app/collection_answering.py", "app/rag_runner.py",
    "app/rag_methods.py", "scripts/run_balanced_query_pilot.py")


def load_cases(path):
    """Use the exact prior questions; only methods and request keys are new."""
    raw = path.read_bytes()
    if sha256_bytes(raw) != SOURCE_CASES_SHA256:
        raise ValueError("QB_PILOT_SOURCE_CASES_PIN_MISMATCH")
    data = json.loads(raw)
    if (data.get("purpose") != "PREREGISTERED_DEVELOPMENT_ANSWER_PILOT_NOT_GOLD_OR_HOLDOUT"
            or any(data.get(field) is not False for field in ("human_reviewed", "holdout", "gold"))
            or not isinstance(data.get("cases"), list) or len(data["cases"]) != 3):
        raise ValueError("QB_PILOT_PREREGISTRATION_INVALID")
    cases = data["cases"]
    if (any(set(case) != {"id", "question", "language", "methods"}
            or case["methods"] != ["hybrid", "hybrid_multi_query"] for case in cases)
            or tuple(case["id"] for case in cases) != CASE_IDS
            or tuple(case["language"] for case in cases) != ("en", "vi", "en")):
        raise ValueError("QB_PILOT_RUNTIME_INPUT_FIELDS_INVALID")
    requests = [RunInputV1(original_question=case["question"], language=case["language"],
        method_ids=METHODS, run_kind="answer",
        idempotency_key=f"qb-answer-v1-{index}-{SOURCE_CASES_SHA256[:16]}")
        for index, case in enumerate(cases)]
    return data, requests


def check_pair(report, *, generation_recipe_sha256=None):
    """Structural fairness only: chosen citations still need semantic review."""
    if (report.get("status") != "COMPLETED" or report.get("cost_complete") is not True
            or len(report.get("methods", [])) != 2
            or any(row.get("status") != "COMPLETED" for row in report["methods"])):
        raise ValueError("QB_PILOT_PAIR_NOT_COMPLETED_STOP_NO_RETRY")
    q, balanced = report["methods"]
    if (tuple(row.get("effective_method") for row in (q, balanced)) != METHODS
            or tuple(row.get("requested_method") for row in (q, balanced)) != METHODS
            or q["config"].get("schema_version") != 3 or balanced["config"].get("schema_version") != 4):
        raise ValueError("QB_PILOT_METHOD_OR_SCHEMA_MISMATCH")
    selector = ContextSelectionRecipe()
    if (balanced["config"].get("context_selection_recipe") != selector.model_dump(mode="json")
            or balanced["config"].get("context_selection_recipe_sha256") != selector.fingerprint):
        raise ValueError("QB_PILOT_RUNTIME_POLICY_MISMATCH")
    if any(q["config"].get(field) != balanced["config"].get(field) for field in COMMON_FIELDS):
        raise ValueError("QB_PILOT_COMMON_CONFIGURATION_MISMATCH")
    if generation_recipe_sha256 is not None and q["config"]["generation_recipe_sha256"] != generation_recipe_sha256:
        raise ValueError("QB_PILOT_GENERATOR_PIN_MISMATCH")
    q_trace, qb_trace = q.get("multi_query_trace"), balanced.get("multi_query_trace")
    if (q_trace is None or q_trace != qb_trace or q_trace.get("base_search_query") != report.get("search_query")
            or q.get("ranked_chunk_ids") != q_trace.get("ranked_chunk_ids")):
        raise ValueError("QB_PILOT_SAME_FROZEN_POOL_MISMATCH")
    q_plan, qb_plan = q.get("planning_trace"), balanced.get("planning_trace")
    if (not q_plan or not qb_plan or q_plan.get("receipt") != qb_plan.get("receipt")
            or q_plan.get("recipe_sha256") != qb_plan.get("recipe_sha256")
            or q_plan.get("subqueries") != qb_plan.get("subqueries")
            or qb_plan.get("cache_hit") is not True or qb_plan.get("new_cost_micro_usd") != 0
            or report.get("planner_sharing") != "Q_FAMILY_V1_SINGLE_DURABLE_PLANNER_OPERATION"):
        raise ValueError("QB_PILOT_SHARED_PLANNER_MISMATCH")
    # Validate the actual typed receipt, source-pool, ordering and response
    # bindings rather than accepting matching unverified fingerprint strings.
    for row in (q, balanced):
        MethodResultV1.model_validate_json(canonical_json(row))


class BoundedPilotGateway:
    """Additional local call ceilings; does not alter managed ledger budgets."""

    def __init__(self, gateway):
        self._gateway = gateway
        self.provider, self.routes, self.ledger = gateway.provider, gateway.routes, gateway.ledger
        self.stage_attempts = Counter()
        self.halted = False

    def generate(self, context, *, system, prompt, max_output_tokens=800):
        stage = context.stage
        if (self.halted or stage not in STAGE_LIMITS or context.attempt != 1
                or context.run_id != RUN or context.tenant_id != "demo"
                or max_output_tokens != OUTPUT_LIMITS[stage]
                or self.stage_attempts[stage] >= STAGE_LIMITS[stage]
                or sum(self.stage_attempts.values()) >= 10):
            self.halted = True
            raise ValueError("QB_PILOT_PRE_DISPATCH_CEILING_OR_SCOPE_BLOCKED")
        self.stage_attempts[stage] += 1
        try:
            return self._gateway.generate(context, system=system, prompt=prompt,
                                          max_output_tokens=max_output_tokens)
        except Exception:
            self.halted = True  # A failed/uncertain operation must never trigger a later call.
            raise


def historical_artifact_hashes(directory):
    output = directory.resolve()
    hashes = {}
    for relative in HISTORICAL_DIRECTORIES:
        old = (ROOT / relative).resolve()
        if output.is_relative_to(old) or old.is_relative_to(output):
            raise ValueError("QB_PILOT_OUTPUT_OVERLAPS_HISTORICAL_ARTIFACTS")
        for name in HISTORICAL_NAMES:
            path = ROOT / relative / name
            hashes[f"{relative}/{name}"] = sha256_bytes(path.read_bytes())
    return hashes


def _assert_history_unchanged(pins):
    if any(sha256_bytes((ROOT / path).read_bytes()) != checksum for path, checksum in pins.items()):
        raise ValueError("QB_PILOT_HISTORICAL_ARTIFACT_CHANGED")


def _preflight(routes, prices_path):
    account = deepseek_preflight(OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}))
    if not account["available"] or account["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("QB_PILOT_ACCOUNT_ROUTE_MISMATCH")
    gateway = managed_gateway(prices_path=prices_path)
    ledger = gateway.ledger
    counters = {"run": ledger.store.get(ledger.run_key(RUN)),
        "funding": ledger.store.get(ledger.external_funding_key("kg-r72")),
        "month": ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))}
    if any(counter is None or counter["halted"]
           or counter.get("reserved_micro_usd") != PRESERVED_RESERVATIONS[name]
           or counter["settled_micro_usd"] + counter["reserved_micro_usd"] >= counter["cap_micro_usd"]
           for name, counter in counters.items()):
        raise ValueError("QB_PILOT_LEDGER_NOT_READY_OR_PENDING_RESERVATION")
    return gateway, {"provider": account, "counters": counters}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight", action="store_true", help="Read provider/counters only, no inference")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "results/context-allocation/answer-pilot-v1")
    args = parser.parse_args(argv)
    directory = args.output_directory
    if (any((directory / name).exists() for name in ("execution-started.json", "summary.json"))
            or any(directory.glob("case-*.json"))):
        raise ValueError("QB_PILOT_EXECUTION_ALREADY_STARTED_USE_REPLAY")
    history = historical_artifact_hashes(directory)
    cases_path = ROOT / "configs/evaluation/multi-query-answer-pilot-v1.json"
    inputs, requests = load_cases(cases_path)
    pipeline_path = ROOT / "results/corpus-expansion-v2/pipeline.json"
    paths = json.loads(pipeline_path.read_bytes())
    price_path = ROOT / "configs/prices/external-direct-2026-10-24.json"
    prices = ExternalPriceSheet.model_validate_json(price_path.read_bytes())
    route_path = ROOT / "configs/models/external-direct-v1.json"
    routes = ExternalRoutes.model_validate_json(route_path.read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region,
                    now=datetime.now(UTC), live=True)
    recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    selector, planner_recipe = ContextSelectionRecipe(), PlannerRecipe()
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id=encoder.recipe("document").model_id, cache_shards=4)
    implementations = {path: sha256_bytes((ROOT / path).read_bytes()) for path in IMPLEMENTATIONS}
    compare_path = ROOT / "scripts/compare_context_allocations.py"
    context_experiment_path = ROOT / "results/context-allocation/development-v1/experiment.json"
    plan = {"schema_version": 1, "kind": "PREREGISTERED_THREE_PAIR_Q_QB_DEVELOPMENT_PILOT",
        "cases_sha256": sha256_bytes(cases_path.read_bytes()), "case_ids": [case["id"] for case in inputs["cases"]],
        "requests": [request.model_dump(mode="json") for request in requests], "methods": list(METHODS),
        "collection_sha256": collection.manifest_sha256, "pipeline_sha256": sha256_bytes(pipeline_path.read_bytes()),
        "lexical_manifest_sha256": sha256_bytes(Path(paths["lexical_manifest_path"]).read_bytes()),
        "generation_recipe": recipe.model_dump(mode="json"), "generation_recipe_sha256": recipe.fingerprint,
        "planner_recipe": planner_recipe.model_dump(mode="json"), "planner_recipe_sha256": planner_recipe.fingerprint,
        "planner_sharing": "Q_FAMILY_V1_SINGLE_DURABLE_PLANNER_OPERATION",
        "context_selection_recipe": selector.model_dump(mode="json"),
        "context_selection_recipe_sha256": selector.fingerprint,
        "query_recipe_sha256": encoder.recipe("query").fingerprint,
        "price_sheet_sha256": sha256_bytes(price_path.read_bytes()), "routes_sha256": sha256_bytes(route_path.read_bytes()),
        "price_sheet_checked_at": prices.checked_at.isoformat(), "price_sheet_expires_at": prices.expires_at.isoformat(),
        "historical_artifact_sha256": history, "implementation_sha256": implementations,
        "previous_r_q_artifact_sha256": {
            ("Q/" if path.startswith(HISTORICAL_DIRECTORIES[0] + "/") else "R/") + Path(path).name: checksum
            for path, checksum in history.items()},
        "selector_module_sha256": implementations["app/context_selection.py"],
        "comparison_workflow_sha256": sha256_bytes(compare_path.read_bytes()) if compare_path.exists() else None,
        "context_experiment_sha256": sha256_bytes(context_experiment_path.read_bytes()) if context_experiment_path.exists() else None,
        "run_id": RUN, "max_new_generation_calls": 6, "max_new_planner_calls": 3,
        "max_shared_vi_rewrites": 1, "max_new_model_dispatches": 10,
        "pre_dispatch_stage_ceilings": STAGE_LIMITS, "sequential_only": True, "resume_allowed": False,
        "preserved_legacy_reservations_micro_usd": PRESERVED_RESERVATIONS,
        "quality_metrics": None, "semantic_review": "NOT_REVIEWED", "human_reviewed": False,
        "holdout": False, "gold": False, "cloud_deployed": False,
        "budget_policy": "USE_EXISTING_SHARED_LEDGER_COUNTERS_WITHOUT_CREATING_RESETTING_OR_RAISING_CAPS",
        "design_notice": "Known development questions; same Q pool and planner, RR_ID_ONLY ordering only. No source targets or expected facts enter runtime inputs. Execution success is not answer quality."}
    plan_path = directory / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_bytes()) != plan:
            raise ValueError("QB_PILOT_PLAN_PIN_MISMATCH")
    else:
        write_once(plan_path, plan)
    if not args.execute and not args.preflight:
        print(json.dumps({"status": "PLANNED_NO_PAID_OR_AWS_CALL", "cases": 3, "path": str(plan_path)}))
        return 0
    preflight = {"created_at": datetime.now(UTC).isoformat(), "model_dispatch_attempts": 0,
                 "ledger_writes": 0, "plan_sha256": sha256_bytes(plan_path.read_bytes())}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        _assert_history_unchanged(history)
        raw_gateway, observations = _preflight(routes, price_path)
        preflight.update(observations, status="READY_READ_ONLY_NO_INFERENCE")
    except Exception as exc:
        preflight.update(status="BLOCKED_BEFORE_EXECUTION", error_category=safe_failure_code(exc))
        write_once(directory / f"preflight-{stamp}.json", preflight)
        print(json.dumps({"status": preflight["status"], "error_category": preflight["error_category"], "model_calls": 0}))
        return 2
    preflight_path = directory / f"preflight-{stamp}.json"
    write_once(preflight_path, preflight)
    if args.preflight:
        print(json.dumps({"status": preflight["status"], "model_calls": 0, "path": str(preflight_path)}))
        return 0
    write_once(directory / "execution-started.json", {"created_at": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_bytes(plan_path.read_bytes()), "preflight_path": str(preflight_path),
        "preflight_sha256": sha256_bytes(preflight_path.read_bytes()), "resume_allowed": False})
    gateway = BoundedPilotGateway(raw_gateway)
    before, reports, error = gateway.provider.dispatch_attempts, [], None
    try:
        planner = QueryPlanCache(ROOT / "data/cache/collection-query-plans", namespace="LIVE")
        with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
            for index, request in enumerate(requests):
                _assert_history_unchanged(history)
                case_before = gateway.provider.dispatch_attempts
                report = run_methods(collection, lexical, request, tenant_id="demo", owner_id="local-ldh-owner",
                    run_id=RUN, recipe=recipe, routes=routes, encoder=encoder, query_plan_cache=planner,
                    context_selection_recipe=selector,
                    cache=AnswerCache(ROOT / "data/cache/collection-answers", namespace="LIVE"),
                    embedding_cache=EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE"),
                    rewrite_cache=RewriteCache(ROOT / "data/cache/collection-rewrites", namespace="LIVE"), gateway=gateway)
                report.update(pilot_case_id=inputs["cases"][index]["id"],
                    model_dispatch_attempts_this_invocation=gateway.provider.dispatch_attempts - case_before)
                write_once(directory / f"case-{index + 1}.json", report)
                reports.append(report)
                check_pair(report, generation_recipe_sha256=recipe.fingerprint)
                if gateway.halted or gateway.provider.dispatch_attempts - before > 10:
                    raise ValueError("QB_PILOT_CALL_CEILING_OR_HALTED")
                print(json.dumps({"case": index + 1, "status": report["status"],
                    "new_known_cost_micro_usd": report["new_known_cost_micro_usd"]}), flush=True)
        _assert_history_unchanged(history)
    except Exception as exc:
        error = safe_failure_code(exc)
    summary = {"schema_version": 1, "status": "COMPLETED" if len(reports) == 3 and not error else "PARTIAL_STOPPED_NO_RETRY",
        "created_at": datetime.now(UTC).isoformat(), "error_category": error, "case_reports": len(reports),
        "model_dispatch_attempts": gateway.provider.dispatch_attempts - before,
        "gateway_generate_attempts_by_stage": dict(gateway.stage_attempts),
        "gateway_halted": gateway.halted,
        "new_known_cost_micro_usd": sum(report["new_known_cost_micro_usd"] for report in reports),
        "cost_complete": not error and all(report["cost_complete"] for report in reports),
        "cost_reconciliation": "REQUIRED_SEPARATELY_FROM_MANAGED_OPERATION_RECEIPTS",
        "local_query_onnx_calls": encoder.inference_calls, "semantic_review": "NOT_REVIEWED",
        "quality_metrics": None, "human_reviewed": False, "gold": False, "holdout": False,
        "cloud_deployed": False, "cost_basis": "PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND_NOT_INVOICE"}
    write_once(directory / "summary.json", summary)
    print(json.dumps(summary))
    return 0 if summary["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
