"""Preregistered three H/Q pairs; explicit sequential paid execution, no resume."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.answer_cache import AnswerCache
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import sha256_bytes
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
from app.generation import GenerationRecipeV9
from app.local_bge import BgeEncoder
from app.model_gateway import safe_failure_code
from app.query_planning import QueryPlanCache
from app.rag_methods import RunInputV1
from app.rag_runner import run_methods
from app.rewrite_cache import RewriteCache
from scripts.answer_local import ROOT, RUN, managed_gateway
from scripts.run_reranker_answer_pilot import write_once


def check_pair(report):
    if report["status"] != "COMPLETED" or not report["cost_complete"] or len(report["methods"]) != 2:
        raise ValueError("Q_PILOT_PAIR_NOT_COMPLETED_STOP_NO_RETRY")
    hybrid, multi = report["methods"]
    if (hybrid["effective_method"] != "hybrid" or multi["effective_method"] != "hybrid_multi_query"
            or hybrid["ranked_chunk_ids"] != multi["multi_query_trace"]["queries"][0]["ranked_chunk_ids"]
            or multi["multi_query_trace"]["base_search_query"] != report["search_query"]):
        raise ValueError("Q_PILOT_BASE_ANCHOR_MISMATCH")
    for field in ("collection_manifest_sha256", "lexical_index_version", "document_recipe_sha256",
                  "query_recipe_sha256", "generation_recipe_sha256", "max_context_chunks",
                  "context_proxy_token_budget", "max_output_tokens", "candidates_per_branch", "fusion"):
        if hybrid["config"][field] != multi["config"][field]:
            raise ValueError("Q_PILOT_COMMON_CONFIGURATION_MISMATCH")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "results/multi-query/pilot-v1")
    args = parser.parse_args(argv)
    directory = args.output_directory
    if (directory / "execution-started.json").exists() or (directory / "summary.json").exists():
        raise ValueError("Q_PILOT_EXECUTION_ALREADY_STARTED_USE_REPLAY")
    cases_path = ROOT / "configs/evaluation/multi-query-answer-pilot-v1.json"
    inputs = json.loads(cases_path.read_bytes())
    if (inputs.get("purpose") != "PREREGISTERED_DEVELOPMENT_ANSWER_PILOT_NOT_GOLD_OR_HOLDOUT"
            or inputs["human_reviewed"] is not False or inputs["holdout"] is not False
            or inputs["gold"] is not False or len(inputs["cases"]) != 3):
        raise ValueError("Q_PILOT_PREREGISTRATION_INVALID")
    cases = inputs["cases"]
    if (any(set(case) != {"id", "question", "language", "methods"}
            or case["methods"] != ["hybrid", "hybrid_multi_query"] for case in cases)
            or any(type(case["id"]) is not str or not case["id"].strip() for case in cases)
            or len({case["id"] for case in cases}) != 3
            or sum(case["language"] == "vi" for case in cases) != 1
            or any(case["language"] not in {"en", "vi"} for case in cases)):
        raise ValueError("Q_PILOT_RUNTIME_FIELDS_INVALID")
    requests = [RunInputV1(original_question=case["question"], language=case["language"],
        method_ids=tuple(case["methods"]), run_kind="answer",
        idempotency_key=f"q-answer-v1-{index}-{sha256_bytes(cases_path.read_bytes())[:16]}")
        for index, case in enumerate(cases)]
    paths_file = ROOT / "results/corpus-expansion-v2/pipeline.json"
    paths = json.loads(paths_file.read_bytes())
    price_path = ROOT / "configs/prices/external-direct-2026-10-24.json"
    prices = ExternalPriceSheet.model_validate_json(price_path.read_bytes())
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region,
                    now=datetime.now(UTC), live=True)
    recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    planner = QueryPlanCache(ROOT / "data/cache/collection-query-plans", namespace="LIVE")
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id=encoder.recipe("document").model_id, cache_shards=4)
    plan = {"schema_version": 1, "kind": "PREREGISTERED_THREE_PAIR_H_Q_DEVELOPMENT_PILOT",
        "cases_sha256": sha256_bytes(cases_path.read_bytes()), "case_ids": [case["id"] for case in cases],
        "requests": [request.model_dump(mode="json") for request in requests],
        "collection_sha256": collection.manifest_sha256, "pipeline_sha256": sha256_bytes(paths_file.read_bytes()),
        "generation_recipe_sha256": recipe.fingerprint, "planner_recipe": planner.recipe.model_dump(mode="json"),
        "planner_recipe_sha256": planner.recipe.fingerprint, "query_recipe_sha256": encoder.recipe("query").fingerprint,
        "price_sheet_sha256": sha256_bytes(price_path.read_bytes()), "run_id": RUN,
        "max_new_generation_calls": 6, "max_new_planner_calls": 3, "max_shared_vi_rewrites": 1,
        "max_new_model_dispatches": 10, "sequential_only": True,
        "quality_metrics": None, "human_reviewed": False, "holdout": False, "gold": False,
        "illustrative_cost_micro_usd": 22382, "forecast_basis": "6x8000input800output +3x900input256output +1x600input256output at published peak cache miss; not invoice or aggregate reservation authorization",
        "cloud_deployed": False}
    if (directory / "plan.json").exists():
        if json.loads((directory / "plan.json").read_bytes()) != plan:
            raise ValueError("Q_PILOT_PLAN_PIN_MISMATCH")
    else:
        write_once(directory / "plan.json", plan)
    if not args.execute:
        print(json.dumps({"status": "PLANNED_NO_PAID_OR_AWS_CALL", "cases": 3, "path": str(directory / "plan.json")}))
        return 0
    preflight = {"created_at": datetime.now(UTC).isoformat(), "model_dispatch_attempts": 0}
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    try:
        account = deepseek_preflight(OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}))
        preflight["provider"] = account
        if not account["available"] or account["metadata_sha256"] != routes.account_model_metadata_sha256:
            raise ValueError("Q_PILOT_ACCOUNT_ROUTE_MISMATCH")
        gateway = managed_gateway(prices_path=price_path)
        ledger = gateway.ledger
        counters = {"run": ledger.store.get(ledger.run_key(RUN)),
            "funding": ledger.store.get(ledger.external_funding_key("kg-r72")),
            "month": ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))}
        preflight["counters"] = counters
        if any(counter is None or counter["halted"] for counter in counters.values()):
            raise ValueError("Q_PILOT_LEDGER_NOT_READY")
        preflight["status"] = "READY_READ_ONLY_NO_INFERENCE"
    except Exception as exc:
        preflight.update(status="BLOCKED_BEFORE_EXECUTION", error_category=safe_failure_code(exc))
        write_once(directory / f"preflight-{stamp}.json", preflight)
        print(json.dumps({"status": preflight["status"], "error_category": preflight["error_category"], "model_calls": 0}))
        return 2
    preflight_path = directory / f"preflight-{stamp}.json"
    write_once(preflight_path, preflight)
    write_once(directory / "execution-started.json", {"created_at": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_bytes((directory / "plan.json").read_bytes()),
        "preflight_path": str(preflight_path), "preflight_sha256": sha256_bytes(preflight_path.read_bytes()), "resume_allowed": False})
    before, reports, error = gateway.provider.dispatch_attempts, [], None
    try:
        with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
            for index, request in enumerate(requests):
                case_before = gateway.provider.dispatch_attempts
                report = run_methods(collection, lexical, request, tenant_id="demo", owner_id="local-ldh-owner",
                    run_id=RUN, recipe=recipe, routes=routes, encoder=encoder, query_plan_cache=planner,
                    cache=AnswerCache(ROOT / "data/cache/collection-answers", namespace="LIVE"),
                    embedding_cache=EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE"),
                    rewrite_cache=RewriteCache(ROOT / "data/cache/collection-rewrites", namespace="LIVE"), gateway=gateway)
                report.update(pilot_case_id=cases[index]["id"],
                    model_dispatch_attempts_this_invocation=gateway.provider.dispatch_attempts - case_before)
                write_once(directory / f"case-{index + 1}.json", report)
                reports.append(report)
                check_pair(report)
                if gateway.provider.dispatch_attempts - before > 10:
                    raise ValueError("Q_PILOT_CALL_CEILING_EXCEEDED")
                print(json.dumps({"case": index + 1, "status": report["status"],
                    "new_known_cost_micro_usd": report["new_known_cost_micro_usd"]}), flush=True)
    except Exception as exc:
        error = safe_failure_code(exc)
    summary = {"status": "COMPLETED" if len(reports) == 3 and not error else "PARTIAL_STOPPED_NO_RETRY",
        "created_at": datetime.now(UTC).isoformat(), "error_category": error, "case_reports": len(reports),
        "model_dispatch_attempts": gateway.provider.dispatch_attempts - before,
        "new_known_cost_micro_usd": sum(report["new_known_cost_micro_usd"] for report in reports),
        "cost_complete": not error and all(report["cost_complete"] for report in reports),
        "local_query_onnx_calls": encoder.inference_calls, "semantic_review": "NOT_REVIEWED",
        "human_reviewed": False, "gold": False, "holdout": False, "cloud_deployed": False,
        "cost_basis": "PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND_NOT_INVOICE"}
    write_once(directory / "summary.json", summary)
    print(json.dumps(summary))
    return 0 if summary["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
