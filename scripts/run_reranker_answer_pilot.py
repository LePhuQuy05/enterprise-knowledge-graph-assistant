"""Three preregistered H/R answer pairs; paid execution explicit and never resumed."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.answer_cache import AnswerCache
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalRoutes
from app.generation import GenerationRecipeV9
from app.local_bge import BgeEncoder
from app.model_gateway import safe_failure_code
from app.rag_methods import RunInputV1
from app.rag_runner import run_methods
from app.reranking import OnnxCrossEncoder
from app.rewrite_cache import RewriteCache
from scripts.answer_local import ROOT, RUN, managed_gateway


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json(value) + b"\n")


def load_cases(path):
    data = json.loads(path.read_bytes())
    if (data.get("purpose") != "PREREGISTERED_DEVELOPMENT_ANSWER_PILOT_NOT_GOLD_OR_HOLDOUT"
            or data.get("human_reviewed") is not False or data.get("gold") is not False
            or data.get("holdout") is not False or len(data.get("cases", [])) != 3):
        raise ValueError("R_PILOT_PREREGISTRATION_INVALID")
    requests = []
    identifiers = set()
    for index, case in enumerate(data["cases"]):
        if set(case) != {"id", "question", "language", "methods"} or case["methods"] != ["hybrid", "hybrid_reranked"]:
            raise ValueError("R_PILOT_RUNTIME_INPUT_FIELDS_INVALID")
        if (not isinstance(case["id"], str) or not case["id"].strip()
                or case["id"] in identifiers):
            raise ValueError("R_PILOT_CASE_ID_INVALID")
        identifiers.add(case["id"])
        requests.append(RunInputV1(original_question=case["question"], language=case["language"],
            method_ids=tuple(case["methods"]), run_kind="answer",
            idempotency_key=f"r-answer-v1-{index}-{sha256_bytes(path.read_bytes())[:16]}"))
    if sum(request.language == "vi" for request in requests) != 1 or any(request.language == "auto" for request in requests):
        raise ValueError("R_PILOT_EXPECTS_TWO_EN_ONE_VI")
    return data, requests


def check_pair(report):
    if report["status"] != "COMPLETED" or not report["cost_complete"] or len(report["methods"]) != 2:
        raise ValueError("R_PILOT_PAIR_NOT_COMPLETED_STOP_NO_RETRY")
    hybrid, reranked = report["methods"]
    if (hybrid["effective_method"] != "hybrid" or reranked["effective_method"] != "hybrid_reranked"
            or hybrid["ranked_chunk_ids"] != reranked["rerank_trace"]["baseline_candidate_ids"]):
        raise ValueError("R_PILOT_SAME_POOL_MISMATCH")
    for field in ("collection_manifest_sha256", "lexical_index_version", "document_recipe_sha256",
                  "query_recipe_sha256", "generation_recipe_sha256", "max_context_chunks",
                  "context_proxy_token_budget", "max_output_tokens", "candidates_per_branch", "fusion"):
        if hybrid["config"][field] != reranked["config"][field]:
            raise ValueError("R_PILOT_COMMON_CONFIGURATION_MISMATCH")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cases", type=Path, default=ROOT / "configs/evaluation/reranker-answer-pilot-v1.json")
    parser.add_argument("--output-directory", type=Path, default=ROOT / "results/reranker-answering/pilot-v1")
    args = parser.parse_args(argv)
    output = args.output_directory
    # A crash or partial result requires investigation, never a new provider attempt.
    if (output / "execution-started.json").exists() or (output / "summary.json").exists():
        raise ValueError("R_PILOT_EXECUTION_ALREADY_STARTED_USE_REPLAY")
    cases, requests = load_cases(args.cases)
    pipeline_path = ROOT / "results/corpus-expansion-v2/pipeline.json"
    paths = json.loads(pipeline_path.read_bytes())
    prices_path = ROOT / "configs/prices/external-direct-2026-10-24.json"
    preflight_path = ROOT / "results/reranker-answering/preflight-summary.json"
    preflight = json.loads(preflight_path.read_bytes())
    if preflight["status"] != "READY_FOR_ROOT_SCOPED_SEQUENTIAL_PILOT_DECISION":
        raise ValueError("R_PILOT_PREFLIGHT_NOT_READY")
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    reranker = OnnxCrossEncoder(ROOT / "data/models/ms-marco-MiniLM-L2-v2",
                               ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json")
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id=encoder.recipe("document").model_id, cache_shards=4)
    pins = {"schema_version": 1, "kind": "PREREGISTERED_THREE_PAIR_DEVELOPMENT_PILOT",
        "cases_sha256": sha256_bytes(args.cases.read_bytes()), "case_ids": [case["id"] for case in cases["cases"]],
        "requests": [request.model_dump(mode="json") for request in requests],
        "collection_sha256": collection.manifest_sha256, "pipeline_sha256": sha256_bytes(pipeline_path.read_bytes()),
        "generation_recipe_sha256": recipe.fingerprint, "reranker_recipe_sha256": reranker.recipe.fingerprint,
        "query_recipe_sha256": encoder.recipe("query").fingerprint,
        "price_sheet_sha256": sha256_bytes(prices_path.read_bytes()), "preflight_sha256": sha256_bytes(preflight_path.read_bytes()),
        "run_id": RUN, "max_new_generation_calls": 6, "max_new_rewrite_calls": 1,
        "sequential_only": True, "human_reviewed": False, "holdout": False, "gold": False,
        "quality_metrics": None, "cloud_deployed": False}
    if pins["price_sheet_sha256"] != preflight["price_sheet_sha256"]:
        raise ValueError("R_PILOT_PRICE_PREFLIGHT_MISMATCH")
    plan_path = output / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_bytes()) != pins:
            raise ValueError("R_PILOT_PLAN_PIN_MISMATCH")
    else:
        write_once(plan_path, pins)
    if not args.execute:
        print(json.dumps({"status": "PLANNED_NO_PAID_CALL", "output": str(plan_path), "cases": 3}))
        return 0
    try:
        gateway = managed_gateway(prices_path=prices_path)
    except Exception as exc:
        # This helper only authenticates and reads the existing counters. It
        # cannot dispatch inference or reserve/reset a budget during setup.
        failure = {"status": "BLOCKED_BEFORE_EXECUTION", "created_at": datetime.now(UTC).isoformat(),
            "error_category": safe_failure_code(exc), "model_dispatch_attempts": 0,
            "ledger_writes": 0, "execution_started": False}
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        write_once(output / f"setup-failure-{stamp}.json", failure)
        print(json.dumps(failure))
        return 2
    write_once(output / "execution-started.json", {"started_at": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_bytes(plan_path.read_bytes()), "automatic_resume": False})
    before = gateway.provider.dispatch_attempts
    reports, error = [], None
    try:
        with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
            for index, request in enumerate(requests):
                case_before = gateway.provider.dispatch_attempts
                report = run_methods(collection, lexical, request, tenant_id="demo", owner_id="local-ldh-owner",
                    run_id=RUN, recipe=recipe, routes=routes, encoder=encoder, reranker=reranker,
                    embedding_cache=EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE"),
                    cache=AnswerCache(ROOT / "data/cache/collection-answers", namespace="LIVE"),
                    rewrite_cache=RewriteCache(ROOT / "data/cache/collection-rewrites", namespace="LIVE"), gateway=gateway)
                report["pilot_case_id"] = cases["cases"][index]["id"]
                report["model_dispatch_attempts_this_invocation"] = gateway.provider.dispatch_attempts - case_before
                write_once(output / f"case-{index + 1}.json", report)
                reports.append(report)
                check_pair(report)
                if gateway.provider.dispatch_attempts - before > 7:
                    raise ValueError("R_PILOT_CALL_CEILING_EXCEEDED")
                print(json.dumps({"case": index + 1, "status": report["status"],
                    "new_known_cost_micro_usd": report["new_known_cost_micro_usd"]}), flush=True)
    except Exception as exc:
        error = safe_failure_code(exc)
    summary = {"schema_version": 1, "status": "COMPLETED" if len(reports) == 3 and not error else "PARTIAL_STOPPED_NO_RETRY",
        "created_at": datetime.now(UTC).isoformat(), "error_category": error, "case_reports": len(reports),
        "model_dispatch_attempts": gateway.provider.dispatch_attempts - before,
        "new_known_cost_micro_usd": sum(report["new_known_cost_micro_usd"] for report in reports),
        "cost_complete": not error and all(report["cost_complete"] for report in reports),
        "local_query_onnx_calls": encoder.inference_calls, "local_reranker_onnx_calls": reranker.inference_calls,
        "semantic_review": "NOT_REVIEWED", "quality_metrics": None, "human_reviewed": False,
        "holdout": False, "cloud_deployed": False, "cost_basis": "PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND_NOT_INVOICE"}
    write_once(output / "summary.json", summary)
    print(json.dumps(summary))
    return 0 if summary["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
