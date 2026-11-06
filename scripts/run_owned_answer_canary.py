"""One preregistered EN owned-answer canary, existing managed budgets, no model retry."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.answer_cache import AnswerCache
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import canonical_json, sha256_bytes
from app.cost import TokenUsage
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
from app.generation import GenerationRecipeV9
from app.ledger import Ledger
from app.local_bge import BgeEncoder
from app.owned_runs import OwnedRunRepository
from app.query_auth import ApiError, AuthContext
from app.query_state import QueryRepository
from app.run_answer_executor import CollectionOwnedAnswerExecutor
from app.run_artifacts import LocalRunBlobs, RunArtifactRepository
from app.run_budget import RunBudgetController
from app.run_execution import OwnedRunWorker, RunOutboxDispatcher
from app.run_service import OwnedRunService
from app.run_storage import DynamoRunStore, SQLiteRunQueue
from scripts.answer_local import ROOT, RUN, managed_gateway
from scripts.reconcile_balanced_query_pilot import validate_counter_baseline

DIRECTORY = ROOT / "results/owned-answers/canary-v1"
PRICES = ROOT / "configs/prices/external-direct-2026-10-24.json"
QUESTION = "Which policies must permit cross-account use of an AWS KMS key?"
KEY = "1ada1a64-55f8-4083-9f31-742790795b15"
AUTH = AuthContext("demo", "00000000-0000-4000-8000-000000000002")
IMPLEMENTATIONS = ("app/run_budget.py", "app/run_answer_executor.py", "app/owned_runs.py", "app/run_execution.py",
    "app/run_service.py", "app/run_storage.py", "app/run_artifacts.py", "scripts/run_owned_answer_canary.py")


def write_once(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as output:
        output.write(canonical_json(value) + b"\n")


def validate_saved_plan(stored, current, directory, *, replay):
    # A replay-only runner repair does not rewrite the executed plan. Verify and
    # retain its original source; all generation/data/ledger implementation pins stay exact.
    own_path = "scripts/run_owned_answer_canary.py"
    if replay and stored["implementation_sha256"][own_path] != current["implementation_sha256"][own_path]:
        archived = directory / "execution-runner.py"
        if not archived.is_file() or sha256_bytes(archived.read_bytes()) != stored["implementation_sha256"][own_path]:
            raise ValueError("OWNED_EXECUTED_RUNNER_ARCHIVE_MISSING")
        current = current | {"implementation_sha256": current["implementation_sha256"] | {
            own_path: stored["implementation_sha256"][own_path]}}
    if stored != current:
        raise ValueError("OWNED_CANARY_PLAN_CHANGED")


def same_recorded_result(detail, results, old):
    return canonical_json([detail, results]) == canonical_json([old["run"], old["results"]])


def preflight():
    now = datetime.now(UTC)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json(PRICES.read_bytes())
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=now, live=True)
    provider = deepseek_preflight(OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}))
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("OWNED_PROVIDER_METADATA_NOT_READY")
    original = managed_gateway(prices_path=PRICES)
    store = DynamoRunStore(original.ledger.store.client, original.ledger.store.table_name)
    gateway = ExternalModelGateway(Ledger(store), original.provider, original.routes, original.prices)
    ledger = gateway.ledger
    counters = {"run": store.get(ledger.run_key(RUN)), "funding": store.get(ledger.external_funding_key("kg-r72")),
        "month": store.get(ledger.month_key("kg-r72", now.strftime("%Y-%m")))}
    validate_counter_baseline(counters, now.isoformat())
    generation = rate.quote(TokenUsage(input_tokens=routes.generation_context_ceiling, output_tokens=800))
    rewrite = rate.quote(TokenUsage(input_tokens=routes.generation_context_ceiling, output_tokens=256))
    available = min(c["cap_micro_usd"] - c["settled_micro_usd"] - c["reserved_micro_usd"] for c in counters.values())
    month = counters["month"]
    available = min(available, month["gross_stop_micro_usd"] - month["other_cost_estimate_micro_usd"]
        - month["settled_micro_usd"] - month["reserved_micro_usd"])
    receipt = {"status": "READY" if available >= generation else "INSUFFICIENT_BUDGET",
        "checked_at": now.isoformat(), "provider": provider, "counters": counters,
        "available_micro_usd": available, "generation_reservation_micro_usd": generation,
        "shared_rewrite_reservation_micro_usd": rewrite,
        "en_compare_two_reservation_micro_usd": 2 * generation,
        "vi_single_reservation_micro_usd": generation + rewrite,
        "quota_remaining": QueryRepository(store).quota_remaining(), "model_calls": 0,
        "caps_reset_or_increased": False, "cloud_deployed": False}
    return gateway, receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true", help="Read existing completed results; AWS reads, no model or writes")
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args(argv)
    directory = args.directory
    if args.preflight:
        _, receipt = preflight()
        output = directory / ("preflight-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json")
        write_once(output, receipt)
        print(json.dumps({"output": str(output), **{k: v for k, v in receipt.items() if k not in {"provider", "counters"}}}))
        return 0 if receipt["status"] == "READY" else 2
    if not args.replay and (directory / "execution-started.json").exists():
        raise ValueError("OWNED_EXECUTION_ALREADY_STARTED_NO_AUTOMATIC_RETRY")
    if args.replay:
        original = managed_gateway(prices_path=PRICES)
        store = DynamoRunStore(original.ledger.store.client, original.ledger.store.table_name)
        gateway = ExternalModelGateway(Ledger(store), original.provider, original.routes, original.prices)
        ready = None
    else:
        gateway, ready = preflight()
    if not args.replay and (ready["status"] != "READY" or ready["quota_remaining"] < 1):
        raise ValueError("OWNED_CANARY_ADMISSION_NOT_READY")
    pipeline_path = ROOT / "results/corpus-expansion-v2/pipeline.json"
    paths = json.loads(pipeline_path.read_bytes())
    collection = load_collection(Path(paths["collection_path"]), tenant_id=AUTH.tenant_id,
        embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    controller = RunBudgetController(gateway, budget_run_id=RUN)
    repo = OwnedRunRepository(gateway.ledger.store, answer_budget=controller)
    def authorize(auth):
        if auth != AUTH:
            raise ApiError(403, "ACCESS_DENIED")
    artifacts = RunArtifactRepository(LocalRunBlobs(ROOT / "data/cache/owned-answer-canary/artifacts", read_only=args.replay))
    calls = []
    original_request = gateway.provider.request
    def limited_request(provider, method, path, body=None):
        if args.replay or calls or (provider, method, path) != ("DEEPSEEK", "POST", "/chat/completions"):
            raise ValueError("OWNED_CANARY_MODEL_DISPATCH_BLOCKED")
        calls.append(sha256_bytes(body))
        return original_request(provider, method, path, body)
    gateway.provider.request = limited_request
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        executor = CollectionOwnedAnswerExecutor(collection, lexical, controller=controller, repository=repo,
            authorize=authorize, recipe=GenerationRecipeV9(model_metadata_sha256=gateway.routes.account_model_metadata_sha256),
            routes=gateway.routes, run_id=RUN, encoder=encoder,
            embedding_cache=EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE"),
            cache=AnswerCache(ROOT / "data/cache/collection-answers", namespace="LIVE"))
        service = OwnedRunService(repo, artifacts, lambda: executor.release, authorize=authorize, release_check=executor.check_release)
        payload = {"original_question": QUESTION, "language": "en", "method_ids": ["hybrid"], "run_kind": "answer"}
        plan = {"version": "owned-answer-canary-v1", "idempotency_key": KEY, "request": payload,
            "collection_manifest_sha256": collection.manifest_sha256, "pipeline_sha256": sha256_bytes(pipeline_path.read_bytes()),
            "release": executor.release.model_dump(mode="json"), "price_sheet_sha256": sha256_bytes(PRICES.read_bytes()),
            "implementation_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in IMPLEMENTATIONS},
            "maximum_new_model_calls": 1, "cloud_deployed": False, "authenticated_web": False}
        plan_path = directory / "plan.json"
        if plan_path.exists():
            validate_saved_plan(json.loads(plan_path.read_bytes()), plan, directory, replay=args.replay)
        elif args.replay:
            raise ValueError("OWNED_CANARY_PLAN_MISSING")
        else:
            write_once(plan_path, plan)
        if args.replay:
            old = json.loads((directory / "summary.json").read_bytes())
            run_id = old["run"]["run_id"]
            detail = service.detail(AUTH, run_id)
            results = [service.public_result(AUTH, run_id, "hybrid")]
            if not same_recorded_result(detail, results, old):
                raise ValueError("OWNED_CANARY_REPLAY_CHANGED")
            receipt = {"status": "PASS_MANAGED_READ_REPLAY", "run_id": run_id, "new_model_calls": len(calls),
                "new_model_cost_micro_usd": 0, "aws_reads": True, "writes": False, "cloud_deployed": False,
                "execution_plan_sha256": sha256_bytes(plan_path.read_bytes()),
                "replay_implementation_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in IMPLEMENTATIONS}}
            output = directory / ("replay-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + ".json")
        else:
            archive = directory / "execution-runner.py"
            if not archive.exists():
                with archive.open("xb") as saved:
                    saved.write(Path(__file__).read_bytes())
            write_once(directory / "execution-started.json", {"plan_sha256": sha256_bytes(plan_path.read_bytes()), "preflight": ready})
            row = service.submit(AUTH, payload, KEY)
            write_once(directory / "admission.json", row)
            queue = SQLiteRunQueue(ROOT / "data/cache/owned-answer-canary/queue.sqlite3")
            published = RunOutboxDispatcher(repo, queue.send).publish(AUTH, row["run_id"])
            worker = OwnedRunWorker(repo, artifacts, executor, authorize=authorize, release_check=executor.check_release)
            deliveries = []
            for identifier, message in queue.receive():
                outcome = worker.process(message)
                deliveries.append(outcome)
                if outcome == "ACK":
                    queue.ack(identifier)
            detail = service.detail(AUTH, row["run_id"])
            results = [service.public_result(AUTH, row["run_id"], "hybrid")] if detail["status"] == "COMPLETED" else []
            receipt = {"status": detail["status"], "kind": "LOCAL_WORKER_MANAGED_DYNAMO_OPERATOR_SYNTHETIC_OWNER_NOT_WEB",
                "run": detail, "results": results, "published_messages": published, "deliveries": deliveries,
                "new_model_calls": len(calls), "dispatch_body_sha256": calls,
                "new_model_cost_micro_usd": detail["budget"]["settled_micro_usd"],
                "cost_complete": detail["budget"]["cost_complete"], "cloud_deployed": False,
                "quality_metrics": None, "semantic_review": "NOT_REVIEWED", "authenticated_web": False}
            output = directory / "summary.json"
        write_once(output, receipt)
    print(json.dumps({"status": receipt["status"], "output": str(output), "new_model_calls": len(calls), "cloud_deployed": False}))
    return 0 if receipt["status"] in {"COMPLETED", "PASS_MANAGED_READ_REPLAY"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
