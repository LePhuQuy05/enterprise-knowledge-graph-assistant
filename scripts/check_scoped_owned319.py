"""No-model corpus319 preflight for the new owned release and all eight pools.

Native V6 task facets are replayed from frozen receipts; this is not a new
planner/generation experiment or a quality benchmark.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalModelGateway, ExternalRoutes
from app.ledger import Ledger
from app.profiled_generation import ProfiledGenerationRecipe
from app.profiled_source_context import SourceContextProfile
from app.run_storage import MemoryRunStore
from app.scoped_method_flow import METHODS, prepare_scoped_flow
from app.scoped_owned_runs import ScopedRunRepository
from app.scoped_query_v6 import ScopedPlanV6, validate_source_facets
from app.scoped_run_budget import ScopedRunBudgetController
from app.scoped_run_executor import ScopedOwnedAnswerExecutor
from scripts import run_scoped_flow_canary as native
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.run_collection_benchmark import OfflineRuntime
from tests.test_external_pilot import configured

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/scoped-owned319-preflight-2026-11-05-v1"


def stable_trace(value):
    if isinstance(value, dict):
        return {key: stable_trace(item) for key, item in value.items() if key not in {"duration_ms", "elapsed_ms"}}
    if isinstance(value, (list, tuple)):
        return [stable_trace(item) for item in value]
    return value


def run(replay):
    scope = native.checked_scope()
    collection, paths, graph = load_migrated()
    if collection.manifest_sha256 != scope["collection_sha256"] or graph.release_sha256 != scope["graph_sha256"]:
        raise ValueError("OWNED319_PREFLIGHT_SOURCE_RELEASE_CHANGED")
    routes = ExternalRoutes.model_validate_json((ROOT/"configs/models/external-direct-v1.json").read_bytes())
    old = configured()
    ledger = Ledger(MemoryRunStore(), clock=old.ledger.clock)
    gateway = ExternalModelGateway(ledger, old.provider, routes, old.prices)
    controller = ScopedRunBudgetController(gateway, budget_run_id="owned319-offline-preflight")
    repo = ScopedRunRepository(ledger.store, clock=ledger.clock, answer_budget=controller)
    code = ("app/scoped_run_contracts.py", "app/scoped_owned_runs.py", "app/scoped_run_executor.py",
        "app/scoped_run_service.py", "app/scoped_run_budget.py", "scripts/check_scoped_owned319.py")
    manifest = {"version": "owned319-pools-and-release-preflight-v1", "source_manifest_sha256": collection.manifest_sha256,
        "graph_release_sha256": graph.release_sha256, "native_scope_version": scope["version"],
        "code_sha256": {name: sha256_bytes((ROOT/name).read_bytes()) for name in code},
        "generation_calls": 0, "new_native_planner_calls": 0, "aws_calls": 0, "serving_changed": False}
    immutable(OUT/"scope.json", manifest)
    rows = []
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
        executor = ScopedOwnedAnswerExecutor(collection, lexical, graph, runtime,
            controller=controller, repository=repo, authorize=lambda auth: None)
        immutable(OUT/"release.json", executor.release.model_dump(mode="json"))
        for task in scope["inputs"]:
            name = task["intent"]+"-"+task["language"]
            prep_path = native.OUT/"preparations"/(name+".json")
            saved = json.loads(prep_path.read_bytes())
            plan = ScopedPlanV6.model_validate_json(canonical_json(saved["plan"]))
            receipt = saved["receipt"]
            if plan.origin != "NATIVE_V6" or plan.receipt_sha256 != sha256_bytes(canonical_json(receipt)) or (
                    plan.facets != validate_source_facets(receipt["raw_output"], task["question"], task["search"])):
                raise ValueError("OWNED319_PREFLIGHT_FROZEN_NATIVE_PLAN")
            for method in METHODS:
                flow = prepare_scoped_flow(collection, lexical, graph, runtime, question=task["question"],
                    search=task["search"], language=task["language"], method=method, run_id="owned319-offline-preflight",
                    owner_id="offline-source-check", recipe=ProfiledGenerationRecipe(
                        model_metadata_sha256=routes.account_model_metadata_sha256), profile=SourceContextProfile(max_chunks=12), plan=plan)
                row = {"task": task, "method": method, "mechanism_status": flow.status,
                    "context_chunk_ids": flow.context.chunk_ids, "context_sha256": flow.context.fingerprint,
                    "pool_sha256": flow.trace["pool_fingerprint"], "used_graph": flow.binding.used_graph,
                    "context_proxy_tokens": flow.context.proxy_tokens, "trace": stable_trace(flow.trace),
                    "preparation_origin": "REPLAYED_FROZEN_NATIVE_V6_NOT_NEW_OWNED_STAGE",
                    "preparation_sha256": sha256_bytes(prep_path.read_bytes()), "answer": None}
                path = OUT/"cells"/(name+"-"+method+".json")
                if replay and canonical_json(json.loads(path.read_bytes())) != canonical_json(row):
                    raise ValueError("OWNED319_PREFLIGHT_REPLAY_CHANGED")
                if not replay:
                    immutable(path, row)
                rows.append(row)
            print(json.dumps({"task": name, "methods": len(METHODS), "new_model_calls": 0}), flush=True)
    report = {"status": "OWNED319_OFFLINE_RELEASE_AND_POOLS_CHECKED", "documents": len(executor.documents),
        "chunks": len(executor.chunks), "graph_runtime_facts": len(graph.facts),
        "graph_source_documents": executor.release.graph_source_document_count, "method_count": len(executor.release.configs),
        "formulations": len(scope["inputs"]), "exposed_intents": 2, "method_cells": len(rows),
        "ready_cells": sum(row["mechanism_status"] == "READY" for row in rows),
        "not_ready_cells": sum(row["mechanism_status"].startswith("NOT_READY") for row in rows),
        "hidden_fallbacks": sum(row["trace"]["hidden_fallbacks"] for row in rows),
        "provider_calls": gateway.provider.calls, "new_model_calls": 0, "aws_calls": 0,
        "generated_answers": 0, "quality_metrics": None, "serving_changed": False}
    if gateway.provider.calls:
        raise ValueError("OWNED319_PREFLIGHT_UNEXPECTED_PROVIDER")
    immutable(OUT/("replay.json" if replay else "report.json"), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(name, side_effect=AssertionError("OWNED319_PREFLIGHT_NETWORK_FORBIDDEN")))
        run(args.replay)
