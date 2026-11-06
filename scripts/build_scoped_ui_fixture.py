"""Actual319 retrieval/source/map + synthetic provider for offline UI tests.

Outputs are fixture receipts, never native answer-quality measurements. No
credentials, AWS clients or external provider are used. Ranking sees no labels.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from app.contracts import canonical_json
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.generation import GenerationRecipeV9
from app.ledger import Ledger
from app.local_bge import BgeEncoder
from app.query_auth import AuthContext
from app.rag_runner import _validate_encoder
from app.reranking import OnnxCrossEncoder
from app.run_artifacts import LocalRunBlobs
from app.run_storage import SQLiteRunStore
from app.scoped_cloud_assets import ScopedGraphBundle, open_scoped_graph
from app.scoped_owned_runs import ScopedRunRepository
from app.scoped_query_v6 import ScopedPlanV6
from app.scoped_run_budget import ScopedRunBudgetController
from app.scoped_run_cloud import capabilities
from app.scoped_run_executor import OwnedScopedTask, ScopedOwnedAnswerExecutor
from app.scoped_run_service import ScopedArtifactRepository, ScopedRunService, ScopedRunWorker
from app.workbench_assets import WorkbenchAssetBundle, open_workbench_assets

ROOT = Path(__file__).resolve().parents[1]


def build(out):
    out = Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT/"results"):
        raise ValueError("SCOPED_UI_FIXTURE_NEW_RESULTS_DIRECTORY_REQUIRED")
    out.mkdir(parents=True)
    candidate = ROOT/"results/scoped-cloud319-2026-11-05-v2"
    config = json.loads((candidate/"cloud-config.json").read_bytes())
    corpus_root, graph_root = (candidate/"assets"/config[k]["version"] for k in ("assets", "graph_assets"))
    b = WorkbenchAssetBundle.model_validate_json((corpus_root/"bundle.json").read_bytes())
    g = ScopedGraphBundle.model_validate_json((graph_root/"bundle.json").read_bytes())
    inputs = json.loads((candidate/"probe-inputs/preparations.json").read_bytes())
    task = next(case for case in inputs if case["task"]["language"] == "en" and case["task"]["intent"] == "kms-cross-account")
    vi = next(case for case in inputs if case["task"]["language"] == "vi" and case["task"]["intent"] == "kms-cross-account")
    ledger = Ledger(SQLiteRunStore(out/"fixture-state.sqlite3"), clock=lambda: datetime(2026, 11, 5, 12, tzinfo=UTC))
    ledger.initialize_month("kg-r72", "2026-11")
    ledger.initialize_external_funding("kg-r72", cap_micro_usd=8000000)
    ledger.initialize_run("kg-r72", "demo", "scoped-ui-fixture", cap_micro_usd=2000000)
    prices = ExternalPriceSheet.model_validate_json((ROOT/"configs/prices/external-direct-2026-11-05.json").read_bytes())
    prices = prices.model_copy(update={"purpose": "FIXTURE"})
    routes = ExternalRoutes.model_validate_json((ROOT/"configs/models/external-direct-v1.json").read_bytes())
    state = {"held": False, "calls": 0}

    class Provider:
        live = False

        def request(self, provider, method, path, raw):
            state["calls"] += 1
            request = json.loads(raw)
            system, prompt = request["messages"]
            prompt = json.loads(prompt["content"])
            if "untrusted_sources" in prompt:
                options = [(source, span) for source in prompt["untrusted_sources"] for span in source["spans"]
                    if len(span["text"]) <= 500]
                if state["held"]:
                    options = [(source, span) for source, span in options if "invok" not in span["text"].lower()]
                source, span = options[0]
                content = {"answer_status": "PARTIAL", "reason": "INSUFFICIENT_EVIDENCE", "limitations": [
                    {"code": "MISSING_SUPPLIED_EVIDENCE", "question_part": prompt["untrusted_question"]}],
                    "claims": [{"text": "Lambda can invoke Lambda." if state["held"] else span["text"],
                        "evidence": [{"source_id": source["source_id"], "span_id": span["span_id"]}]}]}
            elif "facet_1" in system["content"]:
                content = json.loads(task["receipt"]["raw_output"])
            else:
                content = {"search_query": task["task"]["search"]}
            return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(content)}}]}

    gateway = ExternalModelGateway(ledger, Provider(), routes, prices)
    controller = ScopedRunBudgetController(gateway, budget_run_id="scoped-ui-fixture")
    repo = ScopedRunRepository(ledger.store, clock=ledger.clock, answer_budget=controller)
    artifacts = ScopedArtifactRepository(LocalRunBlobs(out/"private-fixture-results"))
    auth = AuthContext("demo", "11111111-1111-1111-1111-111111111111")
    cases = []
    with open_workbench_assets(corpus_root, b, tenant_id="demo") as assets:
        graph = open_scoped_graph(graph_root, g, assets.collection)
        encoder = BgeEncoder(assets.root/"models/bge", assets.root/"models/bge/manifest.json", threads=2)
        runtime = SimpleNamespace(recipe=GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256),
            encoder=encoder, query_recipe=_validate_encoder(assets.collection, encoder),
            cache=EmbeddingCache(out/"vectors", namespace="LIVE"),
            reranker=OnnxCrossEncoder(assets.root/"models/reranker", assets.root/"models/reranker/manifest.json", threads=2))
        class ReplayPlanExecutor(ScopedOwnedAnswerExecutor):
            def prepare(self, auth, parent):
                if not any(m.startswith("hybrid_multi_query") for m in parent["request"]["method_ids"]):
                    return super().prepare(auth, parent)
                self.parent_guard(auth, parent)
                source = next(case for case in (task, vi) if case["task"]["question"] == parent["request"]["original_question"])
                plan = ScopedPlanV6.model_validate_json(canonical_json(source["plan"]))
                # Preserve the original native receipt identity. Do not relabel
                # a fixture plan as native to bypass DEMO-collection guards.
                return OwnedScopedTask(source["task"]["search"], source["task"]["language"], plan, ())
        executor = ReplayPlanExecutor(assets.collection, assets.lexical, graph, runtime,
            controller=controller, repository=repo, authorize=lambda _: None)
        service = ScopedRunService(repo, artifacts, lambda: executor.release, authorize=lambda _: None, release_check=executor.check_release)
        failures = []
        class RecordingExecutor:
            safe_failure = executor.safe_failure

            def prepare(self, *args):
                try:
                    return executor.prepare(*args)
                except Exception as exc:
                    failures.append({"stage": "prepare", "class": type(exc).__name__, "message": str(exc)[:1000]})
                    raise

            def __call__(self, *args):
                try:
                    return executor(*args)
                except Exception as exc:
                    failures.append({"class": type(exc).__name__, "message": str(exc)[:1000]})
                    raise
        worker = ScopedRunWorker(repo, artifacts, RecordingExecutor(), authorize=lambda _: None, release_check=executor.check_release)
        for identifier, selected, question, language, held in (
            ("baseline", ("bm25", "vector", "hybrid"), task["task"]["question"], "en", False),
            ("graph", ("hybrid_reranked", "graph_1hop", "graph_2hop"), task["task"]["question"], "en", False),
            ("queries", ("hybrid_multi_query", "hybrid_multi_query_balanced"), task["task"]["question"], "en", False),
            ("vietnamese", ("hybrid", "hybrid_multi_query", "hybrid_multi_query_balanced"), vi["task"]["question"], "vi", False),
            ("held", ("hybrid",), task["task"]["question"], "en", True)):
            state["held"] = held
            row = service.submit(auth, {"original_question": question, "method_ids": list(selected), "language": language,
                "run_kind": "answer"}, str(uuid4()))
            for outbox in repo.outboxes(auth, row["run_id"]):
                if worker.process(outbox["message"]) != "ACK":
                    raise ValueError("SCOPED_UI_FIXTURE_WORKER_RETRY")
            detail = service.detail(auth, row["run_id"])
            if detail["status"] != "COMPLETED":
                (out/"failure.json").write_bytes(canonical_json({"case": identifier, "detail": detail, "failures": failures}))
                raise ValueError("SCOPED_UI_FIXTURE_EXECUTION_FAILED")
            results = {m: service.public_result(auth, row["run_id"], m) for m in selected}
            sources = {m: {s["chunk_id"]: service.source(auth, row["run_id"], m, s["chunk_id"]) for s in result["context_sources"]}
                for m, result in results.items()}
            cases.append({"id": identifier, "run": detail, "results": results, "sources": sources, "trace": service.trace(auth, row["run_id"])})
        payload = {"version": "scoped319-ui-fixture-v1", "notice": "ACTUAL319 RETRIEVAL + SYNTHETIC PROVIDER/ACCOUNT/QUEUE. NOT NATIVE ANSWER QUALITY.",
            "planner": "Original frozen NATIVE_V6 receipts replayed for Q/QB; no new owned planner/rewrite call for those methods.",
            "capabilities": capabilities({"release": executor.release.model_dump(mode="json")}), "cases": cases,
            "provider_calls": 0, "synthetic_provider_calls": state["calls"], "aws_calls": 0}
    (out/"fixture.json").write_bytes(canonical_json(payload))
    print(json.dumps({"status": "OFFLINE_UI_FIXTURE_BUILT", "cases": len(cases), "documents": 319, "chunks": 3024,
        "actual_graph_facts": len(graph.facts), "synthetic_provider_calls": state["calls"], "provider_calls": 0, "aws_calls": 0}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
