"""Finish the pinned 1600-cell EN/VI matrix using saved shared queries, offline.

Reuses 600 immutable core-English receipts; executes only the remaining 1000.
No gateway or AWS client is constructed. Replay rescores receipts without ONNX.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from app.benchmark_queries import (
    AugmentedRetrievalReceipt,
    SharedQueryBundle,
    validate_shared_bundle,
)
from app.collection_answering import prepare_collection_answer
from app.collection_benchmark import (
    ALL_METHODS,
    CORE_METHODS,
    RetrievalReceipt,
    RetrievalTask,
    score_receipts,
    summarize_receipts,
)
from app.collection_graph_runtime import prepare_collection_graph_answer
from app.collection_lexical import load_global_lexical_index
from app.context_selection import ContextSelectionRecipe
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.model_gateway import safe_failure_code
from scripts.prepare_benchmark_queries import failure_reviews, tasks_from
from scripts.run_collection_benchmark import (
    ROOT,
    OfflineRuntime,
    _code_hashes,
    _write_new,
    load_inputs,
)

QUERY_DIR = ROOT / "results/benchmark316/shared-queries-v1"
BASELINE = ROOT / "results/benchmark316/core-six-en-v1"
Q_METHODS = ("hybrid_multi_query", "hybrid_multi_query_balanced")


def hashes():
    return _code_hashes() | {name: sha256_bytes((ROOT / name).read_bytes()) for name in (
        "app/benchmark_queries.py", "app/query_planning.py", "app/rewrite_cache.py",
        "app/multi_query.py", "app/context_selection.py", "scripts/run_augmented_collection_benchmark.py")}


def load_shared_queries(dataset, protocol_raw):
    raw = (QUERY_DIR / "bundle.json").read_bytes()
    receipt = json.loads((QUERY_DIR / "completion.json").read_bytes())
    if receipt["status"] != "COMPLETE" or receipt["bundle_sha256"] != sha256_bytes(raw):
        raise ValueError("BENCHMARK_SHARED_COMPLETION_OR_HASH_MISMATCH")
    bundle = SharedQueryBundle.model_validate_json(raw)
    metadata = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    validate_shared_bundle(bundle, tasks_from(dataset), protocol_sha256=sha256_bytes(protocol_raw),
                           metadata_sha256=metadata["account_model_metadata_sha256"], failure_reviews=failure_reviews(QUERY_DIR))
    return bundle, sha256_bytes(raw), receipt


class AugmentedRuntime(OfflineRuntime):
    def __init__(self, collection, lexical, bundle, bundle_sha):
        super().__init__(collection, lexical, ALL_METHODS)
        self.rows = {(r.scenario_id, r.language): r for r in bundle.rows}
        self.bundle_sha, self.selector = bundle_sha, ContextSelectionRecipe()

    def configuration(self):
        return super().configuration() | {"shared_bundle_sha256": self.bundle_sha,
            "context_selection_recipe": self.selector.model_dump(mode="json"),
            "context_selection_recipe_sha256": self.selector.fingerprint,
            "query_policy": "Original EN; one saved shared VI rewrite; one saved Q/QB plan per input"}

    def vector_for(self, text, task):
        if text not in self.vectors:
            started = perf_counter()
            embedded = self.cache.resolve_local(self.collection.manifest.tenant_id,
                text=query_embedding_input(text), recipe=self.query_recipe, encoder=self.encoder)
            self.vectors[text] = np.asarray(embedded.record.vector, dtype=np.float32)
            self.embedding_events.append({"scenario_id": task.scenario_id, "language": task.language,
                "query_sha256": sha256_bytes(text.encode()), "duration_ms": round((perf_counter() - started) * 1000, 3)})
        return self.vectors[text]

    def __call__(self, task, method):
        row = self.rows[(task.scenario_id, task.language)]
        vector = None if method == "bm25" else self.vector_for(row.search_query, task)
        planned = method in Q_METHODS
        sub_vectors = tuple(self.vector_for(s, task) for s in row.subqueries) if planned else ()
        started = perf_counter()
        if method.startswith("graph_"):
            _, prepared, _ = prepare_collection_graph_answer(self.collection, self.lexical,
                self.graph, task.question, vector, search_query=row.search_query,
                tenant_id=self.collection.manifest.tenant_id, owner_id="benchmark-local", run_id="benchmark316",
                recipe=self.recipe, hops=int(method[6]), allow_experimental=True)
        else:
            _, prepared = prepare_collection_answer(self.collection, self.lexical, task.question, vector,
                tenant_id=self.collection.manifest.tenant_id, owner_id="benchmark-local", run_id="benchmark316",
                recipe=self.recipe, mode=method, search_query=row.search_query, reranker=self.reranker,
                subqueries=row.subqueries if planned else (), subquery_vectors=sub_vectors,
                context_selection_recipe=self.selector if method == "hybrid_multi_query_balanced" else None)
        return {"effective_method": prepared.binding.mode_used, "ranked_chunk_ids": prepared.retrieved_chunk_ids,
            "context_chunk_ids": prepared.packed.chunk_ids, "context_assertion_ids": prepared.packed.assertion_ids,
            "used_graph": prepared.binding.used_graph, "warnings": prepared.binding.warnings,
            "duration_ms": round((perf_counter() - started) * 1000, 3),
            "context_selection_used": method == "hybrid_multi_query_balanced"}


def normalized_baseline(dataset, protocol_raw, bundle_sha):
    manifest = json.loads((BASELINE / "manifest.json").read_bytes())
    raw = (BASELINE / "receipts.jsonl").read_bytes()
    if (manifest["protocol_sha256"] != sha256_bytes(protocol_raw)
            or manifest["receipts_sha256"] != sha256_bytes(raw) or manifest["languages"] != ["en"]):
        raise ValueError("BENCHMARK_BASELINE_CHANGED")
    for name, expected in manifest["artifacts"].items():
        if Path(name).name != name or sha256_bytes((BASELINE / name).read_bytes()) != expected:
            raise ValueError("BENCHMARK_BASELINE_ARTIFACT_CHANGED")
    for name, expected in manifest["code_sha256"].items():
        if sha256_bytes((ROOT / name).read_bytes()) != expected:
            raise ValueError("BENCHMARK_CORE_CODE_CHANGED_REQUIRES_NEW_COMPARISON")
    rows = [RetrievalReceipt.model_validate_json(line) for line in raw.splitlines()]
    expected = {(s.scenario_id, m) for s in dataset.scenarios for m in CORE_METHODS}
    if len(rows) != 600 or {(r.scenario_id, r.requested_method) for r in rows} != expected or any(r.language != "en" for r in rows):
        raise ValueError("BENCHMARK_BASELINE_CASE_MATRIX_MISMATCH")
    return [AugmentedRetrievalReceipt.model_validate_json(canonical_json(r.model_dump(mode="json") | {
        "search_query_sha256": r.question_sha256, "shared_bundle_sha256": bundle_sha,
        "planner_operation_id": None, "rewrite_operation_id": None,
        "execution_source": "REUSED_PINNED_CORE_EN"})) for r in rows]


def execute_remaining(dataset, runtime, *, emit):
    for scenario in dataset.scenarios:
        for language, methods in (("en", Q_METHODS), ("vi", ALL_METHODS)):
            task = RetrievalTask(scenario.scenario_id, language, getattr(scenario, "question_" + language))
            row = runtime.rows[(task.scenario_id, language)]
            for method in methods:
                planned = method in Q_METHODS
                common = dict(schema_version=1, scenario_id=task.scenario_id, language=language,
                    query_track="EN_ORIGINAL" if language == "en" else "VI_SHARED_REWRITE",
                    question_sha256=sha256_bytes(task.question.encode()), requested_method=method,
                    search_query_sha256=sha256_bytes(row.search_query.encode()), shared_bundle_sha256=runtime.bundle_sha,
                    planner_operation_id=row.planner_receipt.operation_id if planned else None,
                    rewrite_operation_id=row.rewrite_receipt.operation_id if language == "vi" else None,
                    subquery_count=len(row.subqueries) if planned else 0)
                try:
                    if planned and row.preparation_status == "PLAN_FAILED":
                        receipt = AugmentedRetrievalReceipt(**common, status="FAILED", effective_method=None,
                            error_category="SHARED_PLANNER_VALIDATION_FAILED", preparation_error=row.failure_code)
                        emit(receipt)
                        continue
                    receipt = AugmentedRetrievalReceipt(**common, status="COMPLETED", **runtime(task, method))
                except Exception as exc:
                    receipt = AugmentedRetrievalReceipt(**common, status="FAILED", effective_method=None,
                                                        error_category=safe_failure_code(exc))
                emit(receipt)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/benchmark316/all-eight-paired-v1")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    protocol_raw, dataset, collection, index, pipeline = load_inputs()
    bundle, bundle_sha, shared_receipt = load_shared_queries(dataset, protocol_raw)
    output = args.output_dir.resolve()
    if args.replay:
        manifest = json.loads((output / "manifest.json").read_bytes())
        if manifest["protocol_sha256"] != sha256_bytes(protocol_raw) or manifest["shared_bundle_sha256"] != bundle_sha:
            raise ValueError("BENCHMARK_AUGMENTED_REPLAY_INPUT_CHANGED")
        for name, expected in manifest["artifacts"].items():
            if Path(name).name != name or sha256_bytes((output / name).read_bytes()) != expected:
                raise ValueError("BENCHMARK_AUGMENTED_REPLAY_ARTIFACT_CHANGED")
        receipts = [AugmentedRetrievalReceipt.model_validate_json(line) for line in (output / "receipts.jsonl").read_bytes().splitlines()]
        scored = score_receipts(dataset, receipts, index)
        actual = summarize_receipts(dataset, scored, languages=("en", "vi"), methods=ALL_METHODS)
        match = canonical_json(actual) == canonical_json(json.loads((output / "summary.json").read_bytes())["methods"])
        print(json.dumps({"status": "REPLAY_MATCH" if match else "REPLAY_MISMATCH", "receipts": len(receipts),
            "onnx_calls": 0, "model_calls": 0, "aws_calls": 0}))
        return 0 if match else 2
    baseline = normalized_baseline(dataset, protocol_raw, bundle_sha)
    output.mkdir(parents=True, exist_ok=False)
    code_hashes = hashes()
    _write_new(output / "started.json", {"status": "STARTED", "protocol_sha256": sha256_bytes(protocol_raw),
        "shared_bundle_sha256": bundle_sha, "code_sha256": code_hashes,
        "expected_new_receipts": 1000, "reused_core_en_receipts": 600, "model_calls": 0, "aws_calls": 0})
    receipts = []
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = AugmentedRuntime(collection, lexical, bundle, bundle_sha)
        configuration = runtime.configuration() | {"lexical_index": lexical.manifest.index_version}
        original_configuration = json.loads((BASELINE / "configuration.json").read_bytes())
        if any(configuration.get(key) != value for key, value in original_configuration.items() if key != "query_policy"):
            raise ValueError("BENCHMARK_REUSED_CORE_CONFIGURATION_MISMATCH")
        _write_new(output / "configuration.json", configuration)
        with (output / "receipts.jsonl").open("xb") as handle:
            def emit(receipt):
                handle.write(canonical_json(receipt.model_dump(mode="json")) + b"\n")
                handle.flush()
                receipts.append(receipt)
                if len(receipts) > 600 and len(receipts) % 10 == 0:
                    print(json.dumps({"receipts": len(receipts), "failed": sum(r.status == "FAILED" for r in receipts)}), flush=True)
            for r in baseline:
                emit(r)
            execute_remaining(dataset, runtime, emit=emit)
    if hashes() != code_hashes:
        raise ValueError("BENCHMARK_AUGMENTED_CODE_CHANGED_KEEP_RECEIPTS")
    if len(receipts) != 1600:
        raise ValueError("BENCHMARK_AUGMENTED_INCOMPLETE_MATRIX")
    scored = score_receipts(dataset, receipts, index)
    _write_new(output / "scored.json", scored)
    _write_new(output / "shared-embedding-events.json", runtime.embedding_events)
    _write_new(output / "summary.json", {"status": "PROVISIONAL_AI_DRAFT_PAIRED_RETRIEVAL_ONLY",
        "methods": summarize_receipts(dataset, scored, languages=("en", "vi"), methods=ALL_METHODS),
        "independent_holdout": False, "human_approved": 0, "answer_quality_measured": False,
        "shared_preparation": shared_receipt, "retrieval_model_calls": 0, "retrieval_aws_calls": 0,
        "baseline_receipts_reused": 600, "new_retrieval_receipts": 1000})
    artifacts = {p.name: sha256_bytes(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    _write_new(output / "manifest.json", {"schema_version": 2, "completed_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha256_bytes(protocol_raw), "shared_bundle_sha256": bundle_sha,
        "baseline_manifest_sha256": sha256_bytes((BASELINE / "manifest.json").read_bytes()),
        "source_manifest_sha256": collection.manifest_sha256, "code_sha256": code_hashes,
        "scenario_ids": [s.scenario_id for s in dataset.scenarios], "languages": ["en", "vi"],
        "methods": list(ALL_METHODS), "receipts": 1600, "reused_receipts": 600, "new_receipts": 1000,
        "artifacts": artifacts, "receipts_sha256": artifacts["receipts.jsonl"],
        "local_embedding_onnx_calls": runtime.encoder.inference_calls,
        "local_reranker_onnx_calls": runtime.reranker.inference_calls,
        "model_calls": 0, "aws_calls": 0, "shared_preparation": shared_receipt})
    print(json.dumps({"status": "COMPLETED", "receipts": len(receipts), "reused": 600, "new": 1000,
        "failed": sum(r.status == "FAILED" for r in receipts), "model_calls": 0, "aws_calls": 0}))
    return 0 if all(r.status == "COMPLETED" for r in receipts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
