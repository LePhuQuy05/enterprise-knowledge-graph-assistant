"""Pinned question-only local benchmark; never imports a gateway or cloud client.

Default English/core-six only. Q/QB and product Vietnamese require a separately
funded shared planner/rewrite stage; they remain NOT_RUN in this receipt matrix.
Replay validates hashes and rescores saved rankings without loading ONNX.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from app.collection_benchmark import (
    ALL_METHODS,
    CORE_METHODS,
    BenchmarkProtocol,
    RetrievalReceipt,
    RetrievalTask,
    assignment_counts,
    evidence_at_k_feasibility,
    score_receipts,
    summarize_receipts,
    validate_protocol,
)
from app.collection_evaluation import (
    CollectionEvidenceIndex,
    validate_annotation_sources,
    validate_control_scope_review,
    validate_scenario_binding,
)
from app.collection_graph_runtime import load_collection_graph, prepare_collection_graph_answer
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import canonical_json, sha256_bytes
from app.generation import GenerationRecipeV9

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/evaluation"


def _write_new(path, value):
    with path.open("xb") as handle:
        handle.write(canonical_json(value) + b"\n")


def _code_hashes():
    paths = ("app/collection_benchmark.py", "app/collection_evaluation.py", "app/collection_answering.py",
        "app/collection_graph_runtime.py", "app/collection_graph.py", "app/collection_lexical.py",
        "app/collections.py", "app/graph_retrieval.py", "app/retrieval.py", "app/context.py", "app/reranking.py",
        "app/local_bge.py", "app/evaluation.py", "app/generation.py", "scripts/run_collection_benchmark.py", "uv.lock")
    return {name: sha256_bytes((ROOT / name).read_bytes()) for name in paths if (ROOT / name).exists()}


class OfflineRuntime:
    """Receives RetrievalTask only; it cannot inspect labels or expected behavior."""

    def __init__(self, collection, lexical, methods):
        from app.embeddings import EmbeddingCache
        from app.local_bge import BgeEncoder
        from app.rag_runner import _validate_encoder
        from app.reranking import OnnxCrossEncoder

        self.collection, self.lexical = collection, lexical
        self.encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5",
                                  ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
        self.query_recipe = _validate_encoder(collection, self.encoder)
        self.cache = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE")
        self.reranker = OnnxCrossEncoder(ROOT / "data/models/ms-marco-MiniLM-L2-v2",
            ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json") if "hybrid_reranked" in methods else None
        self.graph = None
        if any(m.startswith("graph_") for m in methods):
            graph_receipt = json.loads((ROOT / "results/collection-graph/curated-subset-v1.json").read_bytes())
            self.graph = load_collection_graph(Path(graph_receipt["graph_path"]), collection,
                trusted_sha256=graph_receipt["graph_file_sha256"], allow_experimental=True)
        # Retrieval/packing uses the same recipe identity, but no generation is dispatched.
        metadata = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        self.recipe = GenerationRecipeV9(model_metadata_sha256=metadata["account_model_metadata_sha256"])
        self.vectors, self.embedding_events = {}, []

    def configuration(self):
        return {"document_encoder": self.encoder.recipe("document").model_dump(mode="json"),
            "query_encoder": self.query_recipe.model_dump(mode="json"),
            "reranker": self.reranker.recipe.model_dump(mode="json") if self.reranker else None,
            "packing_generation_recipe": self.recipe.model_dump(mode="json"),
            "graph_file_sha256": self.graph.file_sha256 if self.graph else None,
            "graph_assertions": len(self.graph.overlay.assertions) if self.graph else 0,
            "graph_human_reviewed": False, "graph_full_relation_extraction": False,
            "shared_query_embedding": True, "automatic_retry": False, "backend": "LOCAL_WINDOWS_CPU",
            "k": 5, "search_depth": 20, "rrf_k": 60, "graph_time_budget_ms": 300,
            "query_policy": "Original EN; raw VI diagnostic only, no rewritten product-VI claims"}

    def __call__(self, task, method):
        from app.collection_answering import prepare_collection_answer
        from app.embeddings import query_embedding_input

        vector = None
        if method != "bm25":
            if task.question not in self.vectors:
                started = perf_counter()
                result = self.cache.resolve_local(self.collection.manifest.tenant_id,
                    text=query_embedding_input(task.question), recipe=self.query_recipe, encoder=self.encoder)
                self.vectors[task.question] = np.asarray(result.record.vector, dtype=np.float32)
                self.embedding_events.append({"scenario_id": task.scenario_id, "language": task.language,
                    "question_sha256": sha256_bytes(task.question.encode("utf-8")),
                    "duration_ms": round((perf_counter() - started) * 1000, 3)})
            vector = self.vectors[task.question]
        started = perf_counter()
        if method.startswith("graph_"):
            _, prepared, _ = prepare_collection_graph_answer(self.collection, self.lexical,
                self.graph, task.question, vector, tenant_id=self.collection.manifest.tenant_id,
                owner_id="benchmark-local", run_id="benchmark316", recipe=self.recipe,
                hops=int(method[6]), allow_experimental=True)
        else:
            _, prepared = prepare_collection_answer(self.collection, self.lexical, task.question, vector,
                tenant_id=self.collection.manifest.tenant_id, owner_id="benchmark-local", run_id="benchmark316",
                recipe=self.recipe, mode=method, reranker=self.reranker)
        return {"effective_method": prepared.binding.mode_used,
            "ranked_chunk_ids": prepared.retrieved_chunk_ids, "context_chunk_ids": prepared.packed.chunk_ids,
            "context_assertion_ids": prepared.packed.assertion_ids, "used_graph": prepared.binding.used_graph,
            "warnings": prepared.binding.warnings, "duration_ms": round((perf_counter() - started) * 1000, 3)}


def execute_tasks(dataset, runtime, *, languages, methods, emit):
    """Continue past failures, retain them; no retries and no evaluator fields passed."""
    from app.model_gateway import safe_failure_code

    for scenario in dataset.scenarios:
        for language in languages:
            task = RetrievalTask(scenario.scenario_id, language, getattr(scenario, "question_" + language))
            for method in methods:
                common = dict(schema_version=1, scenario_id=task.scenario_id, language=language,
                    query_track="EN_ORIGINAL" if language == "en" else "VI_RAW_DIAGNOSTIC",
                    question_sha256=sha256_bytes(task.question.encode("utf-8")), requested_method=method)
                try:
                    receipt = RetrievalReceipt(**common, status="COMPLETED", **runtime(task, method))
                except Exception as exc:
                    receipt = RetrievalReceipt(**common, status="FAILED", effective_method=None,
                                               error_category=safe_failure_code(exc))
                emit(receipt)


def load_inputs():
    protocol_bytes = (CONFIG / "collection316-benchmark-protocol-v1.json").read_bytes()
    protocol = BenchmarkProtocol.model_validate_json(protocol_bytes)
    dataset = validate_protocol(protocol, (CONFIG / "collection316-benchmark-frozen-v1.json").read_bytes(),
        (CONFIG / "collection316-exposure-inventory-v1.json").read_bytes(),
        authoring_bytes=(CONFIG / "collection316-annotations-v2.json").read_bytes())
    pipeline = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4,
        trusted_manifest_sha256=protocol.source_manifest_sha256)
    index = CollectionEvidenceIndex.from_collection(collection)
    validate_scenario_binding(dataset, (CONFIG / "collection316-scenarios-100-v2.json").read_bytes())
    validate_annotation_sources(dataset, index)
    validate_control_scope_review(dataset, index, (CONFIG / "collection316-corpus-gap-review-v1.json").read_bytes())
    return protocol_bytes, dataset, collection, index, pipeline


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=CORE_METHODS, default=CORE_METHODS)
    parser.add_argument("--languages", nargs="+", choices=("en", "vi"), default=("en",))
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args(argv)
    if len(set(args.methods)) != len(args.methods) or len(set(args.languages)) != len(args.languages):
        parser.error("Duplicate method/language")
    protocol_bytes, dataset, collection, index, pipeline = load_inputs()
    if args.case_ids:
        if len(set(args.case_ids)) != len(args.case_ids) or not set(args.case_ids) <= {s.scenario_id for s in dataset.scenarios}:
            parser.error("Invalid case IDs")
        dataset = dataset.model_copy(update={"scenarios": tuple(s for s in dataset.scenarios if s.scenario_id in args.case_ids)})
    output = args.output_dir.resolve()
    if args.replay:
        manifest = json.loads((output / "manifest.json").read_bytes())
        if manifest["protocol_sha256"] != sha256_bytes(protocol_bytes):
            raise ValueError("BENCHMARK_REPLAY_PROTOCOL_CHANGED")
        raw = (output / "receipts.jsonl").read_bytes()
        if sha256_bytes(raw) != manifest["receipts_sha256"]:
            raise ValueError("BENCHMARK_REPLAY_RECEIPTS_CHANGED")
        for name, expected in manifest["artifacts"].items():
            if Path(name).name != name or sha256_bytes((output / name).read_bytes()) != expected:
                raise ValueError("BENCHMARK_REPLAY_ARTIFACT_CHANGED")
        if args.case_ids:
            parser.error("Replay uses the manifest's case selection, not --case-ids")
        selected = set(manifest["scenario_ids"])
        dataset = dataset.model_copy(update={"scenarios": tuple(s for s in dataset.scenarios if s.scenario_id in selected)})
        receipts = [RetrievalReceipt.model_validate_json(line) for line in raw.splitlines()]
        summary = summarize_receipts(dataset, score_receipts(dataset, receipts, index),
            languages=manifest["languages"], methods=ALL_METHODS)
        match = canonical_json(summary) == canonical_json(json.loads((output / "summary.json").read_bytes())["methods"])
        print(json.dumps({"replay_match": match, "receipts": len(receipts), "model_calls": 0,
                          "onnx_calls": 0, "aws_calls": 0}))
        return 0 if match else 2
    output.mkdir(parents=True, exist_ok=False)
    code_hashes = _code_hashes()
    _write_new(output / "started.json", {"status": "STARTED", "protocol_sha256": sha256_bytes(protocol_bytes),
        "scenario_ids": [s.scenario_id for s in dataset.scenarios], "code_sha256": code_hashes,
        "model_calls": 0, "aws_calls": 0})
    feasibility = {s.scenario_id: evidence_at_k_feasibility(s, index) for s in dataset.scenarios}
    _write_new(output / "feasibility.json", feasibility)
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, args.methods)
        configuration = runtime.configuration() | {"lexical_index": lexical.manifest.index_version}
        _write_new(output / "configuration.json", configuration)
        receipts = []
        with (output / "receipts.jsonl").open("xb") as handle:
            def emit(receipt):
                handle.write(canonical_json(receipt.model_dump(mode="json")) + b"\n")
                handle.flush()
                receipts.append(receipt)
                if len(receipts) % len(args.methods) == 0:
                    print(json.dumps({"scenario": receipt.scenario_id, "receipts": len(receipts),
                        "failed": sum(r.status == "FAILED" for r in receipts)}), flush=True)
            execute_tasks(dataset, runtime, languages=args.languages, methods=args.methods, emit=emit)
    scored = score_receipts(dataset, receipts, index)
    if _code_hashes() != code_hashes:
        raise ValueError("BENCHMARK_CODE_CHANGED_DURING_RUN_KEEP_RECEIPTS")
    methods = summarize_receipts(dataset, scored, languages=args.languages, methods=ALL_METHODS)
    _write_new(output / "scored.json", scored)
    _write_new(output / "shared-embedding-events.json", runtime.embedding_events)
    _write_new(output / "summary.json", {"status": "PROVISIONAL_AI_DRAFT_RETRIEVAL_ONLY", "methods": methods,
        "assignment_counts": assignment_counts(dataset), "human_approved": 0,
        "independent_holdout": False, "answer_quality_measured": False,
        "limitations": ["All labels AI draft; source alternatives incomplete; all cases author-seen",
            "Evaluation split reserved, nonblind; source-integrity PASS is not semantic correctness",
            "Graph is 17-assertion experimental overlay, not full relation extraction",
            "Q/QB and product-VI not run; missing metrics null; controls not retrieval positives",
            "EN local retrieval timing excludes shared embedding; no AWS/answer latency claim"],
        "model_calls": 0, "aws_calls": 0})
    artifacts = {p.name: sha256_bytes(p.read_bytes()) for p in output.iterdir() if p.is_file()}
    _write_new(output / "manifest.json", {"schema_version": 1, "completed_at": datetime.now(UTC).isoformat(),
        "protocol_sha256": sha256_bytes(protocol_bytes), "dataset_sha256": BenchmarkProtocol.model_validate_json(protocol_bytes).frozen_dataset_sha256,
        "source_manifest_sha256": collection.manifest_sha256, "configuration_sha256": artifacts["configuration.json"],
        "receipts_sha256": artifacts["receipts.jsonl"], "artifacts": artifacts, "code_sha256": _code_hashes(),
        "scenario_ids": [s.scenario_id for s in dataset.scenarios], "methods": list(args.methods),
        "languages": list(args.languages), "local_embedding_onnx_calls": runtime.encoder.inference_calls,
        "local_reranker_onnx_calls": runtime.reranker.inference_calls if runtime.reranker else 0,
        "automatic_retry": False, "model_calls": 0, "aws_calls": 0})
    print(json.dumps({"output": str(output), "receipts": len(receipts), "model_calls": 0, "aws_calls": 0}))
    return 0 if all(r.status == "COMPLETED" for r in receipts) else 2


if __name__ == "__main__":
    raise SystemExit(main())
