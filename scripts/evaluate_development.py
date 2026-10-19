"""Offline development retrieval diagnostics; no AWS/provider calls or holdout loader."""

import argparse
import dataclasses
import json
import platform
import subprocess
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.context import pack_context
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import EmbeddingCache, query_embedding_input
from app.evaluation import (
    DevelopmentDataset,
    aggregate_retrieval,
    score_retrieval,
    validate_sources,
)
from app.graph_retrieval import GraphIndex, GraphLimits, GraphRetrievalResult
from app.local_bge import BgeEncoder
from app.retrieval import retrieve_baseline
from app.snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "configs/evaluation/r72-development-v1.json"
SYSTEMS = ("B1_VECTOR", "B2_HYBRID", "G1_DRAFT", "G2_DRAFT")
PIPELINE_FILES = ("app/evaluation.py", "app/retrieval.py", "app/graph_retrieval.py", "app/context.py",
                  "app/local_bge.py", "app/embeddings.py", "app/contracts.py", "scripts/evaluate_development.py")


def unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("DUPLICATE_DATASET_JSON_KEY")
        value[key] = item
    return value


def read_dataset(path):
    raw = path.read_bytes()
    if len(raw) > 512 * 1024:
        raise ValueError("DATASET_SIZE_BOUND")
    value = json.loads(raw, object_pairs_hook=unique_object)
    if value.get("split") != "DEVELOPMENT":
        raise ValueError("DEVELOPMENT_LOADER_CANNOT_OPEN_OTHER_SPLITS")
    dataset = DevelopmentDataset.model_validate_json(canonical_json(value))
    if any(i.review_status == "REJECTED" for i in dataset.items):
        raise ValueError("REJECTED_ITEM_REQUIRES_NEW_REVIEWED_DATASET_VERSION")
    return dataset, sha256_bytes(raw)


def load_inputs(dataset):
    def snapshot(receipt_name):
        receipt = json.loads((ROOT / "results" / receipt_name).read_bytes())
        return load_snapshot(Path(receipt["snapshot_directory"]), tenant_id="demo",
            embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=receipt["manifest_sha256"])

    baseline, graph = snapshot("bge_baseline.json"), snapshot("pilot_graph_snapshot.json")
    validate_sources(dataset, baseline)
    # All four systems use the same graph-bearing snapshot, with graph ignored in B1/B2.
    # Its source documents, chunks, recipe and vectors must equal the published baseline.
    if (baseline.content.documents != graph.content.documents or baseline.content.chunks != graph.content.chunks
            or baseline.manifest.embedding_recipe != graph.manifest.embedding_recipe
            or not np.array_equal(baseline.vectors, graph.vectors)):
        raise ValueError("COMPARISON_SOURCE_OR_VECTOR_MISMATCH")
    return baseline, graph


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def summarize(rows, dataset):
    output = {}
    for system in SYSTEMS:
        own = [r for r in rows if r["system"] == system]
        output[system] = {
            "provisional_ai_draft": aggregate_retrieval(own, dataset.items, approved_only=False),
            "human_approved_only": aggregate_retrieval(own, dataset.items, approved_only=True),
            "graph_used_questions": sum(r["used_graph"] for r in own),
            "graph_empty_fallbacks": sum("GRAPH_EMPTY" in r["warnings"] for r in own),
            "graph_timeout_fallbacks": sum("GRAPH_TIMEOUT" in r["warnings"] for r in own),
            "by_category_provisional": {},
        }
        for category in sorted({i.category for i in dataset.items}):
            items = [i for i in dataset.items if i.category == category]
            ids = {i.question_id for i in items}
            output[system]["by_category_provisional"][category] = aggregate_retrieval(
                [r for r in own if r["question_id"] in ids], items, approved_only=False)
    pairs = {}
    baseline = {r["question_id"]: r for r in rows if r["system"] == "B2_HYBRID"}
    answerable = {i.question_id for i in dataset.items if i.answerable}
    for system in ("B1_VECTOR", "G1_DRAFT", "G2_DRAFT"):
        deltas = [r["metrics"]["complete_evidence_at_k"] - baseline[r["question_id"]]["metrics"]["complete_evidence_at_k"]
                  for r in rows if r["system"] == system and r["question_id"] in answerable]
        pairs[system] = {"denominator": len(deltas), "wins": deltas.count(1), "losses": deltas.count(-1),
                         "ties": deltas.count(0), "delta_percentage_points": 100 * sum(deltas) / len(deltas) if deltas else None}
    return {"systems": output, "paired_vs_hybrid_provisional": pairs,
            "quality_verdict": "NOT_ESTABLISHED_AI_LABELS_AND_GRAPH_UNAPPROVED",
            "production_default_changed": False, "pilot_eval": "NOT_AUTHORED_NOT_RUN", "holdout": "NOT_AUTHORED_NOT_RUN"}


def replay(directory, dataset, dataset_sha, graph):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    raw = (directory / "retrieval.jsonl").read_bytes()
    if (manifest["status"] != "COMPLETE_DIAGNOSTIC_ONLY" or manifest["dataset_sha256"] != dataset_sha
            or manifest["graph_manifest_sha256"] != graph.manifest_sha256
            or manifest["retrieval_sha256"] != sha256_bytes(raw)):
        raise ValueError("REPLAY_INPUT_HASH_MISMATCH")
    rows = [json.loads(line) for line in raw.splitlines()]
    items = {i.question_id: i for i in dataset.items}
    if len(rows) != len(items) * len(SYSTEMS) or any(r["system"] not in SYSTEMS for r in rows):
        raise ValueError("REPLAY_SYSTEM_OR_COUNT_MISMATCH")
    for row in rows:
        recalculated = score_retrieval(items[row["question_id"]], row["ranked_chunk_ids"], row["context_chunk_ids"], graph)
        if recalculated != row["metrics"]:
            raise ValueError("REPLAY_METRICS_CHANGED")
    summary = summarize(rows, dataset)
    if sha256_bytes(canonical_json(summary)) != manifest["summary_sha256"]:
        raise ValueError("REPLAY_SUMMARY_CHANGED")
    result = {"status": "PASS_RECOMPUTED_FROM_RANKINGS", "checked_at": datetime.now(UTC).isoformat(),
              "rows": len(rows), "new_local_onnx_forwards": 0, "model_api_calls": 0, "aws_calls": 0,
              "dataset_sha256": dataset_sha, "summary_sha256": manifest["summary_sha256"]}
    save(directory / "replay.json", result)
    return result


def run(dataset, dataset_sha, baseline, graph):
    code = {name: sha256_bytes((ROOT / name).read_bytes()) for name in PIPELINE_FILES}
    head = subprocess.run(["git", "rev-parse", "--verify", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    config = {"dataset_sha256": dataset_sha, "source_manifest_sha256": baseline.manifest_sha256,
        "graph_manifest_sha256": graph.manifest_sha256, "systems": SYSTEMS, "k": 5,
        "context_max_chunks": 8, "context_proxy_tokens": 4096, "rrf_constant": 60,
        "graph_limits": dataclasses.asdict(GraphLimits()), "pipeline_sha256": code,
        "dependency_lock_sha256": sha256_bytes((ROOT / "uv.lock").read_bytes()),
        "embedding_recipe": graph.manifest.embedding_recipe.model_dump(mode="json"),
        "generation_calls_planned": 0, "aws_calls_planned": 0,
        "graph_release_approved": False, "git_commit": head.stdout.strip() if head.returncode == 0 else None,
        "git_dirty": bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT,
                                       capture_output=True, text=True).stdout.strip()),
        "python": platform.python_version(), "ranking_gold_inputs": "question text and cached query vector only",
        "mrr_definition": "First positive source-span overlap within top5; diagnostic, not complete fact support.",
        "limitations": ["All labels are draft unless explicitly human reviewed; graph contains unapproved assertions.",
            "Some dev questions/intents were used in prior learning/pilots; these are not independent holdout items.",
            "Multi-hop structural checks cannot prove semantic necessity or exhaust alternative sources.",
            "No generation/citation entailment, abstention success, prompt-injection or latency SLO is measured.",
            "One local pass per item/system; timing includes index reconstruction and excludes model load/query encoding."]}
    fingerprint = sha256_bytes(canonical_json(config))
    directory = ROOT / "results/evaluation-dev" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + fingerprint[:8])
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {**config, "run_id": directory.name, "configuration_sha256": fingerprint,
                "status": "RUNNING", "started_at": datetime.now(UTC).isoformat()}
    save(directory / "manifest.json", manifest)  # Freeze before observing any new rankings.
    (directory / "dataset.json").write_bytes(DATASET.read_bytes())
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    if graph.manifest.embedding_recipe != encoder.recipe("document"):
        raise ValueError("EVALUATION_ENCODER_RECIPE_MISMATCH")
    index = GraphIndex(graph)
    cache = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE")
    rows = []
    try:
        with (directory / "retrieval.jsonl").open("xb") as stream:
            for item in dataset.items:
                embedded = cache.resolve_local("demo", text=query_embedding_input(item.question),
                    recipe=encoder.recipe("query"), encoder=encoder)
                vector = np.asarray(embedded.record.vector, dtype=np.float32)
                baseline_ids = None
                for system in SYSTEMS:
                    start = time.monotonic()
                    if system == "B1_VECTOR":
                        ids = tuple(retrieve_baseline(graph, item.question, vector, mode="vector"))
                        result = GraphRetrievalResult(ids, ids, (), (), "vector", False, (), 0, 0)
                    else:
                        result = index.retrieve(item.question, vector, hops={"B2_HYBRID":0,"G1_DRAFT":1,"G2_DRAFT":2}[system])
                        if system == "B2_HYBRID":
                            baseline_ids = result.ranked_chunk_ids
                        elif result.baseline_chunk_ids != baseline_ids:
                            raise ValueError("HYBRID_BASELINE_CHANGED_BETWEEN_SYSTEMS")
                    retrieval_ms = (time.monotonic() - start) * 1000
                    packed = pack_context(graph, result)
                    row = {"question_id": item.question_id, "system": system,
                        "ranked_chunk_ids": result.ranked_chunk_ids, "context_chunk_ids": packed.chunk_ids,
                        "context_assertion_ids": packed.assertion_ids, "context_proxy_tokens": packed.proxy_tokens,
                        "retrieval_ms": retrieval_ms, "query_embedding_cache_hit": embedded.cache_hit,
                        "mode_used": result.mode_used, "used_graph": result.used_graph, "warnings": result.warnings,
                        "context_warnings": packed.warnings, "assertion_visits": result.assertion_visits,
                        "metrics": score_retrieval(item, result.ranked_chunk_ids, packed.chunk_ids, graph)}
                    stream.write(canonical_json(row) + b"\n")
                    stream.flush()
                    rows.append(row)
        summary = summarize(rows, dataset)
        save(directory / "summary.json", summary)
        manifest.update(status="COMPLETE_DIAGNOSTIC_ONLY", completed_at=datetime.now(UTC).isoformat(),
            retrieval_sha256=sha256_bytes((directory / "retrieval.jsonl").read_bytes()),
            summary_sha256=sha256_bytes(canonical_json(summary)), local_onnx_forwards=encoder.inference_calls,
            category_counts=dict(Counter(i.category for i in dataset.items)), rows=len(rows),
            model_api_calls=0, aws_calls=0)
    except Exception as exc:
        manifest.update(status="INCOMPLETE", error_category=type(exc).__name__, rows=len(rows))
        raise
    finally:
        save(directory / "manifest.json", manifest)
    return {"status": manifest["status"], "directory": str(directory), "rows": len(rows),
            "local_onnx_forwards": encoder.inference_calls, "model_api_calls": 0, "aws_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path, help="Recompute a completed local run; zero new encoder forwards")
    args = parser.parse_args()
    dataset, fingerprint = read_dataset(DATASET)
    baseline, graph = load_inputs(dataset)
    if args.replay:
        directory = args.replay.resolve()
        if not directory.is_relative_to((ROOT / "results/evaluation-dev").resolve()):
            raise ValueError("REPLAY_OUTSIDE_EVALUATION_RESULTS")
        result = replay(directory, dataset, fingerprint, graph)
    else:
        result = run(dataset, fingerprint, baseline, graph)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
