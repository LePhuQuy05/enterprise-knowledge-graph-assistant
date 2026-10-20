"""Follow-up development ablation of mention candidates and relation traversal, no serving changes."""

import argparse
import dataclasses
import json
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.context import pack_context
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import EmbeddingCache, query_embedding_input
from app.evaluation import aggregate_retrieval, score_retrieval
from app.graph_retrieval import GraphIndex, GraphLimits
from app.local_bge import BgeEncoder
from scripts.evaluate_development import load_inputs, save
from scripts.evaluate_expanded import ROOT, paired_comparison, read_pair, replay

SYSTEMS = ("A1_MENTIONS_ONLY", "A2_RELATIONS_1HOP", "A3_RELATIONS_2HOP")


def ablation_indexes(graph):
    mentions, relations = GraphIndex(graph), GraphIndex(graph)
    # Keep identical mentions for seed discovery in all modes. Remove only the
    # expansion component under test; immutable snapshot and serving code stay intact.
    mentions.adjacency = defaultdict(list)
    relations.chunks_by_entity = defaultdict(set)
    return mentions, relations


def summarize(rows, controls, datasets):
    expected = {(lang, i.question_id, system) for lang, ds in datasets.items() for i in ds.items for system in SYSTEMS}
    observed = [(r["language"], r["question_id"], r["system"]) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("COMPONENT_ABLATION_ROWS_INVALID")
    output = {"languages": {}, "scope": "FOLLOWUP_DEVELOPMENT_NOT_INDEPENDENT_CONFIRMATION", "labels": "AI_DRAFT",
              "production_default_changed": False}
    for lang, ds in datasets.items():
        baseline = {r["question_id"]: r for r in controls if r["language"] == lang and r["system"] == "B2_HYBRID"}
        if set(baseline) != {i.question_id for i in ds.items}:
            raise ValueError("COMPONENT_ABLATION_CONTROL_ROWS_INVALID")
        output["languages"][lang] = {}
        for system in SYSTEMS:
            own = [r for r in rows if r["language"] == lang and r["system"] == system]
            scores = aggregate_retrieval(own, ds.items, approved_only=False)
            scores["complete_evidence_count"] = sum(r["metrics"]["complete_evidence_at_k"] or 0 for r in own)
            output["languages"][lang][system] = {"provisional_ai_draft": scores,
                "human_approved_only": aggregate_retrieval(own, ds.items, approved_only=True),
                "graph_used_questions": sum(r["used_graph"] for r in own),
                "contexts_with_assertions": sum(bool(r["context_assertion_ids"]) for r in own),
                "positive_hop_candidates": sum(r["positive_hop_candidates"] for r in own),
                "graph_empty_fallbacks": sum("GRAPH_EMPTY" in r["warnings"] for r in own),
                "paired_vs_hybrid": paired_comparison({r["question_id"]: r for r in own}, baseline, ds.items)}
    return output


def evaluate(directory, *, replay_only=False):
    datasets, hashes = read_pair(directory)
    _, graph = load_inputs(datasets["en"])
    replay(directory, datasets, hashes, graph)
    controls = [json.loads(line) for line in (directory / "retrieval.jsonl").read_bytes().splitlines()]
    output = directory / "graph-components-v1"
    if replay_only:
        manifest = json.loads((output / "manifest.json").read_bytes())
        raw = (output / "retrieval.jsonl").read_bytes()
        if (manifest["dataset_sha256"] != hashes or manifest["graph_manifest_sha256"] != graph.manifest_sha256 or
                manifest["retrieval_sha256"] != sha256_bytes(raw) or
                manifest["prior_control_sha256"] != sha256_bytes((directory / "retrieval.jsonl").read_bytes())):
            raise ValueError("COMPONENT_REPLAY_HASH_MISMATCH")
        rows = [json.loads(line) for line in raw.splitlines()]
        lookup = {(lang, i.question_id): i for lang, ds in datasets.items() for i in ds.items}
        for row in rows:
            if row["metrics"] != score_retrieval(lookup[(row["language"], row["question_id"])], row["ranked_chunk_ids"], row["context_chunk_ids"], graph):
                raise ValueError("COMPONENT_REPLAY_SCORE_MISMATCH")
        summary = summarize(rows, controls, datasets)
        if manifest["summary_sha256"] != sha256_bytes(canonical_json(summary)):
            raise ValueError("COMPONENT_REPLAY_SUMMARY_MISMATCH")
        result = {"status": "PASS_RECOMPUTED_FROM_RANKINGS", "rows": len(rows), "new_local_onnx_forwards": 0, "aws_calls": 0, "model_api_calls": 0}
        save(output / "replay.json", result)
        return result
    output.mkdir(parents=True, exist_ok=False)
    config = {"dataset_sha256": hashes, "graph_manifest_sha256": graph.manifest_sha256,
        "prior_control_sha256": sha256_bytes((directory / "retrieval.jsonl").read_bytes()),
        "systems": SYSTEMS, "k": 5, "graph_limits": dataclasses.asdict(GraphLimits()),
        "pipeline_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in (
            "scripts/evaluate_graph_components.py", "app/graph_retrieval.py", "app/context.py", "app/evaluation.py")},
        "started_at": datetime.now(UTC).isoformat(), "status": "RUNNING",
        "rationale": "Predeclared follow-up after mixed graph produced zero packed assertions; no per-question changes or search over weights.",
        "policy": "Same seeds/vectors/corpus/fusion/context/limits; disable adjacency for mentions, disable mention expansion for relations. No gold in ranking."}
    save(output / "manifest.json", config)
    mentions, relations = ablation_indexes(graph)
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    if encoder.recipe("document") != graph.manifest.embedding_recipe:
        raise ValueError("COMPONENT_ENCODER_RECIPE_MISMATCH")
    cache = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE")
    rows = []
    with (output / "retrieval.jsonl").open("xb") as stream:
        for lang, ds in datasets.items():
            for item in ds.items:
                embedded = cache.resolve_local("demo", text=query_embedding_input(item.question), recipe=encoder.recipe("query"), encoder=encoder)
                vector = np.asarray(embedded.record.vector, dtype=np.float32)
                base = next(r for r in controls if r["question_id"] == item.question_id and r["system"] == "B2_HYBRID")
                for system in SYSTEMS:
                    started = time.monotonic()
                    index = mentions if system == "A1_MENTIONS_ONLY" else relations
                    result = index.retrieve(item.question, vector, hops=2 if system == "A3_RELATIONS_2HOP" else 1)
                    ms = (time.monotonic() - started) * 1000
                    if list(result.baseline_chunk_ids) != base["ranked_chunk_ids"]:
                        raise ValueError("COMPONENT_HYBRID_BASELINE_CHANGED")
                    packed = pack_context(graph, result)
                    row = {"language": lang, "question_id": item.question_id, "system": system,
                        "ranked_chunk_ids": result.ranked_chunk_ids, "context_chunk_ids": packed.chunk_ids,
                        "context_assertion_ids": packed.assertion_ids, "used_graph": result.used_graph,
                        "positive_hop_candidates": sum(bool(c.assertion_path) for c in result.graph_candidates),
                        "warnings": result.warnings, "retrieval_ms": ms,
                        "metrics": score_retrieval(item, result.ranked_chunk_ids, packed.chunk_ids, graph)}
                    stream.write(canonical_json(row) + b"\n")
                    stream.flush()
                    rows.append(row)
    summary = summarize(rows, controls, datasets)
    save(output / "summary.json", summary)
    config.update(status="COMPLETE_DIAGNOSTIC_ONLY", rows=len(rows),
        retrieval_sha256=sha256_bytes((output / "retrieval.jsonl").read_bytes()), summary_sha256=sha256_bytes(canonical_json(summary)),
        local_onnx_forwards=encoder.inference_calls, model_api_calls=0, aws_calls=0)
    save(output / "manifest.json", config)
    return {"status": config["status"], "rows": len(rows), "directory": str(output), "local_onnx_forwards": encoder.inference_calls,
            "model_api_calls": 0, "aws_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to((ROOT / "results/evaluation-expanded").resolve()):
        raise ValueError("COMPONENT_RUN_OUTSIDE_RESULTS")
    print(json.dumps(evaluate(directory, replay_only=args.replay)))


if __name__ == "__main__":
    main()
