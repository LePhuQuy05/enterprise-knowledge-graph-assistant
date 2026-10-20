"""Five-system bilingual development diagnostics. Offline, source-bound, no holdout."""

import argparse
import dataclasses
import json
import platform
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.context import pack_context
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import EmbeddingCache, query_embedding_input
from app.evaluation import aggregate_retrieval, score_retrieval
from app.graph_retrieval import GraphIndex, GraphLimits, GraphRetrievalResult
from app.local_bge import BgeEncoder
from app.retrieval import LexicalIndex, retrieve_baseline
from scripts.evaluate_development import PIPELINE_FILES, load_inputs, read_dataset, save

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = ("B0_BM25", "B1_VECTOR", "B2_HYBRID", "G1_DRAFT", "G2_DRAFT")
LANGUAGES = ("en", "vi")
BOOTSTRAP_SEED = 20261020


def validate_pairs(datasets):
    if set(datasets) != set(LANGUAGES):
        raise ValueError("BILINGUAL_DATASETS_REQUIRED")
    en, vi = datasets["en"], datasets["vi"]
    if en.language != "en" or vi.language != "vi" or len(en.items) != len(vi.items):
        raise ValueError("BILINGUAL_PAIR_MISMATCH")
    for a, b in zip(en.items, vi.items, strict=True):
        if b.question_id != a.question_id + "-vi":
            raise ValueError("BILINGUAL_PAIR_ID_MISMATCH")
        if a.model_dump(exclude={"question_id", "question"}) != b.model_dump(exclude={"question_id", "question"}):
            raise ValueError("BILINGUAL_PAIR_LABEL_MISMATCH")
    return datasets


def read_pair(directory=None):
    datasets, hashes = {}, {}
    for language in LANGUAGES:
        path = (directory / f"dataset-{language}.json" if directory else
                ROOT / f"configs/evaluation/expanded-development-{language}-v1.json")
        datasets[language], hashes[language] = read_dataset(path)
    return validate_pairs(datasets), hashes


def paired_comparison(own, baseline, items):
    """Resample whole intent groups, preserving correlated variants and paired systems."""
    deltas = {i.question_id: own[i.question_id]["metrics"]["complete_evidence_at_k"]
              - baseline[i.question_id]["metrics"]["complete_evidence_at_k"] for i in items if i.answerable}
    groups = defaultdict(list)
    for item in items:
        if item.answerable:
            groups[item.intent_group_id].append(deltas[item.question_id])
    values = list(groups.values())
    means = []
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    if values:
        for _ in range(2000):
            drawn = [values[int(j)] for j in rng.integers(0, len(values), size=len(values))]
            means.append(100 * sum(sum(v) for v in drawn) / sum(map(len, drawn)))
    return {"answerable_questions": len(deltas), "intent_groups": len(groups),
            "wins": sum(d == 1 for d in deltas.values()), "losses": sum(d == -1 for d in deltas.values()),
            "ties": sum(d == 0 for d in deltas.values()),
            "delta_percentage_points": 100 * sum(deltas.values()) / len(deltas) if deltas else None,
            "cluster_bootstrap_percentile95_pp": np.quantile(means, [0.025, 0.975]).tolist() if means else None,
            "changed_question_ids": [key for key, delta in deltas.items() if delta],
            "interval_scope": "Conditional descriptive diagnostic on AI-draft labels; not population or holdout evidence."}


def summarize(rows, datasets):
    expected = {(lang, i.question_id, system) for lang, ds in datasets.items() for i in ds.items for system in SYSTEMS}
    observed = [(r["language"], r["question_id"], r["system"]) for r in rows]
    if len(set(observed)) != len(observed) or set(observed) != expected:
        raise ValueError("EXPANDED_INCOMPLETE_OR_DUPLICATE_ROWS")
    output = {"languages": {}, "paired_language_diagnostics": {}, "systems": list(SYSTEMS),
        "unique_scenarios": len(datasets["en"].items), "language_inputs": sum(len(ds.items) for ds in datasets.values()),
        "all_intent_groups": len({i.intent_group_id for i in datasets["en"].items}),
        "answerable_intent_groups": len({i.intent_group_id for i in datasets["en"].items if i.answerable}),
        "quality_verdict": "PROVISIONAL_AI_DRAFT_NOT_HUMAN_GOLD_OR_HOLDOUT",
        "bootstrap_seed": BOOTSTRAP_SEED, "bootstrap_samples": 2000,
        "production_default_changed": False, "holdout": "NOT_AUTHORED_NOT_RUN"}
    for lang, ds in datasets.items():
        own_rows = [r for r in rows if r["language"] == lang]
        systems = {}
        hybrid = {r["question_id"]: r for r in own_rows if r["system"] == "B2_HYBRID"}
        for system in SYSTEMS:
            own = [r for r in own_rows if r["system"] == system]
            provisional = aggregate_retrieval(own, ds.items, approved_only=False)
            provisional["complete_evidence_count"] = sum(r["metrics"]["complete_evidence_at_k"] or 0 for r in own)
            groups = defaultdict(list)
            lookup = {i.question_id: i for i in ds.items}
            for row in own:
                if lookup[row["question_id"]].answerable:
                    groups[lookup[row["question_id"]].intent_group_id].append(row["metrics"]["complete_evidence_at_k"])
            systems[system] = {"provisional_ai_draft": provisional,
                "human_approved_only": aggregate_retrieval(own, ds.items, approved_only=True),
                "intent_group_macro_complete_evidence_at5": float(np.mean([np.mean(v) for v in groups.values()])) if groups else None,
                "graph_used_questions": sum(r["used_graph"] for r in own),
                "graph_empty_fallbacks": sum("GRAPH_EMPTY" in r["warnings"] for r in own),
                "graph_timeout_fallbacks": sum("GRAPH_TIMEOUT" in r["warnings"] for r in own),
                "ranking_changed_vs_hybrid": sum(r["ranked_chunk_ids"] != hybrid[r["question_id"]]["ranked_chunk_ids"] for r in own),
                "retrieval_ms_p50_p95": np.quantile([r["retrieval_ms"] for r in own], [.5, .95]).tolist(),
                "by_category_provisional": {}}
            for category in sorted({i.category for i in ds.items}):
                items = [i for i in ds.items if i.category == category]
                ids = {i.question_id for i in items}
                systems[system]["by_category_provisional"][category] = aggregate_retrieval(
                    [r for r in own if r["question_id"] in ids], items, approved_only=False)
            if system != "B2_HYBRID":
                systems[system]["paired_vs_hybrid"] = paired_comparison(
                    {r["question_id"]: r for r in own}, hybrid, ds.items)
        output["languages"][lang] = {"category_counts": dict(Counter(i.category for i in ds.items)), "systems": systems}
    for system in SYSTEMS:
        en = {r["question_id"]: r for r in rows if r["language"] == "en" and r["system"] == system}
        vi = {r["question_id"][:-3]: r for r in rows if r["language"] == "vi" and r["system"] == system}
        output["paired_language_diagnostics"][system] = paired_comparison(vi, en, datasets["en"].items)
    return output


def replay(directory, datasets, hashes, graph):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    raw = (directory / "retrieval.jsonl").read_bytes()
    if (manifest["status"] != "COMPLETE_DIAGNOSTIC_ONLY" or manifest["dataset_sha256"] != hashes
            or manifest["graph_manifest_sha256"] != graph.manifest_sha256
            or manifest["systems"] != list(SYSTEMS) or manifest["retrieval_sha256"] != sha256_bytes(raw)
            or any(sha256_bytes((directory / f"dataset-{lang}.json").read_bytes()) != value for lang, value in hashes.items())):
        raise ValueError("EXPANDED_REPLAY_INPUT_MISMATCH")
    rows = [json.loads(line) for line in raw.splitlines()]
    summary = summarize(rows, datasets)  # Require each language/question/system exactly once.
    lookup = {(lang, i.question_id): i for lang, ds in datasets.items() for i in ds.items}
    for row in rows:
        item = lookup[(row["language"], row["question_id"])]
        if (row["question_sha256"] != sha256_bytes(item.question.encode()) or
                row["metrics"] != score_retrieval(item, row["ranked_chunk_ids"], row["context_chunk_ids"], graph)):
            raise ValueError("EXPANDED_REPLAY_METRICS_CHANGED")
    if manifest["summary_sha256"] != sha256_bytes(canonical_json(summary)):
        raise ValueError("EXPANDED_REPLAY_SUMMARY_CHANGED")
    if canonical_json(summary) != canonical_json(json.loads((directory / "summary.json").read_bytes())):
        raise ValueError("EXPANDED_REPLAY_SAVED_SUMMARY_CHANGED")
    result = {"status": "PASS_RECOMPUTED_FROM_RANKINGS", "rows": len(rows), "model_api_calls": 0,
              "aws_calls": 0, "new_local_onnx_forwards": 0, "checked_at": datetime.now(UTC).isoformat()}
    save(directory / "replay.json", result)
    return result


def run(datasets, hashes, baseline, graph):
    files = (*PIPELINE_FILES, "scripts/evaluate_expanded.py")
    config = {"dataset_sha256": hashes, "source_manifest_sha256": baseline.manifest_sha256,
        "graph_manifest_sha256": graph.manifest_sha256, "systems": list(SYSTEMS), "k": 5,
        "context_max_chunks": 8, "context_proxy_tokens": 4096, "rrf_constant": 60,
        "graph_limits": dataclasses.asdict(GraphLimits()), "bootstrap_seed": BOOTSTRAP_SEED,
        "pipeline_sha256": {f: sha256_bytes((ROOT / f).read_bytes()) for f in files},
        "dependency_lock_sha256": sha256_bytes((ROOT / "uv.lock").read_bytes()),
        "embedding_recipe": graph.manifest.embedding_recipe.model_dump(mode="json"),
        "python": platform.python_version(), "graph_release_approved": False,
        "query_policy": "Raw question in each language. No translation, gold facts or reference answer in retrieval.",
        "timing_scope": "One local pass; retrieval includes lexical reconstruction, excludes encoder/model load and context packing. No SLO claim.",
        "label_status": "AI_DRAFT; exact quotes checked; human semantic, alternate-source and translation review pending."}
    fingerprint = sha256_bytes(canonical_json(config))
    directory = ROOT / "results/evaluation-expanded" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + "-" + fingerprint[:8])
    directory.mkdir(parents=True, exist_ok=False)
    manifest = {**config, "configuration_sha256": fingerprint, "run_id": directory.name,
                "status": "RUNNING", "started_at": datetime.now(UTC).isoformat()}
    for lang in LANGUAGES:
        source = ROOT / f"configs/evaluation/expanded-development-{lang}-v1.json"
        if sha256_bytes(source.read_bytes()) != hashes[lang]:
            raise ValueError("DATASET_CHANGED_BEFORE_RUN")
        (directory / f"dataset-{lang}.json").write_bytes(source.read_bytes())
    save(directory / "manifest.json", manifest)  # Freeze before observing rankings.
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    if graph.manifest.embedding_recipe != encoder.recipe("document"):
        raise ValueError("EVALUATION_ENCODER_RECIPE_MISMATCH")
    cache, index = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE"), GraphIndex(graph)
    rows = []
    try:
        with (directory / "retrieval.jsonl").open("xb") as stream:
            for lang, dataset in datasets.items():
                for item in dataset.items:
                    embedded = cache.resolve_local("demo", text=query_embedding_input(item.question),
                        recipe=encoder.recipe("query"), encoder=encoder)
                    vector = np.asarray(embedded.record.vector, dtype=np.float32)
                    hybrid_ids = None
                    for system in SYSTEMS:
                        started = time.monotonic()
                        if system in {"B0_BM25", "B1_VECTOR"}:
                            if system == "B0_BM25":
                                lexical = LexicalIndex(graph.content.documents, graph.content.chunks, tenant_id=graph.manifest.tenant_id)
                                ids = tuple(key for key, _ in lexical.search(item.question, k=20))
                                mode = "bm25"
                            else:
                                ids = tuple(retrieve_baseline(graph, item.question, vector, mode="vector"))
                                mode = "vector"
                            result = GraphRetrievalResult(ids, ids, (), (), mode, False, (), 0, 0)
                        else:
                            result = index.retrieve(item.question, vector, hops={"B2_HYBRID": 0, "G1_DRAFT": 1, "G2_DRAFT": 2}[system])
                            if system == "B2_HYBRID":
                                hybrid_ids = result.ranked_chunk_ids
                            elif result.baseline_chunk_ids != hybrid_ids:
                                raise ValueError("HYBRID_BASELINE_CHANGED_BETWEEN_SYSTEMS")
                        retrieval_ms = (time.monotonic() - started) * 1000
                        packed = pack_context(graph, result)
                        row = {"language": lang, "question_id": item.question_id, "system": system,
                            "question_sha256": sha256_bytes(item.question.encode()),
                            "ranked_chunk_ids": result.ranked_chunk_ids, "context_chunk_ids": packed.chunk_ids,
                            "context_assertion_ids": packed.assertion_ids, "context_proxy_tokens": packed.proxy_tokens,
                            "retrieval_ms": retrieval_ms, "query_embedding_cache_hit": embedded.cache_hit,
                            "mode_used": result.mode_used, "used_graph": result.used_graph, "warnings": result.warnings,
                            "context_warnings": packed.warnings, "assertion_visits": result.assertion_visits,
                            "metrics": score_retrieval(item, result.ranked_chunk_ids, packed.chunk_ids, graph)}
                        stream.write(canonical_json(row) + b"\n")
                        stream.flush()
                        rows.append(row)
        summary = summarize(rows, datasets)
        save(directory / "summary.json", summary)
        manifest.update(status="COMPLETE_DIAGNOSTIC_ONLY", completed_at=datetime.now(UTC).isoformat(),
            retrieval_sha256=sha256_bytes((directory / "retrieval.jsonl").read_bytes()),
            summary_sha256=sha256_bytes(canonical_json(summary)), local_onnx_forwards=encoder.inference_calls,
            rows=len(rows), model_api_calls=0, aws_calls=0,
            graph_assertions=len(graph.content.assertions), graph_assertion_source_chunks=len({a.chunk_id for a in graph.content.assertions}),
            documents=len(graph.documents), chunks=len(graph.chunks))
    except Exception as exc:
        manifest.update(status="INCOMPLETE", error_category=type(exc).__name__, rows=len(rows))
        raise
    finally:
        save(directory / "manifest.json", manifest)
    return {"status": manifest["status"], "directory": str(directory), "rows": len(rows),
            "local_onnx_forwards": encoder.inference_calls, "model_api_calls": 0, "aws_calls": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", type=Path)
    args = parser.parse_args()
    directory = args.replay.resolve() if args.replay else None
    if directory and not directory.is_relative_to((ROOT / "results/evaluation-expanded").resolve()):
        raise ValueError("EXPANDED_REPLAY_OUTSIDE_RESULTS")
    datasets, hashes = read_pair(directory)
    baseline, graph = load_inputs(datasets["en"])
    from app.evaluation import validate_sources
    validate_sources(datasets["vi"], baseline)
    print(json.dumps(replay(directory, datasets, hashes, graph) if directory else run(datasets, hashes, baseline, graph)))


if __name__ == "__main__":
    main()
