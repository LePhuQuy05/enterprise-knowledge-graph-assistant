"""Controlled local mention-coverage ablation; draft labels, no generation or AWS."""

import argparse
import dataclasses
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.context import pack_context
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import EmbeddingCache, query_embedding_input
from app.evaluation import aggregate_retrieval, score_retrieval
from app.graph_retrieval import GraphIndex, GraphLimits, GraphRetrievalResult
from app.local_bge import BgeEncoder
from app.retrieval import retrieve_baseline
from app.snapshot import load_snapshot
from scripts.evaluate_development import DATASET, load_inputs, read_dataset, save

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT / "results/evaluation-dev/20261017T214800Z-6197bdfe"
SYSTEMS = ("B1_VECTOR", "B2_HYBRID", "M_SERVICE_MENTIONS")
CODE_FILES = ("app/service_mentions.py", "app/evaluation.py", "app/graph_retrieval.py", "app/context.py",
              "app/retrieval.py", "app/local_bge.py", "scripts/evaluate_service_mentions.py")


def summarize(rows, dataset):
    output = {}
    for system in SYSTEMS:
        own = [r for r in rows if r["system"] == system]
        output[system] = {
            "provisional_ai_draft": aggregate_retrieval(own, dataset.items, approved_only=False),
            "human_approved_only": aggregate_retrieval(own, dataset.items, approved_only=True),
            "mention_expansion_used_questions": sum(r["used_mentions"] for r in own),
            "empty_expansion_fallbacks": sum("GRAPH_EMPTY" in r["warnings"] for r in own),
        }
    base = {r["question_id"]:r for r in rows if r["system"] == "B2_HYBRID"}
    answerable = {i.question_id for i in dataset.items if i.answerable}
    deltas = [r["metrics"]["complete_evidence_at_k"] - base[r["question_id"]]["metrics"]["complete_evidence_at_k"]
              for r in rows if r["system"] == "M_SERVICE_MENTIONS" and r["question_id"] in answerable]
    return {"systems":output, "mention_vs_hybrid_provisional":{
        "denominator":len(deltas), "wins":deltas.count(1), "losses":deltas.count(-1), "ties":deltas.count(0),
        "delta_percentage_points":100 * sum(deltas) / len(deltas) if deltas else None},
        "verdict":"DEVELOPMENT_ABLATION_ONLY_NO_GRAPH_QUALITY_APPROVAL", "production_default_changed":False,
        "holdout":"NOT_RUN", "semantic_relation_assertions":0}


def verify_replay(directory, dataset, dataset_sha, graph):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    raw = (directory / "retrieval.jsonl").read_bytes()
    if (manifest["status"] != "COMPLETE_DIAGNOSTIC_ONLY" or manifest["dataset_sha256"] != dataset_sha
            or manifest["mention_manifest_sha256"] != graph.manifest_sha256
            or manifest["retrieval_sha256"] != sha256_bytes(raw)):
        raise ValueError("MENTION_REPLAY_HASH_MISMATCH")
    rows = [json.loads(line) for line in raw.splitlines()]
    lookup = {i.question_id:i for i in dataset.items}
    if len(rows) != len(lookup) * len(SYSTEMS) or any(r["system"] not in SYSTEMS for r in rows):
        raise ValueError("MENTION_REPLAY_SYSTEM_COUNT_MISMATCH")
    for row in rows:
        if score_retrieval(lookup[row["question_id"]],row["ranked_chunk_ids"],row["context_chunk_ids"],graph) != row["metrics"]:
            raise ValueError("MENTION_REPLAY_METRIC_MISMATCH")
    if sha256_bytes(canonical_json(summarize(rows,dataset))) != manifest["summary_sha256"]:
        raise ValueError("MENTION_REPLAY_SUMMARY_MISMATCH")
    result = {"status":"PASS_RECOMPUTED_FROM_RANKINGS", "checked_at":datetime.now(UTC).isoformat(),
              "rows":len(rows), "new_onnx_forwards":0, "model_api_calls":0, "aws_calls":0}
    save(directory / "replay.json",result)
    return result


def run(dataset, dataset_sha, graph, baseline, prior):
    prior_manifest = json.loads((PRIOR / "manifest.json").read_bytes())
    if prior_manifest["dataset_sha256"] != dataset_sha or prior_manifest["source_manifest_sha256"] != baseline.manifest_sha256:
        raise ValueError("PRIOR_DEVELOPMENT_INPUTS_CHANGED")
    if sha256_bytes((PRIOR / "retrieval.jsonl").read_bytes()) != prior_manifest["retrieval_sha256"]:
        raise ValueError("PRIOR_CONTROL_ARTIFACT_HASH_MISMATCH")
    controls = {(r["question_id"],r["system"]):r for r in prior if r["system"] in SYSTEMS[:2]}
    if len(controls) != len(dataset.items) * 2:
        raise ValueError("PRIOR_CONTROLS_INCOMPLETE")
    config = {"dataset_sha256":dataset_sha,"source_manifest_sha256":baseline.manifest_sha256,
        "mention_manifest_sha256":graph.manifest_sha256,"systems":SYSTEMS,"k":5,
        "graph_limits":dataclasses.asdict(GraphLimits()), "context_max_chunks":8,"context_proxy_tokens":4096,
        "rrf_constant":60, "pipeline_sha256":{name:sha256_bytes((ROOT/name).read_bytes()) for name in CODE_FILES},
        "dependency_lock_sha256":sha256_bytes((ROOT/"uv.lock").read_bytes()),
        "embedding_recipe":graph.manifest.embedding_recipe.model_dump(mode="json"),
        "prior_control_run":str(PRIOR.relative_to(ROOT)), "new_generation_calls_planned":0,
        "relation_assertions_allowed":False, "tuning":"One predeclared mention-only configuration; no per-question overrides.",
        "limits":["AI draft gold; not holdout or an answer-quality benchmark.",
            "Mention occurrence is not a relation, permission grant or dependency.",
            "Alias dictionary covers six named services, not every AWS service.",
            "The original rejected graph and production serving snapshot remain unchanged."]}
    fingerprint = sha256_bytes(canonical_json(config))
    directory = ROOT / "results/service-mention-evaluation" / (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + fingerprint[:8])
    directory.mkdir(parents=True,exist_ok=False)
    manifest = {**config,"status":"RUNNING","started_at":datetime.now(UTC).isoformat(),"run_id":directory.name}
    save(directory / "manifest.json",manifest)
    (directory / "dataset.json").write_bytes(DATASET.read_bytes())
    rows = []
    try:
        encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5",ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
        if encoder.recipe("document") != graph.manifest.embedding_recipe:
            raise ValueError("MENTION_QUERY_ENCODER_RECIPE_MISMATCH")
        index = GraphIndex(graph)
        cache = EmbeddingCache(ROOT / "data/cache/local-bge",namespace="LIVE")
        with (directory / "retrieval.jsonl").open("xb") as stream:
            for item in dataset.items:
                embedded = cache.resolve_local("demo",text=query_embedding_input(item.question),
                    recipe=encoder.recipe("query"),encoder=encoder)
                vector = np.asarray(embedded.record.vector,dtype=np.float32)
                hybrid_ids = None
                for system in SYSTEMS:
                    started = time.monotonic()
                    if system == "B1_VECTOR":
                        ids = tuple(retrieve_baseline(graph,item.question,vector,mode="vector"))
                        result = GraphRetrievalResult(ids,ids,(),(),"vector",False,(),0,0)
                    else:
                        result = index.retrieve(item.question,vector,hops=0 if system == "B2_HYBRID" else 1)
                        if system == "B2_HYBRID":
                            hybrid_ids = result.ranked_chunk_ids
                        elif result.baseline_chunk_ids != hybrid_ids:
                            raise ValueError("MENTION_HYBRID_SEEDS_CHANGED")
                    retrieval_ms = (time.monotonic() - started) * 1000
                    if result.assertion_visits or any(c.assertion_path or c.hop != 0 for c in result.graph_candidates):
                        raise ValueError("MENTION_ABLATION_CANNOT_TRAVERSE_RELATIONS")
                    packed = pack_context(graph,result)
                    row = {"question_id":item.question_id,"system":system,"ranked_chunk_ids":result.ranked_chunk_ids,
                        "context_chunk_ids":packed.chunk_ids,"context_proxy_tokens":packed.proxy_tokens,
                        "metrics":score_retrieval(item,result.ranked_chunk_ids,packed.chunk_ids,graph),
                        "used_mentions":result.used_graph,"warnings":result.warnings,"context_warnings":packed.warnings,
                        "retrieval_ms":retrieval_ms,"query_cache_hit":embedded.cache_hit,"assertion_visits":0,
                        "candidate_chunks":len(result.graph_candidates),"seed_entity_ids":result.seed_entity_ids}
                    if system in SYSTEMS[:2]:
                        original = controls[(item.question_id,system)]
                        if list(row["ranked_chunk_ids"]) != original["ranked_chunk_ids"] or (
                            list(row["context_chunk_ids"]) != original["context_chunk_ids"] or row["metrics"] != original["metrics"]
                        ):
                            raise ValueError("MENTION_CONTROL_DIFFERS_FROM_FROZEN_BASELINE")
                    stream.write(canonical_json(row)+b"\n")
                    stream.flush()
                    rows.append(row)
        summary = summarize(rows,dataset)
        save(directory / "summary.json",summary)
        manifest.update(status="COMPLETE_DIAGNOSTIC_ONLY",completed_at=datetime.now(UTC).isoformat(),rows=len(rows),
            retrieval_sha256=sha256_bytes((directory/"retrieval.jsonl").read_bytes()),
            summary_sha256=sha256_bytes(canonical_json(summary)),local_onnx_forwards=encoder.inference_calls,
            baseline_controls_equal_prior_run=True,model_api_calls=0,aws_calls=0)
    except Exception as exc:
        manifest.update(status="INCOMPLETE",error_category=type(exc).__name__,rows=len(rows))
        raise
    finally:
        save(directory / "manifest.json",manifest)
    return {"status":manifest["status"],"directory":str(directory),"rows":len(rows),
            "local_onnx_forwards":encoder.inference_calls,"model_api_calls":0,"aws_calls":0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay",type=Path)
    args = parser.parse_args()
    dataset,dataset_sha = read_dataset(DATASET)
    baseline,_ = load_inputs(dataset)
    receipt = json.loads((ROOT / "results/service_mention_snapshot.json").read_bytes())
    graph = load_snapshot(Path(receipt["snapshot_directory"]),tenant_id="demo",embedding_model_id="BAAI/bge-small-en-v1.5",
        trusted_manifest_sha256=receipt["manifest_sha256"])
    if (graph.content.documents != baseline.content.documents or graph.content.chunks != baseline.content.chunks
            or graph.manifest.embedding_recipe != baseline.manifest.embedding_recipe
            or not np.array_equal(graph.vectors,baseline.vectors) or graph.content.assertions):
        raise ValueError("MENTION_COMPARISON_INPUTS_DIFFER")
    if args.replay:
        directory = args.replay.resolve()
        if not directory.is_relative_to((ROOT / "results/service-mention-evaluation").resolve()):
            raise ValueError("MENTION_REPLAY_OUTSIDE_RESULTS")
        result = verify_replay(directory,dataset,dataset_sha,graph)
    else:
        prior = [json.loads(line) for line in (PRIOR / "retrieval.jsonl").read_bytes().splitlines()]
        result = run(dataset,dataset_sha,graph,baseline,prior)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
