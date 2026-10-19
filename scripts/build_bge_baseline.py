"""Build/replay real local BGE vectors and a graph-free snapshot; no model API clients."""

import argparse
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.bge_corpus import rechunk_for_bge
from app.contracts import canonical_json, embedding_input, sha256_bytes
from app.corpus import load_corpus
from app.embeddings import EmbeddingCache, query_embedding_input
from app.extraction import AliasDictionary, deepseek_quote_recipe
from app.local_bge import BgeEncoder
from app.retrieval import retrieve_baseline
from app.snapshot import SnapshotContent, build_snapshot, load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true", help="Strict cache-only; does not load ONNX runtime/model")
    args = parser.parse_args()
    output = ROOT / ("results/bge_baseline_replay.json" if args.replay else "results/bge_baseline.json")
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "RUNNING",
              "paid_model_calls": 0, "api_embedding_cost_micro_usd": 0, "cloud_deployments": 0,
              "graph_status": "NOT_BUILT", "relevance_quality": "NOT_EVALUATED", "platform": platform.system()}
    started = time.monotonic()
    encoder = None
    if args.replay:
        from app.contracts import LocalBgeRecipe

        prior = json.loads((ROOT / "results/bge_baseline.json").read_bytes())
        corpus_dir, preparation = Path(prior["corpus_directory"]), prior["rechunking"]
        recipe = LocalBgeRecipe.model_validate_json(canonical_json(prior["embedding_recipe"]))
        query_recipe = recipe.model_copy(update={"input_role": "query", "input_recipe_version": "question-text-v1"})
    else:
        encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
        report["model_load_seconds"] = round(time.monotonic() - started, 3)
        parent = json.loads((ROOT / "results/ingestion_pilot20_v2.json").read_bytes())
        corpus_dir, preparation = rechunk_for_bge(Path(parent["corpus_directory"]), encoder)
        recipe, query_recipe = encoder.recipe("document"), encoder.recipe("query")
    corpus = load_corpus(corpus_dir, tenant_id="demo")
    docs = {d.document_version_id: d for d in corpus.documents}
    cache = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE")
    vectors, hits = {}, 0
    embed_started = time.monotonic()
    for index, chunk in enumerate(corpus.chunks):
        result = cache.resolve_local("demo", text=embedding_input(docs[chunk.document_version_id], chunk),
                                     recipe=recipe, encoder=encoder)
        vectors[chunk.chunk_id] = np.asarray(result.record.vector, dtype=np.float32)
        hits += result.cache_hit
        if index % 25 == 0:
            print(json.dumps({"stage": "EMBED", "completed": index + 1, "total": len(corpus.chunks)}), flush=True)
    report["embedding_seconds"] = round(time.monotonic() - embed_started, 3)
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    extraction = deepseek_quote_recipe(aliases, metadata_sha256=routes["account_model_metadata_sha256"])
    path = build_snapshot(ROOT / "data/snapshots", SnapshotContent(corpus.documents, corpus.chunks, (), (), ()),
        vectors, tenant_id="demo", purpose="DEMO", extraction_recipe=extraction, embedding_recipe=recipe,
        corpus_version=corpus.version, normalizer_version=preparation["normalizer_version"],
        chunking_version=preparation["chunking_version"], tokenizer_version=preparation["tokenizer_version"])
    snapshot = load_snapshot(path, tenant_id="demo", embedding_model_id=recipe.model_id)
    smoke = []
    for question in ("Why can S3 invoke Lambda but the function cannot read an SSE-KMS object?",
                     "Which permissions are required to decrypt an object across AWS accounts?",
                     "How does an explicit deny affect cross-account S3 access?"):
        qstart = time.monotonic()
        q = cache.resolve_local("demo", text=query_embedding_input(question), recipe=query_recipe, encoder=encoder)
        ranked = retrieve_baseline(snapshot, question, np.asarray(q.record.vector, dtype=np.float32))
        smoke.append({"question": question, "query_cache_hit": q.cache_hit, "top5_chunk_ids": ranked[:5],
                      "seconds": round(time.monotonic() - qstart, 3), "semantic_review": "PENDING"})
    report.update(status="STRUCTURAL_PASS_QUALITY_PENDING", embedding_recipe=recipe.model_dump(mode="json"),
        documents=len(corpus.documents), chunks=len(corpus.chunks), corpus_version=corpus.version,
        corpus_directory=str(corpus_dir), rechunking=preparation, document_cache_hits=hits,
        new_document_encodings=len(corpus.chunks)-hits, local_onnx_forward_calls=encoder.inference_calls if encoder else 0,
        snapshot_directory=str(path), snapshot_version=snapshot.manifest.snapshot_version,
        manifest_sha256=sha256_bytes((path / "manifest.json").read_bytes()), retrieval_smoke=smoke,
        total_seconds=round(time.monotonic() - started, 3))
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "documents", "chunks", "embedding_seconds", "document_cache_hits",
                     "new_document_encodings", "local_onnx_forward_calls", "total_seconds", "paid_model_calls")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
