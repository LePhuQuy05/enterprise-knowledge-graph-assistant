"""Verify the ten live pilot vectors entirely offline; never construct an AWS/API client."""

import json
from pathlib import Path

import numpy as np

from app.contracts import ExternalEmbeddingRecipe, canonical_json, embedding_input, sha256_bytes
from app.corpus import load_corpus
from app.embeddings import EmbeddingCache
from app.model_gateway import CallContext

ROOT = Path(__file__).resolve().parents[1]


def main():
    artifacts = ROOT / "data/experiments/external-pilot10-v2"
    plan = json.loads((artifacts / "plan.json").read_bytes())
    if plan["plan_sha256"] != sha256_bytes(canonical_json({k: v for k, v in plan.items() if k != "plan_sha256"})):
        raise ValueError("PLAN_CHANGED")
    ingestion = json.loads((ROOT / "results/ingestion_pilot20_v2.json").read_bytes())
    corpus = load_corpus(Path(ingestion["corpus_directory"]), tenant_id="demo")
    if corpus.version != plan["corpus_version"]:
        raise ValueError("CORPUS_CHANGED")
    documents = {d.document_version_id: d for d in corpus.documents}
    chunks = {c.chunk_id: c for c in corpus.chunks}
    recipe = ExternalEmbeddingRecipe.model_validate_json(canonical_json(plan["embedding_recipe"]))
    cache = EmbeddingCache(ROOT / "data/cache/external-models", namespace="LIVE")
    rows = []
    for selected in plan["selection"]:
        chunk = chunks[selected["chunk_id"]]
        if chunk.text_sha256 != selected["text_sha256"] or chunk.document_version_id != selected["document_version_id"]:
            raise ValueError("SOURCE_BINDING_CHANGED")
        result = cache.resolve(CallContext(tenant_id="demo", run_id="external-pilot10-r72-v1",
                    request_id="offline-embedding-replay-" + chunk.chunk_id, stage="EMBED"),
                    text=embedding_input(documents[chunk.document_version_id], chunk), recipe=recipe)
        if not result.cache_hit or result.new_cost_micro_usd != 0:
            raise ValueError("OFFLINE_REPLAY_NOT_A_HIT")
        rows.append(result.record.vector)
    vectors = np.asarray(rows, dtype=np.float32)
    saved = np.load(artifacts / "vectors.npy", allow_pickle=False)
    norms = np.linalg.norm(vectors, axis=1)
    if not (np.array_equal(vectors, saved) and np.isfinite(vectors).all() and np.allclose(norms, 1.0, atol=1e-5)):
        raise ValueError("VECTOR_ARTIFACT_CHANGED")
    report = {"status": "PASS", "plan_sha256": plan["plan_sha256"], "vectors": len(rows),
              "dimension": vectors.shape[1], "cache_hits": len(rows), "new_calls": 0,
              "new_budget_cost_micro_usd": 0, "vectors_npy_sha256": sha256_bytes((artifacts / "vectors.npy").read_bytes()),
              "relevance_quality": "NOT_EVALUATED", "full_corpus_vectors": False}
    (ROOT / "results/external_pilot_embeddings_replay.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
