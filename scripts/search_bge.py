"""Search the validated BGE snapshot locally; return source excerpts, not generated answers."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.embeddings import EmbeddingCache, query_embedding_input
from app.local_bge import BgeEncoder
from app.retrieval import retrieve_baseline
from app.snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=("vector", "hybrid"), default="hybrid")
    args = parser.parse_args()
    started = time.monotonic()
    receipt = json.loads((ROOT / "results/bge_baseline.json").read_bytes())
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    snapshot = load_snapshot(Path(receipt["snapshot_directory"]), tenant_id="demo", embedding_model_id=encoder.recipe("document").model_id,
                             trusted_manifest_sha256=receipt["manifest_sha256"])
    if snapshot.manifest.embedding_recipe != encoder.recipe("document"):
        raise ValueError("QUERY_ENCODER_DOES_NOT_MATCH_SNAPSHOT_RECIPE")
    result = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE").resolve_local(
        "demo", text=query_embedding_input(args.question), recipe=encoder.recipe("query"), encoder=encoder)
    ranked = retrieve_baseline(snapshot, args.question, np.asarray(result.record.vector, dtype=np.float32), mode=args.mode)
    excerpts = []
    chunks = {c.chunk_id: c for c in snapshot.content.chunks}
    for key in ranked[:5]:
        chunk = chunks[key]
        doc = snapshot.documents[chunk.document_version_id]
        excerpts.append({"chunk_id": key, "title": doc.title, "source_uri": doc.source_uri,
                         "document_version_id": doc.document_version_id,
                         "start_char": chunk.start_char, "end_char": chunk.end_char, "text": chunk.text})
    print(json.dumps({"mode": args.mode, "answer_kind": "RETRIEVED_SOURCE_EXCERPTS_ONLY",
                      "query_cache_hit": result.cache_hit, "model_id": encoder.recipe("query").model_id,
                      "seconds_including_model_load": round(time.monotonic() - started, 3),
                      "paid_model_calls": 0, "sources": excerpts}, indent=2, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
