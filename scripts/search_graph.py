"""Search the partial pilot graph, print sources/paths/context; no answer generation or paid calls."""

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np

from app.context import pack_context
from app.embeddings import EmbeddingCache, query_embedding_input
from app.graph_retrieval import GraphIndex
from app.local_bge import BgeEncoder
from app.snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def search(question, hops=1):
    started = time.monotonic()
    receipt = json.loads((ROOT / "results/pilot_graph_snapshot.json").read_bytes())
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    snapshot = load_snapshot(Path(receipt["snapshot_directory"]), tenant_id="demo",
        embedding_model_id=encoder.recipe("document").model_id, trusted_manifest_sha256=receipt["manifest_sha256"])
    if snapshot.manifest.embedding_recipe != encoder.recipe("document"):
        raise ValueError("QUERY_ENCODER_DOES_NOT_MATCH_SNAPSHOT_RECIPE")
    embedded = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE").resolve_local(
        "demo", text=query_embedding_input(question), recipe=encoder.recipe("query"), encoder=encoder)
    result = GraphIndex(snapshot).retrieve(question, np.asarray(embedded.record.vector, dtype=np.float32), hops=hops)
    packed = pack_context(snapshot, result)
    return {"question": question, "answer_kind": "RETRIEVED_EVIDENCE_ONLY", "partial_graph": True,
            "semantic_review": "PENDING", "paid_model_calls": 0, "cloud_published": False,
            "query_cache_hit": embedded.cache_hit, "seconds": round(time.monotonic() - started, 3),
            "trace": dataclasses.asdict(result), "context": dataclasses.asdict(packed),
            "corpus_chunks": receipt["counts"]["chunks"],
            "extraction_accepted_chunks": receipt["extraction_accepted_chunks"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--hops", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--save-receipt", action="store_true", help="Save this public-document query under results/graph_retrieval_smoke.json")
    args = parser.parse_args()
    report = search(args.question, args.hops)
    if args.save_receipt:
        (ROOT / "results/graph_retrieval_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
