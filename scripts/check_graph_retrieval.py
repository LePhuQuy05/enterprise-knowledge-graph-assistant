"""Frozen development smoke, not a quality benchmark; no inference API or AWS calls."""

import dataclasses
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.context import pack_context
from app.contracts import sha256_bytes
from app.embeddings import EmbeddingCache, query_embedding_input
from app.graph_retrieval import GraphIndex
from app.local_bge import BgeEncoder
from app.snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    receipt = json.loads((ROOT / "results/pilot_graph_snapshot.json").read_bytes())
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    snapshot = load_snapshot(Path(receipt["snapshot_directory"]), tenant_id="demo",
        embedding_model_id=encoder.recipe("document").model_id, trusted_manifest_sha256=receipt["manifest_sha256"])
    if snapshot.manifest.embedding_recipe != encoder.recipe("document"):
        raise ValueError("QUERY_ENCODER_DOES_NOT_MATCH_SNAPSHOT_RECIPE")
    index, cache = GraphIndex(snapshot), EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE")
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "STRUCTURAL_PASS_QUALITY_PENDING",
              "paid_model_calls": 0, "aws_calls": 0, "partial_graph": True,
              "snapshot_version": snapshot.manifest.snapshot_version, "questions": [], "holdout": False,
              "semantic_review": "PENDING", "graph_improvement": "NOT_MEASURED"}
    for question in ("Why can S3 invoke Lambda but the function cannot read an SSE-KMS object?",
                     "How can Amazon S3 send an event to a Lambda Event Function?",
                     "How do IAM policies control access to AWS KMS keys?"):
        embedded = cache.resolve_local("demo", text=query_embedding_input(question), recipe=encoder.recipe("query"), encoder=encoder)
        vector = np.asarray(embedded.record.vector, dtype=np.float32)
        baseline = index.retrieve(question, vector, hops=0)
        modes = []
        for hops in (0, 1, 2):
            result = index.retrieve(question, vector, hops=hops)
            if result.baseline_chunk_ids != baseline.ranked_chunk_ids:
                raise ValueError("Baseline changed between comparison modes")
            context = pack_context(snapshot, result)
            modes.append({"requested_hops": hops, "mode_used": result.mode_used, "used_graph": result.used_graph,
                "baseline_ranking": result.baseline_chunk_ids, "final_ranking": result.ranked_chunk_ids,
                "graph_candidates": [dataclasses.asdict(c) for c in result.graph_candidates],
                "seed_entity_ids": result.seed_entity_ids, "warnings": result.warnings,
                "assertion_visits": result.assertion_visits, "visited_entities": result.visited_entities,
                "context_chunk_ids": context.chunk_ids, "context_assertion_ids": context.assertion_ids,
                "context_sha256": sha256_bytes(context.serialized.encode()), "context_proxy_tokens": context.proxy_tokens,
                "context_warnings": context.warnings})
        report["questions"].append({"question": question, "query_cache_hit": embedded.cache_hit, "modes": modes})
    report["local_onnx_forwards"] = encoder.inference_calls
    (ROOT / "results/graph_retrieval_comparison_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "questions": len(report["questions"]),
        "paid_model_calls": 0, "graph_modes_used": sum(m["used_graph"] for q in report["questions"] for m in q["modes"]),
        "graph_empty_fallbacks": sum("GRAPH_EMPTY" in m["warnings"] for q in report["questions"] for m in q["modes"])}, indent=2))


if __name__ == "__main__":
    main()
