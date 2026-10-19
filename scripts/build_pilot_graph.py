"""Build a clearly partial, unpublished graph snapshot from accepted pilot sources only."""

import json
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import ExtractionRecipe
from app.graph_store import merge_canonical_rows
from app.snapshot import build_snapshot, load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    baseline = json.loads((ROOT / "results/bge_baseline.json").read_bytes())
    pilot = json.loads((ROOT / "results/bge_extraction_pilot_continued.json").read_bytes())
    if pilot["status"] not in {"STRUCTURAL_PASS_SEMANTIC_PENDING", "PARTIAL_ACCEPTED_WITH_QUARANTINE"}:
        raise ValueError("Completed bounded pilot outcomes required")
    base = load_snapshot(Path(baseline["snapshot_directory"]), tenant_id="demo",
                         embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=baseline["manifest_sha256"])
    from app.contracts import sha256_bytes

    raw = (ROOT / "data/experiments/bge-extraction-pilot-v3/canonical.jsonl").read_bytes()
    if sha256_bytes(raw) != pilot["canonical_sha256"] or pilot["corpus_version"] != base.manifest.corpus_version:
        raise ValueError("Canonical pilot receipt hash or corpus mismatch")
    rows = [json.loads(line) for line in raw.splitlines()]
    if sorted(r["chunk_id"] for r in rows) != sorted(r["chunk_id"] for r in pilot["extractions"]):
        raise ValueError("Canonical accepted set differs from pilot receipt")
    content = merge_canonical_rows(base.content, rows)
    recipe = ExtractionRecipe.model_validate_json(json.dumps(pilot["extraction_recipe"]))
    path = build_snapshot(ROOT / "data/snapshots", content,
        {key: base.vectors[i] for i, key in enumerate(base.chunk_ids)}, tenant_id="demo", purpose="DEMO",
        extraction_recipe=recipe, embedding_recipe=base.manifest.embedding_recipe,
        corpus_version=base.manifest.corpus_version, normalizer_version=base.manifest.normalizer_version,
        chunking_version=base.manifest.chunking_version, tokenizer_version=base.manifest.tokenizer_version)
    snapshot = load_snapshot(path, tenant_id="demo", embedding_model_id="BAAI/bge-small-en-v1.5")
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "PARTIAL_GRAPH_STRUCTURAL_PASS",
        "snapshot_directory": str(path), "snapshot_version": snapshot.manifest.snapshot_version,
        "manifest_sha256": snapshot.manifest_sha256, "counts": snapshot.manifest.counts.model_dump(),
        "extraction_accepted_chunks": len(pilot["extractions"]), "extraction_quarantined_chunks": len(pilot["quarantined"]),
        "extraction_unprocessed_chunks": len(content.chunks) - len(pilot["extractions"]) - len(pilot["quarantined"]),
        "extraction_recipe": recipe.version, "semantic_review": "PENDING", "paid_calls": 0,
        "neo4j_roundtrip": "PENDING", "cloud_published": False,
        "note": "Pilot graph covers a subset of the indexed corpus; not a completed release or quality result."}
    (ROOT / "results/pilot_graph_snapshot.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
