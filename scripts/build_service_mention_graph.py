"""Build a separate unpublished mention-only snapshot from the full indexed corpus."""

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.contracts import sha256_bytes
from app.extraction import AliasDictionary
from app.service_mentions import derive_service_mentions
from app.snapshot import build_snapshot, load_snapshot

ROOT = Path(__file__).resolve().parents[1]
ALIAS_PATH = ROOT / "configs/extraction/aws-approved-aliases-v1.json"
RECEIPT = ROOT / "results/service_mention_snapshot.json"


def main():
    source = json.loads((ROOT / "results/bge_baseline.json").read_bytes())
    base = load_snapshot(Path(source["snapshot_directory"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=source["manifest_sha256"])
    build = derive_service_mentions(base, AliasDictionary.load(ALIAS_PATH))
    manifest = base.manifest
    directory = build_snapshot(ROOT / "data/snapshots", build.content,
        {key: base.vectors[i] for i, key in enumerate(base.chunk_ids)}, tenant_id=manifest.tenant_id,
        purpose=manifest.purpose, extraction_recipe=build.recipe, embedding_recipe=manifest.embedding_recipe,
        corpus_version=manifest.corpus_version, normalizer_version=manifest.normalizer_version,
        chunking_version=manifest.chunking_version, tokenizer_version=manifest.tokenizer_version)
    snapshot = load_snapshot(directory, tenant_id="demo", embedding_model_id="BAAI/bge-small-en-v1.5")
    if (snapshot.content.documents != base.content.documents or snapshot.content.chunks != base.content.chunks
            or not np.array_equal(base.vectors, snapshot.vectors) or snapshot.content.assertions):
        raise ValueError("MENTION_SNAPSHOT_SOURCE_VECTOR_OR_ASSERTION_MISMATCH")
    by_entity = Counter(m.entity_id for m in snapshot.content.mentions)
    report = {"status": "PASS_UNPUBLISHED_MENTION_ONLY_SNAPSHOT", "checked_at": datetime.now(UTC).isoformat(),
        "source_manifest_sha256": base.manifest_sha256, "snapshot_directory": str(directory),
        "snapshot_version": snapshot.manifest.snapshot_version, "manifest_sha256": snapshot.manifest_sha256,
        "alias_file_sha256": sha256_bytes(ALIAS_PATH.read_bytes()), "recipe": build.recipe.model_dump(mode="json"),
        "counts": snapshot.manifest.counts.model_dump(mode="json"), "inspected_chunks": build.inspected_chunks,
        "covered_chunks": len(build.covered_chunk_ids), "uncovered_chunks": build.inspected_chunks - len(build.covered_chunk_ids),
        "per_service_mentions": {snapshot.entities[key].canonical_name: count for key, count in by_entity.items()},
        "source_records_and_vectors_equal_baseline": True, "model_api_calls": 0, "aws_calls": 0,
        "cloud_published": False, "raw_pilot_graph_modified": False,
        "semantic_relation_precision": None, "scope": "Literal service occurrences, not relation extraction or human validation."}
    if RECEIPT.exists():
        previous = json.loads(RECEIPT.read_bytes())
        if previous["manifest_sha256"] != report["manifest_sha256"]:
            raise ValueError("EXISTING_MENTION_SNAPSHOT_CHANGED_CREATE_NEW_VERSION")
    else:
        RECEIPT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k:report[k] for k in ("status", "counts", "covered_chunks", "uncovered_chunks",
                                          "model_api_calls", "aws_calls", "cloud_published")}))


if __name__ == "__main__":
    main()
