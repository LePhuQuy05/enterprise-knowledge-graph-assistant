"""Prepare and index a coverage pilot from archived sources; no paid APIs or deployment.

Network fetching is intentionally a separate bounded command. Run this after
the archive writer finishes, so audit transactions cannot block fetch commits.
"""

import argparse
import json
import time
from pathlib import Path

from app.archive import ArchivePolicy
from app.collection_indexing import index_corpus_directories
from app.contracts import canonical_json
from app.corpus_expansion import prepare_expansion
from app.corpus_inventory import audit_archive
from app.ingestion.pipeline import _write

ROOT = Path(__file__).resolve().parents[1]


def merge_references(paths, destination: Path) -> dict:
    """Never silently replace a source version while assembling an input manifest."""
    paths = tuple(map(Path, paths))
    if destination.resolve() in {path.resolve() for path in paths}:
        raise ValueError("CANDIDATE_OUTPUT_OVERWRITES_INPUT")
    records, duplicate_observations = {}, 0
    for path in paths:
        if path.stat().st_size > 50 * 1024 * 1024:
            raise ValueError("CANDIDATE_MANIFEST_SIZE_LIMIT")
        with path.open("rb") as stream:
            for line in stream:
                if not line.strip():
                    continue
                if len(line) > 65536:
                    raise ValueError("CANDIDATE_REFERENCE_SIZE_LIMIT")
                row = json.loads(line)
                if (not isinstance(row, dict) or not isinstance(row.get("source_uri"), str)
                        or row.get("structural_validation") != "PASS"):
                    raise ValueError("CANDIDATE_REFERENCE_INVALID")
                uri = row["source_uri"]
                if uri in records:
                    previous = records[uri]
                    # Latest fetch metadata is observational; the immutable
                    # document referenced by both manifests must remain equal.
                    observational = {"archive_resource_fetched_at", "archive_resource_effective_uri"}
                    if {k: v for k, v in previous.items() if k not in observational} != {
                        k: v for k, v in row.items() if k not in observational
                    }:
                        raise ValueError("CANDIDATE_SOURCE_VERSION_CONFLICT")
                    duplicate_observations += 1
                records[uri] = row
                if len(records) > 50_000:
                    raise ValueError("CANDIDATE_MANIFEST_RECORD_LIMIT")
    raw = b"".join(canonical_json(records[uri]) + b"\n" for uri in sorted(records))
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write(destination.parent.resolve(), destination.name, raw, immutable=False)
    return {"records": len(records), "duplicate_observations": duplicate_observations,
            "output": str(destination)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-candidates", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/aws-official-archive")
    parser.add_argument("--data", type=Path, default=ROOT / "data/corpus-expansion")
    parser.add_argument("--index-root", type=Path,
                        help="Reuse an existing immutable snapshot/vector-cache directory across releases")
    parser.add_argument("--reuse-collection", type=Path,
                        help="Optional prior collection for verified exact-input vector reuse")
    parser.add_argument("--results", type=Path, default=ROOT / "results/corpus-expansion")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--per-service", type=int, default=20)
    parser.add_argument("--total-limit", type=int, default=500)
    parser.add_argument("--shard-documents", type=int, default=100)
    args = parser.parse_args()
    started = time.monotonic()
    policy = ArchivePolicy.model_validate_json(
        (ROOT / "configs/corpus/aws-official-archive-v1.json").read_bytes())
    if args.selection.stat().st_size > 2 * 1024 * 1024:
        parser.error("selection too large")
    selection = json.loads(args.selection.read_bytes())
    urls = tuple(row["url"] for row in selection["sources"])
    args.results.mkdir(parents=True, exist_ok=True)
    fresh = args.results / "selected-candidates.jsonl"
    inventory = audit_archive(args.archive, policy, candidate_output=fresh, selected_urls=urls)
    _write(args.results.resolve(), "selected-inventory.json", canonical_json(inventory), immutable=False)
    merged = merge_references([args.base_candidates, fresh], args.results / "combined-candidates.jsonl")
    print(json.dumps({"stage": "AUDITED", "scope": inventory["audit_scope"],
                      "counts": inventory["counts"], "merged": merged["records"]}), flush=True)
    preparation = prepare_expansion(
        args.archive, policy, args.results / "combined-candidates.jsonl", args.selection, args.data,
        per_service=args.per_service, total_limit=args.total_limit,
        shard_documents=args.shard_documents,
    )
    print(json.dumps({"stage": "PREPARED", "documents": preparation["selected_documents"],
                      "chunks": preparation["selected_chunks"],
                      "shortage": preparation["total_shortage"]}), flush=True)
    report = {"preparation_status": preparation["status"], "merge": merged,
              "documents": preparation["selected_documents"], "chunks": preparation["selected_chunks"],
              "service_counts": preparation["service_counts"], "paid_model_calls": 0,
              "cloud_deployments": 0, "serving_status": "NOT_CONNECTED"}
    if not args.prepare_only:
        from app.local_bge import BgeEncoder

        encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5",
                             ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
        indexed = index_corpus_directories(
            preparation["corpus_directories"], args.index_root or args.data / "index", encoder,
            reuse_collection_path=args.reuse_collection,
            progress=lambda row: print(json.dumps(row), flush=True),
        )
        _write(args.results.resolve(), "index.json", canonical_json(indexed), immutable=False)
        report.update(collection_path=indexed["collection_path"], new_vectors=indexed["new_vectors"],
                      onnx_forward_calls=encoder.inference_calls,
                      prior_input_vectors_reused=indexed["prior_input_vectors_reused"],
                      block_checkpoint_vectors_reused=indexed["block_checkpoint_vectors_reused"],
                      graph_status="NOT_BUILT", semantic_review="NOT_REVIEWED")
        from app.collection_lexical import build_global_lexical_index
        from app.collections import load_collection

        collection = load_collection(
            Path(indexed["collection_path"]), tenant_id="demo",
            embedding_model_id=encoder.recipe("document").model_id,
        )
        lexical = build_global_lexical_index(collection, args.data / "lexical")
        report["lexical_manifest_path"] = str(lexical)
        report["local_retrieval_modes"] = ["vector", "hybrid_global_bm25"]
    report["seconds"] = round(time.monotonic() - started, 3)
    _write(args.results.resolve(), "pipeline.json", canonical_json(report), immutable=False)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
