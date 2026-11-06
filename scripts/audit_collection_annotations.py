"""Audit source-bound annotation drafts locally; never execute retrieval or a model."""

import argparse
import json
from pathlib import Path

from app.collection_evaluation import (
    CollectionEvaluationDataset,
    CollectionEvidenceIndex,
    validate_annotation_sources,
    validate_control_scope_review,
    validate_scenario_binding,
)
from app.collections import load_collection
from app.contracts import canonical_json, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "configs/evaluation/collection316-annotations-v2.json")
    parser.add_argument("--scenarios", type=Path, default=ROOT / "configs/evaluation/collection316-scenarios-100-v2.json")
    parser.add_argument("--pipeline", type=Path, default=ROOT / "results/corpus-expansion-v2/pipeline.json")
    parser.add_argument("--scope-review", type=Path, default=ROOT / "configs/evaluation/collection316-corpus-gap-review-v1.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args(argv)
    inputs = (args.dataset, args.scenarios, args.pipeline, args.scope_review)
    if args.output.resolve() in {p.resolve() for p in inputs if p is not None}:
        parser.error("Output cannot overwrite an input artifact")
    raw = args.dataset.read_bytes()
    if len(raw) > 4 * 1024 * 1024:
        raise ValueError("ANNOTATION_DATASET_SIZE_LIMIT")
    dataset = CollectionEvaluationDataset.model_validate_json(raw)
    validate_scenario_binding(dataset, args.scenarios.read_bytes())
    pipeline = json.loads(args.pipeline.read_bytes())
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=1)
    index = CollectionEvidenceIndex.from_collection(collection)
    complete_gaps = [s for s in dataset.scenarios if s.kind == "corpus_gap"
                     and s.annotation_status == "DRAFT_COMPLETE"]
    if complete_gaps:
        if args.scope_review is None:
            raise ValueError("CORPUS_GAP_SCOPE_REVIEW_INPUT_REQUIRED")
        scope_raw = args.scope_review.read_bytes()
        if len(scope_raw) > 1024 * 1024:
            raise ValueError("CORPUS_GAP_SCOPE_REVIEW_SIZE_LIMIT")
        validate_control_scope_review(dataset, index, scope_raw)
    counts = validate_annotation_sources(dataset, index)
    pending = [s.scenario_id for s in dataset.scenarios if s.annotation_status != "DRAFT_COMPLETE"]
    result = {
        "schema_version": 1, "status": "PASS_SOURCE_INTEGRITY_DRAFT_ONLY",
        "dataset_sha256": sha256_bytes(raw), "source_manifest_sha256": index.manifest_sha256,
        "counts": counts, "pending_scenarios": pending,
        "annotation_complete": not pending, "human_gold_available": any(
            s.review_status == "HUMAN_APPROVED" for s in dataset.scenarios),
        "final_evaluation_ready": not pending and dataset.split_status == "GROUPED_FROZEN",
        "benchmark_run": False, "model_calls": 0, "aws_calls": 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(result) + b"\n")
    print(json.dumps(result))
    return 2 if args.require_complete and pending else 0


if __name__ == "__main__":
    raise SystemExit(main())
