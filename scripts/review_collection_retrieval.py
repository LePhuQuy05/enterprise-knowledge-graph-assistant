"""Export bounded bilingual source-review worksheets using only local BGE CPU."""

import argparse
import json
from pathlib import Path

from app.collection_lexical import load_global_lexical_index
from app.collection_review import ReviewScenarioSet, build_review_worksheet
from app.collections import load_collection
from app.contracts import canonical_json
from app.ingestion.pipeline import _write
from app.local_bge import BgeEncoder

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--lexical-manifest", type=Path, required=True)
    parser.add_argument("--scenarios", type=Path,
                        default=ROOT / "configs/corpus/aws-coverage-scenarios-v1.json")
    parser.add_argument("--tenant-id", default="demo")
    parser.add_argument("--cache-shards", type=int, default=3)
    parser.add_argument("--cache-mib", type=int, default=128)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.cache_shards <= 128 or not 1 <= args.cache_mib <= 512:
        parser.error("cache-shards must be1..128 and cache-mib1..512; these do not guarantee process RSS")
    if not args.scenarios.is_file() or args.scenarios.stat().st_size > 1024 * 1024:
        parser.error("scenario file missing or larger than 1 MiB")
    scenarios = ReviewScenarioSet.model_validate_json(args.scenarios.read_bytes())
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5",
                         ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    document_recipe, query_recipe = encoder.recipe("document"), encoder.recipe("query")
    collection = load_collection(
        args.collection, tenant_id=args.tenant_id, embedding_model_id=document_recipe.model_id,
        cache_shards=args.cache_shards, cache_bytes=args.cache_mib * 1024 * 1024,
    )
    if collection.manifest.embedding_recipe != document_recipe:
        parser.error("local encoder assets/runtime do not match the collection recipe")
    vectors = {}
    with load_global_lexical_index(args.lexical_manifest, collection) as lexical:
        # Reject all overlong queries before the first local inference, without truncation.
        for case in scenarios.cases:
            for language in ("en", "vi"):
                if encoder.count(getattr(case, language), role="query") > 512:
                    parser.error(f"query exceeds 512 tokens: {case.id}:{language}")
        for case in scenarios.cases:
            for language in ("en", "vi"):
                vectors[f"{case.id}:{language}"] = encoder.encode(
                    [getattr(case, language)], role="query", batch_size=1,
                )[0]
        report = build_review_worksheet(
            collection, lexical, scenarios, vectors, tenant_id=args.tenant_id,
            document_recipe=document_recipe, query_recipe=query_recipe,
        )
    report["local_query_onnx_calls"] = encoder.inference_calls
    report["cache_limits"] = {"shards": args.cache_shards, "serialized_bytes": args.cache_mib * 1024 * 1024}
    raw = canonical_json(report) + b"\n"
    if len(raw) > 64 * 1024 * 1024:
        parser.error("worksheet exceeds 64 MiB output limit")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _write(args.output.parent.resolve(), args.output.name, raw, immutable=False)
    print(json.dumps({"output": str(args.output), "queries": report["query_count"],
                      "rankings": report["ranking_count"], "human_reviewed_queries": 0,
                      "contains_answers": False, "paid_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
