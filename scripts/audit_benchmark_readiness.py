"""Inspect real pinned corpus/graph artifacts; no credentials, model or AWS access."""

import argparse
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from app.benchmark_inventory import cost_scenario, readiness_methods
from app.collections import load_collection
from app.contracts import canonical_json, sha256_bytes
from app.snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def audit(root=ROOT):
    pipeline = json.loads((root / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=1)
    counts = dict.fromkeys(("documents", "chunks", "entities", "mentions", "assertions"), 0)
    chunks, chars, proxy = set(), 0, 0
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        for key in counts:
            counts[key] += len(getattr(shard.content, key))
        for chunk in shard.content.chunks:
            chunks.add(chunk.text_sha256)
            chars += len(chunk.text)
            proxy += chunk.token_count
    if counts != collection.manifest.counts.model_dump():
        raise ValueError("BENCHMARK_COLLECTION_COUNTS_MISMATCH")
    pilot_receipt = json.loads((root / "results/pilot_graph_snapshot.json").read_bytes())
    pilot = load_snapshot(Path(pilot_receipt["snapshot_directory"]), tenant_id="demo",
                          embedding_model_id="BAAI/bge-small-en-v1.5",
                          trusted_manifest_sha256=pilot_receipt["manifest_sha256"])
    if pilot.manifest.counts.model_dump() != pilot_receipt["counts"]:
        raise ValueError("BENCHMARK_PILOT_COUNTS_MISMATCH")
    manifest_raw = (root / "benchmarks/expanded-development-v1/manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    summary_raw = (root / "benchmarks/expanded-development-v1/summary.json").read_bytes()
    # This producer hashes canonical JSON, not the indented export's file bytes.
    summary = json.loads(summary_raw)
    summary_canonical_sha = sha256_bytes(canonical_json(summary))
    if summary_canonical_sha != manifest["summary_sha256"] or (
        manifest["graph_manifest_sha256"] != pilot.manifest_sha256
    ):
        raise ValueError("BENCHMARK_PRIOR_RESULTS_HASH_MISMATCH")
    old = []
    for method in ("B0_BM25", "B1_VECTOR", "B2_HYBRID", "G1_DRAFT", "G2_DRAFT"):
        row = {"id": method}
        for language in ("en", "vi"):
            metrics = summary["languages"][language]["systems"][method]["provisional_ai_draft"]
            row[language] = {"complete": metrics["complete_evidence_count"],
                             "denominator": metrics["answerable_denominator"]}
        old.append(row)
    prices = json.loads((root / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    rate = next(row for row in prices["rates"] if row["provider"] == "DEEPSEEK")
    routes = json.loads((root / "configs/models/external-direct-v1.json").read_bytes())
    price_args = {"input_micro_per_million": rate["input_micro_usd_per_unit"],
                  "output_micro_per_million": rate["output_micro_usd_per_unit"]}
    if rate["tokens_per_unit"] != 1_000_000:
        raise ValueError("BENCHMARK_FORECAST_PRICE_UNIT_MISMATCH")
    release = json.loads((root / "results/owned-cloud/approved-release-v1/completion.json").read_bytes())
    deployed = release["status"].startswith("PASS_AWS_OWNED_RETRIEVAL_RELEASE")
    return {
        "schema_version": 1, "scope": "READINESS_NOT_COLLECTION_QUALITY",
        "checked_at": datetime.now(UTC).isoformat(),
        "collection": {"version": collection.manifest.collection_version,
                       "manifest_sha256": collection.manifest_sha256, **counts,
                       "service_groups": len(pipeline["service_counts"]),
                       "unique_chunk_texts": len(chunks), "chunk_characters_including_overlap": chars,
                       "chunk_proxy_tokens_including_overlap": proxy,
                       "graph_state": "NOT_BUILT" if not counts["assertions"] else "REQUIRES_REVIEW"},
        "methods": readiness_methods(assertion_count=counts["assertions"], recorded_cloud_release=deployed),
        "deployment_observed_at": release["at"],
        "pilot": {**pilot.manifest.counts.model_dump(), "manifest_sha256": pilot.manifest_sha256,
                  "graph_source_chunks": len({a.chunk_id for a in pilot.content.assertions}),
                  "review": "AI_DRAFT", "results": old,
                  "run_id": manifest["run_id"], "summary_canonical_sha256": summary_canonical_sha,
                  "summary_file_sha256": sha256_bytes(summary_raw)},
        "collection_quality": None, "model_calls": 0, "aws_calls": 0,
        "forecast": {"prices_version": prices["version"], "prices_expire_at": prices["expires_at"],
            "prices_source": "https://api-docs.deepseek.com/quick_start/pricing/",
            "assumption": "Extraction 2500 input / 1000 output tokens; 20% additional settled attempts. Not measured.",
            "pilot48": cost_scenario(48, input_tokens=2500, output_tokens=1000,
                                     extra_attempt_fraction=Decimal("0.2"), **price_args),
            "unique_text_full_pass": cost_scenario(len(chunks), input_tokens=2500, output_tokens=1000,
                                                  extra_attempt_fraction=Decimal("0.2"), **price_args),
            "gateway_single_extract_reservation": cost_scenario(1,
                input_tokens=routes["generation_context_ceiling"], output_tokens=2048, **price_args),
            "new_funding_authorized": False, "provider_balance": None},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("BENCHMARK_AUDIT_IMMUTABLE_CHOOSE_NEW_OUTPUT")
    report = audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("collection", "forecast", "model_calls", "aws_calls")}))


if __name__ == "__main__":
    main()
