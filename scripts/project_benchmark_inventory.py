"""Export an allowlisted, public readiness record; no private paths or funding counters."""

import argparse
import json
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes


def project(raw):
    value = json.loads(raw)
    if value["scope"] != "READINESS_NOT_COLLECTION_QUALITY" or value["collection_quality"] is not None:
        raise ValueError("BENCHMARK_PROJECTION_REQUIRES_READINESS_NOT_QUALITY")
    top = {key: value[key] for key in ("schema_version", "scope", "checked_at", "deployment_observed_at",
                                      "collection_quality")}
    top["collection"] = {key: value["collection"][key] for key in (
        "version", "manifest_sha256", "documents", "chunks", "service_groups", "entities", "mentions", "assertions")}
    top["pilot"] = {key: value["pilot"][key] for key in (
        "documents", "chunks", "assertions", "review", "run_id", "summary_canonical_sha256", "summary_file_sha256")}
    top["pilot"]["results"] = [
        {"id": row["id"], **{lang: {key: row[lang][key] for key in ("complete", "denominator")}
                             for lang in ("en", "vi")}} for row in value["pilot"]["results"]]
    top["methods"] = [{key: row[key] for key in (
        "id", "label", "mechanism", "local", "cloud_retrieval", "cloud_answer", "collection_quality")}
                      for row in value["methods"]]
    top["audit_sha256"] = sha256_bytes(raw)
    return top


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(project(args.audit.read_bytes())) + b"\n")
    print(json.dumps({"public_projection": str(args.output), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
