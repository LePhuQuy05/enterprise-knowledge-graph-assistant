"""Prepare public, portable workbench assets locally; no uploads or model calls."""

import argparse
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.workbench_assets import (
    DISK_MARGIN_BYTES,
    materialize_workbench_assets,
    plan_workbench_assets,
)

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Copy only verified allowlisted files into a new bundle directory")
    parser.add_argument("--directory", type=Path, default=ROOT / "data/build/workbench-assets-v1")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.report.exists():
        raise ValueError("WORKBENCH_BUILD_REPORT_EXISTS")
    pipeline = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    bundle, sources = plan_workbench_assets(Path(pipeline["collection_path"]), Path(pipeline["lexical_manifest_path"]),
        ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json",
        ROOT / "data/models/ms-marco-MiniLM-L2-v2", ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json")
    target = args.directory / bundle.version
    args.directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(args.directory).free < bundle.total_bytes + DISK_MARGIN_BYTES:
        raise ValueError("WORKBENCH_BUILD_DISK_LIMIT")
    manifest = materialize_workbench_assets(target, bundle, sources) if args.execute else None
    report = {"status": "LOCAL_BUNDLE_VERIFIED" if manifest else "PLANNED_LOCAL_ONLY", "checked_at": datetime.now(UTC).isoformat(),
        "bundle_version": bundle.version, "bundle_directory": str(target), "total_bytes": bundle.total_bytes,
        "files": len(bundle.files), "source_collection_manifest_sha256": bundle.collection_manifest_sha256,
        "bundle_manifest_sha256": sha256_bytes(manifest.read_bytes()) if manifest else None,
        "bundle_manifest_bytes": manifest.stat().st_size if manifest else None,
        "documents": pipeline["documents"], "chunks": pipeline["chunks"], "service_groups": len(pipeline["service_counts"]),
        "source_paths": {name: str(path) for name, path in sources.items()},
        "aws_calls": 0, "model_calls": 0, "cloud_deployed": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("xb") as output:
        output.write(canonical_json(report) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "source_paths"}))


if __name__ == "__main__":
    main()
