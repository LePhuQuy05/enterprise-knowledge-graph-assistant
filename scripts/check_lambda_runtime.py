"""Run inside the pinned Lambda Python image with Linux-built dependencies."""

import importlib
import json
import os
import platform
import sys
from pathlib import Path

DEPENDENCIES = Path(os.environ.get("KG_RUNTIME_DEPS_DIR", "/asset/python"))
sys.path.insert(0, str(DEPENDENCIES))
sys.path.insert(0, "/workspace")

from app.fixtures import FIXTURE_EMBEDDING_MODEL, FIXTURE_TENANT, build_fixture  # noqa: E402
from app.snapshot import load_snapshot  # noqa: E402


def main() -> int:
    versions = {}
    for name in ("boto3", "numpy", "pydantic", "pydantic_core", "tiktoken", "rank_bm25"):
        module = importlib.import_module(name)
        versions[name] = getattr(module, "__version__", "not_declared")
    path = build_fixture(Path("/asset/fixture"))
    loaded = load_snapshot(path, tenant_id=FIXTURE_TENANT,
                           embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    size = sum(p.stat().st_size for p in DEPENDENCIES.rglob("*") if p.is_file())
    report = dict(status="PASS" if size < 250 * 1024 * 1024 else "FAIL",
                  platform=platform.platform(), machine=platform.machine(), python=sys.version,
                  versions=versions, dependency_bytes=size, dependency_limit_bytes=250 * 1024 * 1024,
                  snapshot_version=loaded.manifest.snapshot_version,
                  fixture_counts=loaded.manifest.counts.model_dump(), aws_calls=0,
                  limitation="Dependency/import smoke, not a Lambda handler invocation or deploy")
    Path("/asset/report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
