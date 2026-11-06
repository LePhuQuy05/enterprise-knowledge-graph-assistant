"""Build a local static-only ZIP. A live baseline match is a separate gate."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

from app.contracts import canonical_json, sha256_bytes
from scripts.build_benchmark_web import patch_assets

ROOT = Path(__file__).resolve().parents[1]


def build(out):
    out = Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT/"results"):
        raise ValueError("SCOPED_WEB_NEW_RESULTS_DIRECTORY_REQUIRED")
    base = ROOT/"results/funding-transition/review-v1/query-funding-transition.zip"
    recorded = json.loads((ROOT/"results/scoped-runtime-linux-2026-11-05-v3/stage.json").read_bytes())
    with base.open("rb") as stream:
        baseline_sha = hashlib.file_digest(stream, "sha256").hexdigest()
    if baseline_sha != recorded["base_package_sha256"]:
        raise ValueError("SCOPED_WEB_RECORDED_BASELINE_CHANGED")
    tree = ast.parse((ROOT/"scripts/build_query_package.py").read_text(encoding="utf-8"))
    assignment = next(node for node in ast.walk(tree) if isinstance(node, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "static_paths" for target in node.targets))
    routes = ast.literal_eval(assignment.value)
    routes.update({f"/assets/benchmark316-cases/aws100-{i:03}.json": (f"benchmark316-cases/aws100-{i:03}.json", "application/json; charset=utf-8") for i in range(1, 101)})
    # Keep any historical public route that is not replaced by the new UI.
    with ZipFile(base) as archive:
        assets = json.loads(archive.read("configs/frontend-static.json"))
    for route, (name, mime) in routes.items():
        path = (ROOT/"frontend/tracer"/name).resolve()
        if not path.is_relative_to(ROOT/"frontend/tracer") or path.is_symlink():
            raise ValueError("SCOPED_WEB_UNSAFE_SOURCE")
        assets[route] = {"body": path.read_text(encoding="utf-8"), "content_type": mime}
    if len(assets) > 200 or any(len(v["body"].encode()) > 512*1024 for v in assets.values()):
        raise ValueError("SCOPED_WEB_ASSET_BOUND")
    out.mkdir(parents=True)
    target = out/"scoped-web-static.zip"
    changed = patch_assets(base, target, assets)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"status": "LOCAL_STATIC_CANDIDATE_NEEDS_LIVE_BASELINE_MATCH", "base_package_sha256": baseline_sha,
        "package_sha256": digest, "package_bytes": target.stat().st_size, **changed,
        "backend_code_unchanged": True, "assets": {route: {"sha256": sha256_bytes(v["body"].encode()), "bytes": len(v["body"].encode()),
            "content_type": v["content_type"]} for route, v in sorted(assets.items())},
        "feature_flag_default": False, "environment_changed": False, "native_answer_canary_passed": False,
        "live_baseline_verified": False, "can_deploy": False, "aws_calls": 0, "provider_calls": 0, "cloud_deployed": False}
    (out/"report.json").write_bytes(canonical_json(report))
    print(json.dumps({k: report[k] for k in ("status", "package_sha256", "package_bytes", "changed_zip_entries", "can_deploy")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
