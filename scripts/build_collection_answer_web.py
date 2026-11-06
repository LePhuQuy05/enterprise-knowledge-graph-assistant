"""Static-only UI migration candidate; existing query API/backend bytes stay pinned."""

import ast
import hashlib
import json
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from scripts.build_benchmark_web import patch_assets

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = json.loads((ROOT / "results/owned-cloud/approved-release-v1/scope-package.json").read_bytes())
    source = Path(base["package_path"])
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != base["sha256"]:
            raise ValueError("COLLECTION_WEB_BASELINE_CHANGED")
    tree = ast.parse((ROOT / "scripts/build_query_package.py").read_text(encoding="utf-8"))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "static_paths" for t in n.targets))
    paths = ast.literal_eval(assignment.value)
    assets = {route: {"body": (ROOT / "frontend/tracer" / name).read_text(encoding="utf-8"), "content_type": mime}
              for route, (name, mime) in paths.items()}
    if len(assets) != 16 or any(len(v["body"].encode()) > 512 * 1024 for v in assets.values()):
        raise ValueError("COLLECTION_WEB_ASSET_BOUNDS")
    target = ROOT / "data/build/collection316-web-v7/query-runtime.zip"
    report_path = ROOT / "results/collection316-serving/web-package-v7.json"
    if target.exists() or report_path.exists():
        raise ValueError("COLLECTION_WEB_CANDIDATE_EXISTS_KEEP_HISTORY")
    checked = patch_assets(source, target, assets)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"status": "PASS_LOCAL_STATIC_CANDIDATE_NOT_DEPLOYED", **checked,
        "package_path": str(target), "sha256": digest, "bytes": target.stat().st_size,
        "base_package_sha256": base["sha256"], "backend_unchanged": True, "aws_calls": 0, "model_calls": 0,
        "assets": {route: {"sha256": sha256_bytes(v["body"].encode()), "content_type": v["content_type"]}
                   for route, v in assets.items()}}
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("xb") as handle:
        handle.write(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "sha256", "changed_assets")}))


if __name__ == "__main__":
    main()
