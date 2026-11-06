"""Prepare a static-only candidate from the verified deployed ZIP; never deploy."""

import ast
import hashlib
import json
from copy import copy
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from app.contracts import canonical_json, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
STATIC = "configs/frontend-static.json"


def patch_assets(source, target, assets):
    if target.exists():
        raise ValueError("BENCHMARK_PACKAGE_EXISTS")
    with ZipFile(source) as before:
        names = before.namelist()
        if len(names) != len(set(names)) or names.count(STATIC) != 1:
            raise ValueError("BENCHMARK_ZIP_INVENTORY_INVALID")
        prior = json.loads(before.read(STATIC))
        if not set(prior) <= set(assets):
            raise ValueError("BENCHMARK_CANNOT_REMOVE_EXISTING_ROUTES")
        target.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=6) as out:
            for info in before.infolist():
                # writestr mutates header offsets on ZipInfo; never reuse the reader's instance.
                out.writestr(copy(info), canonical_json(assets) + b"\n" if info.filename == STATIC else before.read(info))
        with ZipFile(target) as after:
            changes = [name for name in names if before.read(name) != after.read(name)]
            if after.namelist() != names or changes != [STATIC]:
                raise ValueError("BENCHMARK_NON_STATIC_CHANGE")
            size = sum(i.file_size for i in after.infolist())
            if size >= 250 * 1024 * 1024:
                raise ValueError("BENCHMARK_PACKAGE_TOO_LARGE")
    return {"changed_zip_entries": changes, "unchanged_entries": len(names) - 1,
            "uncompressed_bytes": size, "changed_assets": sorted(k for k in assets if assets[k] != prior.get(k))}


def main():
    base = json.loads((ROOT / "results/owned-cloud/approved-release-v1/scope-package.json").read_bytes())
    source = Path(base["package_path"])
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != base["sha256"]:
            raise ValueError("BENCHMARK_LIVE_BASELINE_SHA_MISMATCH")
    tree = ast.parse((ROOT / "scripts/build_query_package.py").read_text(encoding="utf-8"))
    mapping = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == "static_paths" for t in n.targets))
    paths = ast.literal_eval(mapping.value)
    assets = {route: {"body": (ROOT / "frontend/tracer" / name).read_text(encoding="utf-8"), "content_type": mime}
              for route, (name, mime) in paths.items()}
    if len(assets) != 15 or any(len(v["body"].encode()) > 512 * 1024 for v in assets.values()):
        raise ValueError("BENCHMARK_ASSET_BOUNDS_INVALID")
    target = ROOT / "data/build/benchmark-web-v2/query-runtime.zip"
    checked = patch_assets(source, target, assets)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"status": "PASS_LOCAL_STATIC_CANDIDATE_NOT_DEPLOYED", **checked,
        "base_package_sha256": base["sha256"], "package_path": str(target), "sha256": digest,
        "bytes": target.stat().st_size, "backend_unchanged": True, "model_calls": 0, "aws_calls": 0,
        "assets": {route: {"sha256": sha256_bytes(v["body"].encode()), "content_type": v["content_type"]}
                   for route, v in assets.items()}}
    path = ROOT / "results/benchmark-collection316/web-package-v2.json"
    path.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "changed_zip_entries", "changed_assets", "sha256")}))


if __name__ == "__main__":
    main()
