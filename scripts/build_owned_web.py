"""Replace only packaged public frontend assets in the exact live price-repair baseline."""

import ast
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from app.contracts import canonical_json, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]


def main():
    build = json.loads((ROOT / "results/workbench-cloud/query-price-refresh-v1.json").read_bytes())
    source = Path(build["package_path"])
    with source.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != build["sha256"]:
            raise ValueError("WEB_BASELINE_ZIP_CHANGED")
    tree = ast.parse((ROOT / "scripts/build_query_package.py").read_text(encoding="utf-8"))
    assignment = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == "static_paths" for target in n.targets))
    paths = ast.literal_eval(assignment.value)
    assets = {route: {"body": (ROOT / "frontend/tracer" / name).read_text(encoding="utf-8"), "content_type": mime}
              for route, (name, mime) in paths.items()}
    if len(assets) != 13 or any(len(a["body"].encode()) > 512 * 1024 for a in assets.values()):
        raise ValueError("WEB_ASSET_ALLOWLIST_INVALID")
    target = ROOT / "data/build/owned-web-v1/query-runtime.zip"
    if target.exists():
        raise ValueError("OWNED_WEB_BUILD_EXISTS")
    target.parent.mkdir(parents=True)
    raw = canonical_json(assets) + b"\n"
    with ZipFile(source) as baseline, ZipFile(target, "w", compression=ZIP_DEFLATED, compresslevel=6) as candidate:
        for info in baseline.infolist():
            candidate.writestr(info, raw if info.filename == "configs/frontend-static.json" else baseline.read(info.filename))
    with ZipFile(source) as baseline, ZipFile(target) as candidate:
        if baseline.namelist() != candidate.namelist():
            raise ValueError("WEB_STATIC_PATCH_INVENTORY_CHANGED")
        changes = [name for name in baseline.namelist() if baseline.read(name) != candidate.read(name)]
        if changes != ["configs/frontend-static.json"]:
            raise ValueError("WEB_PATCH_CHANGED_NON_FRONTEND_ENTRY")
        from app import tracer

        prior = tracer.STATIC_ASSETS
        try:
            tracer.STATIC_ASSETS = assets
            for route in assets:
                response = tracer.handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": route}, None)
                if response["statusCode"] != 200 or response["body"] != assets[route]["body"]:
                    raise ValueError("WEB_STATIC_ROUTE_READBACK_FAILED")
        finally:
            tracer.STATIC_ASSETS = prior
        uncompressed = sum(info.file_size for info in candidate.infolist())
        if uncompressed >= 250 * 1024 * 1024:
            raise ValueError("WEB_PACKAGE_TOO_LARGE")
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {"status": "PASS_WEB_STATIC_ONLY_PACKAGE", "checked_at": datetime.now(UTC).isoformat(),
        "package_path": str(target), "sha256": digest, "bytes": target.stat().st_size, "uncompressed_bytes": uncompressed,
        "base_package_sha256": build["sha256"], "changed_zip_entries": changes, "unchanged_entries": len(baseline.namelist()) - 1,
        "public_assets": {k: {"sha256": sha256_bytes(v["body"].encode()), "content_type": v["content_type"]} for k, v in assets.items()},
        "model_calls": 0, "aws_calls": 0, "backend_code_unchanged": True, "cloud_deployed": False}
    path = ROOT / "results/owned-cloud/web-package-v1.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "sha256", "bytes", "changed_zip_entries")}))


if __name__ == "__main__":
    main()
