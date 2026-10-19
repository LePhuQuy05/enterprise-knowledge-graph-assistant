"""Build a Linux/amd64 Lambda ZIP from locked wheels and an explicit public source allowlist."""

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(args, *, timeout=300):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=timeout)
    if result.returncode:
        # Build output contains only public packages, not AWS environment/credentials.
        raise RuntimeError("QUERY_PACKAGE_BUILD_FAILED: " + result.stdout[-1800:] + result.stderr[-800:])
    return result.stdout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Download locked public wheels in the pinned Lambda image")
    parser.add_argument("--repack", action="store_true", help="Reuse the verified existing dependency ZIP; update approved application files offline")
    args = parser.parse_args()
    if not args.execute:
        print("Plan: export locked production wheels; isolated Docker mount; no credentials, model APIs or AWS calls.")
        return 0
    from scripts.export_dashboard import OUTPUT as DASHBOARD_OUTPUT
    from scripts.export_dashboard import build_projection

    if not DASHBOARD_OUTPUT.exists() or DASHBOARD_OUTPUT.read_bytes() != build_projection():
        raise ValueError("DASHBOARD_PROJECTION_STALE_REVIEW_EXPORT_BEFORE_BUILD")
    build = ROOT / "data/build/query-runtime-v1"
    stage = build / "source"
    for sub in ("app", "configs/models", "configs/prices", "data/tokenizer-cache"):
        (stage / sub).mkdir(parents=True, exist_ok=True)
    # Never recursively copy the repository, results/, .aws/, private/, model weights or caches.
    approved_paths = []
    for path in (ROOT / "app").rglob("*.py"):
        if path.is_symlink() or not path.resolve().is_relative_to((ROOT / "app").resolve()):
            raise ValueError("UNSAFE_SOURCE_PATH")
        target = stage / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        approved_paths.append(target)
    for name in ("configs/models/bge-small-en-v1.5-onnx.json", "configs/models/external-direct-v1.json",
                 "configs/prices/external-direct-2026-10-17.json"):
        shutil.copyfile(ROOT / name, stage / name)
        approved_paths.append(stage / name)
    static_paths = {"/": ("index.html", "text/html; charset=utf-8"),
        "/assets/style.css": ("style.css", "text/css; charset=utf-8"),
        "/assets/app.js": ("app.js", "text/javascript; charset=utf-8"),
        "/assets/auth.mjs": ("auth.mjs", "text/javascript; charset=utf-8"),
        "/assets/client.mjs": ("client.mjs", "text/javascript; charset=utf-8"),
        "/assets/ui.mjs": ("ui.mjs", "text/javascript; charset=utf-8"),
        "/assets/investigation.mjs": ("investigation.mjs", "text/javascript; charset=utf-8"),
        "/assets/evidence.mjs": ("evidence.mjs", "text/javascript; charset=utf-8"),
        "/assets/brief.mjs": ("brief.mjs", "text/javascript; charset=utf-8"),
        "/assets/dashboard.mjs": ("dashboard.mjs", "text/javascript; charset=utf-8"),
        "/assets/onboarding.mjs": ("onboarding.mjs", "text/javascript; charset=utf-8"),
        "/assets/dashboard-data.json": ("dashboard-data.json", "application/json; charset=utf-8")}
    assets = {route: {"body": (ROOT / "frontend/tracer" / name).read_text(encoding="utf-8"), "content_type": mime}
              for route, (name, mime) in static_paths.items()}
    static_target = stage / "configs/frontend-static.json"
    static_target.write_text(json.dumps(assets) + "\n", encoding="utf-8")
    approved_paths.append(static_target)
    import hashlib

    from app.contracts import sha256_bytes
    from app.ingestion.text import TOKENIZER_CACHE, TOKENIZER_SHA256, TOKENIZER_URL, ProxyTokenizer

    ProxyTokenizer()  # Explicitly fail if the local pinned asset has not been prepared.
    asset = TOKENIZER_CACHE / hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()
    if sha256_bytes(asset.read_bytes()) != TOKENIZER_SHA256:
        raise ValueError("PROXY_TOKENIZER_HASH_MISMATCH")
    shutil.copyfile(asset, stage / "data/tokenizer-cache" / asset.name)
    approved_paths.append(stage / "data/tokenizer-cache" / asset.name)
    uv = shutil.which("uv") or str(ROOT / "data/tools/uv/bin/uv.exe")
    run([str(uv), "export", "--frozen", "--no-dev", "--no-emit-project", "--format", "requirements-txt",
         "--output-file", str(build / "requirements.txt")])
    # stage_digest detects source changes; no source file timestamps enter the ZIP.
    files = {p.relative_to(stage).as_posix(): sha256_bytes(p.read_bytes()) for p in approved_paths}
    (build / "source_inventory.json").write_text(json.dumps(files, indent=2) + "\n", encoding="utf-8")
    image = json.loads((ROOT / "configs/runtime_images.json").read_bytes())["lambda_python"]
    started = datetime.now(UTC)
    if args.repack:
        prior = json.loads((ROOT / "results/query_package_build.json").read_bytes())
        if prior["uv_lock_sha256"] != sha256_bytes((ROOT / "uv.lock").read_bytes()):
            raise ValueError("DEPENDENCY_LOCK_CHANGED_FULL_REBUILD_REQUIRED")
        with (build / "query-runtime.zip").open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != prior["sha256"]:
                raise ValueError("EXISTING_PACKAGE_CHANGED")
        command = "python /builder.py --repack"
    else:
        command = ("python -m pip install --disable-pip-version-check --no-cache-dir --only-binary=:all: "
            "--no-compile --require-hashes --target /tmp/query-package -r /build/requirements.txt && python /builder.py")
    run(["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none" if args.repack else "bridge",
        "--mount", f"type=bind,source={stage},target=/source,readonly",
        "--mount", f"type=bind,source={build},target=/build",
        "--mount", f"type=bind,source={ROOT / 'scripts/package_query_linux.py'},target=/builder.py,readonly",
        "--entrypoint", "/bin/sh", image, "-c",
        command], timeout=600)
    report = json.loads((build / "package_report.json").read_bytes())
    report.update(started_at=started.isoformat(), finished_at=datetime.now(UTC).isoformat(), image=image,
        uv_lock_sha256=sha256_bytes((ROOT / "uv.lock").read_bytes()), credentials_mounted=False,
        aws_calls=0, paid_model_calls=0, package_path=str(build / "query-runtime.zip"))
    (ROOT / "results/query_package_build.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
