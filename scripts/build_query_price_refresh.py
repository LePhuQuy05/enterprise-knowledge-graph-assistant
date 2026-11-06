"""Patch only the deployed ZIP's price path plus one verified price JSON; no AWS writes."""

import argparse
import copy
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalPriceSheet
from scripts.build_query_package import run

ROOT = Path(__file__).resolve().parents[1]
OLD_PRICE = "configs/prices/external-direct-2026-10-17.json"
NEW_PRICE = "configs/prices/external-direct-2026-10-24.json"
HANDLER = "app/query_handler.py"


def patch_archive(source, target, price_raw):
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as new:
        if len(old.namelist()) != len(set(old.namelist())) or NEW_PRICE in old.namelist():
            raise ValueError("PRICE_REFRESH_UNEXPECTED_BASE_ARCHIVE")
        original = old.read(HANDLER)
        needle, replacement = Path(OLD_PRICE).name.encode(), Path(NEW_PRICE).name.encode()
        if original.count(needle) != 1:
            raise ValueError("PRICE_REFRESH_EXPECTED_ONE_LITERAL_PATH")
        for entry in old.infolist():
            content = original.replace(needle, replacement) if entry.filename == HANDLER else old.read(entry)
            new.writestr(copy.copy(entry), content)
        new.writestr(NEW_PRICE, price_raw)
    with zipfile.ZipFile(source) as old, zipfile.ZipFile(target) as new:
        changed = [name for name in old.namelist() if old.read(name) != new.read(name)]
        added = sorted(set(new.namelist()) - set(old.namelist()))
        if changed != [HANDLER] or added != [NEW_PRICE]:
            raise ValueError("PRICE_REFRESH_ZIP_DELTA_NOT_EXACT")
        return {"changed": changed, "added": added, "removed": [], "unchanged_entries": len(old.namelist()) - 1,
            "uncompressed_bytes": sum(entry.file_size for entry in new.infolist())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "data/build/query-price-refresh-v1")
    parser.add_argument("--report", type=Path, default=ROOT / "results/workbench-cloud/query-price-refresh-v1.json")
    args = parser.parse_args(argv)
    if args.report.exists() or (args.directory / "query-price-refresh.zip").exists():
        raise ValueError("PRICE_REFRESH_BUILD_EXISTS")
    baseline = json.loads((ROOT / "results/query_package_build.json").read_bytes())
    source = Path(baseline["package_path"])
    with source.open("rb") as data:
        if hashlib.file_digest(data, "sha256").hexdigest() != baseline["sha256"]:
            raise ValueError("PRICE_REFRESH_BASE_PACKAGE_CHANGED")
    raw = (ROOT / NEW_PRICE).read_bytes()
    prices = ExternalPriceSheet.model_validate_json(raw)
    prices.rate_for("deepseek-flash", "deepseek-flash", "external-default", now=datetime.now(UTC), live=True)
    args.directory.mkdir(parents=True, exist_ok=True)
    target = args.directory / "query-price-refresh.zip"
    delta = patch_archive(source, target, raw)
    # Only public budget rows are needed to prove gateway construction offline; no secret value is copied.
    audit = json.loads((ROOT / "results/owned-answers/canary-v1/source-stage-ledger-audit-v1.json").read_bytes())
    (args.directory / "counter-fixture.json").write_bytes(canonical_json(audit["counters"]) + b"\n")
    image = json.loads((ROOT / "configs/runtime_images.json").read_bytes())["lambda_python"]
    run(["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none", "--memory", "1024m",
        "--mount", f"type=bind,source={args.directory.resolve()},target=/build",
        "--mount", f"type=bind,source={ROOT / 'scripts/check_query_price_refresh_linux.py'},target=/probe.py,readonly",
        "--entrypoint", "python", image, "/probe.py"], timeout=180)
    with target.open("rb") as data:
        digest = hashlib.file_digest(data, "sha256").hexdigest()
    report = {"status": "PASS_EXACT_PRICE_REFRESH_NATIVE", "checked_at": datetime.now(UTC).isoformat(),
        "package_path": str(target.resolve()), "package_bytes": target.stat().st_size, "sha256": digest,
        "base_package_sha256": baseline["sha256"], "delta": delta,
        "native": json.loads((args.directory / "native-report.json").read_bytes()), "price_sheet_version": prices.version,
        "price_sheet_expires_at": prices.expires_at.isoformat(), "price_sheet_sha256": sha256_bytes(raw),
        "planned_changes": ["Existing QueryFunction.Code.S3Key only; no IAM/environment/Cognito/API/limits/data changes"],
        "aws_calls": 0, "model_calls": 0, "cloud_deployed": False}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("xb") as output:
        output.write(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "sha256", "package_bytes", "price_sheet_expires_at", "aws_calls", "model_calls")}))


if __name__ == "__main__":
    main()
