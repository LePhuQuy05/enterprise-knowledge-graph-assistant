"""Append current prices to the reviewed static ZIP; preserve funded legacy logic."""

import argparse
import copy
import hashlib
import json
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json
from app.external_gateway import ExternalPriceSheet

ROOT = Path(__file__).resolve().parents[1]
OLD_PRICE = "configs/prices/external-direct-2026-10-24.json"
NEW_PRICE = "configs/prices/external-direct-2026-11-05.json"
HANDLER = "app/query_handler.py"


def patch_archive(source, target, price_raw):
    prices = ExternalPriceSheet.model_validate_json(price_raw)
    prices.rate_for("deepseek-flash", "deepseek-flash", "external-default", now=datetime.now(UTC), live=True)
    with zipfile.ZipFile(source) as before:
        names = before.namelist()
        if len(names) != len(set(names)) or NEW_PRICE in names:
            raise ValueError("SCOPED_PRICE_UNEXPECTED_ARCHIVE")
        if any(Path(n).is_absolute() or ".." in Path(n).parts or "\\" in n for n in names):
            raise ValueError("SCOPED_PRICE_UNSAFE_ARCHIVE")
        raw = before.read(HANDLER)
        needle, replacement = Path(OLD_PRICE).name.encode(), Path(NEW_PRICE).name.encode()
        if raw.count(needle) != 1 or replacement in raw:
            raise ValueError("SCOPED_PRICE_ONE_LITERAL_REQUIRED")
        if sum(e.file_size for e in before.infolist()) + len(price_raw) >= 250*1024*1024:
            raise ValueError("SCOPED_PRICE_PACKAGE_BOUND")
        with zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED) as after:
            for entry in before.infolist():
                after.writestr(copy.copy(entry), raw.replace(needle, replacement) if entry.filename == HANDLER else before.read(entry))
            entry = zipfile.ZipInfo(NEW_PRICE, date_time=(1980, 1, 1, 0, 0, 0))
            entry.create_system, entry.external_attr, entry.compress_type = 3, 0o644 << 16, zipfile.ZIP_DEFLATED
            after.writestr(entry, price_raw)
    with zipfile.ZipFile(source) as before, zipfile.ZipFile(target) as after:
        changed = sorted(n for n in before.namelist() if before.read(n) != after.read(n))
        added = sorted(set(after.namelist())-set(before.namelist()))
        if changed != [HANDLER] or added != [NEW_PRICE]:
            raise ValueError("SCOPED_PRICE_DELTA_NOT_EXACT")
        return {"changed_from_static_zip": changed, "added_from_static_zip": added, "removed": [],
                "uncompressed_bytes": sum(e.file_size for e in after.infolist())}


def build(out):
    out = Path(out).resolve()
    if out.exists() or not out.is_relative_to(ROOT/"results"):
        raise ValueError("SCOPED_PRICE_NEW_RESULTS_DIRECTORY_REQUIRED")
    static = ROOT/"results/scoped-ui319-2026-11-05-v1/web-candidate-v2"
    source = static/"scoped-web-static.zip"
    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    report = json.loads((static/"report.json").read_bytes())
    if sha(source) != report["package_sha256"]:
        raise ValueError("SCOPED_PRICE_STATIC_BASE_DRIFT")
    out.mkdir(parents=True)
    target = out/"scoped-web-current-rates.zip"
    raw = (ROOT/NEW_PRICE).read_bytes()
    delta = patch_archive(source, target, raw)
    record = {"status": "LOCAL_COMBINED_WEB_CANDIDATE_NEEDS_NATIVE_CHECK_AND_APPROVAL",
        "checked_at": datetime.now(UTC).isoformat(), "base_static_sha256": report["package_sha256"],
        "live_funded_baseline_sha256": report["base_package_sha256"], "package_sha256": sha(target),
        "package_bytes": target.stat().st_size, "price_sha256": hashlib.sha256(raw).hexdigest(), **delta,
        "delta_from_live_baseline": ["configs/frontend-static.json", HANDLER, NEW_PRICE],
        "funding_auth_history_limits_unchanged": True, "old_rate_assets_preserved": True,
        "model_calls": 0, "aws_calls": 0, "deployed": False, "can_execute_without_approval": False}
    (out/"report.json").write_bytes(canonical_json(record))
    print(json.dumps({k: record[k] for k in ("status", "package_sha256", "delta_from_live_baseline", "deployed")}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    build(parser.parse_args().out)
