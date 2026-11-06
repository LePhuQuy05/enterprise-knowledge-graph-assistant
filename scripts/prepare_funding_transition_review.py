"""Read the deployed query ZIP and prepare a minimal local compatibility patch.

Read-only AWS and a presigned code download; no uploads, changesets, deploy,
counter mutation, secrets read, or model calls. Default prints the plan only.
"""

import argparse
import ast
import base64
import copy
import difflib
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from app.contracts import canonical_json, sha256_bytes
from app.funding_policy import COMPATIBLE_AUTHORIZATION_ID, TRANSITION_PROFILE

ROOT = Path(__file__).resolve().parents[1]


def ast_outside_gateway(raw):
    tree = ast.parse(raw)
    tree.body = [n for n in tree.body if not (isinstance(n, ast.FunctionDef) and n.name == "gateway")
        and not (isinstance(n, ast.ImportFrom) and n.module == "app.funding_policy")]
    return ast.dump(tree, include_attributes=False)


def build_patch(original, *, handler_bytes, policy_bytes):
    source = zipfile.ZipFile(io.BytesIO(original))
    names = source.namelist()
    if len(names) != len(set(names)) or "app/query_handler.py" not in names:
        raise ValueError("FUNDING_RELEASE_ZIP_CONTENTS")
    for name in names:
        p = PurePosixPath(name)
        if p.is_absolute() or ".." in p.parts or "\\" in name:
            raise ValueError("FUNDING_RELEASE_ZIP_UNSAFE_PATH")
    old_handler = source.read("app/query_handler.py")
    if ast_outside_gateway(old_handler) != ast_outside_gateway(handler_bytes):
        raise ValueError("FUNDING_RELEASE_UNRELATED_HANDLER_CHANGE")
    out = io.BytesIO()
    replacements = {"app/query_handler.py": handler_bytes, "app/funding_policy.py": policy_bytes}
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for entry in source.infolist():
            target.writestr(copy.copy(entry), replacements.pop(entry.filename, source.read(entry.filename)))
        for name, raw in replacements.items():
            info = zipfile.ZipInfo(name, date_time=(2026, 10, 26, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            target.writestr(info, raw)
    patched = out.getvalue()
    with zipfile.ZipFile(io.BytesIO(patched)) as result:
        changed = sorted(name for name in set(names) | set(result.namelist()) if name not in names or source.read(name) != result.read(name))
    if changed != ["app/funding_policy.py", "app/query_handler.py"]:
        raise ValueError("FUNDING_RELEASE_PATCH_NOT_MINIMAL")
    return patched, old_handler, changed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-aws", action="store_true")
    parser.add_argument("--base-zip", type=Path, help="Reuse a local ZIP only after checking its hash against fresh GetFunction metadata")
    args = parser.parse_args()
    if not args.read_aws:
        print("PLANNED_READ_ONLY_QUERY_FUNCTION_ZIP_REVIEW_NO_AWS_WRITES")
        return
    import boto3
    from botocore.config import Config

    output = ROOT / "results/funding-transition/review-v1"
    output.mkdir(parents=True, exist_ok=True)
    if (output / "plan.json").exists():
        raise ValueError("FUNDING_RELEASE_REVIEW_ALREADY_FROZEN")
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    print("READ_ONLY_IDENTITY_CHECK", flush=True)
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("FUNDING_RELEASE_OPERATOR_IDENTITY")
    print("READ_ONLY_FUNCTION_METADATA", flush=True)
    function = session.client("lambda", config=cfg).get_function(FunctionName="kg-r72-query")
    config = function["Configuration"]
    (output / "read-only-state.json").write_bytes(canonical_json({"function_name": config["FunctionName"],
        "revision_id": config["RevisionId"], "code_sha256": config["CodeSha256"], "code_size": config["CodeSize"],
        "environment_keys": sorted(config.get("Environment", {}).get("Variables", {})), "aws_writes": 0}) + b"\n")
    if args.base_zip:
        if args.base_zip.is_symlink() or not args.base_zip.resolve().is_relative_to(ROOT) or args.base_zip.stat().st_size > 80 * 1024 * 1024:
            raise ValueError("FUNDING_RELEASE_CACHED_BASE_PATH_OR_SIZE")
        original = args.base_zip.read_bytes()
    else:
        url = function["Code"]["Location"]
        parsed = urlsplit(url)
        if parsed.scheme != "https" or not parsed.hostname or not parsed.hostname.endswith(".amazonaws.com"):
            raise ValueError("FUNDING_RELEASE_CODE_DOWNLOAD_HOST")
        print(json.dumps({"status": "DOWNLOADING_BASE_ZIP", "expected_bytes": config["CodeSize"]}), flush=True)
        started, pieces, received = time.monotonic(), [], 0
        with urllib.request.urlopen(url, timeout=15) as response:
            if response.geturl() != url:
                raise ValueError("FUNDING_RELEASE_UNEXPECTED_REDIRECT")
            while True:
                if time.monotonic() - started > 120 or received > 80 * 1024 * 1024:
                    raise ValueError("FUNDING_RELEASE_DOWNLOAD_DEADLINE_OR_SIZE")
                block = response.read(256 * 1024)
                if not block:
                    break
                pieces.append(block)
                received += len(block)
            original = b"".join(pieces)
    if len(original) > 80 * 1024 * 1024 or base64.b64encode(hashlib.sha256(original).digest()).decode() != config["CodeSha256"]:
        raise ValueError("FUNDING_RELEASE_BASE_CODE_CHECKSUM_OR_SIZE")
    new_handler, new_policy = (ROOT / "app/query_handler.py").read_bytes(), (ROOT / "app/funding_policy.py").read_bytes()
    patched, old_handler, changed = build_patch(original, handler_bytes=new_handler, policy_bytes=new_policy)
    (output / "base.zip").write_bytes(original)
    (output / "query-funding-transition.zip").write_bytes(patched)
    diff = "".join(difflib.unified_diff(old_handler.decode().splitlines(True), new_handler.decode().splitlines(True),
        fromfile="deployed/app/query_handler.py", tofile="proposed/app/query_handler.py"))
    (output / "handler.diff").write_text(diff, encoding="utf-8")
    report = {"status": "LOCAL_PACKAGE_REVIEW_READY_NOT_DEPLOYED", "checked_at": datetime.now(UTC).isoformat(),
        "function_name": config["FunctionName"], "handler": config["Handler"], "revision_id": config["RevisionId"],
        "base_code_sha256_base64": config["CodeSha256"], "base_zip_sha256": sha256_bytes(original),
        "proposed_zip_sha256": sha256_bytes(patched), "changed_zip_entries": changed,
        "proposed_profile_after_code_update": TRANSITION_PROFILE, "active_funding_cap_unchanged_micro_usd": 2000000,
        "subsequent_allocation_requires_authorization_audit_id": COMPATIBLE_AUTHORIZATION_ID,
        "future_authorized_ceiling_micro_usd": 8000000, "aws_writes": 0, "provider_calls": 0,
        "ui_auth_iam_accounts_quotas_changed": False, "owned_collection_answering_not_enabled_by_this_patch": True,
        "release_order": ["Verify current code/revision/config and all active funding consumers again",
            "Deploy only the reviewed ZIP; active cap remains 2 USD", "Enable transition profile, probe both cap states without model in native tests",
            "Verify deployed reader configuration before a separate CAS allocation using the new audit ID",
            "Do not roll back to a legacy reader at cap 8; restore cap only if settled+reserved<=2 USD, otherwise keep the compatible reader"],
        "user_release_approval_required": True}
    (output / "plan.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "function_name", "changed_zip_entries", "aws_writes", "proposed_zip_sha256")}))


if __name__ == "__main__":
    main()
