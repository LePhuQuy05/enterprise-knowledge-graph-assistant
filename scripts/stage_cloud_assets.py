"""Stage immutable public corpus/model/runtime artifacts; no release pointer or inference."""

import argparse
import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.contracts import sha256_bytes
from app.local_assets import verify_bundle
from app.snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
BUCKET = "kg-r72-foundation-artifacts-oqar3e2yyr8z"


def plan_assets():
    baseline = json.loads((ROOT / "results/bge_baseline.json").read_bytes())
    snapshot_dir = Path(baseline["snapshot_directory"])
    snapshot = load_snapshot(snapshot_dir, tenant_id="demo", embedding_model_id="BAAI/bge-small-en-v1.5",
                             trusted_manifest_sha256=baseline["manifest_sha256"])
    model_path = ROOT / "configs/models/bge-small-en-v1.5-onnx.json"
    model = json.loads(model_path.read_bytes())
    verify_bundle(ROOT / "data/models/bge-small-en-v1.5", model)
    model_sha = sha256_bytes(model_path.read_bytes())
    build = json.loads((ROOT / "results/query_package_build.json").read_bytes())
    package = Path(build["package_path"])
    with package.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != build["sha256"]:
            raise ValueError("PACKAGE_HASH_CHANGED")
    prefix = f"tenants/demo/snapshots/{snapshot.manifest.snapshot_version}/"
    rows = [{"kind": "MODEL", "key": f"models/bge-small-en-v1.5/{model_sha}/" + f["path"],
             "path": str(ROOT / "data/models/bge-small-en-v1.5" / f["path"]), "bytes": f["bytes"], "sha256": f["sha256"]}
            for f in model["files"]]
    rows.extend({"kind": "SNAPSHOT", "key": prefix + name, "path": str(snapshot_dir / name),
                 "bytes": (snapshot_dir / name).stat().st_size, "sha256": sha256_bytes((snapshot_dir / name).read_bytes())}
                for name in ("manifest.json", *(f.name for f in snapshot.manifest.files)))
    package_key = f"packages/query/{build['sha256']}.zip"
    rows.append({"kind": "PACKAGE", "key": package_key, "path": str(package), "bytes": package.stat().st_size,
                 "sha256": build["sha256"]})
    config = {"schema_version": 1, "bucket": BUCKET, "tenant_id": "demo", "model_manifest_sha256": model_sha,
        "snapshot_version": snapshot.manifest.snapshot_version, "snapshot_manifest_sha256": baseline["manifest_sha256"],
        "snapshot_manifest_bytes": (snapshot_dir / "manifest.json").stat().st_size, "snapshot_prefix": prefix,
        "probe_question": baseline["retrieval_smoke"][0]["question"],
        "expected_top5": baseline["retrieval_smoke"][0]["top5_chunk_ids"]}
    return {"checked_at": datetime.now(UTC).isoformat(), "status": "PLANNED", "objects": rows,
            "runtime_config": config, "package_key": package_key,
            "total_bytes": sum(r["bytes"] for r in rows), "active_release_pointer_changed": False,
            "paid_model_calls": 0, "public_access": False}


def upload_and_verify(client, row):
    try:
        with Path(row["path"]).open("rb") as stream:
            client.put_object(Bucket=BUCKET, Key=row["key"], Body=stream, ContentLength=row["bytes"],
                ServerSideEncryption="AES256", Metadata={"sha256": row["sha256"]}, IfNoneMatch="*",
                ChecksumSHA256=base64.b64encode(bytes.fromhex(row["sha256"])).decode())
        result = "UPLOADED"
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"PreconditionFailed", "412"}:
            raise
        result = "EXISTING_VERIFIED"
    response = client.get_object(Bucket=BUCKET, Key=row["key"])
    body, size, digest = response["Body"], 0, hashlib.sha256()
    try:
        if response["ContentLength"] != row["bytes"] or response.get("ServerSideEncryption") != "AES256":
            raise ValueError("UPLOADED_ASSET_LENGTH_OR_ENCRYPTION")
        while data := body.read(1024 * 1024):
            size += len(data)
            if size > row["bytes"]:
                raise ValueError("UPLOADED_ASSET_OVERRUN")
            digest.update(data)
        if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
            raise ValueError("UPLOADED_ASSET_CHECKSUM")
    finally:
        body.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    report = plan_assets()
    path = ROOT / "results/cloud_asset_staging.json"
    if args.execute:
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=15)
        if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
            raise ValueError("DEVELOPER_IDENTITY_REQUIRED")
        s3 = session.client("s3", config=cfg)
        session.client("cloudformation", config=cfg).describe_stacks(StackName="kg-r72-foundation")
        report["uploads"] = []
        for row in report["objects"]:
            state = upload_and_verify(s3, row)
            report["uploads"].append({"key": row["key"], "sha256": row["sha256"], "status": state})
            path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"stage": row["kind"], "verified_objects": len(report["uploads"])}), flush=True)
        report["status"] = "STAGED_READBACK_PASS"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "total_bytes", "paid_model_calls", "active_release_pointer_changed")}, indent=2))


if __name__ == "__main__":
    main()
