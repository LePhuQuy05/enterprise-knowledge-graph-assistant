"""Private S3 snapshot readback with immutable prefixes and bounded source validation."""

import hashlib
import json
import re
from pathlib import Path

from app.contracts import DATA_FILES, MAX_SNAPSHOT_BYTES, SnapshotManifest, sha256_bytes
from app.snapshot import load_snapshot


def download_checked(client, *, bucket, key, target: Path, expected_sha256: str, expected_bytes: int):
    if target.is_symlink() or target.is_junction():
        raise ValueError("CLOUD_ASSET_UNSAFE_CACHE_PATH")
    if target.is_file():
        if target.stat().st_size != expected_bytes:
            raise ValueError("CLOUD_ASSET_CACHED_SIZE_MISMATCH")
        with target.open("rb") as stream:
            if hashlib.file_digest(stream, "sha256").hexdigest() != expected_sha256:
                raise ValueError("CLOUD_ASSET_CACHED_CHECKSUM_MISMATCH")
        return
    response = client.get_object(Bucket=bucket, Key=key)
    body = response["Body"]
    temp = target.with_suffix(target.suffix + ".partial")
    size, digest = 0, hashlib.sha256()
    try:
        if response["ContentLength"] != expected_bytes:
            raise ValueError("CLOUD_ASSET_SIZE_MISMATCH")
        target.parent.mkdir(parents=True, exist_ok=True)
        with temp.open("wb") as out:
            while chunk := body.read(64 * 1024):
                size += len(chunk)
                if size > expected_bytes:
                    raise ValueError("CLOUD_ASSET_BODY_TOO_LARGE")
                digest.update(chunk)
                out.write(chunk)
        if size != expected_bytes or digest.hexdigest() != expected_sha256:
            raise ValueError("CLOUD_ASSET_CHECKSUM_MISMATCH")
        temp.replace(target)
    finally:
        body.close()
        temp.unlink(missing_ok=True)


def load_s3_snapshot(client, *, bucket: str, prefix: str, manifest_sha256: str, manifest_bytes: int,
                     snapshot_version: str, tenant_id: str, cache_root: Path):
    if (not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) or not re.fullmatch(r"snap_[0-9a-f]{64}", snapshot_version)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", tenant_id)
            or prefix != f"tenants/{tenant_id}/snapshots/{snapshot_version}/" or not 1 <= manifest_bytes <= 64 * 1024):
        raise ValueError("CLOUD_SNAPSHOT_PREFIX_OR_MANIFEST_LIMIT")
    destination = cache_root / manifest_sha256
    destination.mkdir(parents=True, exist_ok=True)
    download_checked(client, bucket=bucket, key=prefix + "manifest.json", target=destination / "manifest.json",
                     expected_sha256=manifest_sha256, expected_bytes=manifest_bytes)
    raw = (destination / "manifest.json").read_bytes()
    if sha256_bytes(raw) != manifest_sha256:
        raise ValueError("CLOUD_SNAPSHOT_MANIFEST_CHANGED")
    manifest = SnapshotManifest.model_validate_json(raw)
    if manifest.tenant_id != tenant_id or manifest.snapshot_version != snapshot_version or manifest.purpose != "DEMO":
        raise ValueError("CLOUD_SNAPSHOT_SCOPE_OR_PURPOSE")
    if {f.name for f in manifest.files} != set(DATA_FILES) or sum(f.byte_size for f in manifest.files) > MAX_SNAPSHOT_BYTES:
        raise ValueError("CLOUD_SNAPSHOT_FILE_LIMIT")
    for row in manifest.files:
        download_checked(client, bucket=bucket, key=prefix + row.name, target=destination / row.name,
                         expected_sha256=row.sha256, expected_bytes=row.byte_size)
    return load_snapshot(destination, tenant_id=tenant_id, embedding_model_id=manifest.embedding_recipe.model_id,
                         trusted_manifest_sha256=manifest_sha256)


def read_deepseek_secret(client, *, secret_arn: str):
    from pydantic import SecretStr

    result = client.get_secret_value(SecretId=secret_arn, VersionStage="AWSCURRENT")
    raw = result.get("SecretString", "")
    if result.get("ARN") != secret_arn or not 1 <= len(raw.encode()) <= 8192:
        raise ValueError("MODEL_SECRET_IDENTITY_OR_SIZE")
    value = json.loads(raw)
    if (set(value) != {"schema_version", "provider", "api_key"} or type(value["schema_version"]) is not int
            or value["schema_version"] != 1 or value["provider"] != "DEEPSEEK"):
        raise ValueError("MODEL_SECRET_FORMAT")
    key = value["api_key"]
    if not isinstance(key, str) or not 16 <= len(key) <= 256 or any(c.isspace() for c in key) or not key.isascii():
        raise ValueError("MODEL_SECRET_KEY_FORMAT")
    return SecretStr(key)
