"""Fetch an immutable, checksum-verified model bundle into Lambda /tmp before loading."""

import hashlib
import json
from pathlib import Path

from app.contracts import sha256_bytes
from app.local_assets import verify_bundle


def ensure_bge_bundle(client, *, bucket: str, manifest_path: Path, cache_root: Path) -> Path:
    """Trusted deployment manifest only. IAM must restrict S3 reads to this model prefix."""
    raw_manifest = manifest_path.read_bytes()
    config = json.loads(raw_manifest)
    manifest_hash = sha256_bytes(raw_manifest)
    approved = {"onnx/model.onnx", "tokenizer.json", "config.json", "tokenizer_config.json",
                "special_tokens_map.json", "1_Pooling/config.json", "sentence_bert_config.json", "README.md"}
    if (len(config["files"]) != 8 or {row["path"] for row in config["files"]} != approved
            or config["model_id"] != "BAAI/bge-small-en-v1.5"
            or config["revision"] != "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
            or sum(row["bytes"] for row in config["files"]) > 140 * 1024 * 1024):
        raise ValueError("BGE_STORE_MANIFEST_INVALID")
    destination = cache_root / manifest_hash
    destination.mkdir(parents=True, exist_ok=True)
    prefix = "models/bge-small-en-v1.5/" + manifest_hash + "/"
    for row in config["files"]:
        path = destination / row["path"]
        if path.exists():
            continue  # Whole-bundle verification below rejects corrupt cached files.
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.resolve().is_relative_to(destination.resolve()):
            raise ValueError("BGE_STORE_PATH_INVALID")
        response = client.get_object(Bucket=bucket, Key=prefix + row["path"])
        body = response["Body"]
        temporary = path.with_suffix(path.suffix + ".partial")
        digest, size = hashlib.sha256(), 0
        try:
            if response["ContentLength"] != row["bytes"]:
                raise ValueError("BGE_STORE_CONTENT_LENGTH_INVALID")
            with temporary.open("wb") as stream:
                while chunk := body.read(64 * 1024):
                    size += len(chunk)
                    if size > row["bytes"]:
                        raise ValueError("BGE_STORE_BODY_TOO_LARGE")
                    digest.update(chunk)
                    stream.write(chunk)
            if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                raise ValueError("BGE_STORE_CHECKSUM_INVALID")
            temporary.replace(path)
        finally:
            body.close()
            temporary.unlink(missing_ok=True)
    verify_bundle(destination, config)
    return destination
