"""Bounded checks for the pinned public BGE model bundle."""

import hashlib
from pathlib import Path


def verify_bundle(directory: Path, config: dict) -> None:
    approved = {"onnx/model.onnx", "tokenizer.json", "config.json", "tokenizer_config.json",
                "special_tokens_map.json", "1_Pooling/config.json", "sentence_bert_config.json", "README.md"}
    if ({r["path"] for r in config["files"]} != approved or len(config["files"]) != len(approved)
            or config["model_id"] != "BAAI/bge-small-en-v1.5"
            or config["revision"] != "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"):
        raise ValueError("BGE_MANIFEST_INVALID")
    for row in config["files"]:
        path = directory / row["path"]
        if (not path.is_file() or path.is_symlink() or not path.resolve().is_relative_to(directory.resolve())
                or not 1 <= row["bytes"] <= 140 * 1024 * 1024 or path.stat().st_size != row["bytes"]):
            raise ValueError("BGE_ASSET_MISSING_OR_SIZE_INVALID")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != row["sha256"]:
            raise ValueError("BGE_ASSET_CHECKSUM_INVALID")
