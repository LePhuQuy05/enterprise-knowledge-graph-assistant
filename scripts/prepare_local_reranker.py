"""Explicit bounded download of a pinned public reranker; never called by serving code."""

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json

ROOT = Path(__file__).resolve().parents[1]
MODEL = "cross-encoder/ms-marco-MiniLM-L2-v2"
REVISION = "1b5cd67b15209f24824c50370e0397743aa9b787"
FILES = ("onnx/model.onnx", "tokenizer.json", "config.json", "tokenizer_config.json",
         "special_tokens_map.json", "README.md", "vocab.txt")
ONNX_SHA = "97ae94c94faf17d97398a7af38e44ee0ef326eff99ece940423b187e885eae6a"


def validate_file(path, row):
    if path.is_symlink() or not path.is_file() or path.stat().st_size != row["size"]:
        raise ValueError("RERANKER_DOWNLOAD_SIZE_OR_PATH")
    raw = path.read_bytes()
    if "lfs" in row:
        if hashlib.sha256(raw).hexdigest() != row["lfs"]["sha256"]:
            raise ValueError("RERANKER_DOWNLOAD_LFS_HASH")
    elif hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest() != row["blobId"]:
        raise ValueError("RERANKER_DOWNLOAD_GIT_BLOB_HASH")
    return {"path": row["rfilename"], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-reference", action="store_true", help="Also fetch local safetensors for independent parity")
    args = parser.parse_args(argv)
    directory = ROOT / "data/models/ms-marco-MiniLM-L2-v2"
    directory.mkdir(parents=True, exist_ok=True)
    if directory.is_symlink() or directory.is_junction() or shutil.disk_usage(directory).free < 512 * 1024 * 1024:
        raise ValueError("RERANKER_DOWNLOAD_DIRECTORY_OR_FREE_SPACE")
    api = f"https://huggingface.co/api/models/{MODEL}/revision/{REVISION}?blobs=true"
    with urllib.request.urlopen(api, timeout=30) as response:
        raw = response.read(1024 * 1024 + 1)
    if len(raw) > 1024 * 1024:
        raise ValueError("RERANKER_DOWNLOAD_METADATA_BOUND")
    metadata = json.loads(raw)
    if metadata["sha"] != REVISION or metadata["cardData"]["license"] != "apache-2.0":
        raise ValueError("RERANKER_DOWNLOAD_REVISION_OR_LICENSE")
    indexed = {row["rfilename"]: row for row in metadata["siblings"]}
    if indexed["onnx/model.onnx"]["lfs"]["sha256"] != ONNX_SHA:
        raise ValueError("RERANKER_DOWNLOAD_ONNX_PIN")
    selected = FILES + (("model.safetensors",) if args.with_reference else ())
    rows, receipt_rows = [], []
    for name in selected:
        row = indexed[name]
        if not 1 <= row["size"] <= 70 * 1024 * 1024:
            raise ValueError("RERANKER_DOWNLOAD_FILE_BOUND")
        target = directory / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.resolve().is_relative_to(directory.resolve()) or target.parent.is_symlink() or target.parent.is_junction():
            raise ValueError("RERANKER_DOWNLOAD_PATH_BOUND")
        reused = target.exists()
        if not reused:
            temp = target.with_name(target.name + "." + uuid.uuid4().hex + ".part")
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}"
            received = 0
            with urllib.request.urlopen(url, timeout=30) as response, temp.open("xb") as stream:
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > row["size"]:
                        raise ValueError("RERANKER_DOWNLOAD_RESPONSE_BOUND")
                    stream.write(chunk)
            validate_file(temp, row)
            os.replace(temp, target)
        verified = validate_file(target, row)
        if name in FILES:
            rows.append(verified)
        receipt_rows.append({**verified, "reused_existing": reused})
    config = {"version": "ms-marco-minilm-l2-v2-onnx-fp32-v1", "model_id": MODEL,
              "revision": REVISION, "max_tokens": 512, "license": "Apache-2.0",
              "license_url": "https://www.apache.org/licenses/LICENSE-2.0",
              "model_card_url": f"https://huggingface.co/{MODEL}/blob/{REVISION}/README.md",
              "files": rows}
    config_path = ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json"
    content = canonical_json(config) + b"\n"
    if config_path.exists() and config_path.read_bytes() != content:
        raise ValueError("RERANKER_DOWNLOAD_CONFIG_IMMUTABLE")
    config_path.write_bytes(content)
    receipt = {"checked_at": datetime.now(UTC).isoformat(), "model_id": MODEL, "revision": REVISION,
               "files": receipt_rows, "artifact_manifest_sha256": hashlib.sha256(content).hexdigest(),
               "reference_weights_requested": args.with_reference, "model_api_calls": 0, "aws_calls": 0}
    output = ROOT / "results/reranker/model-download.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(canonical_json(receipt) + b"\n")
    print(json.dumps({"model": MODEL, "revision": REVISION, "files": len(receipt_rows),
                      "bytes": sum(row["bytes"] for row in receipt_rows), "config": str(config_path)}))


if __name__ == "__main__":
    main()
