"""Explicit public, pinned model download. No API keys, model API calls or remote Python."""

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path

from app.contracts import sha256_bytes
from app.local_assets import verify_bundle

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/models/bge-small-en-v1.5-onnx.json"
MODEL_DIRECTORY = ROOT / "data/models/bge-small-en-v1.5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Otherwise verify already prepared assets offline")
    args = parser.parse_args()
    raw_config = CONFIG.read_bytes()
    config = json.loads(raw_config)
    started = time.monotonic()
    if args.download:
        MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
        for row in config["files"]:
            path = MODEL_DIRECTORY / row["path"]
            if path.exists():
                if path.stat().st_size != row["bytes"] or sha256_bytes(path.read_bytes()) != row["sha256"]:
                    raise ValueError("BGE_EXISTING_ASSET_INVALID")
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".partial")
            digest, size = hashlib.sha256(), 0
            url = f"https://huggingface.co/{config['model_id']}/resolve/{config['revision']}/{row['path']}"
            try:
                with urllib.request.urlopen(url, timeout=45) as response, temporary.open("wb") as stream:
                    while chunk := response.read(64 * 1024):
                        size += len(chunk)
                        if size > row["bytes"]:
                            raise ValueError("BGE_DOWNLOAD_TOO_LARGE")
                        digest.update(chunk)
                        stream.write(chunk)
                if size != row["bytes"] or digest.hexdigest() != row["sha256"]:
                    raise ValueError("BGE_DOWNLOAD_INTEGRITY_INVALID")
                temporary.replace(path)
                print(json.dumps({"asset": row["path"], "bytes": size, "status": "VERIFIED"}), flush=True)
            finally:
                temporary.unlink(missing_ok=True)
    verify_bundle(MODEL_DIRECTORY, config)
    report = {"status": "PASS", "model_id": config["model_id"], "revision": config["revision"],
              "manifest_sha256": sha256_bytes(raw_config), "files": len(config["files"]),
              "total_bytes": sum(r["bytes"] for r in config["files"]),
              "seconds": round(time.monotonic() - started, 3), "paid_model_calls": 0}
    (ROOT / "results/bge_model_preparation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
