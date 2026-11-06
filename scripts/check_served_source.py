"""Check public source provenance pins offline; this never deploys or calls models."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / "configs/releases/collection319-v2.json").read_bytes())
    checked = {"worker_sources": 0, "frontend_assets": 0}
    for group, mode in (("worker_sources", "bytes"), ("frontend_assets", "served_utf8")):
        for name, expected in manifest[group].items():
            path = ROOT / name
            if path.is_symlink() or not path.resolve().is_relative_to(ROOT) or not path.is_file():
                raise ValueError("RELEASE_SOURCE_PATH_INVALID:" + name)
            data = path.read_bytes() if mode == "bytes" else path.read_text(encoding="utf-8").encode()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("RELEASE_SOURCE_PIN_MISMATCH:" + name)
            checked[group] += 1
    print(json.dumps({"status": "PASS_SERVED_SOURCE_PINS", **checked, "aws_calls": 0,
                      "model_calls": 0, "limitation": "Source/static provenance, not live health or answer accuracy"}))


if __name__ == "__main__":
    main()
