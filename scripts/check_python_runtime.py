"""Check the pinned local interpreter and native imports without AWS/model calls."""

import hashlib
import importlib
import json
import platform
import sys
from datetime import datetime, timedelta, timezone
from importlib.metadata import version
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    expected_python = (root / ".python-version").read_text(encoding="utf-8").strip()
    errors = []
    if platform.python_version() != expected_python:
        errors.append(f"Expected Python {expected_python}, got {platform.python_version()}")

    packages = {}
    imports = {
        "boto3": "boto3",
        "numpy": "numpy",
        "pydantic": "pydantic",
        "pydantic-core": "pydantic_core",
        "rank-bm25": "rank_bm25",
        "tiktoken": "tiktoken._tiktoken",
        "awscrt": "awscrt.auth",
    }
    for distribution, module in imports.items():
        try:
            importlib.import_module(module)
            packages[distribution] = version(distribution)
        except (ImportError, OSError) as exc:
            errors.append(f"{distribution}: {type(exc).__name__}: {exc}")

    if "numpy" in packages:
        import numpy as np

        scores = np.eye(2, dtype=np.float32) @ np.array([1.0, 0.0], dtype=np.float32)
        if scores.dtype != np.float32 or not np.array_equal(scores, [1.0, 0.0]):
            errors.append("NumPy float32 matrix multiplication failed")

    report = {
        "checked_at": datetime.now(timezone(timedelta(hours=7))).isoformat(timespec="seconds"),
        "status": "PASS" if not errors else "FAIL",
        "scope": "Pinned interpreter and local native imports; not Linux packaging or app tests",
        "python": platform.python_version(),
        "expected_python": expected_python,
        "platform": f"{sys.platform}/{platform.machine()}",
        "packages": packages,
        "lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "aws_api_calls": 0,
        "model_inference_calls": 0,
        "errors": errors,
    }
    output = root / "results" / "python_bootstrap.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
