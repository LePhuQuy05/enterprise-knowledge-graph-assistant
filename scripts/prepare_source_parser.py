"""Prepare a pinned, isolated, local-only syntax parser without changing app deps."""

import json
import shutil
import subprocess
import urllib.request
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/tools/source-parser"
SPACY = "3.8.18"
MODEL = "en_core_web_sm-3.8.0-py3-none-any.whl"
MODEL_SHA = "1932429db727d4bff3deed6b34cfc05df17794f4a52eeb26cf8928f7c1a0fb85"
URL = "https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/" + MODEL


def main():
    uv = shutil.which("uv")
    if not uv:
        raise ValueError("SOURCE_PARSER_UV_REQUIRED")
    BASE.mkdir(parents=True, exist_ok=True)
    wheel = BASE / MODEL
    if not wheel.exists():
        request = urllib.request.Request(URL, headers={"User-Agent": "kg-source-syntax-local/1"})
        with urllib.request.urlopen(request, timeout=60) as response:
            raw = response.read(20000001)
        if len(raw) > 20000000 or sha256_bytes(raw) != MODEL_SHA:
            raise ValueError("SOURCE_PARSER_MODEL_BOUND_OR_CHECKSUM")
        wheel.write_bytes(raw)
    if sha256_bytes(wheel.read_bytes()) != MODEL_SHA:
        raise ValueError("SOURCE_PARSER_MODEL_CHECKSUM")
    python = BASE / ".venv/Scripts/python.exe"
    if not python.exists():
        subprocess.run([uv, "venv", "--python", "3.12", str(BASE / ".venv")], check=True)
    subprocess.run([uv, "pip", "install", "--python", str(python), "spacy==" + SPACY, str(wheel)], check=True)
    freeze = subprocess.run([uv, "pip", "freeze", "--python", str(python)], check=True, capture_output=True, text=True).stdout
    (BASE / "installed-requirements.txt").write_text(freeze, encoding="utf-8")
    result = {"version": "source-parser-isolated-v1", "spacy_version": SPACY, "model": "en_core_web_sm", "model_version": "3.8.0",
        "model_wheel_sha256": MODEL_SHA, "model_url": URL, "upstream_release": "https://github.com/explosion/spacy-models/releases/tag/en_core_web_sm-3.8.0",
        "installed_requirements_sha256": sha256_bytes(freeze.encode()), "python": str(python.relative_to(ROOT)),
        "app_dependencies_changed": False, "cloud_dependency_added": False}
    (BASE / "installation.json").write_bytes(canonical_json(result) + b"\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
