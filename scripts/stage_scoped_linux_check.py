"""Stage source + existing Linux dependencies, without credentials or network."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.ingestion.text import TOKENIZER_CACHE, TOKENIZER_SHA256, TOKENIZER_URL

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/scoped-runtime-linux-2026-11-05-v3"
BASE = ROOT / "results/funding-transition/review-v1/query-funding-transition.zip"


def stage(out, reuse_dependencies=None):
    out = Path(out).resolve()
    if not out.is_relative_to(ROOT/"results") or out.exists():
        raise ValueError("LINUX_STAGE_EXISTS_USE_THE_RECORDED_INPUTS")
    out.mkdir(parents=True)
    source = out / "source"
    deps = Path(reuse_dependencies).resolve() if reuse_dependencies else out/"dependencies"
    if reuse_dependencies and (not deps.is_relative_to(ROOT/"results") or not (deps/"py.py").is_file()):
        raise ValueError("LINUX_REUSE_DEPENDENCIES_INVALID")
    source.mkdir()
    if not reuse_dependencies:
        deps.mkdir()
    pins = {}
    for folder in ("app", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            relative = path.relative_to(ROOT)
            target = source / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            pins[relative.as_posix()] = sha256_bytes(path.read_bytes())
    for path in (ROOT / "configs").rglob("*.json"):
        relative = path.relative_to(ROOT)
        target = source / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        pins[relative.as_posix()] = sha256_bytes(path.read_bytes())
    for path in (ROOT/"tests/fixtures").rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".csv", ".txt", ".html"}:
            relative = path.relative_to(ROOT)
            target = source/relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
            pins[relative.as_posix()] = sha256_bytes(path.read_bytes())
    if not reuse_dependencies:
        extract_dependencies(deps)
    asset = TOKENIZER_CACHE/hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()
    if sha256_bytes(asset.read_bytes()) != TOKENIZER_SHA256:
        raise ValueError("LINUX_PUBLIC_TOKENIZER_PIN")
    target = source/"data/tokenizer-cache"/asset.name
    target.parent.mkdir(parents=True)
    shutil.copyfile(asset, target)
    pins[target.relative_to(source).as_posix()] = TOKENIZER_SHA256
    manifest = {"version": "scoped-runtime-linux-check-v3", "base_package_sha256": sha256_bytes(BASE.read_bytes()),
        "source_sha256": pins, "dependencies_root": str(deps),
        "dependencies_source": "EXISTING_LINUX_PACKAGE_PLUS_PURE_PYTHON_TEST_TOOLS",
        "pytest_py_shim_sha256": sha256_bytes((deps/"py.py").read_bytes()),
        "network": "none", "mounts": "STAGED_PUBLIC_SOURCE_AND_DEPENDENCIES_ONLY_NO_HOME_OR_PRIVATE_RESULTS",
        "model_calls": 0, "aws_calls": 0, "production_changed": False}
    (out/"stage.json").write_bytes(canonical_json(manifest))
    (out/"output").mkdir()
    print(json.dumps({"status": "STAGED", "source_files": len(pins), "directory": str(out), "dependencies": str(deps)}))


def extract_dependencies(deps):
    with zipfile.ZipFile(BASE) as archive:
        for info in archive.infolist():
            name = Path(info.filename)
            if (name.is_absolute() or ".." in name.parts or "\\" in info.filename):
                raise ValueError("LINUX_BASE_ARCHIVE_MEMBER_INVALID")
            if not name.parts or name.parts[0] in {"app", "configs", "data"} or info.is_dir():
                continue
            target = deps / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    # Pytest and its test-only dependencies are pure Python. Keep Linux runtime
    # dependencies from the existing package, never the Windows .venv binaries.
    site = ROOT / ".venv/Lib/site-packages"
    for package in ("pytest", "_pytest", "pluggy", "iniconfig", "pygments"):
        for path in (site / package).rglob("*.py"):
            target = deps / path.relative_to(site)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    shutil.copyfile(site/"py.py", deps/"py.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--reuse-dependencies", type=Path)
    args = parser.parse_args()
    stage(args.out, args.reuse_dependencies)
