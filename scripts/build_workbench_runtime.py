"""Repack verified Linux dependencies into a separate workbench ZIP and run an isolated native probe."""

import argparse
import hashlib
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from scripts.build_query_package import run

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--directory", type=Path, default=ROOT / "data/build/workbench-runtime-v1")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--owned-workflow", action="store_true", help="Also test real API/outbox/indexed state/worker with file-backed cloud clients")
    parser.add_argument("--collection-answers", action="store_true", help="Also test EN/VI owned answers with a simulated provider, no external calls")
    args = parser.parse_args(argv)
    if args.collection_answers and not args.owned_workflow:
        parser.error("--collection-answers requires --owned-workflow")
    if not args.execute:
        print("Plan: verified existing Lambda dependency ZIP + public source/config allowlist; network-none Docker; no AWS or model APIs.")
        return 0
    if args.report.exists() or (args.directory / "package_report.json").exists():
        raise ValueError("WORKBENCH_NATIVE_BUILD_ALREADY_EXISTS")
    baseline = json.loads((ROOT / "results/query_package_build.json").read_bytes())
    source_zip = Path(baseline["package_path"])
    with source_zip.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    if baseline["status"] != "PASS" or digest != baseline["sha256"] or (
        sha256_bytes((ROOT / "uv.lock").read_bytes()) != baseline["uv_lock_sha256"]
    ):
        raise ValueError("WORKBENCH_REQUIRES_VERIFIED_LOCKED_LINUX_DEPENDENCIES")
    assets = json.loads((ROOT / "results/workbench-cloud/assets-build-v1.json").read_bytes())
    asset_dir = Path(assets["bundle_directory"])
    if assets["status"] != "LOCAL_BUNDLE_VERIFIED" or sha256_bytes((asset_dir / "bundle.json").read_bytes()) != assets["bundle_manifest_sha256"]:
        raise ValueError("WORKBENCH_ASSETS_NOT_READY")
    build = args.directory
    stage = build / "source"
    build.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source_zip, build / "query-runtime.zip")
    source_paths = {}
    for path in sorted((ROOT / "app").rglob("*.py")):
        if path.is_symlink() or not path.resolve().is_relative_to((ROOT / "app").resolve()):
            raise ValueError("WORKBENCH_UNSAFE_PACKAGE_SOURCE")
        source_paths[path.relative_to(ROOT).as_posix()] = path
    for name in ("configs/models/bge-small-en-v1.5-onnx.json", "configs/models/ms-marco-MiniLM-L2-v2-onnx.json",
        "configs/models/external-direct-v1.json", "configs/prices/external-direct-2026-10-24.json"):
        source_paths[name] = ROOT / name
    from app.ingestion.text import TOKENIZER_CACHE, TOKENIZER_SHA256, TOKENIZER_URL

    tokenizer = TOKENIZER_CACHE / hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()
    if not tokenizer.is_file() or sha256_bytes(tokenizer.read_bytes()) != TOKENIZER_SHA256:
        raise ValueError("WORKBENCH_PREPARED_TOKENIZER_REQUIRED_NO_AUTO_DOWNLOAD")
    source_paths["data/tokenizer-cache/" + tokenizer.name] = tokenizer
    inventory = {}
    for name, path in source_paths.items():
        target = stage / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        inventory[name] = sha256_bytes(target.read_bytes())
    (build / "source_inventory.json").write_bytes(canonical_json(inventory) + b"\n")
    references = {}
    receipts = {"kms": ROOT / "results/owned-runs/retrieval-ldh-v1.json", "lambda": ROOT / "results/owned-runs/retrieval-hr-v1.json"}
    for name, path in receipts.items():
        receipt = json.loads(path.read_bytes())
        if receipt["status"] != "COMPLETED" or receipt["new_model_calls"] or receipt["new_aws_calls"]:
            raise ValueError("WORKBENCH_NATIVE_REFERENCE_REQUIRES_RECORDED_RETRIEVAL")
        for method in receipt["results"]:
            references[name + ":" + method["method_id"]] = {key: method[key] for key in ("config_sha256", "ranked_chunk_ids", "context_chunk_ids")}
    (build / "reference.json").write_bytes(canonical_json({"methods": references,
        "receipt_sha256": {name: sha256_bytes(path.read_bytes()) for name, path in receipts.items()}}) + b"\n")
    image = json.loads((ROOT / "configs/runtime_images.json").read_bytes())["lambda_python"]
    started = datetime.now(UTC)
    extra_mounts = (["--mount", f"type=bind,source={ROOT / 'scripts/check_owned_cloud_linux.py'},target=/owned-probe.py,readonly"]
                    if args.owned_workflow else [])
    if args.collection_answers:
        extra_mounts += ["--mount", f"type=bind,source={ROOT / 'scripts/check_collection_answer_linux.py'},target=/answer-probe.py,readonly"]
    run(["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none", "--memory", "1024m", "--cpus", "2", "--pids-limit", "256",
        "--mount", f"type=bind,source={stage.resolve()},target=/source,readonly",
        "--mount", f"type=bind,source={build.resolve()},target=/build",
        "--mount", f"type=bind,source={asset_dir.resolve()},target=/public-assets,readonly",
        "--mount", f"type=bind,source={ROOT / 'scripts/package_query_linux.py'},target=/builder.py,readonly",
        "--mount", f"type=bind,source={ROOT / 'scripts/check_workbench_linux.py'},target=/native-probe.py,readonly",
        *extra_mounts, "--entrypoint", "/bin/sh", image, "-c", "python /builder.py --repack && python /native-probe.py" +
        (" && python /owned-probe.py" if args.owned_workflow else "") +
        (" && python /answer-probe.py" if args.collection_answers else "")], timeout=600)
    package = json.loads((build / "package_report.json").read_bytes())
    native = json.loads((build / "native-report.json").read_bytes())
    if package["status"] != "PASS" or native["status"] != "PASS_PINNED_LINUX_COLLECTION_RETRIEVAL":
        raise ValueError("WORKBENCH_NATIVE_PROBE_NOT_PASS")
    report = {"status": "PASS_NATIVE_WORKBENCH_PACKAGE", "started_at": started.isoformat(), "finished_at": datetime.now(UTC).isoformat(),
        "image": image, "source_inventory_sha256": sha256_bytes((build / "source_inventory.json").read_bytes()),
        "uv_lock_sha256": baseline["uv_lock_sha256"], "base_dependency_zip_sha256": baseline["sha256"],
        "package_path": str((build / "query-runtime.zip").resolve()), "package": package, "native": native,
        "container_memory_limit_mib": 1024, "container_cpus": 2, "credentials_mounted": False,
        "network": "none", "aws_calls": 0, "model_calls": 0, "cloud_deployed": False}
    if args.owned_workflow:
        owned = json.loads((build / "owned-report.json").read_bytes())
        if owned["status"] != "PASS_NATIVE_OWNED_WORKFLOW":
            raise ValueError("OWNED_NATIVE_WORKFLOW_NOT_PASS")
        report["owned_workflow"] = owned
    if args.collection_answers:
        answer = json.loads((build / "collection-answer-report.json").read_bytes())
        if answer["status"] != "PASS_NATIVE_COLLECTION_ANSWER_FIXTURE":
            raise ValueError("COLLECTION_NATIVE_ANSWER_NOT_PASS")
        report["collection_answers"] = answer
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.report.open("xb") as output:
        output.write(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "image", "package_path", "aws_calls", "model_calls", "cloud_deployed")}))


if __name__ == "__main__":
    main()
