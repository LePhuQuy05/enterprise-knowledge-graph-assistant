"""Isolated Linux/amd64 probe using file-backed S3 fixtures, actual ONNX models and no network."""

import json
import resource
import sys
import time
import zipfile
from pathlib import Path

sys.path.insert(0, "/tmp/query-package")

from app.contracts import canonical_json, sha256_bytes  # noqa: E402
from app.workbench_assets import load_s3_workbench_assets  # noqa: E402
from app.workbench_preflight import exercise_retrieval  # noqa: E402


class FileObjects:
    def __init__(self, directory, version):
        self.directory, self.version, self.calls = directory, version, 0

    def get_object(self, **kwargs):
        prefix = "workbench/" + self.version + "/"
        if kwargs["Bucket"] != "isolated-file-backed-s3-fixture" or not kwargs["Key"].startswith(prefix):
            raise ValueError("NATIVE_FILE_STORE_NOT_ALLOWLISTED")
        relative = kwargs["Key"][len(prefix):]
        target = self.directory / relative
        if not target.resolve().is_relative_to(self.directory.resolve()):
            raise ValueError("NATIVE_FILE_STORE_PATH")
        self.calls += 1
        return {"Body": target.open("rb"), "ContentLength": target.stat().st_size}


def main():
    started = time.monotonic()
    reference = json.loads(Path("/build/reference.json").read_bytes())
    manifest = Path("/public-assets/bundle.json").read_bytes()
    bundle = json.loads(manifest)
    store = FileObjects(Path("/public-assets"), bundle["version"])
    pins = dict(bucket="isolated-file-backed-s3-fixture", version=bundle["version"],
        manifest_sha256=sha256_bytes(manifest), manifest_bytes=len(manifest), tenant_id="demo", cache_root=Path("/tmp/workbench-probe-assets"))
    with load_s3_workbench_assets(store, **pins) as assets:
        cold_loaded = time.monotonic()
        cold = exercise_retrieval(assets, Path("/tmp/workbench-probe-state"))
    downloads = store.calls
    with load_s3_workbench_assets(store, **pins) as assets:
        warm = exercise_retrieval(assets, Path("/tmp/workbench-probe-state"))
    if store.calls != downloads or warm["query_onnx_calls"] or warm["reranker_onnx_calls"] or (
        not all(row["idempotent_replay"] and not row["published_messages"] for row in warm["cases"])
    ):
        raise ValueError("WORKBENCH_NATIVE_REPLAY_NOT_COMPUTE_FREE")
    for row in cold["cases"]:
        for method in row["methods"]:
            expected = reference["methods"][row["case"] + ":" + method["method"]]
            for field in ("config_sha256", "ranked_chunk_ids", "context_chunk_ids"):
                if method[field] != expected[field]:
                    raise ValueError("WORKBENCH_NATIVE_WINDOWS_RESULT_PARITY")
    if canonical_json(cold["cases"][0]["methods"]) != canonical_json(warm["cases"][0]["methods"]):
        raise ValueError("WORKBENCH_NATIVE_RESULT_REPLAY_CHANGED")
    package = Path("/build/query-runtime.zip")
    with zipfile.ZipFile(package) as archive:
        if "app/workbench_preflight.py" not in archive.namelist() or any("model.onnx" in n for n in archive.namelist()):
            raise ValueError("WORKBENCH_PACKAGE_HANDLER_OR_WEIGHTS")
    report = {"status": "PASS_PINNED_LINUX_COLLECTION_RETRIEVAL", "python": sys.version.split()[0],
        "bundle_version": bundle["version"], "collection_manifest_sha256": bundle["collection_manifest_sha256"],
        "files": len(bundle["files"]), "cold_file_store_downloads": downloads, "warm_new_downloads": 0,
        "asset_copy_and_validation_seconds": round(cold_loaded - started, 3),
        "total_seconds": round(time.monotonic() - started, 3),
        "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "cold": cold, "warm": warm, "aws_calls": 0, "model_calls": 0,
        "actual_lambda_context": False, "actual_s3_network": False,
        "limitations": ["File-backed S3 adapter, not AWS S3 latency or IAM verification", "Container CPU/memory limits are not Lambda performance measurements",
            "No generation or semantic answer-quality evaluation", "Private IAM-only preflight, no HTTP queue/browser integration"]}
    Path("/build/native-report.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"cold", "warm"}}))


if __name__ == "__main__":
    main()
