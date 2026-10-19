"""Run inside the pinned Lambda image with only public model/snapshot assets mounted."""

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np

from app.local_bge import BgeEncoder
from app.retrieval import retrieve_baseline
from app.snapshot import load_snapshot


def main():
    started = time.monotonic()
    encoder = BgeEncoder(Path("/model"), Path("/workspace/bge.json"), threads=2)
    loaded_seconds = time.monotonic() - started
    inputs = json.loads(Path("/asset/parity_inputs.json").read_bytes())
    docs = encoder.encode(inputs["documents"], role="document")
    queries = encoder.encode(inputs["queries"], role="query")
    reference = np.load("/asset/windows_vectors.npz", allow_pickle=False)
    difference = max(float(np.max(np.abs(docs - reference["documents"]))),
                     float(np.max(np.abs(queries - reference["queries"]))))
    if difference > 1e-4:
        raise ValueError("BGE_WINDOWS_LINUX_PARITY_FAILED")
    # A different padding/batch shape must not change the sentence representation materially.
    separate = np.stack([encoder.encode([text], role="document")[0] for text in inputs["documents"]])
    if not np.allclose(docs, separate, atol=1e-5, rtol=1e-5):
        raise ValueError("BGE_BATCH_PADDING_INVARIANCE_FAILED")
    snapshot = load_snapshot(Path("/snapshot"), tenant_id="demo", embedding_model_id=encoder.recipe("document").model_id,
                             trusted_manifest_sha256=inputs["snapshot_manifest_sha256"])
    if snapshot.manifest.embedding_recipe != encoder.recipe("document"):
        raise ValueError("BGE_QUERY_SNAPSHOT_RECIPE_MISMATCH")
    first_query = encoder.encode([inputs["retrieval_question"]], role="query")[0]
    top5 = retrieve_baseline(snapshot, inputs["retrieval_question"], first_query)[:5]
    if top5 != inputs["windows_top5"]:
        raise ValueError("BGE_LINUX_RETRIEVAL_PARITY_FAILED")
    warm_start = time.monotonic()
    encoder.encode([inputs["retrieval_question"]], role="query")
    warm_seconds = time.monotonic() - warm_start
    directory = Path("/tmp/kg-bge")
    dependency_bytes = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
    model_bytes = sum(p.stat().st_size for p in Path("/model").rglob("*") if p.is_file())
    report = {"status": "PASS", "python": sys.version.split()[0], "load_seconds": round(loaded_seconds, 3),
              "warm_query_seconds": round(warm_seconds, 3), "cross_platform_max_absolute_difference": difference,
              "batch_padding_invariance": "PASS", "snapshot_reload_and_top5_parity": "PASS",
              "dependency_bytes": dependency_bytes, "model_bytes": model_bytes,
              "peak_process_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
              "combined_dependency_model_bytes": dependency_bytes + model_bytes,
              "zip_uncompressed_limit_bytes": 250 * 1024 * 1024,
              "model_delivery": "S3_IMMUTABLE_MODEL_TO_TMP" if dependency_bytes + model_bytes > 250 * 1024 * 1024 else "PACKAGE_OR_S3_TO_TMP",
              "paid_model_api_calls": 0, "aws_deployments": 0, "actual_lambda_latency_measured": False,
              "embedding_recipe": encoder.recipe("document").model_dump(mode="json")}
    Path("/asset/report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
