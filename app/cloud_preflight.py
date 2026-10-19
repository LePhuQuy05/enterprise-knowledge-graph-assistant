"""IAM-invoke-only cloud package probe: S3/snapshot/BGE/secret, never paid model inference."""

import json
import os
import resource
import time
from pathlib import Path

import boto3
from botocore.config import Config

from app.bge_model_store import ensure_bge_bundle
from app.cloud_assets import load_s3_snapshot, read_deepseek_secret
from app.contracts import sha256_bytes
from app.local_bge import BgeEncoder
from app.retrieval import retrieve_baseline

ROOT = Path(__file__).resolve().parents[1]
_STATE = None


def handler(event, context):
    global _STATE
    started = time.monotonic()
    if event not in ({}, {"action": "verify"}):
        return {"status": "REJECTED", "error_category": "FIXED_PREFLIGHT_ACTION_REQUIRED", "paid_model_calls": 0}
    report = {"status": "FAIL", "paid_model_calls": 0, "warm_cache": _STATE is not None}
    try:
        config = json.loads(os.environ["KG_ASSET_CONFIG_JSON"])
        if config["tenant_id"] != "demo" or config["snapshot_version"].startswith("snap_") is not True:
            raise ValueError("PREFLIGHT_SCOPE")
        local_manifest = ROOT / "configs/models/bge-small-en-v1.5-onnx.json"
        if sha256_bytes(local_manifest.read_bytes()) != config["model_manifest_sha256"]:
            raise ValueError("DEPLOYED_MODEL_MANIFEST_CHANGED")
        sdk = Config(retries={"total_max_attempts": 1}, connect_timeout=2, read_timeout=5)
        secrets = boto3.client("secretsmanager", region_name="us-east-1", config=sdk)
        key = read_deepseek_secret(secrets, secret_arn=os.environ["KG_DEEPSEEK_SECRET_ARN"])
        report["secret_readable_and_format_valid"] = bool(key.get_secret_value())
        if _STATE is None:
            s3 = boto3.client("s3", region_name="us-east-1", config=sdk)
            assets_started = time.monotonic()
            directory = ensure_bge_bundle(s3, bucket=config["bucket"], manifest_path=local_manifest,
                                           cache_root=Path("/tmp/kg-models"))
            report["model_download_verify_seconds"] = round(time.monotonic() - assets_started, 3)
            encoder = BgeEncoder(directory, local_manifest, threads=2)
            snapshot = load_s3_snapshot(s3, bucket=config["bucket"], prefix=config["snapshot_prefix"],
                manifest_sha256=config["snapshot_manifest_sha256"], manifest_bytes=config["snapshot_manifest_bytes"],
                snapshot_version=config["snapshot_version"], tenant_id="demo", cache_root=Path("/tmp/kg-snapshots"))
            if snapshot.manifest.embedding_recipe != encoder.recipe("document"):
                raise ValueError("CLOUD_ENCODER_SNAPSHOT_RECIPE_MISMATCH")
            _STATE = (encoder, snapshot)
        encoder, snapshot = _STATE
        vector = encoder.encode([config["probe_question"]], role="query")[0]
        ranked = retrieve_baseline(snapshot, config["probe_question"], vector)[:5]
        if ranked != config["expected_top5"]:
            raise ValueError("CLOUD_LOCAL_RETRIEVAL_PARITY_FAILED")
        from app.ingestion.text import ProxyTokenizer

        report.update(status="PASS", snapshot_version=snapshot.manifest.snapshot_version,
            documents=len(snapshot.content.documents), chunks=len(snapshot.content.chunks),
            dimension=encoder.recipe("document").dimension, top5_parity="PASS", top5_chunk_ids=ranked,
            proxy_tokenizer_ready=ProxyTokenizer().count("public source") > 0,
            process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
            elapsed_seconds=round(time.monotonic() - started, 3), cloud_answers_enabled=False,
            remaining_lambda_ms=context.get_remaining_time_in_millis() if context else None)
    except Exception as exc:
        report.update(error_category=type(exc).__name__, elapsed_seconds=round(time.monotonic() - started, 3))
        if isinstance(exc, ValueError) and str(exc).isupper() and len(str(exc)) < 100:
            report["safe_error"] = str(exc)
    return report
