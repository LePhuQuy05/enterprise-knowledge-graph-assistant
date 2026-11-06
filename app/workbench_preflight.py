"""IAM-only retrieval preflight. No model gateway, secrets, public API or shared-state writes."""

import json
import os
import time
from pathlib import Path

from app.answer_cache import AnswerCache
from app.contracts import sha256_bytes
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalRoutes
from app.generation import GenerationRecipeV9
from app.local_bge import BgeEncoder
from app.owned_runs import OwnedRunRepository
from app.query_auth import ApiError, AuthContext
from app.reranking import OnnxCrossEncoder
from app.run_artifacts import LocalRunBlobs, RunArtifactRepository
from app.run_collection_adapter import CollectionRetrievalExecutor
from app.run_execution import OwnedRunWorker, RunOutboxDispatcher
from app.run_service import OwnedRunService
from app.run_storage import SQLiteRunQueue, SQLiteRunStore
from app.workbench_assets import load_s3_workbench_assets

ROOT = Path(__file__).resolve().parents[1]
AUTH = AuthContext("demo", "00000000-0000-4000-8000-000000000003")
CASES = (
    ("kms", "11111111-0000-4000-8000-000000000001", "Which policies must permit cross-account use of an AWS KMS key?", ("bm25", "vector", "hybrid")),
    ("lambda", "11111111-0000-4000-8000-000000000002", "Does connecting a Lambda function to a public subnet give it a public IP address and internet access?", ("hybrid", "hybrid_reranked")),
)


def exercise_retrieval(assets, state_dir):
    """Fixed two cases, five child methods. Repeated invocation reads the same completed runs."""
    state_dir = Path(state_dir) / assets.bundle.version
    encoder = BgeEncoder(assets.root / "models/bge", assets.root / "models/bge/manifest.json", threads=2)
    reranker = OnnxCrossEncoder(assets.root / "models/reranker", assets.root / "models/reranker/manifest.json", threads=2)
    if encoder.recipe("document") != assets.collection.manifest.embedding_recipe:
        raise ValueError("WORKBENCH_NATIVE_EMBEDDING_RECIPE_MISMATCH")
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    executor = CollectionRetrievalExecutor(assets.collection, assets.lexical,
        recipe=GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256), routes=routes,
        run_id="external-pilot10-r72-v1", encoder=encoder, reranker=reranker,
        embedding_cache=EmbeddingCache(state_dir / "embeddings", namespace="LIVE"),
        cache=AnswerCache(state_dir / "unused-answer-cache", namespace="LIVE"))
    repo = OwnedRunRepository(SQLiteRunStore(state_dir / "state.sqlite3"))
    artifacts = RunArtifactRepository(LocalRunBlobs(state_dir / "results"))
    queue = SQLiteRunQueue(state_dir / "queue.sqlite3")
    def authorize(auth):
        if auth != AUTH:
            raise ApiError(403, "ACCESS_DENIED")
    service = OwnedRunService(repo, artifacts, lambda: executor.release, authorize=authorize, release_check=executor.check_release)
    worker = OwnedRunWorker(repo, artifacts, executor, authorize=authorize, release_check=executor.check_release)
    outputs, source_count = [], 0
    for name, key, question, methods in CASES:
        row = service.submit(AUTH, {"original_question": question, "method_ids": list(methods), "language": "en"}, key)
        published = RunOutboxDispatcher(repo, queue.send).publish(AUTH, row["run_id"])
        for identifier, message in queue.receive():
            if worker.process(message) != "ACK":
                raise ValueError("WORKBENCH_PREFLIGHT_CHILD_NOT_COMPLETED")
            queue.ack(identifier)
        detail = service.detail(AUTH, row["run_id"])
        if detail["status"] != "COMPLETED":
            raise ValueError("WORKBENCH_PREFLIGHT_RUN_FAILED")
        results = []
        for method in methods:
            result = service.result(AUTH, row["run_id"], method)
            for source in result.context_sources:
                shard = assets.collection.get_shard(source["shard_id"])
                doc = shard.documents[source["document_version_id"]]
                if source["text"] != doc.normalized_text[source["start_char"]:source["end_char"]] or (
                    sha256_bytes(source["text"].encode()) != source["text_sha256"]
                    or source["shard_manifest_sha256"] != shard.manifest_sha256 or source["source_uri"] != doc.source_uri
                ):
                    raise ValueError("WORKBENCH_PREFLIGHT_SOURCE_SLICE")
                source_count += 1
            results.append({"method": method, "config_sha256": result.config_sha256,
                "ranked_chunk_ids": list(result.ranked_chunk_ids), "context_chunk_ids": list(result.context_chunk_ids),
                "source_count": len(result.context_sources), "result_sha256": repo.child(AUTH, row["run_id"], method)["result_ref"]["sha256"]})
        outputs.append({"case": name, "status": detail["status"], "idempotent_replay": row["idempotent_replay"],
            "published_messages": published, "methods": results})
    return {"cases": outputs, "source_appearances": source_count, "query_onnx_calls": encoder.inference_calls,
        "reranker_onnx_calls": reranker.inference_calls, "model_calls": 0, "shared_state_writes": False,
        "synthetic_owner_not_browser_auth": True, "quality_metrics": None}


def validate_release(config):
    if type(config) is not dict or set(config) != {"bucket", "version", "manifest_sha256", "manifest_bytes", "tenant_id"} or (
        config["tenant_id"] != "demo" or type(config["bucket"]) is not str
    ):
        raise ValueError("WORKBENCH_PREFLIGHT_RELEASE_CONFIG")
    return config


def handler(event, context):
    started = time.monotonic()
    if event != {"operation": "RETRIEVAL_PREFLIGHT"}:
        return {"status": "REJECTED", "error": "PREFLIGHT_EVENT_NOT_ALLOWED", "model_calls": 0}
    try:
        if os.environ.get("MODEL_ENABLED") != "false":
            raise ValueError("PREFLIGHT_REQUIRES_MODEL_DISABLED")
        if context.get_remaining_time_in_millis() < 65_000:
            raise ValueError("PREFLIGHT_INSUFFICIENT_REMAINING_TIME")
        release = validate_release(json.loads(os.environ["KG_WORKBENCH_RELEASE_JSON"]))
        import boto3
        from botocore.config import Config

        client = boto3.client("s3", region_name="us-east-1", config=Config(retries={"total_max_attempts": 1},
            connect_timeout=2, read_timeout=5))
        with load_s3_workbench_assets(client, **release, cache_root=Path("/tmp/kg-workbench-assets")) as assets:
            report = exercise_retrieval(assets, Path("/tmp/kg-workbench-preflight"))
        return report | {"status": "PASS", "elapsed_seconds": round(time.monotonic() - started, 3),
            "bundle_version": release["version"], "aws_s3_reads": True, "actual_lambda_context": True,
            "new_resources_created_by_handler": False}
    except Exception:
        return {"status": "FAIL", "error": "WORKBENCH_PREFLIGHT_FAILED", "model_calls": 0,
            "elapsed_seconds": round(time.monotonic() - started, 3)}
