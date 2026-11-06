"""Run durable local owned jobs against the real collection, with zero paid model calls."""

import argparse
import json
from pathlib import Path

from app.answer_cache import AnswerCache
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import canonical_json
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalRoutes
from app.generation import GenerationRecipeV9
from app.local_bge import BgeEncoder
from app.owned_runs import OwnedRunRepository, OwnedRunRequestV1
from app.query_auth import ApiError, AuthContext
from app.reranking import OnnxCrossEncoder
from app.run_artifacts import LocalRunBlobs, RunArtifactRepository
from app.run_collection_adapter import CollectionRetrievalExecutor
from app.run_execution import OwnedRunWorker, RunOutboxDispatcher
from app.run_service import OwnedRunService
from app.run_storage import SQLiteRunQueue, SQLiteRunStore

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--methods", nargs="+", choices=("bm25", "vector", "hybrid", "hybrid_reranked"), default=("bm25", "vector", "hybrid"))
    parser.add_argument("--language", choices=("auto", "en", "vi"), default="auto")
    parser.add_argument("--idempotency-key", required=True, help="Canonical UUID, reuse only for the same intended run")
    parser.add_argument("--pipeline", type=Path, default=ROOT / "results/corpus-expansion-v2/pipeline.json")
    parser.add_argument("--state-dir", type=Path, default=ROOT / "data/cache/owned-runs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("OWNED_RUN_RECEIPT_EXISTS")
    # Explicit local persona; this CLI does not authenticate a real browser/Cognito account.
    auth = AuthContext("demo", "00000000-0000-4000-8000-000000000001")
    def authorize(candidate):
        if candidate != auth:
            raise ApiError(403, "ACCESS_DENIED")
    request = OwnedRunRequestV1(original_question=args.question, method_ids=tuple(args.methods), language=args.language)
    pipeline = json.loads(args.pipeline.read_bytes())
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json") if any(
        m != "bm25" for m in request.method_ids) else None
    reranker = OnnxCrossEncoder(ROOT / "data/models/ms-marco-MiniLM-L2-v2",
        ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json") if "hybrid_reranked" in request.method_ids else None
    repo = OwnedRunRepository(SQLiteRunStore(args.state_dir / "state.sqlite3"))
    queue = SQLiteRunQueue(args.state_dir / "queue.sqlite3")
    artifacts = RunArtifactRepository(LocalRunBlobs(args.state_dir / "artifacts"))
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        executor = CollectionRetrievalExecutor(collection, lexical,
            recipe=GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256),
            routes=routes, run_id="external-pilot10-r72-v1", encoder=encoder,
            embedding_cache=EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE") if encoder else None,
            cache=AnswerCache(args.state_dir / "answer-cache", namespace="LIVE"), reranker=reranker)
        service = OwnedRunService(repo, artifacts, lambda: executor.release,
            authorize=authorize, release_check=executor.check_release)
        row = service.submit(auth, request.model_dump(mode="json"), args.idempotency_key)
        published = RunOutboxDispatcher(repo, queue.send).publish(auth, row["run_id"])
        worker = OwnedRunWorker(repo, artifacts, executor, authorize=authorize, release_check=executor.check_release)
        deliveries = []
        for identifier, message in queue.receive():
            outcome = worker.process(message)
            deliveries.append({"method": message["method_id"], "outcome": outcome})
            if outcome == "ACK":
                queue.ack(identifier)
        detail = service.detail(auth, row["run_id"])
        results = [service.public_result(auth, row["run_id"], method["method_id"]) for method in detail["methods"]
            if method["result_available"]]
        receipt = {"status": detail["status"], "kind": "LOCAL_OWNED_RETRIEVAL_SYNTHETIC_OWNER_NOT_WEB",
            "run": detail, "idempotent_replay": row["idempotent_replay"], "results": results,
            "published_messages": published, "deliveries": deliveries,
            "documents": collection.manifest.counts.documents, "chunks": collection.manifest.counts.chunks,
            "new_model_calls": 0, "new_aws_calls": 0, "new_model_cost_micro_usd": 0,
            "query_onnx_calls": encoder.inference_calls if encoder else 0,
            "reranker_onnx_calls": reranker.inference_calls if reranker else 0,
            "quality_metrics": None, "cloud_deployed": False, "authenticated_web": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        output.write(canonical_json(receipt) + b"\n")
    print(json.dumps({"status": receipt["status"], "run_id": row["run_id"], "output": str(args.output),
        "published_messages": published, "idempotent_replay": row["idempotent_replay"],
        "new_model_calls": 0, "new_aws_calls": 0, "cloud_deployed": False}))
    return 0 if receipt["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
