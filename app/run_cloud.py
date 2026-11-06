"""Retrieval-only cloud composition, durable recovery and owner-authorized API.

No external model gateway, secret access or browser credentials in queue messages.
Runtime clients/configuration are constructed only inside an explicit handler.
"""

import json
import os
import re
from pathlib import Path

from app.contracts import canonical_json
from app.owned_runs import TERMINAL, OwnedRunReleaseV1, OwnedRunRepository, RunMessageV1
from app.query_auth import ApiError, AuthContext, require_current_membership
from app.run_artifacts import RunArtifactRepository, S3RunBlobs
from app.run_execution import OwnedRunWorker, RunOutboxDispatcher, SqsRunSender
from app.run_service import OwnedRunApi, OwnedRunService
from app.run_work_index import CloudDynamoRunStore
from app.tracer import response

ROOT = Path(__file__).resolve().parents[1]


def slot_message(slot):
    match = re.fullmatch(r"TENANT#([a-z][a-z0-9_-]{0,63})#USER#([0-9a-f-]{36})", slot.get("owner_partition", ""))
    if slot.get("record_type") != "OWNED_RUN_WORK_SLOT" or slot.get("status") != "ACTIVE" or not match:
        raise ValueError("RUN_WORK_SLOT_INVALID")
    return RunMessageV1(tenant_id=match[1], owner_id=match[2], run_id=slot["run_id"],
        method_id=slot["method_id"], admission_token=slot["admission_token"])


class CloudRunCoordinator:
    def __init__(self, repository, send, authorize):
        self.repo, self.send, self.authorize = repository, send, authorize
        self.dispatcher = RunOutboxDispatcher(repository, send)

    def notify(self, auth, run_id):
        """After committed admission. Failure leaves discoverable slots and PENDING outbox."""
        self.authorize(auth)
        self.repo.recover(auth, run_id)
        return self.dispatcher.publish(auth, run_id)

    def sweep(self):
        slots = self.repo.store.work_slots()
        summary = {"examined": len(slots), "pending_published": 0, "safe_redeliveries": 0,
                   "terminal": 0, "unknown": 0, "errors": 0, "model_calls": 0}
        for slot in slots:
            if slot.get("status") == "FREE":
                continue
            try:
                msg = slot_message(slot)
                auth = AuthContext(msg.tenant_id, msg.owner_id)
                parent = self.repo.get(auth, msg.run_id)
                if parent["admission_token"] != msg.admission_token:
                    raise ValueError("RUN_WORK_SLOT_BINDING_LOST")
                try:
                    self.authorize(auth)
                except ApiError as exc:
                    if exc.status not in {401, 403}:
                        raise
                    self.repo.cancel(auth, msg.run_id)
                self.repo.recover(auth, msg.run_id)
                child = self.repo.child(auth, msg.run_id, msg.method_id)
                if child["status"] in TERMINAL:
                    summary["unknown" if child["status"] == "UNKNOWN" else "terminal"] += 1
                    continue
                outbox = next(row for row in self.repo.outboxes(auth, msg.run_id)
                              if row["message"]["method_id"] == msg.method_id)
                if outbox["status"] == "PENDING":
                    summary["pending_published"] += self.dispatcher.publish(auth, msg.run_id)
                elif child["status"] == "QUEUED" or (child["status"] == "CLAIMED" and
                                                       child["lease_until"] <= int(self.repo.now().timestamp())):
                    # Known pre-execution work can be redelivered. EXECUTING never reruns.
                    self.send(msg.model_dump(mode="json"))
                    summary["safe_redeliveries"] += 1
            except Exception:
                # One corrupt/expired pointer must not hide the other bounded slots.
                summary["errors"] += 1
        return summary


class CloudRunService(OwnedRunService):
    def __init__(self, *args, coordinator, answer_policy=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinator = coordinator
        self.answer_policy = answer_policy

    def submit(self, auth, payload, key):
        if self.answer_policy is not None and isinstance(payload, dict) and payload.get("run_kind") == "answer":
            from app.owned_runs import OwnedRunRequestV1

            try:
                validated = OwnedRunRequestV1.model_validate_json(canonical_json(payload)).model_dump(mode="json")
            except ValueError:
                raise ApiError(400, "INVALID_REQUEST") from None
            self.authorize(auth)
            self.answer_policy.check_admission(self.repo.answer_budget, validated)
        detail = super().submit(auth, payload, key)
        try:
            self.coordinator.notify(auth, detail["run_id"])
            detail["publication"] = "ATTEMPTED_AFTER_COMMIT"
        except Exception:
            detail["publication"] = "RECOVERY_PENDING"
        return detail


def cloud_components():
    import boto3
    from botocore.config import Config

    if os.environ.get("MODEL_ENABLED") not in {"false", "true"} or os.environ.get("KG_RUNS_ENABLED") != "true":
        raise ApiError(503, "RUNS_DISABLED")
    config = json.loads(os.environ["KG_RUN_CLOUD_JSON"])
    required = {"queue_url", "results_bucket", "assets", "release", "public_config", "corpus"}
    if set(config) not in (required, required | {"answer_policy"}) or config["assets"]["tenant_id"] != "demo":
        raise ValueError("RUN_CLOUD_CONFIG_INVALID")
    if "answer_policy" not in config and os.environ.get("MODEL_ENABLED") != "false":
        raise ApiError(503, "RUNS_DISABLED")
    release = OwnedRunReleaseV1.model_validate_json(canonical_json(config["release"]))
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=2, read_timeout=4)
    db = boto3.client("dynamodb", region_name="us-east-1", config=cfg)
    store = CloudDynamoRunStore(db, "kg-r72-query-state")
    from app.ledger import DynamoLedgerStore

    metadata = DynamoLedgerStore(db, "kg-r72-metadata")
    def authorize(auth):
        if auth.tenant_id != config["assets"]["tenant_id"]:
            raise ApiError(403, "ACCESS_DENIED")
        return require_current_membership(auth, metadata)
    answer_policy, controller = None, None
    if "answer_policy" in config:
        from app.cloud_owned_answers import load_collection_answer_policy, make_quote_controller

        answer_policy = load_collection_answer_policy(config["answer_policy"])
        controller = make_quote_controller(store, answer_policy)
    repo = OwnedRunRepository(store, answer_budget=controller)
    def check(pin):
        if OwnedRunReleaseV1.model_validate(pin) != release:
            raise ApiError(503, "COLLECTION_UNAVAILABLE")
    artifacts = RunArtifactRepository(S3RunBlobs(boto3.client("s3", region_name="us-east-1", config=cfg), config["results_bucket"]))
    coordinator = CloudRunCoordinator(repo, SqsRunSender(boto3.client("sqs", region_name="us-east-1", config=cfg), config["queue_url"]), authorize)
    service = CloudRunService(repo, artifacts, lambda: release, authorize=authorize, release_check=check,
                              coordinator=coordinator, answer_policy=answer_policy)
    return config, service, metadata, coordinator


def api_handler(event, context):
    try:
        config, service, metadata, _ = cloud_components()
        def capabilities(auth):
            if auth.tenant_id != config["assets"]["tenant_id"]:
                raise ApiError(403, "ACCESS_DENIED")
            answering = service.answer_policy is not None
            return {"schema_version": 1, "catalog_version": "owned-collection-capabilities-v2" if answering else "owned-retrieval-capabilities-v1", "available": True,
                "collection_version": config["release"]["collection_version"], "corpus": config["corpus"],
                "source_scope": "local_collection", "methods": list(config["release"]["method_config_sha256"]),
                "run_kinds": ["retrieval", "answer"] if answering else ["retrieval"],
                "answer_methods": list(service.answer_policy.methods) if answering else [],
                "maximum_answer_methods": 1 if answering else 0,
                "maximum_methods": 3, "monthly_child_limit": 300,
                "answering_enabled": answering, "vi_query_transform": "SHARED_ENGLISH_REWRITE_FOR_ANSWERS" if answering else "NONE_DIAGNOSTIC_ENGLISH_ENCODER",
                "quality_metrics": None}
        return OwnedRunApi(service, metadata, config["public_config"], capabilities=capabilities).handler(event)
    except Exception:
        return response(503, ApiError(503, "RUN_STATE_UNAVAILABLE").body())


def watchdog_handler(event, context):
    if event != {"operation": "RUN_WATCHDOG"}:
        raise ValueError("RUN_WATCHDOG_EVENT_INVALID")
    _, _, _, coordinator = cloud_components()
    report = coordinator.sweep()
    print(json.dumps({"event": "run_watchdog", **report}), flush=True)
    if report["errors"] or report["unknown"]:
        raise RuntimeError("RUN_WATCHDOG_NEEDS_ATTENTION")
    return report


def worker_handler(event, context):
    # Fail closed before touching state for malformed invocations; the SQS event source is IAM-only.
    records = event.get("Records") if type(event) is dict else None
    if type(records) is not list or len(records) != 1:
        raise ValueError("RUN_WORKER_REQUIRES_ONE_SQS_RECORD")
    if context.get_remaining_time_in_millis() < 65_000:
        raise ValueError("RUN_WORKER_TIME_BUDGET")
    config, service, _, _ = cloud_components()
    from app.cloud_owned_answers import CloudAnswerDispatch, make_cloud_answer_executor

    # Terminal redeliveries/cancellations are resolved before loading corpus, keys or ONNX.
    execute = CloudAnswerDispatch(service=service, config=config, context=context,
        retrieval_factory=make_retrieval_executor, answer_factory=make_cloud_answer_executor)
    worker = OwnedRunWorker(service.repo, service.artifacts, execute,
        authorize=service.authorize, release_check=service.release_check)
    try:
        return worker.sqs_batch(event)
    finally:
        execute.close()


def make_retrieval_executor(assets):
    from app.answer_cache import AnswerCache
    from app.embeddings import EmbeddingCache
    from app.external_gateway import ExternalRoutes
    from app.generation import GenerationRecipeV9
    from app.local_bge import BgeEncoder
    from app.reranking import OnnxCrossEncoder
    from app.run_collection_adapter import CollectionRetrievalExecutor

    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    directory = Path("/tmp/kg-run-cache") / assets.bundle.version
    return CollectionRetrievalExecutor(assets.collection, assets.lexical,
        recipe=GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256), routes=routes,
        run_id="external-pilot10-r72-v1", encoder=BgeEncoder(assets.root / "models/bge", assets.root / "models/bge/manifest.json", threads=2),
        reranker=OnnxCrossEncoder(assets.root / "models/reranker", assets.root / "models/reranker/manifest.json", threads=2),
        embedding_cache=EmbeddingCache(directory / "embeddings", namespace="LIVE"),
        cache=AnswerCache(directory / "unused-answer-cache", namespace="LIVE"))
