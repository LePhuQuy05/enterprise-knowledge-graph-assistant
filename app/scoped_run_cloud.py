"""Opt-in V2 AWS composition, isolated from the deployed V1 handlers/routes.

API/watchdog have quote-only controllers; only the queue worker initializes
portable assets, a provider credential and a live transport. Importing this
module performs no AWS call and enables no route or funding allocation.
"""

import os
import re
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.answer_qualification import policy_fingerprint
from app.contracts import Record, Sha256, canonical_json, sha256_bytes
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.ledger import Ledger
from app.query_auth import ApiError, require_current_membership
from app.run_artifacts import S3RunBlobs
from app.run_service import OwnedRunApi
from app.scoped_cloud_coordinator import ScopedCloudCoordinator, ScopedCloudService, ScopedSqsSender
from app.scoped_cloud_state import ScopedCloudDynamoRunStore
from app.scoped_owned_runs import ScopedRunRepository
from app.scoped_run_budget import ScopedRunBudgetController
from app.scoped_run_contracts import ScopedRunReleaseV2
from app.scoped_run_service import ScopedArtifactRepository, ScopedRunWorker
from app.tracer import response

ROOT = Path(__file__).resolve().parents[1]


class CorpusAssetPins(Record):
    bucket: Annotated[str, Field(pattern=r"^kg-r72-[a-z0-9-]{1,54}$")]
    version: Annotated[str, Field(pattern=r"^wb_[0-9a-f]{64}$")]
    manifest_sha256: Sha256
    manifest_bytes: Annotated[int, Field(gt=0, le=1048576)]
    tenant_id: Literal["demo"] = "demo"


class GraphAssetPins(Record):
    bucket: Annotated[str, Field(pattern=r"^kg-r72-[a-z0-9-]{1,54}$")]
    version: Annotated[str, Field(pattern=r"^sg_[0-9a-f]{64}$")]
    manifest_sha256: Sha256
    manifest_bytes: Annotated[int, Field(gt=0, le=16384)]


class PriceAssetPin(Record):
    path: Annotated[str, Field(pattern=r"^configs/prices/external-direct-20[0-9]{2}-[0-9]{2}-[0-9]{2}\.json$")]
    sha256: Sha256


class ScopedCloudConfigV2(Record):
    schema_version: Literal[2] = 2
    queue_url: Annotated[str, Field(pattern=r"^https://sqs\.us-east-1\.amazonaws\.com/[0-9]{12}/kg-r72-scoped-[a-zA-Z0-9_-]+$")]
    results_bucket: Annotated[str, Field(pattern=r"^kg-r72-scoped-[a-z0-9-]{1,47}$")]
    assets: CorpusAssetPins
    graph_assets: GraphAssetPins
    prices: PriceAssetPin
    release: ScopedRunReleaseV2
    public_config: dict

    @model_validator(mode="after")
    def release_binding(self):
        if self.release.budget_run_id != "collection319-scoped-answer-v2" or (
                self.release.qualification_policy_sha256 != policy_fingerprint()
                or any(c.fixture_only for c in self.release.configs.values())):
            raise ValueError("SCOPED_CLOUD_LIVE_RELEASE_POLICY")
        if self.assets.bucket != self.graph_assets.bucket or self.public_config.get("region") not in {None, "us-east-1"}:
            raise ValueError("SCOPED_CLOUD_ASSET_OR_REGION_BINDING")
        return self


def make_scoped_quote_controller(store, prices_pin, *, clock=None):
    pin = PriceAssetPin.model_validate(prices_pin)
    raw = (ROOT/pin.path).read_bytes()
    if sha256_bytes(raw) != pin.sha256:
        raise ValueError("SCOPED_CLOUD_PRICE_ASSET_CHANGED")
    prices = ExternalPriceSheet.model_validate_json(raw)
    routes = ExternalRoutes.model_validate_json((ROOT/"configs/models/external-direct-v1.json").read_bytes())
    routes = routes.model_copy(update={"live_inference_enabled": True, "live_inference_gate": "OWNED_SCOPED_ANSWER_RELEASE_V2"})
    ledger = Ledger(store, clock=clock or (lambda: datetime.now(UTC)))
    # Rate freshness is checked at admission/dispatch. Expired prices must not
    # disable history, cancellation, slot cleanup or reconciliation reads.
    return ScopedRunBudgetController(ledger=ledger, routes=routes, prices=prices,
        budget_run_id="collection319-scoped-answer-v2", quote_live=True)


def load_cloud_config():
    path = os.environ.get("KG_SCOPED_CONFIG_PATH", "")
    digest = os.environ.get("KG_SCOPED_CONFIG_SHA256", "")
    if not re.fullmatch(r"configs/releases/scoped-cloud-[0-9a-f]{64}\.json", path) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("SCOPED_CLOUD_CONFIG_PIN")
    target = ROOT/path
    if not target.is_file() or not 0 < target.stat().st_size <= 128*1024:
        raise ValueError("SCOPED_CLOUD_CONFIG_BOUND")
    raw = target.read_bytes()
    if sha256_bytes(raw) != digest or path != f"configs/releases/scoped-cloud-{digest}.json":
        raise ValueError("SCOPED_CLOUD_CONFIG_PIN")
    return ScopedCloudConfigV2.model_validate_json(raw).model_dump(mode="json")


def cloud_components():
    import boto3
    from botocore.config import Config

    from app.ledger import DynamoLedgerStore

    if os.environ.get("KG_SCOPED_RUNS_ENABLED") != "true" or os.environ.get("MODEL_ENABLED") not in {"true", "false"}:
        raise ApiError(503, "SCOPED_RUNS_DISABLED")
    # The release/config is larger than Lambda's environment-variable limit;
    # package it as pinned JSON and keep only its path/hash in the environment.
    config = load_cloud_config()
    release = ScopedRunReleaseV2.model_validate(config["release"])
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=2, read_timeout=4)
    # Login credentials can carry an account ID, making new SDK versions prefer
    # an account endpoint. The approved ledger/transport pins the regional URL.
    db = boto3.client("dynamodb", region_name="us-east-1", config=cfg,
        endpoint_url="https://dynamodb.us-east-1.amazonaws.com")
    store = ScopedCloudDynamoRunStore(db, "kg-r72-query-state")
    metadata = DynamoLedgerStore(db, "kg-r72-metadata")
    controller = make_scoped_quote_controller(store, config["prices"])
    repo = ScopedRunRepository(store, clock=controller.ledger.clock, answer_budget=controller)

    def authorize(auth):
        if auth.tenant_id != config["assets"]["tenant_id"]:
            raise ApiError(403, "ACCESS_DENIED")
        return require_current_membership(auth, metadata)

    def check(value):
        if ScopedRunReleaseV2.model_validate(value) != release:
            raise ApiError(503, "COLLECTION_UNAVAILABLE")

    artifacts = ScopedArtifactRepository(S3RunBlobs(boto3.client("s3", region_name="us-east-1", config=cfg), config["results_bucket"]))
    coordinator = ScopedCloudCoordinator(repo, ScopedSqsSender(boto3.client("sqs", region_name="us-east-1", config=cfg),
        config["queue_url"]), authorize)
    service = ScopedCloudService(repo, artifacts, lambda: release, authorize=authorize, release_check=check, coordinator=coordinator)
    return config, service, metadata, coordinator


class ScopedRunApi:
    """Reuse verified authorization/parsing with a strictly separate route prefix."""

    def __init__(self, *args, **kwargs):
        self.base = OwnedRunApi(*args, **kwargs)

    def handler(self, event):
        path = event.get("rawPath", "")
        if path == "/v2/scoped-run-capabilities":
            mapped = "/v1/run-capabilities"
        elif path == "/v2/scoped-runs" or path.startswith("/v2/scoped-runs/"):
            mapped = "/v1/runs"+path[len("/v2/scoped-runs"):]
        else:
            return response(404, ApiError(404, "NOT_FOUND").body())
        return self.base.handler(event | {"rawPath": mapped})


def capabilities(config):
    release = ScopedRunReleaseV2.model_validate(config["release"])
    return {"schema_version": 2, "catalog_version": "owned-scoped-qualified-v2", "available": True,
        "collection_version": release.collection_version, "corpus": {"documents": release.document_count,
            "chunks": release.chunk_count}, "graph": {"facts": release.graph_fact_count,
            "source_documents": release.graph_source_document_count, "experimental": True, "human_reviewed": False},
        "methods": list(release.configs), "answer_methods": list(release.configs), "run_kinds": ["answer"],
        "maximum_methods": 3, "maximum_answer_methods": 3, "monthly_child_limit": 300,
        "answering_enabled": True, "source_scope": "local_collection", "semantic_verified": False,
        "quality_metrics": None, "qualification_states": ["NO_MECHANICAL_FLAGS", "REVIEW_REQUIRED", "NOT_READY"]}


def api_handler(event, context):
    try:
        config, service, metadata, _ = cloud_components()
        return ScopedRunApi(service, metadata, config["public_config"], capabilities=lambda auth: capabilities(config)).handler(event)
    except Exception:
        return response(503, ApiError(503, "SCOPED_STATE_UNAVAILABLE").body())


def watchdog_handler(event, context):
    if event != {"operation": "SCOPED_WATCHDOG_V2"}:
        raise ValueError("SCOPED_WATCHDOG_EVENT_INVALID")
    _, _, _, coordinator = cloud_components()
    report = coordinator.sweep()
    print(canonical_json({"event": "scoped_watchdog_v2", **report}).decode(), flush=True)
    if report["errors"] or report["unknown"]:
        raise RuntimeError("SCOPED_WATCHDOG_NEEDS_ATTENTION")
    return report


class ScopedCloudDispatch:
    def __init__(self, service, config, context, *, assets_factory=None, executor_factory=None):
        self.service, self.config, self.context = service, config, context
        self.assets_factory, self.executor_factory = assets_factory or cloud_assets, executor_factory or make_scoped_executor
        self.stack, self.executor = ExitStack(), None

    def close(self):
        self.stack.close()

    def _executor(self):
        if self.executor is None:
            assets = self.stack.enter_context(self.assets_factory(self.config))
            candidate = self.executor_factory(assets, self.service, self.config, self.context)
            candidate.check_release(self.config["release"])
            self.executor = candidate
        return self.executor

    def prepare(self, auth, parent):
        return self._executor().prepare(auth, parent)

    def safe_failure(self, auth, parent, method):
        return self.service.repo.answer_budget.safe_failure(auth, parent, method)

    def __call__(self, auth, parent, method, guard):
        guard()
        return self._executor()(auth, parent, method, guard)


def cloud_assets(config):
    import boto3
    from botocore.config import Config

    from app.scoped_cloud_assets import load_s3_scoped_assets

    client = boto3.client("s3", region_name="us-east-1", config=Config(retries={"total_max_attempts": 1},
        connect_timeout=2, read_timeout=8))
    return load_s3_scoped_assets(client, assets=config["assets"], graph_assets=config["graph_assets"],
        cache_root=Path("/tmp/kg-scoped-assets"))


def make_scoped_executor(assets, service, config, context):
    import time

    import boto3
    from botocore.config import Config

    from app.cloud_assets import read_deepseek_secret
    from app.embeddings import EmbeddingCache
    from app.external_gateway import ExternalModelGateway
    from app.external_http import OfficialHttpsTransport
    from app.generation import GenerationRecipeV9
    from app.local_bge import BgeEncoder
    from app.rag_runner import _validate_encoder
    from app.reranking import OnnxCrossEncoder
    from app.scoped_run_executor import ScopedOwnedAnswerExecutor

    if os.environ.get("MODEL_ENABLED") != "true":
        raise ApiError(503, "MODEL_DISABLED")
    controller = service.repo.answer_budget
    release = ScopedRunReleaseV2.model_validate(config["release"])
    service.repo.check_profile(release)
    base = assets.base
    encoder = BgeEncoder(base.root/"models/bge", base.root/"models/bge/manifest.json", threads=2)
    runtime = SimpleNamespace(recipe=GenerationRecipeV9(model_metadata_sha256=controller.routes.account_model_metadata_sha256),
        encoder=encoder, query_recipe=_validate_encoder(base.collection, encoder),
        cache=EmbeddingCache(Path("/tmp/kg-scoped-vector-cache")/base.bundle.version, namespace="LIVE"),
        reranker=OnnxCrossEncoder(base.root/"models/reranker", base.root/"models/reranker/manifest.json", threads=2))
    remaining = context.get_remaining_time_in_millis()/1000
    if remaining < 40:
        raise ApiError(503, "SCOPED_ANSWER_TIME_BUDGET")
    key = read_deepseek_secret(boto3.client("secretsmanager", region_name="us-east-1",
        config=Config(retries={"total_max_attempts": 1}, connect_timeout=2, read_timeout=3)),
        secret_arn=os.environ["KG_DEEPSEEK_SECRET_ARN"])
    transport = OfficialHttpsTransport({"DEEPSEEK": key}, timeout_seconds=25,
        deadline_monotonic=time.monotonic()+remaining-12)
    controller.bind_gateway(ExternalModelGateway(controller.ledger, transport, controller.routes, controller.prices))
    executor = ScopedOwnedAnswerExecutor(base.collection, base.lexical, assets.graph, runtime,
        controller=controller, repository=service.repo, authorize=service.authorize)
    executor.check_release(config["release"])
    return executor


def worker_handler(event, context):
    if type(event) is not dict or type(event.get("Records")) is not list or len(event["Records"]) != 1:
        raise ValueError("SCOPED_WORKER_REQUIRES_ONE_SQS_RECORD")
    if context.get_remaining_time_in_millis() < 65_000:
        raise ValueError("SCOPED_WORKER_TIME_BUDGET")
    config, service, _, _ = cloud_components()
    execute = ScopedCloudDispatch(service, config, context)
    worker = ScopedRunWorker(service.repo, service.artifacts, execute,
        authorize=service.authorize, release_check=service.release_check)
    try:
        return worker.sqs_batch(event)
    finally:
        execute.close()
