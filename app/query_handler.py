"""AWS serving composition; static routes stay light, queries load pinned BGE lazily."""

import json
import os
import time
from pathlib import Path

import boto3
from botocore.config import Config

from app import tracer
from app.account_registration import VerifiedEmailEnrollment
from app.aws_knowledge import AwsKnowledgeClient
from app.cloud_assets import load_s3_snapshot, read_deepseek_secret
from app.contracts import sha256_bytes
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport
from app.generation import GenerationRecipeV9
from app.ledger import DynamoLedgerStore, Ledger
from app.query_api import QueryApi
from app.query_auth import ApiError, reauthorize
from app.query_service import RUN, QueryService
from app.query_state import DynamoQueryStore, QueryRepository
from app.query_telemetry import QueryTelemetry, route_label
from app.releases import ReleaseRegistry

ROOT = Path(__file__).resolve().parents[1]
_RUNTIME = None
_CLIENTS = None
_STATIC_LOADED = False
_COLD_START = True


def clients():
    global _CLIENTS
    if _CLIENTS is None:
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=1, read_timeout=2)
        _CLIENTS = {name: boto3.client(name, region_name="us-east-1", config=cfg)
                    for name in ("dynamodb", "secretsmanager", "s3", "cognito-idp")}
    return _CLIENTS


def load_runtime(release):
    global _RUNTIME
    from app.bge_model_store import ensure_bge_bundle
    from app.local_bge import BgeEncoder

    config = release["asset_config"]
    manifest = ROOT / "configs/models/bge-small-en-v1.5-onnx.json"
    if config["model_manifest_sha256"] != sha256_bytes(manifest.read_bytes()):
        raise ApiError(503, "SNAPSHOT_UNAVAILABLE")
    cache_key = release["tenant_id"], release["manifest_sha256"]
    if _RUNTIME is None or _RUNTIME[0] != cache_key:
        directory = ensure_bge_bundle(clients()["s3"], bucket=config["bucket"], manifest_path=manifest,
                                      cache_root=Path("/tmp/kg-models"))
        encoder = BgeEncoder(directory, manifest, threads=2)
        snapshot = load_s3_snapshot(clients()["s3"], bucket=config["bucket"], prefix=config["snapshot_prefix"],
            manifest_sha256=release["manifest_sha256"], manifest_bytes=config["snapshot_manifest_bytes"],
            snapshot_version=release["snapshot_version"], tenant_id=release["tenant_id"], cache_root=Path("/tmp/kg-snapshots"))
        if snapshot.manifest.embedding_recipe != encoder.recipe("document"):
            raise ApiError(503, "SNAPSHOT_UNAVAILABLE")
        _RUNTIME = cache_key, encoder, snapshot
    return _RUNTIME[1:]


def gateway(timeout, *, deadline_monotonic=None):
    if os.environ.get("MODEL_ENABLED") != "true":
        raise ApiError(503, "MODEL_DISABLED")
    ledger = Ledger(DynamoLedgerStore(clients()["dynamodb"], "kg-r72-query-state"))
    for key, cap in ((ledger.run_key(RUN), 500_000), (ledger.external_funding_key("kg-r72"), 2_000_000)):
        row = ledger.store.get(key)
        if not row or row.get("cap_micro_usd") != cap or row.get("halted"):
            raise ApiError(429, "BUDGET_LIMIT")
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes())
    key = read_deepseek_secret(clients()["secretsmanager"], secret_arn=os.environ["KG_DEEPSEEK_SECRET_ARN"])
    transport = OfficialHttpsTransport({"DEEPSEEK": key}, timeout_seconds=timeout,
                                       deadline_monotonic=deadline_monotonic)
    return ExternalModelGateway(ledger, transport, routes.model_copy(update={
        "live_inference_enabled": True, "live_inference_gate": "AUTHENTICATED_DURABLE_QUERY_ADMISSION"}), prices)


def handler(event, context):
    global _STATIC_LOADED, _COLD_START
    started = time.monotonic()
    cold_start, _COLD_START = _COLD_START, False
    method = event.get("requestContext", {}).get("http", {}).get("method")
    path = event.get("rawPath", "")
    if method == "GET" and (path in {"/", "/auth/callback", "/config.json"} or path.startswith("/assets/")):
        if not _STATIC_LOADED:
            tracer.STATIC_ASSETS = json.loads((ROOT / "configs/frontend-static.json").read_bytes())
            _STATIC_LOADED = True
        return tracer.handler(event, context)
    telemetry = QueryTelemetry(invocation_id=getattr(context, "aws_request_id", None),
                               route=route_label(event), cold_start=cold_start)
    try:
        if os.environ.get("MODEL_ENABLED") != "true":
            raise ApiError(503, "MODEL_DISABLED")
        metadata = DynamoLedgerStore(clients()["dynamodb"], "kg-r72-metadata")
        repo = QueryRepository(DynamoQueryStore(clients()["dynamodb"], "kg-r72-query-state"))
        routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        recipe = GenerationRecipeV9(model_metadata_sha256=routes["account_model_metadata_sha256"])
        def runtime(release):
            key = release["tenant_id"], release["manifest_sha256"]
            telemetry.event("query_runtime_cache", runtime_cache_hit=_RUNTIME is not None and _RUNTIME[0] == key)
            return load_runtime(release)

        def authorization_check(auth):
            reauthorize(auth, event, public_config=tracer.PUBLIC_CONFIG, metadata_store=metadata,
                        required_scope="kg/query" if method == "POST" else "kg/read")

        def gateway_factory(timeout):
            # Absolute monotonic boundary is set before SDK setup/reserve, not reset at socket dispatch.
            return gateway(timeout, deadline_monotonic=time.monotonic() + max(0.0, remaining() - 5))

        service = QueryService(repo, ReleaseRegistry(metadata), runtime, gateway_factory, recipe,
                               authorization_check=authorization_check, telemetry=telemetry,
                               knowledge_factory=lambda timeout: AwsKnowledgeClient(timeout=timeout,
                                   deadline=time.monotonic() + max(0, remaining() - 8)))
        enrollment = (VerifiedEmailEnrollment(clients()["cognito-idp"], metadata, tracer.PUBLIC_CONFIG)
                      if tracer.PUBLIC_CONFIG.get("registration_enabled") is True else None)
        api = QueryApi(service, metadata, tracer.PUBLIC_CONFIG, telemetry=telemetry, enroll_member=enrollment)

        def remaining():
            return max(0, min(24 - (time.monotonic() - started), context.get_remaining_time_in_millis() / 1000 - 2))

        return api.handler(event, remaining_seconds=remaining)
    except ApiError as exc:
        result = tracer.response(exc.status, exc.body())
    except Exception:
        result = tracer.response(503, ApiError(503, "STATE_UNAVAILABLE").body())
    telemetry.complete(result)
    return result
