"""Bounded HTTP API v2 adapter; no browser-controlled identity, model or bucket."""

import json
import re

from app.external_http import strict_json
from app.query_auth import ApiError, authenticate, reauthorize
from app.query_state import REQUEST_ID
from app.query_telemetry import NullTelemetry
from app.tracer import response


class QueryApi:
    def __init__(self, service, metadata_store, public_config, *, telemetry=None, enroll_member=None):
        self.service, self.metadata_store, self.public_config = service, metadata_store, public_config
        self.telemetry = telemetry or NullTelemetry()
        self.enroll_member = enroll_member

    def handle(self, event):
        try:
            method = event.get("requestContext", {}).get("http", {}).get("method")
            path = event.get("rawPath", "")
            detail = re.fullmatch(r"/v1/queries/(" + REQUEST_ID + ")", path)
            source = re.fullmatch(r"/v1/sources/(chk_[0-9a-f]{64})", path)
            if not ((method == "GET" and (path in {"/v1/status", "/v1/queries"} or detail or source)) or (
                method == "POST" and path == "/v1/queries"
            )):
                raise ApiError(404, "NOT_FOUND")
            required_scope = "kg/query" if method == "POST" else "kg/read"
            if path == "/v1/status" and event.get("queryStringParameters"):
                raise ApiError(400, "INVALID_REQUEST")
            with self.telemetry.stage("authorize"):
                auth = authenticate(event, public_config=self.public_config, metadata_store=self.metadata_store,
                                    required_scope=required_scope,
                                    enroll_member=self.enroll_member if path == "/v1/status" else None)
            params = event.get("queryStringParameters") or {}
            if not isinstance(params, dict):
                raise ApiError(400, "INVALID_REQUEST")
            if method == "POST":
                headers = {k.lower(): v for k, v in event.get("headers", {}).items()}
                if params or headers.get("content-type", "").split(";")[0].strip().lower() != "application/json":
                    raise ApiError(400, "INVALID_REQUEST")
                body = event.get("body", "")
                if type(body) is not str or event.get("isBase64Encoded") or len(body.encode("utf-8")) > 16 * 1024:
                    raise ApiError(400, "INVALID_REQUEST")
                try:
                    payload = strict_json(body.encode("utf-8"))
                except Exception:
                    raise ApiError(400, "INVALID_REQUEST") from None
                data = self.service.submit(auth, payload, headers.get("idempotency-key"),
                                           remaining_seconds=self.remaining_seconds)
            elif path == "/v1/status":
                if params:
                    raise ApiError(400, "INVALID_REQUEST")
                with self.telemetry.stage("status_read"):
                    release = self.service.releases.current(auth.tenant_id)
                    self.telemetry.bind(snapshot_version=release["snapshot_version"])
                    data = {"status": "READY", "tenant_id": auth.tenant_id, "user_id": auth.user_id,
                        "fixture_only": False, "model_enabled": True, "available_modes": ["vector", "hybrid"],
                        "snapshot_version": release["snapshot_version"], "published_at": release["published_at"],
                        "quota_remaining": self.service.repo.quota_remaining(),
                        "message": "Document answers, citations and your own history are available."}
                    data["available_corpus_scopes"] = ["snapshot"] + (["aws_official"] if self.service.knowledge_factory else [])
                    data["answer_languages"] = ["auto", "en", "vi"]
            elif path == "/v1/queries":
                if set(params) - {"limit", "cursor"} or not re.fullmatch(r"[0-9]{1,2}", params.get("limit", "10")):
                    raise ApiError(400, "INVALID_REQUEST")
                data = self.service.history(auth, limit=int(params.get("limit", "10")), cursor=params.get("cursor"))
            elif detail:
                if params:
                    raise ApiError(400, "INVALID_REQUEST")
                data = self.service.detail(auth, detail[1])
            else:
                if set(params) != {"request_id"}:
                    raise ApiError(400, "INVALID_REQUEST")
                data = self.service.source(auth, params["request_id"], source[1])
            with self.telemetry.stage("authorization_refresh"):
                reauthorize(auth, event, public_config=self.public_config, metadata_store=self.metadata_store,
                            required_scope=required_scope)
            if len(json.dumps(data).encode()) > 128 * 1024:
                raise ApiError(503, "RESPONSE_TOO_LARGE")
            return response(200, data)
        except ApiError as exc:
            result = response(exc.status, exc.body())
            if exc.code == "REQUEST_IN_PROGRESS":
                result["headers"]["retry-after"] = "2"
            return result
        except Exception:
            return response(503, ApiError(503, "STATE_UNAVAILABLE").body())

    def handler(self, event, *, remaining_seconds):
        # One QueryApi object per invocation: deadline closure is never shared across requests.
        self.remaining_seconds = remaining_seconds
        result = self.handle(event)
        self.telemetry.complete(result)
        return result
