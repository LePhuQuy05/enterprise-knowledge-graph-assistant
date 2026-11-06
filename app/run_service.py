"""Owned, bounded read projections for an unreleased retrieval run API."""

import base64
import re

from app.contracts import canonical_json
from app.external_http import strict_json
from app.owned_runs import RUN_ID, OwnedRunRequestV1
from app.query_auth import ApiError, authenticate, reauthorize
from app.query_language import answer_language
from app.run_execution import validate_result
from app.tracer import response


class OwnedRunService:
    def __init__(self, repository, artifacts, release_provider, *, authorize, release_check):
        self.repo, self.artifacts, self.release_provider = repository, artifacts, release_provider
        self.authorize, self.release_check = authorize, release_check

    def submit(self, auth, payload, key):
        try:
            # JSON arrays become tuples without weakening strict Python record validation.
            request = OwnedRunRequestV1.model_validate_json(canonical_json(payload))
        except Exception:
            raise ApiError(400, "INVALID_REQUEST") from None
        self.authorize(auth)
        row, fresh = self.repo.admit(auth, key, request, self.release_provider())
        return self.detail(auth, row["run_id"]) | {"idempotent_replay": not fresh}

    def detail(self, auth, run_id):
        self.authorize(auth)
        self.release_check(self.repo.get(auth, run_id)["release"])
        summary = self.repo.summary(auth, run_id)
        summary["query_transform"] = "NONE"
        summary["warnings"] = []
        request = summary["request"]
        answering = request["run_kind"] == "answer"
        if not answering:
            summary["warnings"].append("RETRIEVAL_ONLY_NO_GENERATION")
        if not answering and answer_language(request["original_question"], request["language"]) == "vi":
            summary["warnings"].append("DIRECT_VI_RETRIEVAL_DIAGNOSTIC_ENGLISH_ENCODER_ASCII_BM25")
        if answering:
            parent = self.repo.get(auth, run_id)
            summary["budget"] = self.repo.answer_budget.view(auth, parent)
            if answer_language(request["original_question"], request["language"]) == "vi":
                shared = next(s for s in summary["budget"]["stages"] if s["slot"] == "REWRITE:shared")
                # Settlement proves accounted usage, not that rewrite validation passed.
                summary["query_transform"] = "SHARED_REWRITE_SETTLED" if shared["status"] == "SETTLED" else "SHARED_REWRITE_PENDING"
        return summary

    def cancel(self, auth, run_id):
        self.authorize(auth)
        self.repo.cancel(auth, run_id)
        return self.detail(auth, run_id)

    def result(self, auth, run_id, method):
        self.detail(auth, run_id)
        child = self.repo.child(auth, run_id, method)
        if child["status"] != "COMPLETED":
            raise ApiError(409, "RUN_RESULT_UNAVAILABLE")
        raw = self.artifacts.read(auth, run_id, method, child["result_ref"])
        result = validate_result(self.repo.get(auth, run_id), method, strict_json(raw))
        self.authorize(auth)
        return result

    def public_result(self, auth, run_id, method):
        result = self.result(auth, run_id, method)
        public = {"method_id": method, "status": result.status, "config_sha256": result.config_sha256,
            "collection_version": result.collection_version, "ranked_chunk_ids": result.ranked_chunk_ids,
            "context_chunk_ids": result.context_chunk_ids, "context_proxy_tokens": result.context_proxy_tokens,
            "context_sources": [{k: v for k, v in source.items() if k != "text"} for source in result.context_sources],
            "quality_metrics": None, "new_cost_micro_usd": result.new_cost_micro_usd, "semantic_review": "NOT_REVIEWED"}
        public["execution"] = {"events": [event.model_dump(mode="json") for event in result.trace],
            "config": result.config.model_dump(mode="json"),
            "rerank": result.rerank_trace.model_dump(mode="json") if result.rerank_trace else None}
        if result.response is not None:
            public["run_id"] = run_id
            public["response"] = result.response.model_dump(mode="json")
        return public

    def source(self, auth, run_id, method, chunk_id):
        result = self.result(auth, run_id, method)
        source = next((s for s in result.context_sources if s["chunk_id"] == chunk_id), None)
        if source is None:
            raise ApiError(404, "SOURCE_NOT_FOUND")
        return source

    def trace(self, auth, run_id):
        detail = self.detail(auth, run_id)
        results = []
        for child in detail["methods"]:
            if child["result_available"]:
                result = self.result(auth, run_id, child["method_id"])
                results.append({"method_id": child["method_id"], "config_sha256": result.config_sha256,
                    "events": [x.model_dump(mode="json") for x in result.trace],
                    "rerank_trace": result.rerank_trace.model_dump(mode="json") if result.rerank_trace else None})
        return {"run_id": run_id, "methods": results, "quality_metrics": None}

    def history(self, auth, *, limit=10, cursor=None):
        self.authorize(auth)
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ApiError(400, "INVALID_LIMIT")
        after = None
        if cursor is not None:
            try:
                if type(cursor) is not str or len(cursor) > 2048 or not re.fullmatch(r"[A-Za-z0-9_-]+", cursor):
                    raise ValueError()
                value = strict_json(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
                if set(value) != {"PK", "SK", "exp"} or value["PK"] != auth.partition or (
                    type(value["exp"]) is not int or value["exp"] <= int(self.repo.now().timestamp())
                    or not re.fullmatch(r"RUNHIST#[0-9TZ:.\-]+#" + RUN_ID, value["SK"])
                ):
                    raise ValueError()
                after = value["SK"]
            except Exception:
                raise ApiError(400, "INVALID_CURSOR") from None
        rows, last = self.repo.store.run_history_keys(auth.partition, limit=limit, after=after)
        items = []
        for row in rows:
            try:
                items.append(self.detail(auth, row["SK"].split("#")[-1]))
            except ApiError as exc:
                if exc.status not in {404, 503}:
                    raise
        next_cursor = base64.urlsafe_b64encode(canonical_json({"PK": auth.partition, "SK": last,
            "exp": int(self.repo.now().timestamp()) + 3600})).decode().rstrip("=") if last else None
        return {"items": items, "next_cursor": next_cursor}


class OwnedRunApi:
    """Signature verification is upstream; no enrollment or new production routes here."""

    def __init__(self, service, metadata_store, public_config, *, capabilities=None):
        self.service, self.metadata_store, self.public_config = service, metadata_store, public_config
        self.capabilities = capabilities

    def handler(self, event):
        try:
            method = event.get("requestContext", {}).get("http", {}).get("method")
            path = event.get("rawPath", "")
            detail = re.fullmatch(r"/v1/runs/(" + RUN_ID + r")(?:(/cancel|/trace)|/methods/([a-z0-9_]+)(?:/sources/(chk_[0-9a-f]{64}))?)?", path)
            capability_read = path == "/v1/run-capabilities" and method == "GET" and self.capabilities is not None
            allowed = capability_read or (path == "/v1/runs" and method in {"GET", "POST"}) or (
                detail and ((method == "GET" and detail[2] != "/cancel") or (method == "POST" and detail[2] == "/cancel")))
            if not allowed:
                raise ApiError(404, "NOT_FOUND")
            scope = "kg/query" if method == "POST" else "kg/read"
            auth = authenticate(event, public_config=self.public_config, metadata_store=self.metadata_store, required_scope=scope)
            params = event.get("queryStringParameters") or {}
            if type(params) is not dict or (params and (path != "/v1/runs" or method != "GET")):
                raise ApiError(400, "INVALID_REQUEST")
            code = 200
            if method == "POST" and path == "/v1/runs":
                headers = {k.lower(): v for k, v in event.get("headers", {}).items()}
                body = event.get("body", "")
                if type(body) is not str or len(body.encode("utf-8")) > 16384 or event.get("isBase64Encoded") or (
                    headers.get("content-type", "").split(";")[0].strip().lower() != "application/json"
                ):
                    raise ApiError(400, "INVALID_REQUEST")
                try:
                    payload = strict_json(body.encode("utf-8"))
                except Exception:
                    raise ApiError(400, "INVALID_REQUEST") from None
                data = self.service.submit(auth, payload, headers.get("idempotency-key"))
                code = 202
            elif method == "POST":
                if event.get("body") not in (None, "", "{}") or event.get("isBase64Encoded"):
                    raise ApiError(400, "INVALID_REQUEST")
                data = self.service.cancel(auth, detail[1])
            elif capability_read:
                data = self.capabilities(auth)
            elif path == "/v1/runs":
                if set(params) - {"limit", "cursor"} or not re.fullmatch(r"[0-9]{1,2}", params.get("limit", "10")):
                    raise ApiError(400, "INVALID_REQUEST")
                data = self.service.history(auth, limit=int(params.get("limit", "10")), cursor=params.get("cursor"))
            elif detail[2] == "/trace":
                data = self.service.trace(auth, detail[1])
            elif detail[4]:
                data = self.service.source(auth, detail[1], detail[3], detail[4])
            elif detail[3]:
                data = self.service.public_result(auth, detail[1], detail[3])
            else:
                data = self.service.detail(auth, detail[1])
            reauthorize(auth, event, public_config=self.public_config, metadata_store=self.metadata_store, required_scope=scope)
            if len(canonical_json(data)) > 128 * 1024:
                raise ApiError(503, "RESPONSE_TOO_LARGE")
            result = response(code, data)
            if data.get("status") not in {"COMPLETED", "PARTIAL", "FAILED", "CANCELLED", "EXPIRED", "UNKNOWN"} and "run_id" in data:
                result["headers"]["retry-after"] = "2"
            return result
        except ApiError as exc:
            return response(exc.status, exc.body())
        except Exception:
            return response(503, ApiError(503, "RUN_STATE_UNAVAILABLE").body())
