"""Bounded JSON-line telemetry: closed fields, no request/source/provider payloads."""

import json
import math
import re
import time
from contextlib import contextmanager
from datetime import UTC, datetime

STAGES = frozenset({"authorize", "authorization_refresh", "release_read", "admission", "runtime_load", "query_embedding",
    "retrieval_context", "gateway_setup", "model_call", "receipt_write", "answer_validation",
    "final_write", "detail_read", "history_read", "source_read", "status_read", "query_rewrite", "official_search"})
ROUTES = frozenset({"query_create", "query_list", "query_detail", "source", "status", "other"})
ERRORS = frozenset({"NOT_FOUND", "INVALID_REQUEST", "MODE_UNAVAILABLE", "ACCESS_TOKEN_REQUIRED",
    "SCOPE_REQUIRED", "AUTHORIZATION_UNAVAILABLE", "ACTIVE_MEMBERSHIP_REQUIRED", "ACCESS_DENIED",
    "INVALID_IDEMPOTENCY_KEY", "IDEMPOTENCY_KEY_EXPIRED", "IDEMPOTENCY_CONFLICT", "QUERY_QUOTA_EXCEEDED",
    "STATE_BUSY", "QUERY_NOT_FOUND", "REQUEST_ALREADY_TERMINAL", "RECEIPT_ALREADY_PRESENT",
    "INVALID_LIMIT", "INVALID_CURSOR", "REQUEST_IN_PROGRESS", "RECONCILIATION_REQUIRED",
    "SNAPSHOT_UNAVAILABLE", "QUERY_TIMEOUT", "QUESTION_TOKEN_LIMIT", "MODEL_UNAVAILABLE",
    "MODEL_DISABLED", "BUDGET_LIMIT", "GENERATION_INVALID", "MODEL_OR_STATE_UNAVAILABLE",
    "SOURCE_NOT_FOUND", "RESPONSE_TOO_LARGE", "STATE_UNAVAILABLE", "INTERNAL_ERROR"})
PATTERNS = {"invocation_id": r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}",
    "request_id": r"q_[0-9a-f]{64}", "snapshot_version": r"snap_[0-9a-f]{64}",
    "operation_id": r"op_[0-9a-f]{64}", "recipe_sha256": r"[0-9a-f]{64}"}
ENUMS = {"route": ROUTES, "stage": STAGES, "stage_status": {"OK", "ERROR"},
    "record_status": {"PROCESSING", "COMPLETED", "FAILED"},
    "answer_status": {"ANSWERED", "PARTIAL", "ABSTAINED"}, "mode": {"vector", "hybrid"},
    "error_code": ERRORS}
BOOLEANS = frozenset({"cold_start", "runtime_cache_hit", "idempotent_replay"})
LIMITS = {"elapsed_ms": 120_000, "status_code": 599, "input_tokens": 1_000_000,
          "output_tokens": 1_000_000, "cost_micro_usd": 2_000_000, "context_chunks": 1000}


def route_label(event):
    """Return a fixed label; never reflect raw paths or query parameters into logs."""
    method = event.get("requestContext", {}).get("http", {}).get("method")
    path = event.get("rawPath", "")
    if method == "POST" and path == "/v1/queries":
        return "query_create"
    if method == "GET":
        if path in {"/v1/status", "/v1/queries"}:
            return "status" if path == "/v1/status" else "query_list"
        if type(path) is str and re.fullmatch(r"/v1/queries/q_[0-9a-f]{64}", path):
            return "query_detail"
        if type(path) is str and re.fullmatch(r"/v1/sources/chk_[0-9a-f]{64}", path):
            return "source"
    return "other"


def safe_fields(fields):
    result = {}
    for key, value in fields.items():
        if key in PATTERNS and type(value) is str and re.fullmatch(PATTERNS[key], value):
            result[key] = value
        elif key in ENUMS and type(value) is str and value in ENUMS[key]:
            result[key] = value
        elif key in BOOLEANS and type(value) is bool:
            result[key] = value
        elif key in LIMITS and type(value) in {int, float} and 0 <= value <= LIMITS[key] and math.isfinite(value):
            if key == "elapsed_ms":
                result[key] = round(value, 3)
            elif type(value) is int and (key != "status_code" or value >= 100):
                result[key] = value
    return result


def write_event(row):
    # Existing Lambda logging mode stays TEXT; application messages are standalone JSON lines.
    # Do not configure/debug the root logger, boto3 or HTTP transport with request payloads.
    print(json.dumps(row, separators=(",", ":"), allow_nan=False), flush=True)


class NullTelemetry:
    def bind(self, **fields):
        pass

    def event(self, name, **fields):
        pass

    def complete(self, response):
        pass

    @contextmanager
    def stage(self, name):
        yield


class QueryTelemetry:
    def __init__(self, *, invocation_id=None, route="other", cold_start=False, sink=write_event, clock=time.monotonic):
        self.sink, self.clock = sink, clock
        self.started = clock()
        self.context = safe_fields({"invocation_id": invocation_id, "route": route, "cold_start": cold_start})
        self.finished = False

    def bind(self, **fields):
        # Identity is intentionally absent: request_id correlates to access-controlled durable state.
        self.context.update(safe_fields({k: v for k, v in fields.items()
            if k in {"request_id", "snapshot_version", "mode", "recipe_sha256"}}))

    def event(self, name, **fields):
        if type(name) is not str or name not in {"query_stage", "query_model_receipt", "query_runtime_cache", "query_complete"}:
            return
        try:
            row = self.context | safe_fields(fields) | {"event": name, "schema_version": 1,
                "timestamp": datetime.now(UTC).isoformat()}
            if len(json.dumps(row).encode()) <= 2048:
                self.sink(row)
        except Exception:
            pass  # Telemetry failure must never trigger a second provider dispatch or change durable state.

    @contextmanager
    def stage(self, name):
        started, status = self.clock(), "OK"
        try:
            yield
        except BaseException:
            status = "ERROR"
            raise
        finally:
            self.event("query_stage", stage=name, stage_status=status, elapsed_ms=(self.clock() - started) * 1000)

    def complete(self, response):
        if self.finished:
            return
        self.finished = True
        fields = {"status_code": response.get("statusCode"), "elapsed_ms": (self.clock() - self.started) * 1000}
        try:
            data = json.loads(response["body"])
            if isinstance(data, dict):
                fields.update(request_id=data.get("request_id"), record_status=data.get("status"),
                    idempotent_replay=data.get("idempotent_replay"))
                if isinstance(data.get("response"), dict):
                    fields["answer_status"] = data["response"].get("answer_status")
                if isinstance(data.get("error"), dict):
                    code = data["error"].get("code")
                    fields["error_code"] = code if type(code) is str and code in ERRORS else "INTERNAL_ERROR"
        except (ValueError, TypeError, KeyError):
            fields["error_code"] = "INTERNAL_ERROR"
        self.event("query_complete", **fields)
