import json
from types import SimpleNamespace

import pytest

from app.query_telemetry import QueryTelemetry, route_label, safe_fields

INVOCATION = "11111111-2222-3333-4444-555555555555"
REQUEST = "q_" + "a" * 64
SECRET = "SYNTHETIC_SECRET_NEVER_LOG_9ac2"


def test_closed_fields_strip_payloads_injection_and_unbounded_numbers():
    fields = {"request_id": REQUEST, "elapsed_ms": 12.34567, "input_tokens": 30,
        "cold_start": True, "mode": "hybrid", "question": SECRET, "Authorization": SECRET,
        "raw_output": SECRET, "tenant_id": SECRET, "user_id": SECRET, "error": SECRET,
        "error_code": SECRET, "operation_id": SECRET, "snapshot_version": SECRET,
        "recipe_sha256": SECRET, "stage": SECRET, "status_code": True, "cost_micro_usd": 10**500,
        "output_tokens": float("nan"), "context_chunks": float("inf"), "runtime_cache_hit": "true"}
    assert safe_fields(fields) == {"request_id": REQUEST, "elapsed_ms": 12.346, "input_tokens": 30,
                                   "cold_start": True, "mode": "hybrid"}
    assert SECRET not in json.dumps(safe_fields(fields))


def test_stage_failure_is_timed_without_exception_text_and_complete_once():
    rows, clock = [], [1.0]
    trace = QueryTelemetry(invocation_id=INVOCATION, route="query_create", cold_start=True,
                           sink=rows.append, clock=lambda: clock[0])
    trace.bind(request_id=REQUEST, question=SECRET, mode="hybrid")
    with pytest.raises(RuntimeError):
        with trace.stage("model_call"):
            clock[0] += 0.25
            raise RuntimeError(SECRET)
    response = {"statusCode": 503, "body": json.dumps({"request_id": REQUEST,
        "error": {"code": "MODEL_OR_STATE_UNAVAILABLE", "message": SECRET}, "question": SECRET})}
    trace.complete(response)
    trace.complete(response)
    assert [r["event"] for r in rows] == ["query_stage", "query_complete"]
    assert rows[0]["stage_status"] == "ERROR" and rows[0]["elapsed_ms"] == 250
    assert rows[1]["error_code"] == "MODEL_OR_STATE_UNAVAILABLE"
    assert all(r["invocation_id"] == INVOCATION and r["request_id"] == REQUEST for r in rows)
    assert SECRET not in json.dumps(rows) and all(len(json.dumps(r).encode()) <= 2048 for r in rows)


def test_broken_sink_never_changes_request_control_flow():
    def broken_sink(row):
        raise OSError(SECRET)

    trace = QueryTelemetry(sink=broken_sink)
    with trace.stage("runtime_load"):
        pass
    trace.complete({"statusCode": 503, "body": '{"error":{"code":"STATE_UNAVAILABLE"}}'})
    assert trace.finished


@pytest.mark.parametrize("method,path,expected", [
    ("POST", "/v1/queries", "query_create"), ("GET", "/v1/queries", "query_list"),
    ("GET", "/v1/status", "status"), ("GET", "/v1/queries/" + REQUEST, "query_detail"),
    ("GET", "/v1/sources/chk_" + "b" * 64, "source"),
    ("GET", "/v1/queries/" + SECRET, "other"), ("DELETE", "/v1/queries", "other")])
def test_route_label_never_reflects_paths(method, path, expected):
    assert route_label({"requestContext": {"http": {"method": method}}, "rawPath": path}) == expected


def test_unknown_error_text_and_invalid_event_are_not_logged():
    rows = []
    trace = QueryTelemetry(invocation_id=SECRET, sink=rows.append)
    trace.event(SECRET, question=SECRET)
    trace.event([], question=SECRET)
    trace.complete({"statusCode": 500, "body": json.dumps({"error": {"code": SECRET}})})
    assert len(rows) == 1 and rows[0]["error_code"] == "INTERNAL_ERROR"
    assert "invocation_id" not in rows[0] and SECRET not in json.dumps(rows)


def test_default_sink_emits_one_valid_json_line(capsys):
    trace = QueryTelemetry(invocation_id=INVOCATION)
    trace.complete({"statusCode": 401, "body": '{"error":{"code":"ACCESS_TOKEN_REQUIRED"}}'})
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1 and json.loads(lines[0])["error_code"] == "ACCESS_TOKEN_REQUIRED"


def test_runtime_composition_emits_closed_event_on_early_failure(monkeypatch, capsys):
    from app import query_handler

    monkeypatch.setenv("MODEL_ENABLED", "false")
    context = SimpleNamespace(aws_request_id=INVOCATION)
    result = query_handler.handler({"rawPath": "/v1/queries", "body": SECRET,
        "requestContext": {"http": {"method": "POST"}}, "headers": {"Authorization": SECRET}}, context)
    row = json.loads(capsys.readouterr().out)
    assert result["statusCode"] == 503 and row["error_code"] == "MODEL_DISABLED"
    assert row["invocation_id"] == INVOCATION and SECRET not in json.dumps(row)
