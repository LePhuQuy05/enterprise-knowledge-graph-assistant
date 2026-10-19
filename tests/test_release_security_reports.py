import json

import pytest

from scripts.check_release_security import parse_event, parse_report, summarize_latency


def test_report_parser_keeps_numbers_and_request_id_only():
    line = ("REPORT RequestId: 11111111-2222-3333-4444-555555555555\tDuration: 126.54 ms\t"
        "Billed Duration: 200 ms\tMemory Size: 1024 MB\tMax Memory Used: 490 MB\tInit Duration: 70.12 ms")
    assert parse_report(line) == {"invocation_id": "11111111-2222-3333-4444-555555555555",
        "duration_ms": 126.54, "billed_duration_ms": 200, "memory_mb": 1024,
        "max_memory_used_mb": 490, "init_duration_ms": 70.12}
    assert parse_report("SYNTHETIC_PRIVATE_TEXT") is None


@pytest.mark.parametrize("field,value", [("question", "private"), ("input_tokens", -1),
                                        ("request_id", "private"), ("timestamp", "private")])
def test_cloudwatch_collector_rejects_unsafe_or_invalid_telemetry(field, value):
    row = {"event": "query_complete", "schema_version": 1,
           "timestamp": "2026-10-18T00:00:00+00:00", "status_code": 200}
    assert parse_event(json.dumps(row)) == row
    row[field] = value
    with pytest.raises(ValueError, match="UNSAFE_TELEMETRY_SCHEMA"):
        parse_event(json.dumps(row))


def test_sample_summary_is_explicitly_not_a_latency_slo():
    assert summarize_latency([30, 10, 20]) == {"samples": 3, "min_ms": 10, "median_ms": 20, "max_ms": 30,
        "scope": "Small read-only smoke sample, not a capacity or latency SLO."}


def test_log_collection_cannot_promote_an_incomplete_probe_run(tmp_path):
    from scripts.check_release_security import collect_logs

    class Logs:
        def describe_log_groups(self, **kwargs):
            return {"logGroups": [{"logGroupName": "/aws/lambda/kg-r72-query", "retentionInDays": 7}]}

        def filter_log_events(self, **kwargs):
            return {"events": []}

    # No expected invocations: log collection ends immediately but must not report an empty PASS.
    (tmp_path / "report.json").write_text(json.dumps({"status": "INCOMPLETE", "start_ms": 1,
        "canary": "synthetic", "invocations": [], "checks": []}))
    assert collect_logs(Logs(), tmp_path)["status"] == "INCOMPLETE"
