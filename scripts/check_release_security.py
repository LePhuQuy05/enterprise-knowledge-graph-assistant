"""Opt-in bounded, model-free AWS probes; never read passwords, tokens or provider keys."""

import argparse
import base64
import json
import re
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from botocore.config import Config

from app.ledger import DynamoLedgerStore, Ledger
from app.query_auth import AuthContext
from app.query_service import RUN
from app.query_state import QueryRepository
from app.query_telemetry import safe_fields
from scripts.foundation import _session

ROOT = Path(__file__).resolve().parents[1]
GROUP = "/aws/lambda/kg-r72-query"
UUID = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
EVENTS = {"query_stage", "query_complete", "query_runtime_cache", "query_model_receipt"}


def parse_report(line):
    rid = re.search(r"REPORT RequestId: (" + UUID + ")", line)
    if not rid:
        return None
    row = {"invocation_id": rid[1]}
    for key, label in (("duration_ms", "Duration"), ("billed_duration_ms", "Billed Duration"),
                       ("memory_mb", "Memory Size"), ("max_memory_used_mb", "Max Memory Used"),
                       ("init_duration_ms", "Init Duration")):
        match = re.search(r"(?:^|\t|  )" + label + r": ([0-9.]+) (?:ms|MB)", line)
        if match:
            row[key] = float(match[1])
    return row


def parse_event(line):
    try:
        row = json.loads(line)
    except (ValueError, TypeError):
        return None
    if not isinstance(row, dict) or type(row.get("event")) is not str or row.get("event") not in EVENTS or (
        type(row.get("schema_version")) is not int or row["schema_version"] != 1
    ):
        return None
    allowed = {"event", "schema_version", "timestamp"} | set(safe_fields(row))
    if set(row) != allowed or not re.fullmatch(r"[0-9T:.+Z\-]{20,40}", str(row.get("timestamp", ""))):
        raise ValueError("UNSAFE_TELEMETRY_SCHEMA")
    if len(line.encode()) > 2048:
        raise ValueError("OVERSIZED_TELEMETRY_EVENT")
    return row


def summarize_latency(values):
    return {"samples": len(values), "min_ms": min(values), "median_ms": round(statistics.median(values), 3),
            "max_ms": max(values), "scope": "Small read-only smoke sample, not a capacity or latency SLO."}


def run(session, output):
    output.mkdir(parents=True, exist_ok=False)

    def save(data):
        (output / "report.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=38)
    client, logs = session.client("lambda", config=cfg), session.client("logs", config=cfg)
    db = session.client("dynamodb", config=cfg)
    store = DynamoLedgerStore(db, "kg-r72-query-state")
    config = client.get_function_configuration(FunctionName="kg-r72-query")
    build = json.loads((ROOT / "results/query_package_build.json").read_bytes())
    if base64.b64decode(config["CodeSha256"]).hex() != build["sha256"] or (
        config["Handler"] != "app.query_handler.handler" or config["State"] != "Active"
        or config["LastUpdateStatus"] != "Successful"
    ):
        raise ValueError("EXPECTED_QUERY_CODE_NOT_ACTIVE")
    outputs = json.loads((ROOT / "results/query-cloud/deployment.json").read_bytes())["outputs"]
    users = session.client("cognito-idp", config=cfg).list_users(UserPoolId=outputs["UserPoolId"], Limit=10)["Users"]
    identities = {u["Username"]: next(a["Value"] for a in u["Attributes"] if a["Name"] == "sub")
                  for u in users if u["Username"] in {"demo-a", "demo-b"}}
    if set(identities) != {"demo-a", "demo-b"}:
        raise ValueError("EXPECTED_REVIEWERS_REQUIRED")
    prior = json.loads((ROOT / "results/query-cloud/prompt-v2/answer_smoke_response.json").read_bytes())["result"]
    rid = prior["request_id"]
    auth = AuthContext("demo", identities["demo-a"])
    row = store.get((auth.partition, "REQUEST#" + rid))
    if not row or row.get("status") != "COMPLETED":
        raise ValueError("EXISTING_COMPLETED_SMOKE_REQUIRED_NO_NEW_QUERY")
    chunk = row["allowed_chunk_ids"][0]
    period = datetime.now(UTC).strftime("%Y-%m")

    def counters():
        return {"funding": store.get(Ledger.external_funding_key("kg-r72")),
                "run": store.get(Ledger.run_key(RUN)), "quota": store.get(QueryRepository.quota_key(period))}

    before, calls, checks = counters(), [], []
    canary = "SYNTHETIC_DO_NOT_LOG_" + uuid4().hex
    start_ms = int(time.time() * 1000) - 1000
    report = {"status": "RUNNING", "started_at": datetime.now(UTC).isoformat(), "start_ms": start_ms,
        "package_sha256": build["sha256"], "direct_iam_synthetic_claims": True,
        "positive_browser_jwt_test": False, "new_model_calls_requested": 0,
        "canary": canary, "invocations": calls, "checks": checks}
    save(report)

    def event(method, path, *, user="demo-a", params=None, body=None, scope="kg/read kg/query"):
        return {"requestContext": {"http": {"method": method}, "authorizer": {"jwt": {"claims": {
            "sub": identities[user], "iss": "https://cognito-idp.us-east-1.amazonaws.com/" + outputs["UserPoolId"],
            "client_id": outputs["BrowserClientId"], "token_use": "access", "scope": scope,
            "exp": str(int(time.time()) + 300)}}}}, "rawPath": path,
            "headers": {"content-type": "application/json", "idempotency-key": str(uuid4()), "authorization": canary},
            "queryStringParameters": params, "body": json.dumps(body) if body is not None else None}

    def invoke(name, request, expected):
        started = time.monotonic()
        response = client.invoke(FunctionName="kg-r72-query", InvocationType="RequestResponse", LogType="Tail",
                                 Payload=json.dumps(request).encode())
        with response["Payload"] as stream:
            raw = stream.read(256 * 1024 + 1)
        if response.get("FunctionError") or len(raw) > 256 * 1024:
            raise ValueError("BOUNDED_PROBE_FUNCTION_FAILURE")
        wire = json.loads(raw)
        tail = base64.b64decode(response.get("LogResult", "")).decode("utf-8", errors="replace")
        metrics = next((v for line in tail.splitlines() if (v := parse_report(line))), None)
        events = [v for line in tail.splitlines() if (v := parse_event(line))]
        result = {"name": name, "expected_http_status": expected, "http_status": wire["statusCode"],
            "sdk_elapsed_ms": round((time.monotonic() - started) * 1000, 3),
            "lambda_report": metrics, "tail_events": events, "canary_absent_in_tail": canary not in tail}
        # No response/source/exception bodies are persisted. Compare only the expected control outcome.
        return result

    def probe(name, request, expected):
        result = invoke(name, request, expected)
        calls.append(result)
        checks.append({"name": name, "passed": result["http_status"] == expected and result["canary_absent_in_tail"]})
        save(report)

    try:
        for i in range(4):
            probe("source_read_" + str(i + 1), event("GET", "/v1/sources/" + chunk, params={"request_id": rid}), 200)
        probe("own_detail", event("GET", "/v1/queries/" + rid), 200)
        probe("other_owner_detail", event("GET", "/v1/queries/" + rid, user="demo-b"), 404)
        probe("other_owner_source", event("GET", "/v1/sources/" + chunk, user="demo-b", params={"request_id": rid}), 404)
        probe("unapproved_chunk", event("GET", "/v1/sources/chk_" + "0" * 64, params={"request_id": rid}), 404)
        probe("invalid_cursor", event("GET", "/v1/queries", params={"cursor": "not-a-cursor"}), 400)
        probe("invalid_history_limit", event("GET", "/v1/queries", params={"limit": "-1"}), 400)
        probe("extra_source_params", event("GET", "/v1/sources/" + chunk, params={"request_id": rid, "tenant_id": canary}), 400)
        request = event("POST", "/v1/queries", body={"question": canary}, scope="kg/read")
        probe("wrong_scope", request, 403)
        for kind in ("missing", "id", "expired"):
            request = event("GET", "/v1/status")
            if kind == "missing":
                request["requestContext"].pop("authorizer")
            else:
                request["requestContext"]["authorizer"]["jwt"]["claims"]["token_use" if kind == "id" else "exp"] = (
                    "id" if kind == "id" else "1")
            probe("auth_" + kind, request, 401)
        for name, body in (("forged_owner", {"question": canary, "owner_id": identities["demo-b"]}),
                           ("unavailable_graph", {"question": canary, "mode": "graph_2hop"}),
                           ("oversize_question", {"question": canary * 100})):
            probe(name, event("POST", "/v1/queries", body=body), 400)
        request = event("POST", "/v1/queries")
        request["body"] = '{"question":"' + canary + '","question":"duplicate"}'
        probe("duplicate_json_key", request, 400)
        # Small bounded read concurrency; never a provider/load test or an attempt to exhaust quota.
        with ThreadPoolExecutor(max_workers=2) as pool:
            rows = list(pool.map(lambda i: invoke("parallel_detail_" + str(i), event("GET", "/v1/queries/" + rid), 200), range(6)))
        for result in rows:
            calls.append(result)
            checks.append({"name": result["name"], "passed": result["http_status"] == 200 and result["canary_absent_in_tail"]})
        for method, path, token in (("GET", "/v1/status", None), ("GET", "/v1/queries", "Bearer fake.invalid.jwt"),
                                    ("POST", "/v1/queries", None)):
            request = Request(outputs["PublicUrl"].rstrip("/") + path, method=method,
                headers={"authorization": token} if token else {}, data=b'{}' if method == "POST" else None)
            try:
                with urlopen(request, timeout=10) as response:
                    status = response.status
            except HTTPError as exc:
                status = exc.code
                exc.close()
            checks.append({"name": "http_" + method + path + ("_forged" if token else "_missing"), "passed": status == 401})
        after = counters()
        report["model_and_admission_counters_unchanged"] = before == after
        checks.append({"name": "no_new_model_charge_or_admission", "passed": before == after})
        report["counter_snapshot"] = {"funding_settled_micro_usd": after["funding"]["settled_micro_usd"],
            "funding_reserved_micro_usd": after["funding"]["reserved_micro_usd"],
            "funding_cap_micro_usd": after["funding"]["cap_micro_usd"], "run_cap_micro_usd": after["run"]["cap_micro_usd"],
            "monthly_admitted_count": after["quota"]["count"] if after["quota"] else 0}
        report["source_read_sdk_latency"] = summarize_latency([r["sdk_elapsed_ms"] for r in calls if r["name"].startswith("source_read_")])
        report["status"] = "PROBES_PASS_PENDING_LOG_PERSISTENCE" if all(c["passed"] for c in checks) else "FAIL"
        save(report)
        collect_logs(logs, output)
    except Exception as exc:
        report.update(status="INCOMPLETE", error_category=type(exc).__name__)
        save(report)
        raise
    return json.loads((output / "report.json").read_bytes())


def collect_logs(logs, output):
    path = output / "report.json"
    report = json.loads(path.read_bytes())
    expected = {r["lambda_report"]["invocation_id"] for r in report["invocations"] if r["lambda_report"]}
    group = next((g for g in logs.describe_log_groups(logGroupNamePrefix=GROUP)["logGroups"] if g["logGroupName"] == GROUP), {})
    app_events, platform, unexpected, canary_absent = {}, {}, 0, True
    pages_truncated = False
    for attempt in range(8):
        args = {"logGroupName": GROUP, "startTime": report["start_ms"], "limit": 1000}
        for page in range(5):
            response = logs.filter_log_events(**args)
            for item in response["events"]:
                line = item["message"].strip()
                canary_absent &= report["canary"] not in line
                row = parse_event(line)
                if row and row.get("invocation_id") in expected:
                    app_events[item["eventId"]] = row
                metric = parse_report(line)
                if metric and metric["invocation_id"] in expected:
                    platform[metric["invocation_id"]] = metric
            token = response.get("nextToken")
            if not token or token == args.get("nextToken"):
                break
            args["nextToken"] = token
            if page == 4:
                pages_truncated = True
        complete = {r["invocation_id"] for r in app_events.values() if r["event"] == "query_complete"}
        if expected <= complete and expected <= platform.keys():
            break
        time.sleep(2)
    rows = list(app_events.values())
    unexpected = sum(r["event"] == "query_model_receipt" or (r["event"] == "query_stage" and r.get("stage") == "model_call") for r in rows)
    complete = {r["invocation_id"] for r in rows if r["event"] == "query_complete"}
    result = {"log_group": GROUP, "retention_days": group.get("retentionInDays"),
        "expected_invocations": len(expected), "persisted_completion_invocations": len(complete),
        "persisted_report_invocations": len(platform), "canary_absent_in_inspected_logs": canary_absent,
        "new_model_stages_or_receipts": unexpected, "bounded_pages_truncated": pages_truncated,
        "application_events": rows, "lambda_reports": list(platform.values())}
    passed = len(expected) == len(report["invocations"]) == 25 and len(report["checks"]) == 29 and (
        report.get("model_and_admission_counters_unchanged") is True) and expected <= complete and (
        expected <= platform.keys() and group.get("retentionInDays") == 7 and canary_absent
        and not unexpected and not pages_truncated)
    result["status"] = "PASS" if passed else "INCOMPLETE"
    (output / "cloudwatch.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    report.update(checked_at=datetime.now(UTC).isoformat(), cloudwatch_status=result["status"],
        status="PASS_MODEL_FREE_OPERATIONAL_SMOKE" if passed and all(c["passed"] for c in report["checks"]) else "INCOMPLETE",
        limitations=["Synthetic claims through IAM invocation do not prove positive HTTP JWT validation.",
            "No model calls: not a prompt-injection resistance or answer-quality benchmark.",
            "Six concurrent read tasks use at most two workers; this is not a load/capacity test.",
            "CloudWatch inspection is bounded to the probe time window; it is not a historic secret audit.",
            "Model/quota counter changes can reflect concurrent reviewer activity; investigate attribution before retrying."])
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Run read-only and rejected-input AWS probes, no model inference")
    parser.add_argument("--collect", type=Path, help="Recollect logs for an existing report without repeating invocations")
    args = parser.parse_args()
    if not args.execute and not args.collect:
        print(json.dumps({"status": "PLANNED_NOT_EXECUTED", "maximum_direct_invocations": 25, "model_calls": 0}))
        return 0
    base = ROOT / "results/release-security"
    if args.collect:
        output = args.collect.resolve()
        if not output.is_relative_to(base.resolve()) or not (output / "report.json").is_file():
            raise ValueError("EXPECTED_EXISTING_RELEASE_REPORT")
        result = collect_logs(_session("kg-dev").client("logs"), output)
    else:
        output = base / datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        result = run(_session("kg-dev"), output)
    print(json.dumps({"status": result["status"], "report": str(output / "report.json"),
                      "invocations": len(result["invocations"]), "cloudwatch_status": result.get("cloudwatch_status")}))
    return 0 if result["status"] == "PASS_MODEL_FREE_OPERATIONAL_SMOKE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
