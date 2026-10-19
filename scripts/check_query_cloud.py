"""One opt-in live cloud answer, replay/isolation probes and managed reconciliation.

IAM direct invocations carry synthetic authorizer claims: this is not a positive HTTP JWT/browser test.
Never reads reviewer passwords, bearer tokens or provider API keys.
"""

import argparse
import json
import time
from datetime import UTC, datetime
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import uuid4

from botocore.config import Config

from app.ledger import DynamoLedgerStore, Ledger
from app.query_auth import AuthContext
from app.query_service import RUN
from scripts.foundation import _session
from scripts.query_cloud import OUTPUT, ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="One paid query within the existing 0.50 USD run/2 USD funding")
    parser.add_argument("--experiment", choices=["initial", "prompt-v2"], default="initial",
                        help="Explicit versioned experiment after diagnosing the initial failure; no runtime repair/retry")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PLANNED_NOT_EXECUTED", "maximum_new_paid_queries": 1}))
        return 0
    target = OUTPUT if args.experiment == "initial" else OUTPUT / "prompt-v2"

    def save_report(name, data):
        target.mkdir(parents=True, exist_ok=True)
        (target / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        return data

    intent_path = target / "smoke_intent.json"
    if intent_path.exists():
        raise ValueError("EXISTING_SMOKE_INTENT_DO_NOT_REPEAT_OR_RETRY_MODEL")
    session = _session("kg-dev")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=38)
    client = session.client("lambda", config=cfg)
    outputs = json.loads((ROOT / "results/tracer/deployment.json").read_bytes())["outputs"]
    identities = {}
    for row in session.client("cognito-idp", config=cfg).list_users(UserPoolId=outputs["UserPoolId"], Limit=10)["Users"]:
        if row["Username"] in {"demo-a", "demo-b"}:
            identities[row["Username"]] = next(a["Value"] for a in row["Attributes"] if a["Name"] == "sub")
    if set(identities) != {"demo-a", "demo-b"}:
        raise ValueError("EXPECTED_REVIEWER_IDENTITIES_REQUIRED")
    query_store = DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state")
    ledger = Ledger(query_store)
    funding_key, run_key = ledger.external_funding_key("kg-r72"), ledger.run_key(RUN)
    before = {"funding": query_store.get(funding_key), "run": query_store.get(run_key)}
    key = str(uuid4())
    save_report("smoke_intent.json", {"status": "STARTED_NO_AUTOMATIC_RETRY", "idempotency_key": key,
        "checked_at": datetime.now(UTC).isoformat(), "maximum_new_paid_queries": 1,
        "positive_jwt_test": False, "direct_iam_invocation": True, "experiment": args.experiment})
    checks, calls = [], []

    def event(method, path, user="demo-a", *, body=None, params=None):
        return {"rawPath": path, "requestContext": {"http": {"method": method}, "authorizer": {"jwt": {"claims": {
            "sub": identities[user], "iss": "https://cognito-idp.us-east-1.amazonaws.com/" + outputs["UserPoolId"],
            "client_id": outputs["BrowserClientId"], "token_use": "access", "exp": str(int(time.time()) + 300),
            "scope": "kg/read kg/query"}}}}, "headers": {"content-type": "application/json", "idempotency-key": key},
            "body": json.dumps(body) if body is not None else None, "queryStringParameters": params}

    def invoke(name, request):
        started = time.monotonic()
        result = client.invoke(FunctionName="kg-r72-query", InvocationType="RequestResponse", Payload=json.dumps(request).encode())
        stream = result["Payload"]
        try:
            raw = stream.read(256 * 1024 + 1)
        finally:
            stream.close()
        if result.get("FunctionError") or len(raw) > 256 * 1024:
            raise ValueError("FUNCTION_OR_RESPONSE_SIZE_FAILURE")
        wire = json.loads(raw)
        data = json.loads(wire["body"])
        calls.append({"check": name, "http_status": wire["statusCode"], "elapsed_seconds": round(time.monotonic() - started, 3)})
        save_report("smoke_progress.json", {"status": "RUNNING", "invocations": calls, "checks": checks})
        return wire["statusCode"], data

    try:
        request = event("POST", "/v1/queries", body={
            "question": "Which policies must permit cross-account use of an AWS KMS key?", "mode": "hybrid"})
        code, data = invoke("one_live_query", request)
        save_report("answer_smoke_response.json", {"http_status": code, "result": data,
            "limitation": "Paid model response; structural/source checks only, semantic human review pending."})
        if code != 200 or data.get("status") != "COMPLETED":
            raise ValueError("FIRST_QUERY_FAILED_NO_NEW_MODEL_ATTEMPT")
        rid = data["request_id"]
        auth = AuthContext("demo", identities["demo-a"])
        record = query_store.get((auth.partition, "REQUEST#" + rid))
        if not record or not record.get("receipt"):
            raise ValueError("DURABLE_QUERY_RECEIPT_REQUIRED")
        operation = ledger.operation(record["receipt"]["operation_id"])
        if operation.status != "SETTLED":
            raise ValueError("QUERY_OPERATION_NOT_SETTLED")
        settled = {"funding": query_store.get(funding_key), "run": query_store.get(run_key)}
        code, replay = invoke("same_key_replay", request)
        checks.append({"name": "replay_same_response", "passed": code == 200 and replay.get("idempotent_replay") is True
                       and replay.get("response") == data["response"]})
        for owner, expected in (("demo-a", 200), ("demo-b", 404)):
            status, _ = invoke("detail_" + owner, event("GET", "/v1/queries/" + rid, owner))
            checks.append({"name": "detail_" + owner, "passed": status == expected})
        citations = data["response"]["citations"]
        if citations:
            source_path = "/v1/sources/" + citations[0]["chunk_id"]
            for owner, expected in (("demo-a", 200), ("demo-b", 404)):
                status, source = invoke("source_" + owner, event("GET", source_path, owner, params={"request_id": rid}))
                checks.append({"name": "source_" + owner, "passed": status == expected and (
                    owner != "demo-a" or citations[0]["quote"] in source.get("text", ""))})
        for owner in ("demo-a", "demo-b"):
            status, history = invoke("history_" + owner, event("GET", "/v1/queries", owner))
            found = any(item["request_id"] == rid for item in history.get("items", []))
            checks.append({"name": "history_" + owner, "passed": status == 200 and found == (owner == "demo-a")})
        no_claims = event("GET", "/v1/status")
        no_claims["requestContext"].pop("authorizer")
        checks.append({"name": "missing_claims", "passed": invoke("missing_claims", no_claims)[0] == 401})
        for path in ("/v1/status", "/v1/queries", "/v1/queries/" + rid):
            try:
                with urlopen(Request(outputs["PublicUrl"].rstrip("/") + path), timeout=10) as result:
                    status = result.status
            except HTTPError as exc:
                status = exc.code
                exc.close()
            checks.append({"name": "http_no_token_" + path, "passed": status == 401})
        after = {"funding": query_store.get(funding_key), "run": query_store.get(run_key)}
        checks.append({"name": "replay_reads_no_new_model_charge", "passed": settled == after})
        report = {"status": "PASS" if all(c["passed"] for c in checks) else "FAIL", "checks": checks,
            "invocations": calls, "request_id": rid, "operation_id": operation.operation_id,
            "new_paid_query_calls": 1, "new_usage_upper_bound_micro_usd": operation.actual_micro_usd,
            "funding_settled_micro_usd": after["funding"]["settled_micro_usd"],
            "funding_reserved_micro_usd": after["funding"]["reserved_micro_usd"],
            "funding_settled_delta_micro_usd": after["funding"]["settled_micro_usd"] - before["funding"]["settled_micro_usd"],
            "run_cap_micro_usd": after["run"]["cap_micro_usd"], "funding_cap_micro_usd": after["funding"]["cap_micro_usd"],
            "checked_at": datetime.now(UTC).isoformat(), "positive_jwt_test": False,
            "limitation": "Direct IAM Lambda probes use synthetic authorizer claims. HTTP missing-token rejection verified; positive browser JWT/query UI and semantic review pending."}
        report["experiment"] = args.experiment
        save_report("smoke.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    except Exception as exc:
        report = {"status": "INCOMPLETE", "error_category": type(exc).__name__, "invocations": calls,
                  "no_model_retry": True, "new_paid_query_calls_maximum": 1}
        report["experiment"] = args.experiment
        save_report("smoke.json", report)
        print(json.dumps(report))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
