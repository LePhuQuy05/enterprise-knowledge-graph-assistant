"""Bounded opt-in AWS-wide live smoke. IAM probes are not a positive browser JWT test."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from botocore.config import Config

from app.contracts import canonical_json
from app.ledger import DynamoLedgerStore, Ledger
from app.query_auth import AuthContext
from app.query_service import RUN
from scripts.foundation import _session

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/aws-knowledge/cloud-smoke"
CASES = {
    "ec2-en": ("How do security groups differ from network ACLs in an Amazon VPC?", "en"),
    "rds-vi": ("Amazon RDS Multi-AZ khác gì với Read Replica?", "vi"),
    "kms-vi": ("Vì sao S3 kích hoạt Lambda thành công nhưng hàm vẫn không đọc được object SSE-KMS ở tài khoản khác?", "vi"),
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    target = OUT / args.case
    if not args.execute:
        print(json.dumps({"status": "PLANNED", "maximum_new_queries": 1, "maximum_model_calls": 2}))
        return
    if (target / "intent.json").exists():
        raise ValueError("EXISTING_INTENT_NO_AUTOMATIC_RETRY")
    session = _session("kg-dev")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=36)
    client = session.client("lambda", config=cfg)
    outputs = json.loads((ROOT / "results/tracer/deployment.json").read_bytes())["outputs"]
    identities = {}
    for user in session.client("cognito-idp", config=cfg).list_users(UserPoolId=outputs["UserPoolId"], Limit=10)["Users"]:
        if user["Username"] in {"demo-a", "demo-b"}:
            identities[user["Username"]] = next(a["Value"] for a in user["Attributes"] if a["Name"] == "sub")
    if set(identities) != {"demo-a", "demo-b"}:
        raise ValueError("EXPECTED_REVIEWER_IDENTITIES_REQUIRED")
    store = DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state")
    ledger = Ledger(store)
    def counters():
        return {"funding": store.get(ledger.external_funding_key("kg-r72")), "run": store.get(ledger.run_key(RUN))}
    before = counters()
    if any(not row or row.get("halted") for row in before.values()):
        raise ValueError("BUDGET_UNAVAILABLE")
    key, calls = str(uuid4()), []
    save(target / "intent.json", {"case": args.case, "at": datetime.now(UTC).isoformat(),
        "idempotency_key": key, "maximum_new_queries": 1, "positive_browser_jwt_test": False,
        "direct_iam_synthetic_claims": True, "no_automatic_paid_retry": True})

    def event(method, path, *, body=None, params=None, owner="demo-a"):
        return {"rawPath": path, "requestContext": {"http": {"method": method}, "authorizer": {"jwt": {"claims": {
            "sub": identities[owner], "iss": "https://cognito-idp.us-east-1.amazonaws.com/" + outputs["UserPoolId"],
            "client_id": outputs["BrowserClientId"], "token_use": "access", "exp": str(int(time.time()) + 300),
            "scope": "kg/read kg/query"}}}}, "headers": {"content-type": "application/json", "idempotency-key": key},
            "body": json.dumps(body, ensure_ascii=False) if body else None, "queryStringParameters": params}

    def invoke(label, event):
        started = time.monotonic()
        response = client.invoke(FunctionName="kg-r72-query", InvocationType="RequestResponse", Payload=canonical_json(event))
        stream = response["Payload"]
        try:
            raw = stream.read(256 * 1024 + 1)
        finally:
            stream.close()
        if len(raw) > 256 * 1024 or response.get("FunctionError"):
            raise ValueError("FUNCTION_OR_RESPONSE_FAILURE")
        wire = json.loads(raw)
        data = json.loads(wire["body"])
        calls.append({"label": label, "http_status": wire["statusCode"], "elapsed_seconds": round(time.monotonic() - started, 3)})
        save(target / "progress.json", calls)
        return wire["statusCode"], data

    report = {"status": "INCOMPLETE", "case": args.case, "checks": [], "invocations": calls,
        "semantic_review": "AI_REVIEW_PENDING", "positive_browser_jwt_test": False}
    try:
        question, language = CASES[args.case]
        request = event("POST", "/v1/queries", body={"question": question, "corpus_scope": "aws_official", "answer_language": language})
        code, data = invoke("new_query", request)
        save(target / "answer.json", {"http_status": code, "record": data})
        if code != 200 or data.get("status") != "COMPLETED":
            raise ValueError("FIRST_QUERY_FAILED_NO_NEW_PAID_ATTEMPT")
        rid, answer = data["request_id"], data["response"]
        auth = AuthContext("demo", identities["demo-a"])
        row = store.get((auth.partition, "REQUEST#" + rid))
        receipts = [row[k] for k in ("rewrite_receipt", "receipt") if k in row]
        operations = [{"operation_id": r["operation_id"], "status": ledger.operation(r["operation_id"]).status} for r in receipts]
        report["operations"] = operations
        report["checks"].extend([
            {"name": "operations_settled", "passed": all(o["status"] == "SETTLED" for o in operations)},
            {"name": "official_backend", "passed": answer["retrieval_backend"] == "aws_knowledge_mcp"},
            {"name": "answer_language", "passed": answer["answer_language"] == language},
            {"name": "not_graph", "passed": not answer["used_graph"]},
        ])
        settled = counters()
        code, replay = invoke("replay", request)
        report["checks"].append({"name": "same_answer_replay", "passed": code == 200 and replay.get("idempotent_replay") and replay["response"] == answer})
        if answer["citations"]:
            citation = answer["citations"][0]
            path, params = "/v1/sources/" + citation["chunk_id"], {"request_id": rid}
            code, source = invoke("source_owner", event("GET", path, params=params))
            save(target / "source.json", source)
            report["checks"].append({"name": "pinned_source_exact_quote", "passed": code == 200 and citation["quote"] in source.get("text", "")})
            code, _ = invoke("source_foreign_owner", event("GET", path, params=params, owner="demo-b"))
            report["checks"].append({"name": "foreign_owner_404", "passed": code == 404})
        else:
            report["checks"].append({"name": "answer_has_sources", "passed": False})
        after = counters()
        report["checks"].append({"name": "replay_source_no_model_cost", "passed": after == settled})
        report["cost_delta_micro_usd"] = {name: after[name]["settled_micro_usd"] - before[name]["settled_micro_usd"] for name in before}
        report["answer_status"] = answer["answer_status"]
        report["status"] = "PASS_LIVE_STRUCTURAL_SMOKE" if all(c["passed"] for c in report["checks"]) else "FAIL"
    except Exception as exc:
        report["error_category"] = type(exc).__name__
        if isinstance(exc, ValueError) and str(exc).isupper():
            report["safe_error"] = str(exc)
        raise
    finally:
        save(target / "report.json", report)
    print(json.dumps({"status": report["status"], "case": args.case, "checks": len(report["checks"]), "output": str(target)}))


if __name__ == "__main__":
    main()
