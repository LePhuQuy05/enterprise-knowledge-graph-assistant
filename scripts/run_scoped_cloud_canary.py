"""Execute only the approved six frozen parents via privileged Lambda API invokes.

Real SQS workers/provider/managed ledger; synthetic authorizer claims are not a
JWT signature proof. No retries, transport recovery or publication of the UI.
"""

import argparse
import base64
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
from app.ledger import Counter, DynamoLedgerStore, Ledger
from app.query_auth import membership_context
from app.scoped_run_contracts import ScopedRunRequestV2, ScopedSourceV2

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT/"results/scoped-release319-2026-11-05-v2"


def checked_cases(plan):
    cases = plan["canary"]["cases"]
    if len(cases) != 6 or plan["canary"]["max_new_model_calls"] != 21:
        raise ValueError("SCOPED_CANARY_APPROVED_MATRIX_REQUIRED")
    keys, pairs, calls = set(), set(), 0
    for case in cases:
        request = ScopedRunRequestV2.model_validate_json(canonical_json(case["request"]))
        if case["idempotency_key"] in keys or request.language not in {"en", "vi"}:
            raise ValueError("SCOPED_CANARY_DUPLICATE_OR_LANGUAGE")
        keys.add(case["idempotency_key"])
        bound = len(request.method_ids)+int(request.language == "vi")+int(any(m.startswith("hybrid_multi_query") for m in request.method_ids))
        if case["max_model_calls"] != bound:
            raise ValueError("SCOPED_CANARY_CALL_BOUND")
        calls += bound
        for method in request.method_ids:
            pair = request.language, method
            if pair in pairs:
                raise ValueError("SCOPED_CANARY_DUPLICATE_METHOD")
            pairs.add(pair)
    if calls != 21 or len(pairs) != 16:
        raise ValueError("SCOPED_CANARY_INCOMPLETE_MATRIX")
    return cases


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    plan = json.loads((DIRECTORY/"plan.json").read_bytes())
    approval = json.loads((DIRECTORY/"approval.json").read_bytes())
    if approval["plan_sha256"] != sha256_bytes((DIRECTORY/"plan.json").read_bytes()):
        raise ValueError("SCOPED_CANARY_APPROVAL_PIN")
    cases = checked_cases(plan)
    if not args.execute:
        print("APPROVED_SCOPE_VALID_NO_AWS_OR_MODEL_CALLS")
        return
    if (DIRECTORY/"canary-started.json").exists():
        raise ValueError("SCOPED_CANARY_ALREADY_STARTED_RECONCILE_NO_AUTOMATIC_RETRY")
    deployment = json.loads((DIRECTORY/"backend-deployment.json").read_bytes())
    if deployment["status"] != "CREATE_COMPLETE":
        raise ValueError("SCOPED_CANARY_BACKEND_NOT_READY")
    import boto3
    from botocore.config import Config

    cfg = Config(connect_timeout=4, read_timeout=35, retries={"total_max_attempts": 1})
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if session.client("sts", config=cfg).get_caller_identity()["Arn"] != "arn:aws:iam::357433444130:user/kg-dev":
        raise ValueError("SCOPED_CANARY_OPERATOR")
    client = session.client("dynamodb", endpoint_url="https://dynamodb.us-east-1.amazonaws.com", config=cfg)
    store, metadata = DynamoLedgerStore(client, "kg-r72-query-state"), DynamoLedgerStore(client, "kg-r72-metadata")
    ledger = Ledger(store)
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    if funding.cap_micro_usd != 8000000 or funding.cap_micro_usd-funding.reserved_micro_usd-funding.settled_micro_usd < 2000000:
        raise ValueError("SCOPED_CANARY_FUNDING")
    cloud = json.loads((ROOT/"results/scoped-cloud319-2026-11-05-v3/cloud-config.json").read_bytes())
    public = cloud["public_config"]
    user = session.client("cognito-idp", config=cfg).admin_get_user(UserPoolId=public["issuer"].rsplit("/", 1)[1], Username="demo-a")
    sub = next(a["Value"] for a in user["UserAttributes"] if a["Name"] == "sub")
    auth = membership_context(metadata, sub)
    if auth.tenant_id != "demo" or not user["Enabled"] or user["UserStatus"] != "CONFIRMED":
        raise ValueError("SCOPED_CANARY_EXISTING_OWNER")
    lam = session.client("lambda", config=cfg)
    for name in ["api", "worker", "watchdog"]:
        live = lam.get_function_configuration(FunctionName="kg-r72-scoped-runs-"+name)
        if base64.b64decode(live["CodeSha256"]).hex() != plan["cloud_package_sha256"] or live["State"] != "Active":
            raise ValueError("SCOPED_CANARY_CLOUD_PACKAGE")
    run_key = ledger.run_key(plan["budget"]["nested_run_id"])
    previous = store.get(run_key)
    if previous is not None:
        counter = Counter.model_validate(previous)
        if counter.cap_micro_usd != 2000000 or counter.settled_micro_usd or counter.reserved_micro_usd or counter.halted:
            raise ValueError("SCOPED_CANARY_COUNTER_NOT_EMPTY_NO_RESET")
    else:
        ledger.initialize_run("kg-r72", "demo", plan["budget"]["nested_run_id"], cap_micro_usd=2000000)
    (DIRECTORY/"canary-budget-created.json").write_bytes(canonical_json({"created_at": datetime.now(UTC).isoformat(),
        "counter": store.get(run_key), "created": previous is None, "shared_cap_unchanged": True}))
    report = {"status": "STARTED", "checked_at": datetime.now(UTC).isoformat(), "parents": [],
              "privileged_lambda_invocation": True, "jwt_signature_verified": False, "ui_enabled": False}
    (DIRECTORY/"canary-started.json").write_bytes(canonical_json({"plan_sha256": approval["plan_sha256"],
        "runner_sha256": sha256_bytes(Path(__file__).read_bytes()), "owner_sub_sha256": sha256_bytes(sub.encode()),
        "counter_before": store.get(run_key), "checked_at": report["checked_at"]}))

    def save():
        (DIRECTORY/"canary-summary.json").write_bytes(canonical_json(report))

    def invoke(method, path, *, payload=None, key=None):
        event = {"rawPath": path, "requestContext": {"http": {"method": method}, "authorizer": {"jwt": {"claims": {
            "sub": sub, "username": "demo-a", "token_use": "access", "iss": public["issuer"],
            "client_id": public["client_id"], "scope": "kg/read kg/query", "exp": str(int(time.time())+900)}}}},
            "headers": {"content-type": "application/json"}}
        if key:
            event["headers"]["idempotency-key"] = key
        if payload is not None:
            event["body"] = json.dumps(payload, ensure_ascii=False)
        response = lam.invoke(FunctionName="kg-r72-scoped-runs-api", InvocationType="RequestResponse", Payload=canonical_json(event))
        if response.get("FunctionError"):
            raise ValueError("SCOPED_CANARY_API_FUNCTION_ERROR")
        data = json.loads(response["Payload"].read())
        if data["statusCode"] not in {200, 202}:
            error = json.loads(data.get("body", "{}"))
            raise ValueError("SCOPED_CANARY_API_"+str(data["statusCode"])+"_"+str(error.get("error", {}).get("code")))
        return json.loads(data["body"])

    try:
        caps = invoke("GET", "/v2/scoped-run-capabilities")
        (DIRECTORY/"canary-capabilities.json").write_bytes(canonical_json(caps))
        if caps["corpus"] != {"documents": 319, "chunks": 3024}:
            raise ValueError("SCOPED_CANARY_CORPUS")
        for index, case in enumerate(cases):
            current = Counter.model_validate(store.get(run_key))
            if current.settled_micro_usd >= 100000:
                report["status"] = "STOPPED_OBSERVED_USAGE_LIMIT"
                save()
                return 2
            if current.reserved_micro_usd or current.halted:
                raise ValueError("SCOPED_CANARY_UNRECONCILED_HOLD_NO_NEW_PARENT")
            membership_context(metadata, sub)
            path = DIRECTORY/f"canary-parent-{index+1}"
            path.mkdir()
            (path/"intent.json").write_bytes(canonical_json(case))
            submitted = invoke("POST", "/v2/scoped-runs", payload=case["request"], key=case["idempotency_key"])
            (path/"submission.json").write_bytes(canonical_json(submitted))
            run_id = submitted["run_id"]
            entry = {"index": index+1, "run_id": run_id, "language": case["request"]["language"], "methods": case["request"]["method_ids"], "status": submitted["status"]}
            report["parents"].append(entry)
            save()
            print("PARENT_SUBMITTED", index+1, entry["language"], entry["methods"], flush=True)
            started, last_print = time.monotonic(), 0
            while time.monotonic()-started < 1250:
                detail = invoke("GET", "/v2/scoped-runs/"+run_id)
                (path/"detail.json").write_bytes(canonical_json(detail))
                entry["status"] = detail["status"]
                save()
                if any(m["status"] in {"FAILED", "UNKNOWN", "EXPIRED", "CANCELLED"} for m in detail["methods"]):
                    raise ValueError("SCOPED_CANARY_METHOD_NOT_COMPLETED_RECONCILE")
                if all(m["status"] == "COMPLETED" for m in detail["methods"]):
                    break
                if time.monotonic()-last_print >= 30:
                    print("PARENT_PROGRESS", index+1, [(m["method_id"], m["status"]) for m in detail["methods"]], flush=True)
                    last_print = time.monotonic()
                time.sleep(8)
            else:
                raise ValueError("SCOPED_CANARY_DEADLINE_NO_NEW_PARENT")
            entry["results"] = []
            for method in entry["methods"]:
                result = invoke("GET", f"/v2/scoped-runs/{run_id}/methods/{method}")
                (path/(method+".json")).write_bytes(canonical_json(result))
                if result["requested_method"] != method or result["effective_method"] != method or result.get("fallback"):
                    raise ValueError("SCOPED_CANARY_METHOD_IDENTITY")
                if result["qualification"] in {"REVIEW_REQUIRED", "NOT_READY"} and result["response"] is not None:
                    raise ValueError("SCOPED_CANARY_WITHHELD_LEAK")
                sources = result["context_sources"]
                if sources:
                    source = invoke("GET", f"/v2/scoped-runs/{run_id}/methods/{method}/sources/{sources[0]['chunk_id']}")
                    ScopedSourceV2.model_validate_json(canonical_json(source))
                    (path/(method+"-source.json")).write_bytes(canonical_json(source))
                entry["results"].append({"method": method, "qualification": result["qualification"], "sources": len(sources),
                    "native_status": result.get("native_answer_status"), "graph_bundles": len(result.get("graph_evidence", []))})
            (path/"trace.json").write_bytes(canonical_json(invoke("GET", "/v2/scoped-runs/"+run_id+"/trace")))
            current = Counter.model_validate(store.get(run_key))
            entry["counter_after"] = current.model_dump(mode="json")
            if current.reserved_micro_usd:
                raise ValueError("SCOPED_CANARY_TERMINAL_HOLDS_REMAIN")
            save()
            print("PARENT_COMPLETE", index+1, "settled_micro_usd", current.settled_micro_usd, flush=True)
        report.update(status="MANAGED_EXECUTION_COMPLETE_NEEDS_SOURCE_AND_FINANCIAL_REVIEW", counter_after=store.get(run_key))
        save()
        print(report["status"], flush=True)
    except Exception as exc:
        report.update(status="STOPPED_RECONCILIATION_REQUIRED", error_category=type(exc).__name__,
                      safe_error=str(exc) if isinstance(exc, ValueError) else "SDK_OR_TRANSPORT_FAILURE_NO_AUTOMATIC_RETRY")
        save()
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
