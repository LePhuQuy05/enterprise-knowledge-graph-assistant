"""Execute the specifically approved funding-reader patch in resumable stages.

No inference or account/IAM/UI changes. Allocation is a separate command and
requires verified deployed code plus a recorded model-free operational smoke.
"""

import argparse
import base64
import json
from datetime import UTC, datetime

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.contracts import canonical_json, sha256_bytes
from app.funding_policy import (
    COMPATIBLE_AUTHORIZATION_ID,
    TRANSITION_PROFILE,
    validate_external_funding,
)
from app.ledger import DynamoLedgerStore, Ledger
from app.ledger_allocation import increase_external_allocation
from scripts.plan_funding_transition_release import DIRECTORY
from scripts.query_cloud import inspect_changes
from scripts.stage_cloud_assets import BUCKET

STACK, FUNCTION = "kg-r72-tracer", "kg-r72-query"


def read(name):
    return json.loads((DIRECTORY / name).read_bytes())


def write(name, value):
    value = {"checked_at": datetime.now(UTC).isoformat(), **value}
    (DIRECTORY / name).write_bytes(canonical_json(value) + b"\n")
    return value


def body(client, **kwargs):
    value = client.get_template(StackName=STACK, TemplateStage="Original", **kwargs)["TemplateBody"]
    return json.loads(value) if isinstance(value, str) else value


def inspect_templates(before, after):
    import copy

    expected = copy.deepcopy(before)
    props = expected["Resources"]["QueryFunction"]["Properties"]
    props["Code"]["S3Key"] = read("release-plan.json")["proposed_s3_key"]
    props["Environment"]["Variables"]["KG_FUNDING_POLICY"] = TRANSITION_PROFILE
    if canonical_json(expected) != canonical_json(after):
        raise ValueError("FUNDING_RELEASE_UNAPPROVED_TEMPLATE_DELTA")
    if props["Code"]["S3Bucket"] != BUCKET:
        raise ValueError("FUNDING_RELEASE_BUCKET")


def runtime_verified(session, plan, *, updated):
    cfg = session.client("lambda", config=aws_config()).get_function_configuration(FunctionName=FUNCTION)
    expected = plan["package_code_sha256_base64"] if updated else plan["base_code_sha256_base64"]
    if cfg["CodeSha256"] != expected or cfg.get("State") != "Active" or cfg.get("LastUpdateStatus") != "Successful":
        raise ValueError("FUNDING_RELEASE_RUNTIME_NOT_EXPECTED_ACTIVE_CODE")
    before = read("previous-template.json")["Resources"]["QueryFunction"]["Properties"]["Environment"]["Variables"]
    if not updated and cfg["RevisionId"] != plan["base_revision_id"]:
        raise ValueError("FUNDING_RELEASE_BASE_REVISION_CHANGED")
    actual_env = cfg.get("Environment", {}).get("Variables", {})
    baseline_path = DIRECTORY / "resolved-environment-before.json"
    if baseline_path.exists():
        resolved_before = read(baseline_path.name)["variables"]
    elif updated:
        raise ValueError("FUNDING_RELEASE_RESOLVED_ENVIRONMENT_BASELINE_REQUIRED")
    else:
        # CloudFormation intrinsics are objects; Lambda reports their resolved
        # strings. Exact approved CodeSha+RevisionId binds this baseline to the
        # reviewed function, while literal variables and all keys still match.
        if set(actual_env) != set(before) or any(actual_env[k] != v for k, v in before.items() if isinstance(v, str)):
            raise ValueError("FUNDING_RELEASE_BASE_ENVIRONMENT_DRIFT")
        resolved_before = actual_env
        write(baseline_path.name, {"revision_id": cfg["RevisionId"], "variables": actual_env})
    expected_env = resolved_before | ({"KG_FUNDING_POLICY": TRANSITION_PROFILE} if updated else {})
    if actual_env != expected_env or cfg["Handler"] != "app.query_handler.handler":
        raise ValueError("FUNDING_RELEASE_RUNTIME_ENVIRONMENT_DRIFT")
    consumers = read("active-consumers.json")["functions"]
    for old in consumers:
        if old["function_name"] == FUNCTION:
            continue
        live = session.client("lambda", config=aws_config()).get_function_configuration(FunctionName=old["function_name"])
        variables = live.get("Environment", {}).get("Variables", {})
        owned = json.loads(variables.get("KG_RUN_CLOUD_JSON", "{}"))
        if (live["CodeSha256"] != old["code_sha256"] or live["RevisionId"] != old["revision_id"]
                or owned.get("answer_policy")):
            raise ValueError("FUNDING_RELEASE_ACTIVE_CONSUMER_CHANGED")
    return {"code_sha256_base64": cfg["CodeSha256"], "revision_id": cfg["RevisionId"],
            "profile": cfg.get("Environment", {}).get("Variables", {}).get("KG_FUNDING_POLICY", "LEGACY_2_USD"),
            "state": cfg["State"], "last_update_status": cfg["LastUpdateStatus"]}


def aws_config():
    return Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=30)


def stage_checksum_verified(client, package, plan):
    expected = base64.b64encode(bytes.fromhex(plan["package_sha256"])).decode()
    try:
        head = client.head_object(Bucket=BUCKET, Key=plan["proposed_s3_key"], ChecksumMode="ENABLED")
        uploaded = False
    except ClientError as exc:
        if exc.response["Error"]["Code"] not in {"404", "NoSuchKey", "NotFound"}:
            raise
        with package.open("rb") as stream:
            client.put_object(Bucket=BUCKET, Key=plan["proposed_s3_key"], Body=stream,
                ContentLength=package.stat().st_size, ServerSideEncryption="AES256", IfNoneMatch="*",
                Metadata={"sha256": plan["package_sha256"]}, ChecksumSHA256=expected)
        head = client.head_object(Bucket=BUCKET, Key=plan["proposed_s3_key"], ChecksumMode="ENABLED")
        uploaded = True
    if (head.get("ChecksumSHA256") != expected or head["ContentLength"] != package.stat().st_size
            or head.get("ServerSideEncryption") != "AES256"
            or head.get("Metadata", {}).get("sha256") != plan["package_sha256"]):
        raise ValueError("FUNDING_RELEASE_S3_CHECKSUM_LENGTH_METADATA_OR_ENCRYPTION")
    return "UPLOADED_S3_CHECKSUM_VERIFIED" if uploaded else "EXISTING_S3_CHECKSUM_VERIFIED"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("stage-plan", "apply", "status", "verify", "activate"))
    args = parser.parse_args()
    plan = read("release-plan.json")
    approval = read("approval.json")
    if (approval.get("package_sha256") != plan["package_sha256"] or approval.get("approved") is not True
            or approval.get("activate_cap_micro_usd") != 8000000):
        raise ValueError("FUNDING_RELEASE_CONCRETE_USER_APPROVAL_REQUIRED")
    package = DIRECTORY / "query-funding-transition.zip"
    if sha256_bytes(package.read_bytes()) != plan["package_sha256"]:
        raise ValueError("FUNDING_RELEASE_PACKAGE_DRIFT")
    before, after = read("previous-template.json"), read("proposed-template.json")
    inspect_templates(before, after)
    if sha256_bytes(canonical_json(after)) != plan["template_sha256"]:
        raise ValueError("FUNDING_RELEASE_TEMPLATE_DRIFT")
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    identity = session.client("sts", config=aws_config()).get_caller_identity()
    if identity["Account"] != "357433444130" or not identity["Arn"].endswith(":user/kg-dev"):
        raise ValueError("FUNDING_RELEASE_OPERATOR_REQUIRED")
    cfn = session.client("cloudformation", config=aws_config())
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=aws_config()), "kg-r72-query-state"))
    if args.command == "stage-plan":
        if (DIRECTORY / "changeset.json").exists():
            raise ValueError("FUNDING_RELEASE_CHANGESET_ALREADY_EXISTS_INSPECT_STATUS")
        runtime_verified(session, plan, updated=False)
        if cfn.describe_stacks(StackName=STACK)["Stacks"][0]["StackStatus"] != "UPDATE_COMPLETE" or body(cfn) != before:
            raise ValueError("FUNDING_RELEASE_LIVE_TEMPLATE_CHANGED")
        funding = validate_external_funding(ledger)
        row = {"path": str(package), "key": plan["proposed_s3_key"], "bytes": package.stat().st_size,
               "sha256": plan["package_sha256"]}
        print("STAGING_APPROVED_PACKAGE", flush=True)
        staged = stage_checksum_verified(session.client("s3", config=aws_config()), package, plan)
        write("staged.json", {"status": staged, "key": row["key"], "sha256": row["sha256"],
                              "funding_before": funding.model_dump(mode="json"), "model_calls": 0})
        name = "funding-reader-" + plan["package_sha256"][:16]
        request = cfn.create_change_set(StackName=STACK, ChangeSetName=name, ChangeSetType="UPDATE",
            TemplateBody=json.dumps(after), Capabilities=["CAPABILITY_NAMED_IAM"],
            ClientToken=name, Description="Approved exact funding reader Code+Environment patch; no IAM/UI/account/cap change")
        write("changeset.json", {"status": "CREATED_NOT_EXECUTED", "change_set_id": request["Id"],
                                  "template_sha256": plan["template_sha256"], "model_calls": 0})
        print("APPROVED_CHANGESET_CREATED_NOT_EXECUTED")
    elif args.command == "apply":
        runtime_verified(session, plan, updated=False)
        if body(cfn) != before:
            raise ValueError("FUNDING_RELEASE_LIVE_TEMPLATE_CHANGED")
        validate_external_funding(ledger)
        change = read("changeset.json")
        ready = cfn.describe_change_set(StackName=STACK, ChangeSetName=change["change_set_id"])
        inspect_changes(ready, code_only=True)
        if body(cfn, ChangeSetName=change["change_set_id"]) != after:
            raise ValueError("FUNDING_RELEASE_REMOTE_CHANGESET_DRIFT")
        write("reviewed-changes.json", {"status": "SCOPE_VERIFIED", "changes": ready["Changes"], "model_calls": 0})
        # Persist intent before dispatch; do not execute again after uncertain results.
        if (DIRECTORY / "execute-intent.json").exists():
            raise ValueError("FUNDING_RELEASE_EXECUTE_ALREADY_CLAIMED_INSPECT_STATUS")
        write("execute-intent.json", {"change_set_id": change["change_set_id"], "model_calls": 0})
        cfn.execute_change_set(StackName=STACK, ChangeSetName=change["change_set_id"],
                              ClientRequestToken="funding-reader-" + plan["package_sha256"][:16])
        print("APPROVED_CHANGESET_EXECUTING")
    elif args.command == "status":
        state = cfn.describe_stacks(StackName=STACK)["Stacks"][0]["StackStatus"]
        events = cfn.describe_stack_events(StackName=STACK)["StackEvents"][:12]
        result = {"stack_status": state, "events": [{k: row.get(k) for k in (
            "LogicalResourceId", "ResourceStatus", "ResourceStatusReason")} for row in events], "model_calls": 0}
        write("status.json", result)
        print(json.dumps(result))
    elif args.command == "verify":
        if cfn.describe_stacks(StackName=STACK)["Stacks"][0]["StackStatus"] != "UPDATE_COMPLETE" or body(cfn) != after:
            raise ValueError("FUNDING_RELEASE_NOT_DEPLOYED_EXACT_TEMPLATE")
        runtime = runtime_verified(session, plan, updated=True)
        funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
        result = {"status": "DEPLOYED_READER_VERIFIED", "runtime": runtime,
                  "funding": funding.model_dump(mode="json"), "model_calls": 0}
        write("deployed-reader.json", result)
        print(json.dumps({"status": result["status"], "cap_micro_usd": funding.cap_micro_usd, "model_calls": 0}))
    else:
        smoke = json.loads((DIRECTORY / "operational-smoke/report.json").read_bytes())
        verified = read("deployed-reader.json")
        http = read("http-static-auth.json")
        iam = read("iam-funding-audit-scope.json")
        if (verified.get("status") != "DEPLOYED_READER_VERIFIED" or smoke.get("status") != "PASS_MODEL_FREE_OPERATIONAL_SMOKE"
                or smoke.get("package_sha256") != plan["package_sha256"]
                or not smoke.get("model_and_admission_counters_unchanged") or http.get("status") != "PASS"
                or iam.get("status") != "FRESH_INLINE_READ_SCOPE_COVERS_COUNTER_AND_AUDIT_SORT_KEYS"):
            raise ValueError("FUNDING_RELEASE_VERIFIED_READER_AND_MODEL_FREE_SMOKE_REQUIRED")
        runtime_verified(session, plan, updated=True)
        if body(cfn) != after or cfn.describe_stacks(StackName=STACK)["Stacks"][0]["StackStatus"] != "UPDATE_COMPLETE":
            raise ValueError("FUNDING_RELEASE_PRE_ALLOCATION_TEMPLATE_DRIFT")
        from app.query_service import RUN

        month_key = ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m"))
        month_before = ledger.store.get(month_key)
        run_before = ledger.store.get(ledger.run_key(RUN))
        if not month_before or month_before["cap_micro_usd"] != 12000000:
            raise ValueError("FUNDING_RELEASE_EXISTING_MONTH_REQUIRED")
        result = increase_external_allocation(ledger, project="kg-r72", old_cap=2000000, new_cap=8000000,
            authorization_id=COMPATIBLE_AUTHORIZATION_ID, maximum_authorized_cap=8000000)
        funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
        run_after = ledger.store.get(ledger.run_key(RUN))
        month_after = ledger.store.get(month_key)
        if month_before != month_after or run_before != run_after:
            raise ValueError("FUNDING_RELEASE_CONCURRENT_MONTH_OR_RUN_CHANGE_INSPECT_NO_RESET")
        result.update(status="AUDITED_8_USD_TOTAL_ACTIVE_READER_VERIFIED", model_calls=0,
            month_unchanged=True, run_unchanged=True,
            remaining_micro_usd=funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd)
        write("allocation-active.json", result)
        print(json.dumps({k: result[k] for k in ("status", "changed", "remaining_micro_usd", "model_calls")}))


if __name__ == "__main__":
    main()
