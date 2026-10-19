"""Review a mutable-only account update. Draft is local; plan never executes a change set."""

import argparse
import base64
import copy
import json
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from scripts.foundation import _cfn, _session, _wait_change_set
from scripts.query_cloud import STACK_NAME, assert_no_cycles, inspect_code_update
from scripts.query_cloud import inspect_changes as inspect_code_changes

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/account-registration"


def enrollment_statements():
    return [
        {"Effect": "Allow", "Action": ["dynamodb:PutItem"],
         "Resource": {"Fn::Sub": "arn:${AWS::Partition}:dynamodb:${AWS::Region}:${AWS::AccountId}:table/kg-r72-metadata"},
         "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": ["USER#*"]},
                       "ForAnyValue:StringEquals": {"dynamodb:EnclosingOperation": ["TransactWriteItems"]}}},
        {"Effect": "Allow", "Action": ["cognito-idp:AdminGetUser"], "Resource": {"Fn::GetAtt": ["ReviewerPool", "Arn"]}},
    ]


def build_update(before, package_code):
    result = copy.deepcopy(before)
    resources = result["Resources"]
    if set(package_code) != {"S3Bucket", "S3Key"} or not re.fullmatch(
        r"packages/query/[0-9a-f]{64}\.zip", package_code["S3Key"]
    ):
        raise ValueError("HASHED_QUERY_PACKAGE_REQUIRED")
    pool = resources["ReviewerPool"]["Properties"]
    if pool["UserPoolTier"] != "LITE" or pool["UsernameConfiguration"] != {"CaseSensitive": False}:
        raise ValueError("KNOWN_COGNITO_POOL_REQUIRED")
    pool["AdminCreateUserConfig"]["AllowAdminCreateUserOnly"] = False
    pool["AutoVerifiedAttributes"] = ["email"]
    pool["AccountRecoverySetting"] = {"RecoveryMechanisms": [{"Name": "verified_email", "Priority": 1}]}
    pool["VerificationMessageTemplate"] = {"DefaultEmailOption": "CONFIRM_WITH_CODE"}
    function = resources["QueryFunction"]["Properties"]
    if function["Code"].get("S3Bucket") != package_code["S3Bucket"]:
        raise ValueError("PACKAGE_BUCKET_MUST_NOT_CHANGE")
    function["Code"] = package_code
    config = function["Environment"]["Variables"]["PUBLIC_CONFIG_JSON"]
    # Preserve every existing CFN identity token; only append one literal JSON field.
    parts = config.get("Fn::Join", [None, None]) if isinstance(config, dict) else None
    if not parts or parts[0] != "" or not isinstance(parts[1], list) or (
        not isinstance(parts[1][-1], str) or not parts[1][-1].endswith("}")
    ):
        raise ValueError("KNOWN_PUBLIC_CONFIG_EXPRESSION_REQUIRED")
    if any(isinstance(p, str) and '"registration_enabled"' in p for p in parts[1]):
        raise ValueError("REGISTRATION_ALREADY_CONFIGURED_REVIEW_A_NEW_UPDATE")
    parts[1][-1] = parts[1][-1][:-1] + ',"registration_enabled":true}'
    statements = resources["QueryRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    statements.extend(enrollment_statements())
    assert_no_cycles(resources)
    return result


def inspect_update(before, after):
    expected = build_update(before, after["Resources"]["QueryFunction"]["Properties"]["Code"])
    if canonical_json(after) != canonical_json(expected):
        raise ValueError("ONLY_EXACT_ACCOUNT_REGISTRATION_CHANGES_ALLOWED")
    return {"modified": ["ReviewerPool", "QueryRole", "QueryFunction"], "added": [], "removed": [],
            "pool_client_identity_preserved": True, "username_schema_unchanged": True,
            "template_sha256": sha256_bytes(canonical_json(after)),
            "before_template_sha256": sha256_bytes(canonical_json(before)), "model_calls": 0}


def inspect_changes(change):
    if change.get("StackName") != STACK_NAME or change.get("Status") != "CREATE_COMPLETE" or (
        change.get("ExecutionStatus") != "AVAILABLE" or change.get("NextToken")
    ):
        raise ValueError("ACCOUNT_CHANGE_SET_NOT_READY")
    changed = set()
    for item in change.get("Changes", []):
        row = item["ResourceChange"]
        name = row["LogicalResourceId"]
        if row["Action"] != "Modify" or row.get("Replacement") != "False" or name not in {
            "ReviewerPool", "QueryRole", "QueryFunction", "LambdaIntegration"
        }:
            raise ValueError("ACCOUNT_UPDATE_REFUSES_RESOURCE_REPLACEMENT_OR_UNRELATED_CHANGE")
        if name == "LambdaIntegration" and (row.get("ResourceType") != "AWS::ApiGatewayV2::Integration" or not row.get("Details") or any(
            d.get("CausingEntity") != "QueryFunction.Arn" or d.get("Target", {}).get("Name") != "IntegrationUri"
            or d.get("Target", {}).get("RequiresRecreation") != "Never" for d in row["Details"]
        )):
            raise ValueError("UNAPPROVED_INTEGRATION_REEVALUATION")
        changed.add(name)
    if not {"ReviewerPool", "QueryRole", "QueryFunction"} <= changed:
        raise ValueError("ACCOUNT_UPDATE_INCOMPLETE")


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def prepare(before, *, aws_plan=False, session=None, code_only=False):
    build = json.loads((ROOT / "results/query_package_build.json").read_bytes())
    blob = Path(build["package_path"]).read_bytes()
    if build["status"] != "PASS" or sha256_bytes(blob) != build["sha256"]:
        raise ValueError("VERIFIED_PACKAGE_BUILD_REQUIRED")
    code = {"S3Bucket": before["Resources"]["QueryFunction"]["Properties"]["Code"]["S3Bucket"],
            "S3Key": "packages/query/" + build["sha256"] + ".zip"}
    prefix = "refinement-" if code_only else ""
    if code_only:
        after = copy.deepcopy(before)
        after["Resources"]["QueryFunction"]["Properties"]["Code"] = code
        report = inspect_code_update(before, after)
        report["before_template_sha256"] = sha256_bytes(canonical_json(before))
    else:
        after = build_update(before, code)
        report = inspect_update(before, after)
    report.update(status="LOCAL_REVIEW_NOT_DEPLOYED", package_sha256=build["sha256"],
                  checked_at=datetime.now(UTC).isoformat(), email_delivery_verified=False)
    if aws_plan:
        from scripts.query_cloud import stage
        assets = stage(session)
        if assets["package_key"] != code["S3Key"]:
            raise ValueError("STAGED_PACKAGE_DIFFERS")
        client = _cfn(session)
        client.validate_template(TemplateBody=json.dumps(after))
        created = client.create_change_set(StackName=STACK_NAME, ChangeSetName="account-registration-" + uuid.uuid4().hex,
            ChangeSetType="UPDATE", TemplateBody=json.dumps(after), Capabilities=["CAPABILITY_NAMED_IAM"],
            Description="Verified-email registration and narrow first-login membership enrollment; preserve identities")
        ready = _wait_change_set(client, created["Id"], stack_name=STACK_NAME)
        if code_only:
            inspect_code_changes(ready, code_only=True)
        else:
            inspect_changes(ready)
        report.update(status="AWS_CHANGE_SET_REVIEWED_NOT_EXECUTED", change_set_id=created["Id"])
    save(prefix + "before_template.json", before)
    save(prefix + "template.json", after)
    save(prefix + "plan.json", report)
    return report


def apply_reviewed(session, *, code_only=False):
    prefix = "refinement-" if code_only else ""
    report = json.loads((OUT / (prefix + "plan.json")).read_bytes())
    if report["status"] != "AWS_CHANGE_SET_REVIEWED_NOT_EXECUTED":
        raise ValueError("FRESH_REVIEWED_ACCOUNT_PLAN_REQUIRED")
    client = _cfn(session)
    change = client.describe_change_set(StackName=STACK_NAME, ChangeSetName=report["change_set_id"])
    if code_only:
        inspect_code_changes(change, code_only=True)
    else:
        inspect_changes(change)
    remote = client.get_template(StackName=STACK_NAME, ChangeSetName=report["change_set_id"],
                                 TemplateStage="Original")["TemplateBody"]
    current = client.get_template(StackName=STACK_NAME, TemplateStage="Original")["TemplateBody"]
    remote = json.loads(remote) if isinstance(remote, str) else remote
    current = json.loads(current) if isinstance(current, str) else current
    if sha256_bytes(canonical_json(current)) != report["before_template_sha256"] or (
        sha256_bytes(canonical_json(remote)) != report["template_sha256"]
    ):
        raise ValueError("LIVE_OR_REVIEWED_TEMPLATE_CHANGED_REPLAN_REQUIRED")
    (inspect_code_update if code_only else inspect_update)(current, remote)
    build = json.loads((ROOT / "results/query_package_build.json").read_bytes())
    if build["sha256"] != report["package_sha256"] or (
        sha256_bytes(Path(build["package_path"]).read_bytes()) != build["sha256"]
    ):
        raise ValueError("REVIEWED_PACKAGE_CHANGED")
    client.execute_change_set(StackName=STACK_NAME, ChangeSetName=report["change_set_id"],
                              ClientRequestToken=uuid.uuid4().hex)
    for _ in range(150):
        stack = client.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
        status = stack["StackStatus"]
        if status == "UPDATE_COMPLETE":
            outputs = {o["OutputKey"]: o["OutputValue"] for o in stack["Outputs"]}
            actual = session.client("lambda").get_function_configuration(FunctionName="kg-r72-query")
            config = json.loads(actual["Environment"]["Variables"]["PUBLIC_CONFIG_JSON"])
            pool = session.client("cognito-idp").describe_user_pool(UserPoolId=outputs["UserPoolId"])["UserPool"]
            if base64.b64decode(actual["CodeSha256"], validate=True).hex() != report["package_sha256"] or (
                config.get("registration_enabled") is not True
                or pool["AdminCreateUserConfig"]["AllowAdminCreateUserOnly"] is not False
                or pool.get("AutoVerifiedAttributes") != ["email"]
            ):
                raise ValueError("ACCOUNT_DEPLOYMENT_READBACK_MISMATCH")
            result = {"status": status, "checked_at": datetime.now(UTC).isoformat(),
                      "template_sha256": report["template_sha256"], "package_sha256": report["package_sha256"],
                      "registration_enabled": True, "outputs": outputs, "identity_resources_preserved": True,
                      "model_calls": 0, "new_accounts_created_by_agent": 0, "real_email_acceptance": "PENDING_USER"}
            save(prefix + "deployment.json", result)
            return result
        if status not in {"UPDATE_IN_PROGRESS", "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS"}:
            raise ValueError("ACCOUNT_UPDATE_NOT_COMPLETE_INSPECT_EVENTS_NO_AUTOMATIC_RETRY")
        time.sleep(2)
    raise ValueError("ACCOUNT_UPDATE_PENDING_DO_NOT_REPEAT_EXECUTION")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["draft", "plan", "apply", "code-plan", "apply-code"])
    parser.add_argument("--execute-approved", action="store_true", help="Execute only after explicit approval of the prepared plan")
    parser.add_argument("--before-template", type=Path, default=ROOT / "results/query-cloud/previous_template.json")
    args = parser.parse_args()
    if args.command in {"apply", "apply-code"}:
        if not args.execute_approved:
            raise ValueError("EXPLICIT_ROLLOUT_APPROVAL_REQUIRED")
        print(json.dumps(apply_reviewed(_session("kg-dev"), code_only=args.command == "apply-code")))
        return
    session = None
    if args.command == "draft":
        before = json.loads(args.before_template.read_bytes())
    else:
        session = _session("kg-dev")
        client = _cfn(session)
        stack = client.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
        if stack["StackStatus"] != "UPDATE_COMPLETE":
            raise ValueError("EXISTING_STACK_NOT_STABLE")
        before = client.get_template(StackName=STACK_NAME, TemplateStage="Original")["TemplateBody"]
        if isinstance(before, str):
            before = json.loads(before)
    print(json.dumps(prepare(before, aws_plan=args.command in {"plan", "code-plan"},
                             session=session, code_only=args.command == "code-plan")))


if __name__ == "__main__":
    main()
