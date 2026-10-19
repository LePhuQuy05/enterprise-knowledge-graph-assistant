"""Review an asset-free change set, then execute only that exact first-create plan.

No account upgrade, IAM mutation, model subscription or inference is performed.
Later application stacks use a separate deployment path; this is state bootstrap only.
"""

import argparse
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.contracts import canonical_json, sha256_bytes
from app.ledger import DynamoLedgerStore, Ledger, OperationSpec

STACK = "kg-r72-foundation"
TABLE = "kg-r72-query-state"
PROJECT = "kg-r72"
TYPES = {"AWS::DynamoDB::Table": 2, "AWS::S3::Bucket": 1, "AWS::S3::BucketPolicy": 1}


def inspect_template(template: dict) -> dict:
    if set(template) - {"AWSTemplateFormatVersion", "Description", "Resources", "Outputs", "Metadata"}:
        raise ValueError("Unexpected template features; inspect before deployment")
    resources = template["Resources"]
    counts = {kind: sum(r["Type"] == kind for r in resources.values()) for kind in TYPES}
    if counts != TYPES or len(resources) != 4:
        raise ValueError("Foundation must contain exactly two tables, a bucket and its TLS policy")
    for resource in resources.values():
        if resource.get("DeletionPolicy") != "Retain" or resource.get("UpdateReplacePolicy") != "Retain":
            raise ValueError("State retention is mandatory")
        props = resource["Properties"]
        if resource["Type"] == "AWS::DynamoDB::Table":
            if props["TableName"] not in {TABLE, "kg-r72-metadata"} or (
                props.get("BillingMode") != "PAY_PER_REQUEST"
                or props.get("TimeToLiveSpecification") != {"AttributeName": "expires_at", "Enabled": True}
                or props.get("ProvisionedThroughput") is not None
            ):
                raise ValueError("Unapproved table identity, capacity or expiry configuration")
        if resource["Type"] == "AWS::S3::Bucket":
            if props.get("PublicAccessBlockConfiguration") != {
                "BlockPublicAcls": True, "BlockPublicPolicy": True,
                "IgnorePublicAcls": True, "RestrictPublicBuckets": True,
            } or props.get("BucketEncryption") != {"ServerSideEncryptionConfiguration": [
                {"ServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}},
            ]}:
                raise ValueError("Private encrypted artifact storage is mandatory")
    return {"template_sha256": sha256_bytes(canonical_json(template)), "resource_counts": counts,
            "iam_resources": 0, "compute_resources": 0, "inference_calls": 0}


def _session(profile):
    session = boto3.Session(profile_name=profile, region_name="us-east-1")
    identity = session.client("sts").get_caller_identity()
    if not identity["Arn"].endswith(":user/kg-dev"):
        raise ValueError("Deployment requires the known non-root kg-dev identity")
    return session


def _cfn(session):
    return session.client("cloudformation", config=Config(
        retries={"total_max_attempts": 2}, connect_timeout=5, read_timeout=15,
    ))


def _wait_change_set(client, change_set, *, stack_name=STACK):
    for _ in range(45):
        response = client.describe_change_set(StackName=stack_name, ChangeSetName=change_set)
        if response["Status"] == "CREATE_COMPLETE":
            return response
        if response["Status"] == "FAILED":
            raise ValueError("Change set failed; inspect CloudFormation events locally")
        time.sleep(1)
    raise ValueError("Change set is still pending; no execution attempted")


def inspect_changes(response):
    if response.get("StackName") != STACK or response.get("Status") != "CREATE_COMPLETE" or (
        response.get("ExecutionStatus") != "AVAILABLE" or response.get("NextToken")
    ):
        raise ValueError("Unexpected change-set identity, status or pagination")
    changes = [c["ResourceChange"] for c in response["Changes"]]
    counts = {kind: sum(c["ResourceType"] == kind for c in changes) for kind in TYPES}
    if counts != TYPES or len(changes) != 4 or any(c["Action"] != "Add" for c in changes):
        raise ValueError("Only the reviewed four-resource first-create plan may execute")


def plan(session, template_path, output, *, stack_name=STACK, inspector=inspect_template,
         change_inspector=inspect_changes, capabilities=()):
    template = json.loads(template_path.read_bytes())
    review = inspector(template)
    client = _cfn(session)
    try:
        client.describe_stacks(StackName=stack_name)
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ValidationError" or (
            "does not exist" not in exc.response["Error"].get("Message", "")
        ):
            raise
    else:
        raise ValueError("Stack already exists; first-create helper does not update or replace state")
    client.validate_template(TemplateBody=json.dumps(template))
    result = client.create_change_set(
        StackName=stack_name, ChangeSetName=stack_name + "-" + uuid.uuid4().hex,
        ChangeSetType="CREATE", TemplateBody=json.dumps(template),
        Description="Reviewed KG R72 stack; no inference or model subscriptions",
        Capabilities=list(capabilities),
        Tags=[{"Key": "Project", "Value": PROJECT}, {"Key": "Owner", "Value": "portfolio"}],
    )
    response = _wait_change_set(client, result["Id"], stack_name=stack_name)
    change_inspector(response)
    review.update(status="REVIEWED_NOT_EXECUTED", change_set_id=result["Id"], stack_name=stack_name,
                  template_path=str(template_path.resolve()), checked_at=datetime.now(UTC).isoformat())
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(review, indent=2) + "\n", encoding="utf-8")
    return {k: v for k, v in review.items() if k not in {"change_set_id", "template_path"}}


def apply(session, plan_path, output, *, stack_name=STACK, inspector=inspect_template,
          change_inspector=inspect_changes):
    review = json.loads(plan_path.read_bytes())
    if review["status"] != "REVIEWED_NOT_EXECUTED" or review["stack_name"] != stack_name:
        raise ValueError("A reviewed first-create plan is required")
    client = _cfn(session)
    response = client.describe_change_set(StackName=stack_name, ChangeSetName=review["change_set_id"])
    change_inspector(response)
    remote = client.get_template(StackName=stack_name, ChangeSetName=review["change_set_id"],
                                 TemplateStage="Original")["TemplateBody"]
    if isinstance(remote, str):
        remote = json.loads(remote)
    if inspector(remote)["template_sha256"] != review["template_sha256"]:
        raise ValueError("Remote template differs from reviewed template")
    client.execute_change_set(StackName=stack_name, ChangeSetName=review["change_set_id"],
                              ClientRequestToken=uuid.uuid4().hex)
    # A bounded wait within this process; caller can monitor output/session meanwhile.
    for _ in range(90):
        stack = client.describe_stacks(StackName=stack_name)["Stacks"][0]
        status = stack["StackStatus"]
        if status == "CREATE_COMPLETE":
            break
        if status not in {"CREATE_IN_PROGRESS", "REVIEW_IN_PROGRESS"}:
            raise ValueError("Stack creation did not complete; inspect CloudFormation events")
        time.sleep(2)
    else:
        raise ValueError("Stack is still pending; do not repeat execution")
    outputs = {item["OutputKey"]: item["OutputValue"] for item in stack["Outputs"]}
    result = {"status": status, "stack_name": stack_name, "source_region": "us-east-1",
              "checked_at": datetime.now(UTC).isoformat(), "outputs": outputs,
              "template_sha256": review["template_sha256"], "inference_calls": 0,
              "retained_on_destroy": [name for name, item in remote['Resources'].items()
                                      if item.get('DeletionPolicy') in {'Retain', 'RetainExceptOnCreate'}]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def ledger_smoke(session, output):
    client = session.client("dynamodb", config=Config(retries={"total_max_attempts": 2}))
    table = client.describe_table(TableName=TABLE)["Table"]
    if table["TableStatus"] != "ACTIVE":
        raise ValueError("Shared ledger table is not active")
    ledger = Ledger(DynamoLedgerStore(client, TABLE))
    period = datetime.now(UTC).strftime("%Y-%m")
    run = "foundation-probe-" + uuid.uuid4().hex
    ledger.initialize_month(PROJECT, period)
    ledger.initialize_run(PROJECT, "demo", run, cap_micro_usd=100_000)
    before = ledger.store.get(ledger.month_key(PROJECT, period))
    spec = OperationSpec(project_id=PROJECT, tenant_id="demo", run_id=run, request_id="probe",
                         stage="SMOKE", attempt=1, input_sha256=sha256_bytes(b"NO_MODEL_CALL"),
                         model_id="no-inference-state-probe", invocation_id="no-inference-state-probe",
                         source_region="us-east-1", price_sheet_version="NO_MODEL_CALL",
                         max_cost_micro_usd=5)
    reserved = ledger.reserve(spec)
    assert ledger.reserve(spec).operation_id == reserved.operation_id
    cancelled = ledger.cancel_before_call(reserved.operation_id)
    assert ledger.cancel_before_call(reserved.operation_id).status == cancelled.status
    after = ledger.store.get(ledger.month_key(PROJECT, period))
    assert cancelled.status == "CANCELLED_BEFORE_CALL" and not cancelled.call_started
    for name in ("reserved_micro_usd", "settled_micro_usd"):
        assert before[name] == after[name]
    result = {"status": "PASS", "environment": "MANAGED_AWS_DYNAMODB",
              "checked_at": datetime.now(UTC).isoformat(), "table": TABLE,
              "operation_id": spec.operation_id, "run_id": run,
              "reservation_replay": True, "cancel_replay": True, "counters_restored": True,
              "inference_calls": 0, "model_charge_recorded_micro_usd": 0,
              "limitation": "State transaction probe; not provider settlement or runtime-role IAM smoke"}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "apply", "ledger-smoke"])
    parser.add_argument("--profile", default="kg-dev")
    parser.add_argument("--template", type=Path, default=Path("cdk.out/kg-r72-foundation.template.json"))
    parser.add_argument("--plan", type=Path, default=Path("results/foundation/plan.json"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        session = _session(args.profile)
        if args.command == "plan":
            result = plan(session, args.template, args.output or args.plan)
        elif args.command == "apply":
            result = apply(session, args.plan, args.output or Path("results/foundation/deployment.json"))
        else:
            result = ledger_smoke(session, args.output or Path("results/foundation/ledger_smoke.json"))
        print(json.dumps(result, indent=2))
    except Exception as exc:
        # Full service exceptions may contain account IDs or other private details.
        print(json.dumps({"status": "INCOMPLETE", "error_category": type(exc).__name__,
                          "inference_calls": 0}))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
