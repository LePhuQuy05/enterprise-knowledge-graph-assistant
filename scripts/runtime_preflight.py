"""Deploy/invoke an isolated private Lambda; never perform paid model inference."""

import argparse
import base64
import copy
import hmac
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from aws_cdk import App
from botocore.config import Config

from app.cloud_assets import read_deepseek_secret
from app.contracts import canonical_json, sha256_bytes
from app.external_http import load_private_key
from infra.runtime_preflight import FUNCTION_NAME, STACK_NAME, RuntimePreflightStack
from scripts.foundation import _cfn, _session, _wait_change_set, apply, plan

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/runtime-preflight"
TYPES = {"AWS::SecretsManager::Secret", "AWS::Logs::LogGroup", "AWS::IAM::Role", "AWS::Lambda::Function"}


def save(name, report):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / name).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def synthesize():
    artifacts = json.loads((ROOT / "results/cloud_asset_staging.json").read_bytes())
    if artifacts["status"] != "STAGED_READBACK_PASS" or len(artifacts["uploads"]) != 16:
        raise ValueError("VERIFIED_CLOUD_ASSETS_REQUIRED")
    app = App(outdir=str(OUTPUT / "cdk"))
    RuntimePreflightStack(app, artifacts)
    assembly = app.synth()
    return Path(assembly.directory) / (STACK_NAME + ".template.json")


def inspect_template(template, expected):
    if canonical_json(template) != canonical_json(expected):
        raise ValueError("TEMPLATE_DIFFERS_FROM_REVIEWED_PRIVATE_PREFLIGHT")
    resources = template["Resources"]
    if len(resources) != 4 or {r["Type"] for r in resources.values()} != TYPES:
        raise ValueError("ONLY_FOUR_PRIVATE_PREFLIGHT_RESOURCES_ALLOWED")
    role = next(r["Properties"] for r in resources.values() if r["Type"] == "AWS::IAM::Role")
    statements = role["Policies"][0]["PolicyDocument"]["Statement"]
    if {a for s in statements for a in s["Action"]} != {
        "s3:GetObject", "secretsmanager:GetSecretValue", "logs:CreateLogStream", "logs:PutLogEvents"
    } or role.get("ManagedPolicyArns"):
        raise ValueError("PREFLIGHT_MUST_NOT_HAVE_MODEL_OR_STATE_WRITE_PERMISSIONS")
    fn = next(r["Properties"] for r in resources.values() if r["Type"] == "AWS::Lambda::Function")
    if fn["FunctionName"] != FUNCTION_NAME or fn["Environment"]["Variables"]["MODEL_ENABLED"] != "false":
        raise ValueError("FIXED_PRIVATE_FUNCTION_REQUIRED")
    return {"template_sha256": sha256_bytes(canonical_json(template)),
            "resource_counts": {t: 1 for t in sorted(TYPES)}, "paid_model_calls": 0, "public_routes": 0}


def inspect_changes(response):
    changes = [c["ResourceChange"] for c in response.get("Changes", [])]
    if response.get("StackName") != STACK_NAME or response.get("Status") != "CREATE_COMPLETE" or (
        response.get("ExecutionStatus") != "AVAILABLE" or response.get("NextToken")
        or len(changes) != 4 or {c["ResourceType"] for c in changes} != TYPES
        or any(c["Action"] != "Add" for c in changes)
    ):
        raise ValueError("ONLY_REVIEWED_FOUR_RESOURCE_FIRST_CREATE_ALLOWED")


def stack_outputs(session):
    stack = _cfn(session).describe_stacks(StackName=STACK_NAME)["Stacks"][0]
    if stack["StackStatus"] not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}:
        raise ValueError("PRIVATE_PREFLIGHT_STACK_NOT_READY")
    resources = _cfn(session).describe_stack_resources(StackName=STACK_NAME)["StackResources"]
    if len(resources) != 4 or {r["ResourceType"] for r in resources} != TYPES:
        raise ValueError("PREFLIGHT_RESOURCE_OWNERSHIP_CHANGED")
    outputs = {o["OutputKey"]: o["OutputValue"] for o in stack["Outputs"]}
    if outputs["FunctionName"] != FUNCTION_NAME:
        raise ValueError("PREFLIGHT_FUNCTION_CHANGED")
    secret = next(r for r in resources if r["ResourceType"] == "AWS::SecretsManager::Secret")
    if outputs["DeepSeekSecretArn"] != secret["PhysicalResourceId"]:
        raise ValueError("SECRET_NOT_OWNED_BY_PREFLIGHT_STACK")
    return outputs


def logging_update(before):
    """Change only the malformed log-stream ARN in the existing private role."""
    inspect_template(before, before)
    after = copy.deepcopy(before)
    statements = after["Resources"]["Role"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    log = next(s for s in statements if s["Action"] == ["logs:CreateLogStream", "logs:PutLogEvents"])
    old = {"Fn::Join": ["", [{"Fn::GetAtt": ["Logs", "Arn"]}, ":log-stream:*"]]}
    fixed = {"Fn::Sub": "arn:${AWS::Partition}:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/lambda/"
             + FUNCTION_NAME + ":log-stream:*"}
    if log["Resource"] not in (old, fixed):
        raise ValueError("UNEXPECTED_PREFLIGHT_LOG_SCOPE")
    log["Resource"] = fixed
    return after


def inspect_logging_changes(response):
    changes = [c["ResourceChange"] for c in response.get("Changes", [])]
    if response.get("StackName") != STACK_NAME or response.get("Status") != "CREATE_COMPLETE" or (
        response.get("ExecutionStatus") != "AVAILABLE" or response.get("NextToken") or not changes
    ):
        raise ValueError("LOGGING_CHANGE_SET_NOT_READY")
    if not any(c["LogicalResourceId"] == "Role" for c in changes):
        raise ValueError("LOGGING_ROLE_CHANGE_REQUIRED")
    for change in changes:
        if change["Action"] != "Modify" or change.get("Replacement") != "False":
            raise ValueError("NO_RESOURCE_REPLACEMENT_ALLOWED")
        if change["LogicalResourceId"] == "Role" and change["ResourceType"] == "AWS::IAM::Role":
            continue
        if change["LogicalResourceId"] != "Function" or change["ResourceType"] != "AWS::Lambda::Function" or (
            not change.get("Details") or any(d.get("CausingEntity") != "Role.Arn" or (
                d.get("Target", {}).get("Name") != "Role" or d["Target"].get("RequiresRecreation") != "Never"
            ) for d in change["Details"])
        ):
            raise ValueError("ONLY_LOG_ROLE_AND_EXACT_ROLE_REFERENCE_MAY_CHANGE")


def log_plan(session):
    client = _cfn(session)
    outputs = stack_outputs(session)
    before = client.get_template(StackName=STACK_NAME, TemplateStage="Original")["TemplateBody"]
    before = json.loads(before) if isinstance(before, str) else before
    after = logging_update(before)
    if before == after:
        return save("logging_plan.json", {"status": "ALREADY_FIXED", "paid_model_calls": 0})
    if {k for k in before["Resources"] if before["Resources"][k] != after["Resources"][k]} != {"Role"}:
        raise ValueError("ONLY_LOG_POLICY_CAN_CHANGE")
    path = OUTPUT / "logging_template.json"
    path.write_text(json.dumps(after, indent=2) + "\n", encoding="utf-8")
    result = client.create_change_set(StackName=STACK_NAME, ChangeSetName="fix-log-stream-" + uuid.uuid4().hex,
        ChangeSetType="UPDATE", TemplateBody=json.dumps(after), Capabilities=["CAPABILITY_NAMED_IAM"],
        Description="Fix only private preflight log-stream ARN; preserve secret, assets and function config")
    response = _wait_change_set(client, result["Id"], stack_name=STACK_NAME)
    inspect_logging_changes(response)
    return save("logging_plan.json", {"status": "REVIEWED_NOT_EXECUTED", "change_set_id": result["Id"],
        "before_sha256": sha256_bytes(canonical_json(before)), "template_sha256": sha256_bytes(canonical_json(after)),
        "template_path": str(path), "outputs_before": outputs, "paid_model_calls": 0,
        "changes": response["Changes"], "checked_at": datetime.now(UTC).isoformat()})


def log_apply(session):
    review = json.loads((OUTPUT / "logging_plan.json").read_bytes())
    if review["status"] != "REVIEWED_NOT_EXECUTED":
        raise ValueError("REVIEWED_LOGGING_UPDATE_REQUIRED")
    client = _cfn(session)
    remote = client.get_template(StackName=STACK_NAME, ChangeSetName=review["change_set_id"],
                                 TemplateStage="Original")["TemplateBody"]
    remote = json.loads(remote) if isinstance(remote, str) else remote
    current = client.get_template(StackName=STACK_NAME, TemplateStage="Original")["TemplateBody"]
    current = json.loads(current) if isinstance(current, str) else current
    if sha256_bytes(canonical_json(current)) != review["before_sha256"] or (
        sha256_bytes(canonical_json(remote)) != review["template_sha256"] or remote != logging_update(current)
    ):
        raise ValueError("LOGGING_REVIEWED_TEMPLATE_CHANGED")
    inspect_logging_changes(client.describe_change_set(StackName=STACK_NAME, ChangeSetName=review["change_set_id"]))
    client.execute_change_set(StackName=STACK_NAME, ChangeSetName=review["change_set_id"], ClientRequestToken=uuid.uuid4().hex)
    for _ in range(60):
        stack = client.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
        if stack["StackStatus"] == "UPDATE_COMPLETE":
            break
        if stack["StackStatus"] not in {"UPDATE_IN_PROGRESS", "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS"}:
            raise ValueError("LOGGING_UPDATE_NOT_COMPLETE")
        time.sleep(2)
    else:
        raise ValueError("LOGGING_UPDATE_STILL_PENDING_DO_NOT_REEXECUTE")
    outputs = stack_outputs(session)
    if outputs != review["outputs_before"]:
        raise ValueError("PREFLIGHT_IDENTITIES_CHANGED")
    return save("logging_deployment.json", {"status": "UPDATE_COMPLETE", "outputs": outputs,
        "template_sha256": review["template_sha256"], "secret_identity_unchanged": True,
        "paid_model_calls": 0, "checked_at": datetime.now(UTC).isoformat()})


def initialize_secret(client, arn, key):
    """Initialize the new stack's empty secret; existing versions compare, never rotate."""
    description = client.describe_secret(SecretId=arn)
    if description["ARN"] != arn or description["Name"] != "kg-r72/deepseek":
        raise ValueError("SECRET_IDENTITY_CHANGED")
    if description.get("VersionIdsToStages"):
        current = client.get_secret_value(SecretId=arn, VersionStage="AWSCURRENT")
        try:
            existing = read_deepseek_secret(client, secret_arn=arn)
        except ValueError:
            raise ValueError("UNEXPECTED_EXISTING_SECRET_DO_NOT_OVERWRITE") from None
        if not hmac.compare_digest(existing.get_secret_value(), key.get_secret_value()):
            raise ValueError("EXISTING_SECRET_KEY_DIFFERS_DO_NOT_ROTATE")
        return {"status": "EXISTING_MATCH_VERIFIED", "arn": arn,
                "version_id": current["VersionId"], "secret_value_logged": False, "paid_model_calls": 0}
    if description.get("RotationEnabled"):
        raise ValueError("UNEXPECTED_SECRET_ROTATION_CONFIGURATION")
    payload = json.dumps({"schema_version": 1, "provider": "DEEPSEEK", "api_key": key.get_secret_value()})
    written = client.put_secret_value(SecretId=arn, SecretString=payload, ClientRequestToken=uuid.uuid4().hex)
    verified = read_deepseek_secret(client, secret_arn=arn)
    if not hmac.compare_digest(verified.get_secret_value(), key.get_secret_value()):
        raise ValueError("SECRET_READBACK_MISMATCH")
    return {"status": "INITIALIZED_AND_READBACK_VERIFIED", "arn": arn,
            "version_id": written["VersionId"], "secret_value_logged": False, "paid_model_calls": 0}


def verify_function_config(config, outputs, artifacts):
    expected_sha = artifacts["package_key"].split("/")[-1].removesuffix(".zip")
    if base64.b64decode(config["CodeSha256"], validate=True).hex() != expected_sha:
        raise ValueError("DEPLOYED_PACKAGE_CHECKSUM_CHANGED")
    variables = config["Environment"]["Variables"]
    if config.get("Handler") != "app.cloud_preflight.handler" or config.get("Runtime") != "python3.12" or (
        config.get("Architectures") != ["x86_64"] or config.get("Timeout") != 28 or config.get("MemorySize") != 1024
        or variables.get("MODEL_ENABLED") != "false"
        or variables.get("KG_DEEPSEEK_SECRET_ARN") != outputs["DeepSeekSecretArn"]
        or json.loads(variables.get("KG_ASSET_CONFIG_JSON", "null")) != artifacts["runtime_config"]
    ):
        raise ValueError("DEPLOYED_PREFLIGHT_CONFIGURATION_CHANGED")


def probe(session):
    outputs = stack_outputs(session)
    client = session.client("lambda", config=Config(retries={"total_max_attempts": 1},
        connect_timeout=3, read_timeout=40))
    config = client.get_function_configuration(FunctionName=FUNCTION_NAME)
    if config["State"] != "Active" or config["LastUpdateStatus"] != "Successful":
        raise ValueError("FUNCTION_NOT_READY")
    artifacts = json.loads((ROOT / "results/cloud_asset_staging.json").read_bytes())
    verify_function_config(config, outputs, artifacts)
    rows = []
    for index in range(2):
        response = client.invoke(FunctionName=FUNCTION_NAME, InvocationType="RequestResponse", LogType="Tail",
                                 Payload=b'{"action":"verify"}')
        body = response["Payload"]
        try:
            raw = body.read(64 * 1024 + 1)
        finally:
            body.close()
        if len(raw) > 64 * 1024:
            raise ValueError("UNEXPECTED_PREFLIGHT_RESPONSE_SIZE")
        payload = json.loads(raw)
        # Keep only performance REPORT lines, never arbitrary logs or service error payloads.
        tail = base64.b64decode(response.get("LogResult", "")).decode("utf-8", errors="replace")
        lines = [line for line in tail.splitlines() if line.startswith("REPORT RequestId:")]
        if response.get("FunctionError"):
            payload = {"status": "FAIL", "error_category": "LAMBDA_FUNCTION_ERROR", "paid_model_calls": 0}
        rows.append({"sequence": index + 1, "aws_request_id": response["ResponseMetadata"]["RequestId"],
                     "result": payload, "lambda_report_lines": lines})
        passed = payload.get("status") == "PASS" and payload.get("paid_model_calls") == 0 and (
            payload.get("top5_parity") == "PASS" and payload.get("secret_readable_and_format_valid") is True
            and payload.get("cloud_answers_enabled") is False and payload.get("proxy_tokenizer_ready") is True
        )
        report = {"status": "PASS" if passed and len(rows) == 2 else "INCOMPLETE",
            "checked_at": datetime.now(UTC).isoformat(), "function": FUNCTION_NAME,
            "package_code_sha256_base64": config["CodeSha256"], "invocations": rows,
            "paid_model_calls": 0, "public_routes": 0,
            "limitation": "Private runtime parity only; no authenticated query/history/provider call"}
        save("probe.json", report)
        print(json.dumps({"sequence": index + 1, "result": payload}), flush=True)
        if not passed:
            raise ValueError("PRIVATE_RUNTIME_PROBE_FAILED")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["synth", "plan", "apply", "secret", "probe", "log-plan", "log-apply"])
    args = parser.parse_args()
    try:
        if args.command == "synth":
            path = synthesize()
            result = inspect_template(json.loads(path.read_bytes()), json.loads(path.read_bytes()))
        else:
            session = _session("kg-dev")
            if args.command in {"log-plan", "log-apply"}:
                result = log_plan(session) if args.command == "log-plan" else log_apply(session)
            elif args.command in {"plan", "apply"}:
                path = synthesize()
                expected = json.loads(path.read_bytes())
                def inspector(template):
                    return inspect_template(template, expected)
                if args.command == "plan":
                    result = plan(session, path, OUTPUT / "plan.json", stack_name=STACK_NAME,
                        inspector=inspector, change_inspector=inspect_changes, capabilities=("CAPABILITY_NAMED_IAM",))
                else:
                    result = apply(session, OUTPUT / "plan.json", OUTPUT / "deployment.json",
                        stack_name=STACK_NAME, inspector=inspector, change_inspector=inspect_changes)
            elif args.command == "secret":
                outputs = stack_outputs(session)
                client = session.client("secretsmanager", config=Config(retries={"total_max_attempts": 1}))
                result = save("secret_setup.json", initialize_secret(client, outputs["DeepSeekSecretArn"],
                                                                      load_private_key(ROOT, "DEEPSEEK")))
            else:
                result = probe(session)
        print(json.dumps(result, indent=2))
        return 0
    except Exception as exc:
        result = {"status": "INCOMPLETE", "error_category": type(exc).__name__, "paid_model_calls": 0}
        if isinstance(exc, ValueError) and str(exc).isupper() and len(str(exc)) < 100:
            result["safe_error"] = str(exc)
        if hasattr(exc, "response"):
            result["service_error_code"] = exc.response.get("Error", {}).get("Code")
        print(json.dumps(result))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
