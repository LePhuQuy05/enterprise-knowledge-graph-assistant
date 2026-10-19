import base64
import copy
import json

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template
from pydantic import SecretStr

from infra.runtime_preflight import STACK_NAME, RuntimePreflightStack
from scripts.runtime_preflight import (
    initialize_secret,
    inspect_changes,
    inspect_logging_changes,
    inspect_template,
    logging_update,
    verify_function_config,
)


def test_log_fix_preserves_secret_and_all_unrelated_template_properties():
    artifacts = {"runtime_config": {"bucket": "fixture-bucket"}, "package_key": "packages/fixed.zip",
                 "objects": [{"kind": "MODEL", "key": "models/fixed"}]}
    new = Template.from_stack(RuntimePreflightStack(App(), artifacts)).to_json()
    old = copy.deepcopy(new)
    log = next(s for s in old["Resources"]["Role"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
               if "logs:PutLogEvents" in s["Action"])
    log["Resource"] = {"Fn::Join": ["", [{"Fn::GetAtt": ["Logs", "Arn"]}, ":log-stream:*"]]}
    assert logging_update(old) == new and logging_update(new) == new
    assert old["Resources"]["DeepSeekKey"]["DeletionPolicy"] == "Retain"
    log["Resource"] = "*"
    with pytest.raises(ValueError, match="UNEXPECTED_PREFLIGHT_LOG_SCOPE"):
        logging_update(old)


@pytest.mark.parametrize("bad", [None, "secret", "replacement", "wrong_reference"])
def test_logging_change_set_guard(bad):
    rows = [{"ResourceChange": {"LogicalResourceId": "Role", "ResourceType": "AWS::IAM::Role", "Action": "Modify",
                              "Replacement": "False"}}]
    response = {"StackName": STACK_NAME, "Status": "CREATE_COMPLETE", "ExecutionStatus": "AVAILABLE", "Changes": rows}
    if bad == "secret":
        rows[0]["ResourceChange"].update(LogicalResourceId="DeepSeekKey", ResourceType="AWS::SecretsManager::Secret")
    elif bad == "replacement":
        rows[0]["ResourceChange"]["Replacement"] = "True"
    elif bad == "wrong_reference":
        rows.append({"ResourceChange": {"LogicalResourceId": "Function", "ResourceType": "AWS::Lambda::Function",
            "Action": "Modify", "Replacement": "False", "Details": [{"CausingEntity": "Other.Arn",
                "Target": {"Name": "Role", "RequiresRecreation": "Never"}}]}})
    if bad is None:
        inspect_logging_changes(response)
    else:
        with pytest.raises(ValueError):
            inspect_logging_changes(response)


def test_deployment_review_rejects_extra_permissions_and_non_add_changes():
    artifacts = {"runtime_config": {"bucket": "fixture-bucket"}, "package_key": "packages/fixed.zip",
                 "objects": [{"kind": "MODEL", "key": "models/fixed"}]}
    expected = Template.from_stack(RuntimePreflightStack(App(), artifacts)).to_json()
    assert inspect_template(expected, expected)["public_routes"] == 0
    altered = copy.deepcopy(expected)
    role = next(r for r in altered["Resources"].values() if r["Type"] == "AWS::IAM::Role")
    role["Properties"]["ManagedPolicyArns"] = ["arn:aws:iam::aws:policy/AdministratorAccess"]
    with pytest.raises(ValueError, match="TEMPLATE_DIFFERS"):
        inspect_template(altered, expected)
    response = {"StackName": STACK_NAME, "Status": "CREATE_COMPLETE", "ExecutionStatus": "AVAILABLE",
        "Changes": [{"ResourceChange": {"ResourceType": r["Type"], "Action": "Add"}}
                    for r in expected["Resources"].values()]}
    inspect_changes(response)
    response["Changes"][0]["ResourceChange"]["Action"] = "Modify"
    with pytest.raises(ValueError, match="FIRST_CREATE"):
        inspect_changes(response)


class Secrets:
    def __init__(self):
        self.raw, self.writes, self.version = None, 0, None

    def describe_secret(self, **kwargs):
        versions = {self.version: ["AWSCURRENT"]} if self.version else {}
        return {"ARN": "fixture-arn", "Name": "kg-r72/deepseek", "VersionIdsToStages": versions}

    def get_secret_value(self, **kwargs):
        assert self.version is not None, "Do not try reading a value before the first version exists"
        return {"ARN": "fixture-arn", "SecretString": self.raw, "VersionId": self.version}

    def put_secret_value(self, **kwargs):
        self.writes += 1
        self.raw, self.version = kwargs["SecretString"], kwargs["ClientRequestToken"]
        return {"VersionId": self.version}


def test_secret_initialization_replay_never_rotates_or_outputs_key():
    client, key = Secrets(), SecretStr("synthetic-unit-test-private-key")
    result = initialize_secret(client, "fixture-arn", key)
    assert result["status"] == "INITIALIZED_AND_READBACK_VERIFIED" and client.writes == 1
    assert key.get_secret_value() not in json.dumps(result)
    assert initialize_secret(client, "fixture-arn", key)["status"] == "EXISTING_MATCH_VERIFIED"
    assert client.writes == 1
    with pytest.raises(ValueError, match="DO_NOT_ROTATE"):
        initialize_secret(client, "fixture-arn", SecretStr("different-unit-test-private-key"))
    assert client.writes == 1


def test_unknown_existing_secret_is_not_overwritten():
    client = Secrets()
    client.raw = "unexpected-existing-content"
    client.version = "external-version"
    with pytest.raises(ValueError, match="DO_NOT_OVERWRITE"):
        initialize_secret(client, "fixture-arn", SecretStr("synthetic-unit-test-private-key"))
    assert client.writes == 0


def test_probe_checks_actual_code_and_configuration_before_invoke():
    artifacts = {"package_key": "packages/query/" + "a" * 64 + ".zip", "runtime_config": {"tenant_id": "demo"}}
    outputs = {"DeepSeekSecretArn": "fixture-arn"}
    config = {"CodeSha256": base64.b64encode(bytes.fromhex("a" * 64)).decode(),
        "Handler": "app.cloud_preflight.handler", "Runtime": "python3.12", "Architectures": ["x86_64"],
        "Timeout": 28, "MemorySize": 1024, "Environment": {"Variables": {
            "MODEL_ENABLED": "false", "KG_DEEPSEEK_SECRET_ARN": "fixture-arn",
            "KG_ASSET_CONFIG_JSON": json.dumps(artifacts["runtime_config"])}}}
    verify_function_config(config, outputs, artifacts)
    changed = copy.deepcopy(config)
    changed["Environment"]["Variables"]["MODEL_ENABLED"] = "true"
    with pytest.raises(ValueError, match="CONFIGURATION_CHANGED"):
        verify_function_config(changed, outputs, artifacts)
    changed = config | {"CodeSha256": base64.b64encode(bytes.fromhex("b" * 64)).decode()}
    with pytest.raises(ValueError, match="CHECKSUM_CHANGED"):
        verify_function_config(changed, outputs, artifacts)
