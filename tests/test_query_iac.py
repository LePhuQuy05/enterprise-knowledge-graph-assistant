import copy

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template

from infra.tracer import TracerStack
from scripts.query_cloud import inspect_changes, inspect_code_update, inspect_update
from scripts.tracer import assert_no_cycles


def test_query_update_preserves_auth_identity_and_scopes_permissions():
    runtime = {"secret_arn": "arn:aws:secretsmanager:us-east-1:111111111111:secret:fixture-key-abcdef",
        "assets": {"runtime_config": {"bucket": "fixture-approved-bucket"}, "package_key": "packages/query/fixed.zip",
                   "objects": [{"kind": "MODEL", "key": "models/fixed/model.onnx"},
                               {"kind": "SNAPSHOT", "key": "tenants/demo/snapshots/fixed/manifest.json"}]}}
    before = Template.from_stack(TracerStack(App())).to_json()["Resources"]
    after = Template.from_stack(TracerStack(App(), runtime=runtime)).to_json()["Resources"]
    assert_no_cycles(after)
    allowed = {"QueryFunction", "QueryRole", "DefaultStage"}
    assert {k for k in before if before[k] != after[k]} == allowed
    assert set(after) - set(before) == {"PrivateQueryCreate", "PrivateQueryList", "PrivateQueryDetail", "PrivateSource"}
    for name in set(after) - set(before):
        props = after[name]["Properties"]
        assert props["AuthorizationType"] == "JWT"
        assert props["AuthorizationScopes"] == ["kg/query" if name == "PrivateQueryCreate" else "kg/read"]
    statements = after["QueryRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    assert {a for row in statements for a in row["Action"]} == {"dynamodb:GetItem", "dynamodb:Query", "dynamodb:PutItem",
        "s3:GetObject", "secretsmanager:GetSecretValue", "logs:CreateLogStream", "logs:PutLogEvents"}
    write = next(r for r in statements if r["Action"] == ["dynamodb:PutItem"])
    assert write["Condition"]["ForAnyValue:StringEquals"]["dynamodb:EnclosingOperation"] == ["TransactWriteItems"]
    assert "Scan" not in str(statements) and "Marketplace" not in str(statements)
    logging = next(r for r in statements if "logs:PutLogEvents" in r["Action"])
    assert logging["Resource"]["Fn::Sub"].endswith("log-group:/aws/lambda/kg-r72-query:log-stream:*")
    fn = after["QueryFunction"]["Properties"]
    assert fn["Handler"] == "app.query_handler.handler" and fn["Environment"]["Variables"]["MODEL_ENABLED"] == "true"
    assert fn["Timeout"] == 28 and fn["MemorySize"] == 1024 and "VpcConfig" not in fn
    assert inspect_update({"Resources": before}, {"Resources": after})["removed"] == []
    malicious = copy.deepcopy(after)
    malicious["ReviewerPool"]["Properties"]["MfaConfiguration"] = "ON"
    with pytest.raises(ValueError, match="PRESERVE_COGNITO"):
        inspect_update({"Resources": before}, {"Resources": malicious})


@pytest.mark.parametrize("bad", [None, "cognito", "replacement", "integration"])
def test_change_set_allows_only_safe_dynamic_integration_and_nullable_add_replacement(bad):
    from scripts.query_cloud import ADDED, STACK_NAME

    rows = [{"ResourceChange": {"LogicalResourceId": key, "Action": "Add", "Replacement": None}}
            for key in ADDED]
    rows += [{"ResourceChange": {"LogicalResourceId": "QueryFunction", "Action": "Modify", "Replacement": "False"}},
             {"ResourceChange": {"LogicalResourceId": "LambdaIntegration", "Action": "Modify", "Replacement": "False",
                "ResourceType": "AWS::ApiGatewayV2::Integration", "Details": [{"CausingEntity": "QueryFunction.Arn",
                    "Target": {"Name": "IntegrationUri", "RequiresRecreation": "Never"}}]}}]
    response = {"StackName": STACK_NAME, "Status": "CREATE_COMPLETE", "ExecutionStatus": "AVAILABLE", "Changes": rows}
    if bad == "cognito":
        rows[-2]["ResourceChange"]["LogicalResourceId"] = "ReviewerPool"
    elif bad == "replacement":
        rows[-2]["ResourceChange"]["Replacement"] = "True"
    elif bad == "integration":
        rows[-1]["ResourceChange"]["Details"][0]["CausingEntity"] = "OtherFunction.Arn"
    if bad is None:
        inspect_changes(response)
    else:
        with pytest.raises(ValueError):
            inspect_changes(response)


def test_code_only_update_cannot_change_env_or_other_resource():
    old = {"Resources": {"QueryFunction": {"Properties": {"Code": {"S3Bucket": "approved", "S3Key": "old"},
        "Environment": {"Variables": {"MODEL_ENABLED": "true"}}}}, "ReviewerPool": {"Properties": {"Name": "fixed"}}}}
    new = copy.deepcopy(old)
    new["Resources"]["QueryFunction"]["Properties"]["Code"]["S3Key"] = "new"
    assert inspect_code_update(old, new)["modified"] == ["QueryFunction"]
    changed = copy.deepcopy(new)
    changed["Resources"]["QueryFunction"]["Properties"]["Environment"]["Variables"]["MODEL_ENABLED"] = "false"
    with pytest.raises(ValueError):
        inspect_code_update(old, changed)
    changed = copy.deepcopy(new)
    changed["Resources"]["ReviewerPool"]["Properties"]["Name"] = "other"
    with pytest.raises(ValueError):
        inspect_code_update(old, changed)


def test_registration_is_explicit_opt_in_and_grants_only_scoped_first_login_rights():
    runtime = {"secret_arn": "arn:aws:secretsmanager:us-east-1:111111111111:secret:fixture-key-abcdef",
        "assets": {"runtime_config": {"bucket": "fixture-approved-bucket"}, "package_key": "packages/query/fixed.zip",
                   "objects": [{"kind": "MODEL", "key": "models/fixed/model.onnx"}]}}
    before = Template.from_stack(TracerStack(App(), runtime=runtime)).to_json()["Resources"]
    after = Template.from_stack(TracerStack(App(), runtime=runtime, registration_enabled=True)).to_json()["Resources"]
    assert {k for k in before if before[k] != after[k]} == {"ReviewerPool", "QueryRole", "QueryFunction"}
    assert after["ReviewerPool"]["Properties"]["AdminCreateUserConfig"]["AllowAdminCreateUserOnly"] is False
    extra = after["QueryRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"][-2:]
    assert extra[0]["Action"] == ["dynamodb:PutItem"]
    assert extra[0]["Condition"]["ForAllValues:StringLike"]["dynamodb:LeadingKeys"] == ["USER#*"]
    assert extra[1] == {"Effect": "Allow", "Action": ["cognito-idp:AdminGetUser"],
                        "Resource": {"Fn::GetAtt": ["ReviewerPool", "Arn"]}}
    assert_no_cycles(after)
