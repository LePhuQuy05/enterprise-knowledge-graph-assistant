import copy

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template

from infra.tracer import TracerStack
from scripts.method_catalog_release import inspect_catalog_update
from scripts.query_cloud import inspect_code_update, inspect_update


@pytest.fixture(scope="module")
def catalog_templates():
    runtime = {"secret_arn": "arn:aws:secretsmanager:us-east-1:111111111111:secret:fixture-key-abcdef",
        "assets": {"runtime_config": {"bucket": "fixture-approved-bucket"}, "package_key": "packages/query/fixed.zip",
                   "objects": [{"kind": "MODEL", "key": "models/fixed/model.onnx"}]}}
    before = Template.from_stack(TracerStack(App(), runtime=runtime, registration_enabled=True)).to_json()
    after = Template.from_stack(TracerStack(App(), runtime=runtime, registration_enabled=True,
        method_catalog_enabled=True)).to_json()
    return before, after


def test_read_catalog_is_explicit_opt_in_with_no_iam_auth_or_model_changes(catalog_templates):
    before, after = catalog_templates
    result = inspect_catalog_update(before, after)
    assert result["added"] == ["PrivateRagMethods"] and result["modified"] == ["DefaultStage"]
    assert not result["cloud_deployed"] and result["model_calls"] == result["aws_calls"] == 0
    assert "PrivateRagMethods" not in before["Resources"]
    with pytest.raises(ValueError):
        inspect_code_update(before, after)
    with pytest.raises(ValueError):
        inspect_update(before, after)  # Historical four-route approval remains narrow.


def test_catalog_can_include_only_a_reviewed_code_object_change(catalog_templates):
    before, after = map(copy.deepcopy, catalog_templates)
    after["Resources"]["QueryFunction"]["Properties"]["Code"]["S3Key"] = "packages/query/new.zip"
    assert inspect_catalog_update(before, after)["modified"] == ["DefaultStage", "QueryFunction"]


@pytest.mark.parametrize("bad", ["scope", "authorizer", "integration", "public", "new_route", "remove",
    "throttle", "dependency", "iam", "env", "cognito", "code_bucket", "output", "metadata"])
def test_catalog_inspector_rejects_unrelated_or_weaker_changes(catalog_templates, bad):
    before, after = map(copy.deepcopy, catalog_templates)
    rows = after["Resources"]
    route = rows["PrivateRagMethods"]["Properties"]
    if bad == "scope":
        route["AuthorizationScopes"] = ["kg/query"]
    elif bad == "authorizer":
        route["AuthorizerId"] = {"Ref": "other"}
    elif bad == "integration":
        route["Target"] = "other"
    elif bad == "public":
        route["AuthorizationType"] = "NONE"
    elif bad == "new_route":
        rows["Other"] = copy.deepcopy(rows["PrivateRagMethods"])
    elif bad == "remove":
        del rows["PrivateSource"]
    elif bad == "throttle":
        rows["DefaultStage"]["Properties"]["DefaultRouteSettings"]["ThrottlingRateLimit"] = 100
    elif bad == "dependency":
        rows["DefaultStage"]["DependsOn"].remove("PrivateRagMethods")
    elif bad == "iam":
        rows["QueryRole"]["Properties"]["Policies"] = []
    elif bad == "env":
        rows["QueryFunction"]["Properties"]["Environment"]["Variables"]["MODEL_ENABLED"] = "false"
    elif bad == "cognito":
        rows["ReviewerPool"]["Properties"]["MfaConfiguration"] = "ON"
    elif bad == "code_bucket":
        rows["QueryFunction"]["Properties"]["Code"]["S3Bucket"] = "other"
    elif bad == "output":
        after["Outputs"]["PublicUrl"]["Value"] = "other"
    else:
        after["Description"] = "unrelated template change"
    with pytest.raises(ValueError):
        inspect_catalog_update(before, after)


@pytest.mark.parametrize("value", [True, "true", 1, None])
def test_catalog_cannot_be_enabled_without_explicit_bool_and_runtime(value):
    with pytest.raises(ValueError, match="EXPLICIT_RUNTIME_OPT_IN"):
        TracerStack(App(), method_catalog_enabled=value)
