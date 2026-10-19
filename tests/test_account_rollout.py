import copy
import json

import pytest

from scripts.account_rollout import build_update, inspect_changes, inspect_update


@pytest.fixture
def before():
    # Portable synthetic template: no ignored results files, account identifiers or secrets.
    return {"Resources": {
        "ReviewerPool": {"Type": "AWS::Cognito::UserPool", "Properties": {
            "UserPoolTier": "LITE", "UsernameConfiguration": {"CaseSensitive": False},
            "AdminCreateUserConfig": {"AllowAdminCreateUserOnly": True}}},
        "QueryFunction": {"Type": "AWS::Lambda::Function", "Properties": {
            "Code": {"S3Bucket": "fixture", "S3Key": "packages/old.zip"}, "Timeout": 28,
            "Environment": {"Variables": {"PUBLIC_CONFIG_JSON": {"Fn::Join": ["", [
                '{"client_id":"', {"Ref": "BrowserClient"}, '","issuer":"https://issuer.invalid"}']]}}}}},
        "QueryRole": {"Type": "AWS::IAM::Role", "Properties": {"Policies": [{"PolicyDocument": {"Statement": []}}]}},
        "BrowserClient": {"Type": "AWS::Cognito::UserPoolClient", "Properties": {"UserPoolId": {"Ref": "ReviewerPool"}}},
    }, "Outputs": {"Fixture": {"Value": "unchanged"}}}


def code(before):
    return {"S3Bucket": before["Resources"]["QueryFunction"]["Properties"]["Code"]["S3Bucket"],
            "S3Key": "packages/query/" + "a" * 64 + ".zip"}


def test_exact_update_preserves_identity_routes_and_all_unrelated_resources(before):
    old = copy.deepcopy(before)
    after = build_update(before, code(before))
    result = inspect_update(before, after)
    assert result["modified"] == ["ReviewerPool", "QueryRole", "QueryFunction"]
    assert before == old
    assert before["Outputs"] == after["Outputs"]
    for name in before["Resources"]:
        if name not in result["modified"]:
            assert before["Resources"][name] == after["Resources"][name]
    pool = after["Resources"]["ReviewerPool"]["Properties"]
    assert pool["AutoVerifiedAttributes"] == ["email"]
    assert "UsernameAttributes" not in pool and "Schema" not in pool
    assert pool["AccountRecoverySetting"]["RecoveryMechanisms"] == [{"Name": "verified_email", "Priority": 1}]


@pytest.mark.parametrize("tamper", ["client", "schema", "role", "function", "route"])
def test_update_guard_rejects_identity_privilege_runtime_or_route_drift(before, tamper):
    after = build_update(before, code(before))
    if tamper == "client":
        after["Resources"]["BrowserClient"]["Properties"]["CallbackURLs"] = ["https://attacker.invalid"]
    elif tamper == "schema":
        after["Resources"]["ReviewerPool"]["Properties"]["UsernameAttributes"] = ["email"]
    elif tamper == "role":
        after["Resources"]["QueryRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"][-1]["Resource"] = "*"
    elif tamper == "function":
        after["Resources"]["QueryFunction"]["Properties"]["Timeout"] = 900
    else:
        after["Resources"]["PublicBackdoor"] = {"Type": "AWS::ApiGatewayV2::Route"}
    with pytest.raises(ValueError, match="ONLY_EXACT"):
        inspect_update(before, after)


def test_change_set_requires_all_three_modifications_and_refuses_replacements():
    rows = [{"ResourceChange": {"LogicalResourceId": name, "Action": "Modify", "Replacement": "False"}}
            for name in ("ReviewerPool", "QueryRole", "QueryFunction")]
    change = {"StackName": "kg-r72-tracer", "Status": "CREATE_COMPLETE", "ExecutionStatus": "AVAILABLE", "Changes": rows}
    inspect_changes(change)
    for value in ("True", "Conditional"):
        altered = copy.deepcopy(change)
        altered["Changes"][0]["ResourceChange"]["Replacement"] = value
        with pytest.raises(ValueError, match="REPLACEMENT"):
            inspect_changes(altered)
    with pytest.raises(ValueError, match="INCOMPLETE"):
        inspect_changes(change | {"Changes": rows[:2]})


@pytest.mark.parametrize("tamper", ["live", "reviewed"])
def test_apply_refuses_stale_or_changed_templates_before_any_execution(before, tmp_path, monkeypatch, tamper):
    from scripts import account_rollout

    after = build_update(before, code(before))
    report = inspect_update(before, after) | {
        "status": "AWS_CHANGE_SET_REVIEWED_NOT_EXECUTED", "change_set_id": "fixture-change"}
    (tmp_path / "plan.json").write_text(json.dumps(report), encoding="utf-8")
    changed = copy.deepcopy(before if tamper == "live" else after)
    changed["Outputs"]["Fixture"]["Value"] = "changed-after-review"
    class Client:
        def describe_change_set(self, **kwargs):
            return {"StackName": "kg-r72-tracer", "Status": "CREATE_COMPLETE", "ExecutionStatus": "AVAILABLE",
                    "Changes": [{"ResourceChange": {"LogicalResourceId": name, "Action": "Modify", "Replacement": "False"}}
                                for name in ("ReviewerPool", "QueryRole", "QueryFunction")]}

        def get_template(self, **kwargs):
            reviewed = "ChangeSetName" in kwargs
            return {"TemplateBody": changed if reviewed == (tamper == "reviewed") else after if reviewed else before}

        def execute_change_set(self, **kwargs):
            pytest.fail("Stale or changed templates must never execute")
    monkeypatch.setattr(account_rollout, "OUT", tmp_path)
    monkeypatch.setattr(account_rollout, "_cfn", lambda _: Client())
    with pytest.raises(ValueError, match="TEMPLATE_CHANGED"):
        account_rollout.apply_reviewed(None)
