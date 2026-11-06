import copy
import json

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template

from infra.owned_runs import ROUTES, OwnedRunsStack


def synthesize(tmp_path, **options):
    artifacts = {"bucket": "kg-r72-fixture-bucket", "bundle_version": "wb_" + "a" * 64,
        "bundle_manifest_sha256": "b" * 64, "bundle_manifest_bytes": 123, "package_sha256": "c" * 64}
    release = {"collection_version": "collection_" + "d" * 64, "collection_manifest_sha256": "e" * 64,
        "method_config_sha256": {"bm25": "f" * 64, "vector": "1" * 64, "hybrid": "2" * 64}}
    stack = OwnedRunsStack(App(outdir=str(tmp_path / "cdk"), analytics_reporting=False), artifacts=artifacts,
        release=release, public_config={"issuer": "fixture", "client_id": "fixture"}, api_id="fixtureapi123",
        authorizer_id="auth123", corpus={"documents": 316, "chunks": 2978, "service_groups": 24}, **options)
    return Template.from_stack(stack).to_json()


def test_sqs_visibility_partial_batch_and_concurrency_are_bounded(tmp_path):
    t = synthesize(tmp_path)["Resources"]
    assert t["Queue"]["Properties"]["VisibilityTimeout"] == 6 * t["Worker"]["Properties"]["Timeout"] == 540
    assert t["Queue"]["Properties"]["RedrivePolicy"]["maxReceiveCount"] == 5
    assert t["Consumer"]["Properties"]["BatchSize"] == 1
    assert t["Consumer"]["Properties"]["ScalingConfig"] == {"MaximumConcurrency": 2}
    assert t["Consumer"]["Properties"]["FunctionResponseTypes"] == ["ReportBatchItemFailures"]
    assert t["DeadLetters"]["Properties"]["MessageRetentionPeriod"] == 14 * 86400


def test_no_secrets_model_permissions_public_bucket_or_new_identity(tmp_path):
    t = synthesize(tmp_path)["Resources"]
    assert not any(r["Type"] in {"AWS::DynamoDB::Table", "AWS::Cognito::UserPool", "AWS::ApiGatewayV2::Api", "AWS::Lambda::Url"} for r in t.values())
    for role in ("ApiRole", "WorkerRole", "WatchdogRole"):
        actions = [a for p in t[role]["Properties"]["Policies"] for st in p["PolicyDocument"]["Statement"] for a in st["Action"]]
        assert not any(a.startswith(("bedrock:", "secretsmanager:")) or a == "*" for a in actions)
        assert "dynamodb:Scan" not in actions
    for name in ("Api", "Worker", "Watchdog"):
        env = t[name]["Properties"]["Environment"]["Variables"]
        assert env["MODEL_ENABLED"] == "false" and env["KG_RUNS_ENABLED"] == "true"
        assert "VpcConfig" not in t[name]["Properties"]
    assert all(t["Results"]["Properties"]["PublicAccessBlockConfiguration"].values())
    assert t["Results"]["DeletionPolicy"] == "Retain"
    assert t["Results"]["Properties"]["LifecycleConfiguration"]["Rules"][0]["ExpirationInDays"] == 31


def test_all_new_routes_reuse_same_api_jwt_and_correct_scope(tmp_path):
    resources = synthesize(tmp_path)["Resources"]
    for identifier, (method, path) in ROUTES.items():
        p = resources[identifier + "Route"]["Properties"]
        assert p["ApiId"] == "fixtureapi123" and p["AuthorizerId"] == "auth123"
        assert p["AuthorizationType"] == "JWT" and p["AuthorizationScopes"] == ["kg/query" if method == "POST" else "kg/read"]
        assert p["RouteKey"] == method + " " + path
    assert resources["Recovery"]["Properties"]["ScheduleExpression"] == "rate(1 minute)"
    assert resources["RecoveryRetry"]["Properties"]["MaximumRetryAttempts"] == 0


def test_transaction_writes_use_putitem_with_enclosing_operation_and_bounded_keys(tmp_path):
    resources = synthesize(tmp_path)["Resources"]
    for role in ("ApiRole", "WorkerRole", "WatchdogRole"):
        statements = resources[role]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
        assert all("dynamodb:TransactWriteItems" not in s["Action"] for s in statements)
        writes = [s for s in statements if "dynamodb:PutItem" in s["Action"]]
        assert len(writes) == 1 and writes[0]["Action"] == ["dynamodb:PutItem"]
        condition = writes[0]["Condition"]
        assert condition["ForAnyValue:StringEquals"]["dynamodb:EnclosingOperation"] == ["TransactWriteItems"]
        assert condition["ForAllValues:StringLike"]["dynamodb:LeadingKeys"] == [
            "TENANT#demo#USER#*", "QUOTA#kg-r72#*", "RUNACTIVE#kg-r72", "RUNWORK#kg-r72"]


def test_template_review_cannot_accept_broadened_role_or_extra_resource(tmp_path):
    from scripts.plan_owned_runs import inspect_owned_template

    template = synthesize(tmp_path)
    assert inspect_owned_template(template, template)["model_calls"] == 0
    changed = copy.deepcopy(template)
    changed["Resources"]["WorkerRole"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"].append(
        {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": "*"})
    with pytest.raises(ValueError, match="UNAPPROVED"):
        inspect_owned_template(template, changed)
    changed = json.loads(json.dumps(template))
    changed["Resources"]["PublicUrl"] = {"Type": "AWS::Lambda::Url", "Properties": {}}
    with pytest.raises(ValueError, match="UNAPPROVED"):
        inspect_owned_template(template, changed)
