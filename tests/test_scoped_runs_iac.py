"""Synth-only operational/security checks; no CloudFormation calls."""

import json

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template

from app.contracts import canonical_json, sha256_bytes
from infra.scoped_runs import ACCOUNT, ROUTES, STACK_NAME, ScopedRunsStack
from tests.test_authorization_evidence import auth_case as auth_case  # noqa: F401
from tests.test_scoped_cloud_handlers import live_config
from tests.test_scoped_owned_pipeline import scoped_stack as scoped_stack  # noqa: F401


def synth(s, tmp_path):
    config = live_config(s) | {"queue_url": f"https://sqs.us-east-1.amazonaws.com/{ACCOUNT}/{STACK_NAME}-work",
        "results_bucket": STACK_NAME+"-results-"+ACCOUNT}
    stack = ScopedRunsStack(App(outdir=str(tmp_path/"cdk"), analytics_reporting=False), config=config,
        package_sha256="f"*64, api_id="fixtureapi123", authorizer_id="auth123",
        deepseek_secret_arn=f"arn:aws:secretsmanager:us-east-1:{ACCOUNT}:secret:kg-r72/deepseek-fixture")
    return Template.from_stack(stack).to_json(), config


def test_separate_v2_routes_and_existing_identity_state_are_preserved(scoped_stack, tmp_path):
    template, _ = synth(scoped_stack, tmp_path)
    rows = template["Resources"]
    assert not any(row["Type"] in {"AWS::DynamoDB::Table", "AWS::Cognito::UserPool", "AWS::ApiGatewayV2::Api",
        "AWS::Lambda::Url", "AWS::EC2::NatGateway", "AWS::Neptune::DBCluster"} for row in rows.values())
    routes = [row["Properties"] for row in rows.values() if row["Type"] == "AWS::ApiGatewayV2::Route"]
    assert len(routes) == 8 and {r["RouteKey"] for r in routes} == {verb+" "+path for verb, path in ROUTES.values()}
    assert all(r["AuthorizationType"] == "JWT" and r["AuthorizerId"] == "auth123" for r in routes)
    assert all(r["AuthorizationScopes"] == ["kg/query" if r["RouteKey"].startswith("POST") else "kg/read"] for r in routes)


def test_only_worker_reads_exact_secret_and_pinned_assets(scoped_stack, tmp_path):
    template, config = synth(scoped_stack, tmp_path)
    rows = template["Resources"]
    for kind in ("Api", "Worker", "Watchdog"):
        role = rows[kind+"Role"]["Properties"]
        statements = [s for p in role["Policies"] for s in p["PolicyDocument"]["Statement"]]
        actions = {action for s in statements for action in s["Action"]}
        assert "dynamodb:Scan" not in actions and "dynamodb:DeleteItem" not in actions
        assert "*" not in actions and not any(a.startswith(("bedrock:", "aws-marketplace:")) for a in actions)
        assert ("secretsmanager:GetSecretValue" in actions) == (kind == "Worker")
        writes = [s for s in statements if "dynamodb:PutItem" in s["Action"]]
        keys = writes[0]["Condition"]["ForAllValues:StringLike"]["dynamodb:LeadingKeys"]
        assert "RUNWORK#kg-r72#scoped-v2" in keys and "RUNWORK#kg-r72" not in keys
        assert "RUN#collection319-scoped-answer-v2" in keys and "RUN#*" not in keys
        env = rows[kind]["Properties"]["Environment"]["Variables"]
        assert env["MODEL_ENABLED"] == ("true" if kind == "Worker" else "false")
        assert "KG_SCOPED_CLOUD_JSON" not in env and sum(len(k)+len(v) for k, v in env.items()) < 4096
        assert env["KG_SCOPED_CONFIG_SHA256"] == sha256_bytes(canonical_json(config))
        assert ("KG_DEEPSEEK_SECRET_ARN" in env) == (kind == "Worker")
    worker = json.dumps(rows["WorkerRole"])
    assert config["assets"]["version"] in worker and config["graph_assets"]["version"] in worker


def test_recovery_latency_retention_and_partial_batch_are_bounded(scoped_stack, tmp_path):
    rows = synth(scoped_stack, tmp_path)[0]["Resources"]
    assert rows["Queue"]["Properties"]["VisibilityTimeout"] == 6*rows["Worker"]["Properties"]["Timeout"] == 1080
    assert rows["Consumer"]["Properties"]["BatchSize"] == 1
    assert rows["Consumer"]["Properties"]["FunctionResponseTypes"] == ["ReportBatchItemFailures"]
    assert rows["Consumer"]["Properties"]["ScalingConfig"] == {"MaximumConcurrency": 2}
    assert rows["RecoveryRetry"]["Properties"]["MaximumRetryAttempts"] == 0
    assert rows["Results"]["DeletionPolicy"] == "Retain"
    assert rows["Results"]["Properties"]["LifecycleConfiguration"]["Rules"][0]["ExpirationInDays"] == 31
    assert all(rows[kind+"Logs"]["Properties"]["RetentionInDays"] == 7 for kind in ("Api", "Worker", "Watchdog"))
    assert len([r for r in rows.values() if r["Type"] == "AWS::CloudWatch::Alarm"]) == 3


def test_mismatched_target_names_rejected_before_stack_build(scoped_stack, tmp_path):
    config = live_config(scoped_stack)
    with pytest.raises(ValueError, match="RESOURCE_BINDING"):
        ScopedRunsStack(App(outdir=str(tmp_path/"cdk"), analytics_reporting=False), config=config,
            package_sha256="f"*64, api_id="fixtureapi123", authorizer_id="auth123",
            deepseek_secret_arn=f"arn:aws:secretsmanager:us-east-1:{ACCOUNT}:secret:kg-r72/deepseek-fixture")
