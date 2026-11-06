"""Separate IAM-only collection preflight; no changes to the running web/auth/state stacks."""

import json
import re

from aws_cdk import Aws, BootstraplessSynthesizer, CfnOutput, Fn, Stack, Tags
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

STACK_NAME = "kg-r72-workbench-preflight"


def validate_artifacts(artifacts):
    if type(artifacts) is not dict or set(artifacts) != {"bucket", "bundle_version", "bundle_manifest_sha256", "bundle_manifest_bytes", "package_sha256"}:
        raise ValueError("WORKBENCH_PREFLIGHT_ARTIFACT_SHAPE")
    if any(type(artifacts[key]) is not str or re.fullmatch(pattern, artifacts[key]) is None for key, pattern in (
        ("bucket", r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]"), ("bundle_version", r"wb_[0-9a-f]{64}"),
        ("bundle_manifest_sha256", r"[0-9a-f]{64}"), ("package_sha256", r"[0-9a-f]{64}"))) or (
        type(artifacts["bundle_manifest_bytes"]) is not int or not 0 < artifacts["bundle_manifest_bytes"] <= 1024 * 1024
    ):
        raise ValueError("WORKBENCH_PREFLIGHT_ARTIFACT_PINS")
    return artifacts


class WorkbenchPreflightStack(Stack):
    def __init__(self, scope: Construct, artifacts, construct_id=STACK_NAME):
        artifacts = validate_artifacts(artifacts)
        super().__init__(scope, construct_id, stack_name=STACK_NAME, synthesizer=BootstraplessSynthesizer(),
            description="IAM-only pinned collection retrieval preflight; no public API, model inference or shared-state writes")
        Tags.of(self).add("Project", "kg-r72")
        Tags.of(self).add("Owner", "portfolio")
        group = logs.CfnLogGroup(self, "Logs", log_group_name="/aws/lambda/" + STACK_NAME, retention_in_days=7)
        role = iam.CfnRole(self, "Role", role_name=STACK_NAME + "-role", assume_role_policy_document={
            "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]},
            policies=[iam.CfnRole.PolicyProperty(policy_name="read-one-workbench-bundle", policy_document={
                "Version": "2012-10-17", "Statement": [
                    {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource":
                        "arn:" + Aws.PARTITION + ":s3:::" + artifacts["bucket"] + "/workbench/" + artifacts["bundle_version"] + "/*"},
                    {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"], "Resource":
                        Fn.sub("arn:${AWS::Partition}:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/lambda/" + STACK_NAME + ":log-stream:*")},
                ]})])
        release = {"bucket": artifacts["bucket"], "version": artifacts["bundle_version"],
            "manifest_sha256": artifacts["bundle_manifest_sha256"], "manifest_bytes": artifacts["bundle_manifest_bytes"], "tenant_id": "demo"}
        function = lambda_.CfnFunction(self, "Function", function_name=STACK_NAME, runtime="python3.12", architectures=["x86_64"],
            handler="app.workbench_preflight.handler", memory_size=1024, timeout=90, role=role.attr_arn,
            ephemeral_storage=lambda_.CfnFunction.EphemeralStorageProperty(size=512),
            code=lambda_.CfnFunction.CodeProperty(s3_bucket=artifacts["bucket"], s3_key="packages/workbench/" + artifacts["package_sha256"] + ".zip"),
            environment=lambda_.CfnFunction.EnvironmentProperty(variables={"MODEL_ENABLED": "false",
                "KG_WORKBENCH_RELEASE_JSON": json.dumps(release, separators=(",", ":"))}))
        function.add_resource_dependency(group)
        CfnOutput(self, "FunctionName", value=function.ref)
