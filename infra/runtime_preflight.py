"""Isolated IAM-only runtime verification, no API route or inference capability."""

import json
from pathlib import Path

from aws_cdk import Aws, BootstraplessSynthesizer, CfnOutput, Fn, RemovalPolicy, Stack, Tags
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_secretsmanager as secrets
from constructs import Construct

STACK_NAME = "kg-r72-runtime-preflight"
FUNCTION_NAME = STACK_NAME
ROOT = Path(__file__).resolve().parents[1]


class RuntimePreflightStack(Stack):
    def __init__(self, scope: Construct, artifacts: dict, construct_id=STACK_NAME):
        super().__init__(scope, construct_id, stack_name=STACK_NAME,
            synthesizer=BootstraplessSynthesizer(), description="Private BGE/S3/secret preflight; no public API or inference")
        Tags.of(self).add("Project", "kg-r72")
        Tags.of(self).add("Owner", "portfolio")
        config = artifacts["runtime_config"]
        secret = secrets.CfnSecret(self, "DeepSeekKey", name="kg-r72/deepseek",
            description="DeepSeek server-only key; initialized separately, never in CloudFormation",
            tags=[{"key": "Project", "value": "kg-r72"}])
        secret.apply_removal_policy(RemovalPolicy.RETAIN)
        group = logs.CfnLogGroup(self, "Logs", log_group_name="/aws/lambda/" + FUNCTION_NAME, retention_in_days=7)
        objects = ["arn:" + Aws.PARTITION + ":s3:::" + config["bucket"] + "/" + row["key"]
                   for row in artifacts["objects"] if row["kind"] in {"MODEL", "SNAPSHOT"}]
        role = iam.CfnRole(self, "Role", role_name=STACK_NAME + "-role", assume_role_policy_document={
            "Version": "2012-10-17", "Statement": [{"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"},
                                                      "Action": "sts:AssumeRole"}]},
            policies=[iam.CfnRole.PolicyProperty(policy_name="read-pinned-preflight-assets", policy_document={
                "Version": "2012-10-17", "Statement": [
                    {"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": sorted(objects)},
                    {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"], "Resource": secret.ref},
                    {"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                     "Resource": Fn.sub("arn:${AWS::Partition}:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/lambda/"
                                        + FUNCTION_NAME + ":log-stream:*")},
                ]})])
        function = lambda_.CfnFunction(self, "Function", function_name=FUNCTION_NAME, runtime="python3.12",
            architectures=["x86_64"], memory_size=1024, timeout=28, role=role.attr_arn,
            ephemeral_storage=lambda_.CfnFunction.EphemeralStorageProperty(size=512),
            handler="app.cloud_preflight.handler",
            code=lambda_.CfnFunction.CodeProperty(s3_bucket=config["bucket"], s3_key=artifacts["package_key"]),
            environment=lambda_.CfnFunction.EnvironmentProperty(variables={
                "KG_ASSET_CONFIG_JSON": json.dumps(config, separators=(",", ":")),
                "KG_DEEPSEEK_SECRET_ARN": secret.ref, "MODEL_ENABLED": "false",
            }))
        function.add_resource_dependency(group)
        CfnOutput(self, "FunctionName", value=function.ref)
        CfnOutput(self, "DeepSeekSecretArn", value=secret.ref)
