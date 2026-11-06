"""Separate opt-in319 qualified-answer stack; synthesizing performs no AWS write."""

import re

from aws_cdk import Aws, BootstraplessSynthesizer, CfnOutput, Fn, RemovalPolicy, Stack, Tags
from aws_cdk import aws_apigatewayv2 as api
from aws_cdk import aws_cloudwatch as cw
from aws_cdk import aws_events as events
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs

from app.contracts import canonical_json, sha256_bytes
from app.scoped_run_cloud import ScopedCloudConfigV2

STACK_NAME = "kg-r72-scoped-runs"
ACCOUNT = "357433444130"
ROUTES = {
    "Capabilities": ("GET", "/v2/scoped-run-capabilities"), "Submit": ("POST", "/v2/scoped-runs"),
    "History": ("GET", "/v2/scoped-runs"), "Detail": ("GET", "/v2/scoped-runs/{run_id}"),
    "Trace": ("GET", "/v2/scoped-runs/{run_id}/trace"), "Cancel": ("POST", "/v2/scoped-runs/{run_id}/cancel"),
    "Result": ("GET", "/v2/scoped-runs/{run_id}/methods/{method}"),
    "Source": ("GET", "/v2/scoped-runs/{run_id}/methods/{method}/sources/{chunk_id}"),
}


class ScopedRunsStack(Stack):
    def __init__(self, scope, *, config, package_sha256, api_id, authorizer_id, deepseek_secret_arn):
        config = ScopedCloudConfigV2.model_validate_json(canonical_json(config)).model_dump(mode="json")
        if not re.fullmatch(r"[0-9a-f]{64}", package_sha256) or not re.fullmatch(r"[a-z0-9]{5,20}", api_id) or (
                not re.fullmatch(r"[a-z0-9]{4,20}", authorizer_id) or not re.fullmatch(
                rf"arn:aws:secretsmanager:us-east-1:{ACCOUNT}:secret:[A-Za-z0-9/_+=.@-]+", deepseek_secret_arn)):
            raise ValueError("SCOPED_INFRA_TARGET_PINS")
        queue_name = STACK_NAME+"-work"
        bucket_name = STACK_NAME+"-results-"+ACCOUNT
        if config["queue_url"] != f"https://sqs.us-east-1.amazonaws.com/{ACCOUNT}/{queue_name}" or (
                config["results_bucket"] != bucket_name):
            raise ValueError("SCOPED_INFRA_RESOURCE_BINDING")
        config_digest = sha256_bytes(canonical_json(config))
        super().__init__(scope, STACK_NAME, stack_name=STACK_NAME, synthesizer=BootstraplessSynthesizer(),
            env={"account": ACCOUNT, "region": "us-east-1"},
            description="Opt-in319 owned qualified answers V2, shared quota/budget and isolated queue/assets/routes")
        Tags.of(self).add("Project", "kg-r72")
        Tags.of(self).add("Owner", "portfolio")
        bucket = s3.CfnBucket(self, "Results", bucket_name=bucket_name,
            bucket_encryption=s3.CfnBucket.BucketEncryptionProperty(server_side_encryption_configuration=[
                s3.CfnBucket.ServerSideEncryptionRuleProperty(server_side_encryption_by_default=
                    s3.CfnBucket.ServerSideEncryptionByDefaultProperty(sse_algorithm="AES256"))]),
            public_access_block_configuration=s3.CfnBucket.PublicAccessBlockConfigurationProperty(
                block_public_acls=True, block_public_policy=True, ignore_public_acls=True, restrict_public_buckets=True),
            ownership_controls=s3.CfnBucket.OwnershipControlsProperty(rules=[
                s3.CfnBucket.OwnershipControlsRuleProperty(object_ownership="BucketOwnerEnforced")]),
            lifecycle_configuration=s3.CfnBucket.LifecycleConfigurationProperty(rules=[s3.CfnBucket.RuleProperty(
                id="expire-qualified-results", status="Enabled", expiration_in_days=31,
                abort_incomplete_multipart_upload=s3.CfnBucket.AbortIncompleteMultipartUploadProperty(days_after_initiation=1))]))
        bucket.apply_removal_policy(RemovalPolicy.RETAIN)
        s3.CfnBucketPolicy(self, "ResultsTls", bucket=bucket.ref, policy_document={"Version": "2012-10-17", "Statement": [
            {"Effect": "Deny", "Principal": "*", "Action": "s3:*", "Resource": [bucket.attr_arn, bucket.attr_arn+"/*"],
                "Condition": {"Bool": {"aws:SecureTransport": "false"}}}]})
        dead = sqs.CfnQueue(self, "DeadLetters", message_retention_period=14*86400, sqs_managed_sse_enabled=True)
        queue = sqs.CfnQueue(self, "Queue", queue_name=queue_name, visibility_timeout=1080, message_retention_period=86400,
            receive_message_wait_time_seconds=20, sqs_managed_sse_enabled=True,
            redrive_policy={"deadLetterTargetArn": dead.attr_arn, "maxReceiveCount": 5})
        table = Fn.sub("arn:${AWS::Partition}:dynamodb:${AWS::Region}:${AWS::AccountId}:table/kg-r72-query-state")
        metadata = Fn.sub("arn:${AWS::Partition}:dynamodb:${AWS::Region}:${AWS::AccountId}:table/kg-r72-metadata")
        keys = ["TENANT#demo#USER#*", "QUOTA#kg-r72#*", "RUNACTIVE#kg-r72", "RUNWORK#kg-r72#scoped-v2",
            "RUN#collection319-scoped-answer-v2", "FUNDING#kg-r72#external-r72-v1", "BUDGET#kg-r72#20??-??", "OP#*"]
        common = [{"Effect": "Allow", "Action": ["dynamodb:GetItem", "dynamodb:Query"], "Resource": table},
            {"Effect": "Allow", "Action": ["dynamodb:PutItem"], "Resource": table,
                "Condition": {"ForAllValues:StringLike": {"dynamodb:LeadingKeys": keys},
                    "ForAnyValue:StringEquals": {"dynamodb:EnclosingOperation": ["TransactWriteItems"]}}},
            {"Effect": "Allow", "Action": ["dynamodb:GetItem"], "Resource": metadata}]
        functions = {}
        for kind, handler, memory, timeout in (("Api", "api_handler", 512, 25), ("Worker", "worker_handler", 1536, 180),
                ("Watchdog", "watchdog_handler", 512, 30)):
            name = STACK_NAME+"-"+kind.lower()
            group = logs.CfnLogGroup(self, kind+"Logs", log_group_name="/aws/lambda/"+name, retention_in_days=7)
            policy = list(common)+[{"Effect": "Allow", "Action": ["logs:CreateLogStream", "logs:PutLogEvents"],
                "Resource": Fn.sub("arn:${AWS::Partition}:logs:${AWS::Region}:${AWS::AccountId}:log-group:/aws/lambda/"+name+":log-stream:*")}]
            if kind in {"Api", "Worker"}:
                policy.append({"Effect": "Allow", "Action": ["s3:GetObject"]+(["s3:PutObject"] if kind == "Worker" else []),
                    "Resource": bucket.attr_arn+"/tenants/demo/users/*/runs/*/qualified-v2/*"})
            if kind == "Worker":
                policy.extend([{"Effect": "Allow", "Action": ["sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:GetQueueAttributes"],
                    "Resource": queue.attr_arn}, {"Effect": "Allow", "Action": ["secretsmanager:GetSecretValue"],
                    "Resource": deepseek_secret_arn}])
                for key, prefix in (("assets", "workbench"), ("graph_assets", "scoped-graph")):
                    asset = config[key]
                    policy.append({"Effect": "Allow", "Action": ["s3:GetObject"], "Resource": Fn.join("", ["arn:",
                        Aws.PARTITION, ":s3:::", asset["bucket"], "/", prefix, "/", asset["version"], "/*"])})
            else:
                policy.append({"Effect": "Allow", "Action": ["sqs:SendMessage"], "Resource": queue.attr_arn})
            role = iam.CfnRole(self, kind+"Role", assume_role_policy_document={"Version": "2012-10-17", "Statement": [
                {"Effect": "Allow", "Principal": {"Service": "lambda.amazonaws.com"}, "Action": "sts:AssumeRole"}]},
                policies=[iam.CfnRole.PolicyProperty(policy_name="scoped-v2-"+kind.lower(),
                    policy_document={"Version": "2012-10-17", "Statement": policy})])
            variables = {"MODEL_ENABLED": "true" if kind == "Worker" else "false", "KG_SCOPED_RUNS_ENABLED": "true",
                "KG_SCOPED_CONFIG_PATH": f"configs/releases/scoped-cloud-{config_digest}.json", "KG_SCOPED_CONFIG_SHA256": config_digest}
            if kind == "Worker":
                variables["KG_DEEPSEEK_SECRET_ARN"] = deepseek_secret_arn
            function = lambda_.CfnFunction(self, kind, function_name=name, runtime="python3.12", architectures=["x86_64"],
                handler="app.scoped_run_cloud."+handler, memory_size=memory, timeout=timeout, role=role.attr_arn,
                ephemeral_storage=lambda_.CfnFunction.EphemeralStorageProperty(size=1024 if kind == "Worker" else 512),
                code=lambda_.CfnFunction.CodeProperty(s3_bucket=config["assets"]["bucket"],
                    s3_key=f"packages/scoped-v2/{package_sha256}.zip"),
                environment=lambda_.CfnFunction.EnvironmentProperty(variables=variables))
            function.add_resource_dependency(group)
            functions[kind] = function
        lambda_.CfnEventSourceMapping(self, "Consumer", function_name=functions["Worker"].attr_arn,
            event_source_arn=queue.attr_arn, batch_size=1, maximum_batching_window_in_seconds=0,
            function_response_types=["ReportBatchItemFailures"],
            scaling_config=lambda_.CfnEventSourceMapping.ScalingConfigProperty(maximum_concurrency=2), enabled=True)
        rule = events.CfnRule(self, "Recovery", schedule_expression="rate(1 minute)", state="ENABLED", targets=[
            events.CfnRule.TargetProperty(id="scoped-slot-recovery", arn=functions["Watchdog"].attr_arn,
                input='{"operation":"SCOPED_WATCHDOG_V2"}', retry_policy=events.CfnRule.RetryPolicyProperty(
                    maximum_retry_attempts=0, maximum_event_age_in_seconds=60))])
        lambda_.CfnPermission(self, "RecoveryInvoke", action="lambda:InvokeFunction", function_name=functions["Watchdog"].ref,
            principal="events.amazonaws.com", source_arn=rule.attr_arn)
        lambda_.CfnEventInvokeConfig(self, "RecoveryRetry", function_name=functions["Watchdog"].ref, qualifier="$LATEST",
            maximum_retry_attempts=0, maximum_event_age_in_seconds=60)
        integration = api.CfnIntegration(self, "Integration", api_id=api_id, integration_type="AWS_PROXY",
            integration_uri=functions["Api"].attr_arn, payload_format_version="2.0", integration_method="POST", timeout_in_millis=29000)
        for identifier, (method, path) in ROUTES.items():
            api.CfnRoute(self, identifier+"Route", api_id=api_id, route_key=method+" "+path, authorization_type="JWT",
                authorizer_id=authorizer_id, authorization_scopes=["kg/query" if method == "POST" else "kg/read"],
                target="integrations/"+integration.ref)
        lambda_.CfnPermission(self, "ApiInvoke", action="lambda:InvokeFunction", function_name=functions["Api"].ref,
            principal="apigateway.amazonaws.com", source_arn=Fn.sub(
                "arn:${AWS::Partition}:execute-api:${AWS::Region}:${AWS::AccountId}:"+api_id+"/*/*/v2/scoped-run*"))
        for identifier, namespace, metric, dimension, threshold in (
            ("DeadLetterAlarm", "AWS/SQS", "ApproximateNumberOfMessagesVisible", {"Name": "QueueName", "Value": dead.attr_queue_name}, 1),
            ("WorkerErrorAlarm", "AWS/Lambda", "Errors", {"Name": "FunctionName", "Value": functions["Worker"].ref}, 1),
            ("RecoveryErrorAlarm", "AWS/Lambda", "Errors", {"Name": "FunctionName", "Value": functions["Watchdog"].ref}, 1)):
            cw.CfnAlarm(self, identifier, namespace=namespace, metric_name=metric,
                dimensions=[cw.CfnAlarm.DimensionProperty(name=dimension["Name"], value=dimension["Value"])],
                statistic="Maximum" if namespace == "AWS/SQS" else "Sum", period=60, evaluation_periods=1,
                threshold=threshold, comparison_operator="GreaterThanOrEqualToThreshold", treat_missing_data="notBreaching")
        CfnOutput(self, "WorkerFunction", value=functions["Worker"].ref)
        CfnOutput(self, "ApiFunction", value=functions["Api"].ref)
        CfnOutput(self, "ResultBucket", value=bucket.ref)
        CfnOutput(self, "QueueUrl", value=queue.ref)
