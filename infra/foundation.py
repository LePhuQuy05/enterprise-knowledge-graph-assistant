"""Asset-free state stack: no bootstrap roles, compute or model subscriptions."""

from aws_cdk import BootstraplessSynthesizer, CfnOutput, RemovalPolicy, Stack, Tags
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_s3 as s3
from constructs import Construct

STACK_NAME = "kg-r72-foundation"
PROJECT = "kg-r72"
QUERY_TABLE = "kg-r72-query-state"
METADATA_TABLE = "kg-r72-metadata"


class FoundationStack(Stack):
    def __init__(self, scope: Construct, construct_id: str = STACK_NAME):
        super().__init__(
            scope, construct_id, stack_name=STACK_NAME,
            synthesizer=BootstraplessSynthesizer(),
            description="KG R72 private snapshot storage and shared state; no inference or compute",
        )
        Tags.of(self).add("Project", PROJECT)
        Tags.of(self).add("Owner", "portfolio")
        for identifier, name in (("QueryState", QUERY_TABLE), ("Metadata", METADATA_TABLE)):
            table = dynamodb.CfnTable(
                self, identifier, table_name=name, billing_mode="PAY_PER_REQUEST",
                attribute_definitions=[
                    dynamodb.CfnTable.AttributeDefinitionProperty(attribute_name=k, attribute_type="S")
                    for k in ("PK", "SK")
                ],
                key_schema=[
                    dynamodb.CfnTable.KeySchemaProperty(attribute_name="PK", key_type="HASH"),
                    dynamodb.CfnTable.KeySchemaProperty(attribute_name="SK", key_type="RANGE"),
                ],
                sse_specification=dynamodb.CfnTable.SSESpecificationProperty(sse_enabled=True),
                time_to_live_specification=dynamodb.CfnTable.TimeToLiveSpecificationProperty(
                    attribute_name="expires_at", enabled=True,
                ),
            )
            table.apply_removal_policy(RemovalPolicy.RETAIN)
            CfnOutput(self, identifier + "TableName", value=table.ref)
        bucket = s3.CfnBucket(
            self, "Artifacts",
            bucket_encryption=s3.CfnBucket.BucketEncryptionProperty(
                server_side_encryption_configuration=[
                    s3.CfnBucket.ServerSideEncryptionRuleProperty(
                        server_side_encryption_by_default=s3.CfnBucket.ServerSideEncryptionByDefaultProperty(
                            sse_algorithm="AES256",
                        ),
                    ),
                ],
            ),
            public_access_block_configuration=s3.CfnBucket.PublicAccessBlockConfigurationProperty(
                block_public_acls=True, block_public_policy=True,
                ignore_public_acls=True, restrict_public_buckets=True,
            ),
            ownership_controls=s3.CfnBucket.OwnershipControlsProperty(
                rules=[s3.CfnBucket.OwnershipControlsRuleProperty(object_ownership="BucketOwnerEnforced")],
            ),
        )
        bucket.apply_removal_policy(RemovalPolicy.RETAIN)
        policy = s3.CfnBucketPolicy(self, "ArtifactsTlsPolicy", bucket=bucket.ref, policy_document={
            "Version": "2012-10-17", "Statement": [{
                "Sid": "DenyInsecureTransport", "Effect": "Deny", "Principal": "*",
                "Action": "s3:*", "Resource": [bucket.attr_arn, bucket.attr_arn + "/*"],
                "Condition": {"Bool": {"aws:SecureTransport": "false"}},
            }],
        })
        policy.apply_removal_policy(RemovalPolicy.RETAIN)
        CfnOutput(self, "ArtifactBucketName", value=bucket.ref)
