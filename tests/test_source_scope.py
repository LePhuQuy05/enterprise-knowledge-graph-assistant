import pytest

from app.source_scope import matches_task


@pytest.mark.parametrize("title,query,allowed", [
    ("Deploying Web Function code from Amazon S3", "Lambda runtime GetObject AccessDenied", False),
    ("Deploying Web Function code from Amazon S3", "Deploy Web Function code from S3", True),
    ("Introducing self-managed Amazon S3 buckets for AWS Lambda function code", "S3 invokes Lambda execution role read object", False),
    ("Introducing self-managed Amazon S3 buckets for AWS Lambda function code", "Lambda deployment package permissions", True),
    ("Configuration options for retrieving sensitive data samples with Macie", "Cross-account KMS decrypt", False),
    ("Configuration options for retrieving sensitive data samples with Macie", "Macie KMS sample permissions", True),
    ("Allowing users in other accounts to use a KMS key", "Cross-account KMS decrypt", True),
])
def test_known_feature_scope_mismatches_are_excluded_not_generalized(title, query, allowed):
    assert matches_task(title, query) is allowed
