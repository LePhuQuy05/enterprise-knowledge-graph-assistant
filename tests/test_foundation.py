import copy

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template

from infra.foundation import FoundationStack
from scripts.foundation import inspect_changes, inspect_template


@pytest.fixture(scope="module")
def template():
    return Template.from_stack(FoundationStack(App(analytics_reporting=False))).to_json()


def test_foundation_is_private_pay_per_request_and_retains_state(template):
    result = inspect_template(template)
    assert result['iam_resources'] == result['compute_resources'] == 0
    assert len(template['Resources']) == 4
    assert 'Parameters' not in template  # No bootstrap-version parameter or asset bucket needed.
    for table in (r for r in template['Resources'].values() if r['Type'] == 'AWS::DynamoDB::Table'):
        assert table['Properties']['KeySchema'] == [
            {'AttributeName': 'PK', 'KeyType': 'HASH'}, {'AttributeName': 'SK', 'KeyType': 'RANGE'},
        ]


def test_template_rejects_added_iam_resource(template):
    mutated = copy.deepcopy(template)
    mutated['Resources']['Escalation'] = {'Type': 'AWS::IAM::Role'}
    with pytest.raises(ValueError):
        inspect_template(mutated)


def test_template_rejects_public_bucket_and_provisioned_table(template):
    mutated = copy.deepcopy(template)
    bucket = next(r for r in mutated['Resources'].values() if r['Type'] == 'AWS::S3::Bucket')
    bucket['Properties']['PublicAccessBlockConfiguration']['BlockPublicPolicy'] = False
    with pytest.raises(ValueError):
        inspect_template(mutated)
    mutated = copy.deepcopy(template)
    table = next(r for r in mutated['Resources'].values() if r['Type'] == 'AWS::DynamoDB::Table')
    table['Properties']['BillingMode'] = 'PROVISIONED'
    with pytest.raises(ValueError):
        inspect_template(mutated)


@pytest.mark.parametrize('alteration', ['delete', 'replace', 'other_stack', 'paginated'])
def test_change_set_execution_rejects_unreviewed_operations(alteration):
    response = {
        'StackName': 'kg-r72-foundation', 'Status': 'CREATE_COMPLETE', 'ExecutionStatus': 'AVAILABLE',
        'Changes': [{'ResourceChange': {'ResourceType': kind, 'Action': 'Add'}}
                    for kind in ['AWS::DynamoDB::Table', 'AWS::DynamoDB::Table',
                                 'AWS::S3::Bucket', 'AWS::S3::BucketPolicy']],
    }
    inspect_changes(response)
    if alteration == 'delete':
        response['Changes'][0]['ResourceChange']['Action'] = 'Remove'
    elif alteration == 'replace':
        response['Changes'][0]['ResourceChange']['Action'] = 'Modify'
    elif alteration == 'other_stack':
        response['StackName'] = 'unrelated-stack'
    else:
        response['NextToken'] = 'next'
    with pytest.raises(ValueError):
        inspect_changes(response)
