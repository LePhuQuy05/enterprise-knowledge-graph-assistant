"""Review/deploy the invite-only Cognito tracer, never the final RAG demo."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from scripts.foundation import _session, apply, plan

STACK = "kg-r72-tracer"
EXPECTED = {"AWS::ApiGatewayV2::Api": 1, "AWS::Cognito::UserPool": 1,
            "AWS::Cognito::UserPoolResourceServer": 1, "AWS::Cognito::UserPoolClient": 1,
            "AWS::Cognito::UserPoolDomain": 1, "AWS::Logs::LogGroup": 1, "AWS::IAM::Role": 1,
            "AWS::Lambda::Function": 1, "AWS::ApiGatewayV2::Integration": 1,
            "AWS::ApiGatewayV2::Authorizer": 1, "AWS::ApiGatewayV2::Route": 5,
            "AWS::ApiGatewayV2::Stage": 1, "AWS::Lambda::Permission": 1}


def cleanup_failed_create(session):
    """Only a rolled-back first-create stack and its empty, tagged retained pool."""
    client = session.client('cloudformation')
    stack = client.describe_stacks(StackName=STACK)['Stacks'][0]
    tags = {item['Key']: item['Value'] for item in stack.get('Tags', [])}
    if stack['StackStatus'] != 'ROLLBACK_COMPLETE' or tags.get('Project') != 'kg-r72':
        raise ValueError('Cleanup is restricted to the failed project creation')
    resources = client.list_stack_resources(StackName=STACK)
    if resources.get('NextToken'):
        raise ValueError('Unexpected rollback pagination')
    retained = []
    for item in resources['StackResourceSummaries']:
        if item['ResourceType'] not in EXPECTED:
            raise ValueError('Unexpected resource in failed stack')
        if item['ResourceStatus'] == 'DELETE_COMPLETE':
            continue
        if (item['ResourceType'], item['LogicalResourceId'], item['ResourceStatus']) != (
            'AWS::Cognito::UserPool', 'ReviewerPool', 'DELETE_SKIPPED',
        ):
            raise ValueError('Cleanup refuses nonempty or still-live stack resources')
        pool_id = item['PhysicalResourceId']
        cognito = session.client('cognito-idp')
        pool = cognito.describe_user_pool(UserPoolId=pool_id)['UserPool']
        if pool['Name'] != 'kg-r72-reviewers' or pool.get('UserPoolTags', {}).get('Project') != 'kg-r72':
            raise ValueError('Retained pool ownership differs from failed project')
        if cognito.list_users(UserPoolId=pool_id, Limit=1)['Users']:
            raise ValueError('Will not delete a pool containing users')
        retained.append(pool_id)
    for pool_id in retained:
        session.client('cognito-idp').delete_user_pool(UserPoolId=pool_id)
    client.delete_stack(StackName=STACK)
    for _ in range(30):
        try:
            result = client.describe_stacks(StackName=stack['StackId'])['Stacks'][0]
            if result['StackStatus'] == 'DELETE_COMPLETE':
                break
        except Exception:
            raise ValueError('Could not verify cleanup; inspect stack status before retrying') from None
        time.sleep(1)
    else:
        raise ValueError('Failed-stack cleanup is still pending')
    return {'status': 'FAILED_CREATE_CLEANED', 'empty_pools_removed': len(retained),
            'inference_calls': 0, 'foundation_unchanged': True}


def assert_no_cycles(resources):
    graph = {}
    def references(node):
        found = set()
        if isinstance(node, dict):
            if node.get('Ref') in resources:
                found.add(node['Ref'])
            target = node.get('Fn::GetAtt', [])
            if isinstance(target, list) and target and target[0] in resources:
                found.add(target[0])
            for value in node.values():
                found.update(references(value))
        elif isinstance(node, list):
            for value in node:
                found.update(references(value))
        return found
    for name, resource in resources.items():
        dependencies = resource.get('DependsOn', [])
        if isinstance(dependencies, str):
            dependencies = [dependencies]
        graph[name] = references(resource) | set(dependencies)
    done = set()
    def visit(name, path):
        if name in path:
            raise ValueError('Circular CloudFormation dependency')
        if name in done:
            return
        for child in graph[name]:
            visit(child, path | {name})
        done.add(name)
    for name in graph:
        visit(name, set())


def inspect_template(template):
    if set(template) - {'AWSTemplateFormatVersion', 'Description', 'Resources', 'Outputs', 'Metadata'}:
        raise ValueError('Unexpected template features')
    resources = template['Resources']
    if dict(Counter(item['Type'] for item in resources.values())) != EXPECTED:
        raise ValueError('Unexpected tracer resource inventory')
    assert_no_cycles(resources)
    def props(kind):
        return [r['Properties'] for r in resources.values() if r['Type'] == kind]
    pool, = props('AWS::Cognito::UserPool')
    client, = props('AWS::Cognito::UserPoolClient')
    domain, = props('AWS::Cognito::UserPoolDomain')
    if pool['UserPoolTier'] != 'LITE' or not pool['AdminCreateUserConfig']['AllowAdminCreateUserOnly']:
        raise ValueError('Only invite-only Cognito Lite is allowed')
    if client.get('GenerateSecret') is not False or client['AllowedOAuthFlows'] != ['code'] or (
        set(client['AllowedOAuthScopes']) != {'openid', 'kg/read', 'kg/query'}
        or client['AccessTokenValidity'] != 5 or domain['ManagedLoginVersion'] != 1
    ):
        raise ValueError('Unexpected browser OAuth configuration')
    function, = props('AWS::Lambda::Function')
    if function['FunctionName'] != 'kg-r72-query' or function['Runtime'] != 'python3.12' or (
        function['MemorySize'] != 1024 or function['Timeout'] != 28
        or function['Architectures'] != ['x86_64'] or 'ReservedConcurrentExecutions' in function
        or 'VpcConfig' in function or function['Environment']['Variables']['MODEL_ENABLED'] != 'false'
    ):
        raise ValueError('Unexpected Lambda configuration')
    role, = props('AWS::IAM::Role')
    trust = role['AssumeRolePolicyDocument']['Statement']
    if trust != [{'Effect': 'Allow', 'Principal': {'Service': 'lambda.amazonaws.com'},
                  'Action': 'sts:AssumeRole'}]:
        raise ValueError('Only Lambda may assume the tracer runtime role')
    statements = role['Policies'][0]['PolicyDocument']['Statement']
    actions = {action for statement in statements for action in statement['Action']}
    if role['RoleName'] != 'kg-r72-query-runtime' or actions != {
        'dynamodb:GetItem', 'logs:CreateLogStream', 'logs:PutLogEvents',
    } or role.get('ManagedPolicyArns') or any(s['Resource'] == '*' for s in statements):
        raise ValueError('Tracer role must not invoke models, write data or deploy resources')
    routes = {p['RouteKey']: p for p in props('AWS::ApiGatewayV2::Route')}
    if set(routes) != {'GET /', 'GET /config.json', 'GET /auth/callback',
                       'GET /assets/{proxy+}', 'GET /v1/status'}:
        raise ValueError('Public route allowlist changed')
    if routes['GET /v1/status']['AuthorizationType'] != 'JWT' or (
        routes['GET /v1/status']['AuthorizationScopes'] != ['kg/read']
    ):
        raise ValueError('Status route must require JWT and read scope')
    logs, = props('AWS::Logs::LogGroup')
    stage, = props('AWS::ApiGatewayV2::Stage')
    if logs['RetentionInDays'] != 7 or stage['StageName'] != '$default':
        raise ValueError('Unexpected logs or endpoint prefix')
    stage_record = next(r for r in resources.values() if r['Type'] == 'AWS::ApiGatewayV2::Stage')
    route_ids = {name for name, r in resources.items() if r['Type'] == 'AWS::ApiGatewayV2::Route'}
    if not route_ids <= set(stage_record.get('DependsOn', [])):
        raise ValueError('Stage route settings require all routes to exist first')
    return {'template_sha256': sha256_bytes(canonical_json(template)), 'resource_counts': EXPECTED,
            'inference_calls': 0, 'model_enabled': False, 'dependency_cycles': False}


def inspect_changes(response):
    if response.get('StackName') != STACK or response.get('Status') != 'CREATE_COMPLETE' or (
        response.get('ExecutionStatus') != 'AVAILABLE' or response.get('NextToken')
    ):
        raise ValueError('Unexpected tracer change set')
    changes = [c['ResourceChange'] for c in response['Changes']]
    if dict(Counter(c['ResourceType'] for c in changes)) != EXPECTED or any(
        c['Action'] != 'Add' for c in changes
    ):
        raise ValueError('Only the exact first-create tracer plan is supported')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['plan', 'apply', 'cleanup-failed-create'])
    parser.add_argument('--profile', default='kg-dev')
    args = parser.parse_args()
    session = _session(args.profile)
    kwargs = dict(stack_name=STACK, inspector=inspect_template, change_inspector=inspect_changes)
    try:
        if args.command == 'cleanup-failed-create':
            result = cleanup_failed_create(session)
        elif args.command == 'plan':
            result = plan(session, Path('cdk.out/kg-r72-tracer.template.json'),
                          Path('results/tracer/plan.json'), capabilities=['CAPABILITY_NAMED_IAM'], **kwargs)
        else:
            result = apply(session, Path('results/tracer/plan.json'),
                           Path('results/tracer/deployment.json'), **kwargs)
        print(json.dumps(result, indent=2))
    except Exception as exc:
        print(json.dumps({'status': 'INCOMPLETE', 'error_category': type(exc).__name__,
                          'inference_calls': 0}))
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
