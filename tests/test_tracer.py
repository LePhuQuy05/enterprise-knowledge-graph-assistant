import copy
import json
import time

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template

from app import tracer
from infra.tracer import TracerStack, handler_source
from scripts.tracer import assert_no_cycles, inspect_template

SUB = "00000000-0000-4000-8000-000000000001"
CONFIG = {"issuer": "https://issuer.example/pool", "client_id": "client"}


def event(path="/v1/status", **claims):
    return {"rawPath": path, "requestContext": {"http": {"method": "GET"}, "authorizer": {"jwt": {
        "claims": {"sub": SUB, "token_use": "access", "iss": CONFIG['issuer'], "client_id": 'client',
                   "exp": str(int(time.time()) + 300), "scope": "openid kg/read", **claims},
    }}}}


@pytest.fixture(autouse=True)
def public_config(monkeypatch):
    monkeypatch.setattr(tracer, 'PUBLIC_CONFIG', CONFIG)


def test_public_static_is_served_before_any_aws_client(monkeypatch):
    monkeypatch.setattr(tracer, 'STATIC_ASSETS', {'/': {'body': '<h1>Fixture</h1>', 'content_type': 'text/html'}})
    def forbidden(*args):
        pytest.fail('Static assets must not touch membership or model services')
    monkeypatch.setattr(tracer, 'membership', forbidden)
    assert tracer.handler(event('/'), None)['statusCode'] == 200
    assert tracer.handler(event('/auth/callback'), None)['statusCode'] == 200
    assert json.loads(tracer.handler(event('/config.json'), None)['body']) == CONFIG
    for path in ['/../docs/README.md', '/assets/../config.json', '/assets/%2e%2e/secrets', '/v1/queries']:
        assert tracer.handler(event(path), None)['statusCode'] == 404


@pytest.mark.parametrize('claims,status', [({'token_use': 'id'}, 401), ({'exp': '0'}, 401),
                                         ({'iss': 'wrong'}, 401), ({'client_id': 'wrong'}, 401),
                                         ({'sub': '../../other'}, 401), ({'scope': 'openid'}, 403)])
def test_bad_claims_rejected_before_membership_read(monkeypatch, claims, status):
    def forbidden(*args):
        pytest.fail('Invalid claims must not reach storage')
    monkeypatch.setattr(tracer, 'membership', forbidden)
    assert tracer.handler(event(**claims), None)['statusCode'] == status


def member(status='ACTIVE'):
    return {'record_type': {'S': 'MEMBERSHIP'}, 'schema_version': {'N': '1'},
            'status': {'S': status}, 'tenant_id': {'S': 'demo'}, 'roles': {'SS': ['reviewer']}}


def test_active_membership_is_resolved_by_verified_subject(monkeypatch):
    def lookup(sub):
        assert sub == SUB
        return member()
    monkeypatch.setattr(tracer, 'membership', lookup)
    request = event()
    request['queryStringParameters'] = {'tenant': 'attacker'}
    response = tracer.handler(request, None)
    payload = json.loads(response['body'])
    assert response['statusCode'] == 200 and payload['tenant_id'] == 'demo'
    assert payload['fixture_only'] and not payload['model_enabled']
    assert response['headers']['cache-control'] == 'no-store'


@pytest.mark.parametrize('item', [None, member('DISABLED'), member() | {'roles': {'SS': ['other']}}])
def test_missing_or_disabled_membership_fails_closed(monkeypatch, item):
    monkeypatch.setattr(tracer, 'membership', lambda sub: item)
    assert tracer.handler(event(), None)['statusCode'] == 403


def test_storage_outage_is_a_service_error_not_abstention(monkeypatch):
    def broken(sub):
        raise TimeoutError('private service error')
    monkeypatch.setattr(tracer, 'membership', broken)
    response = tracer.handler(event(), None)
    assert response['statusCode'] == 503 and 'private' not in response['body']


@pytest.fixture(scope='module')
def template():
    return Template.from_stack(TracerStack(App(analytics_reporting=False))).to_json()


def test_tracer_template_has_no_cycle_and_no_model_permissions(template):
    review = inspect_template(template)
    assert review['model_enabled'] is False
    assert len(json.dumps(template).encode()) < 51200
    compile(handler_source(), 'index.py', 'exec')
    stage = next(r['Properties'] for r in template['Resources'].values()
                 if r['Type'] == 'AWS::ApiGatewayV2::Stage')
    assert stage['DefaultRouteSettings']['ThrottlingBurstLimit'] == 2
    assert stage['RouteSettings']['GET /assets/{proxy+}']['ThrottlingBurstLimit'] >= 8
    stage_resource = next(r for r in template['Resources'].values()
                          if r['Type'] == 'AWS::ApiGatewayV2::Stage')
    route_ids = {name for name, r in template['Resources'].items()
                 if r['Type'] == 'AWS::ApiGatewayV2::Route'}
    assert route_ids <= set(stage_resource['DependsOn'])


def test_circular_auth_api_dependency_is_rejected():
    with pytest.raises(ValueError, match='Circular'):
        assert_no_cycles({'Api': {'Properties': {'Ref': 'Client'}},
                          'Client': {'Properties': {'Ref': 'Api'}}})


def test_tracer_rejects_wildcard_trust_and_missing_stage_dependency(template):
    altered = copy.deepcopy(template)
    role = next(r['Properties'] for r in altered['Resources'].values() if r['Type'] == 'AWS::IAM::Role')
    role['AssumeRolePolicyDocument']['Statement'][0]['Principal'] = '*'
    with pytest.raises(ValueError, match='Only Lambda'):
        inspect_template(altered)
    altered = copy.deepcopy(template)
    stage = next(r for r in altered['Resources'].values() if r['Type'] == 'AWS::ApiGatewayV2::Stage')
    stage['DependsOn'] = []
    with pytest.raises(ValueError, match='all routes'):
        inspect_template(altered)
