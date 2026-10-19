"""HTTP/auth rejection and real execution-role reads; browser OAuth remains a human check."""

import json
import re
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config


def main():
    outputs = json.loads(Path('results/tracer/deployment.json').read_bytes())['outputs']
    origin = outputs['PublicUrl'].rstrip('/')
    if not re.fullmatch(r'https://[a-z0-9]+\.execute-api\.us-east-1\.amazonaws\.com', origin):
        raise ValueError('Unexpected deployed API endpoint')
    checks = []
    config = None
    cases = [('/', 200), ('/assets/app.js', 200), ('/assets/auth.mjs', 200),
             ('/assets/style.css', 200), ('/auth/callback', 200), ('/config.json', 200),
             ('/assets/%2e%2e/configs/funding.json', 404), ('/.env', 404), ('/v1/status', 401)]
    for path, expected in cases:
        try:
            response = urllib.request.urlopen(origin + path, timeout=10)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            raw = response.read(1_048_577)
            assert response.code == expected, (path, response.code)
            assert len(raw) <= 1_048_576
            if expected == 200:
                assert response.headers['cache-control'] == 'no-store'
                assert response.headers['x-content-type-options'] == 'nosniff'
            if path == '/config.json':
                config = json.loads(raw)
                assert set(config) == {'issuer', 'client_id', 'auth_domain', 'callback_url', 'logout_url'}
            checks.append({'path': path, 'status': response.code, 'expected': expected})
    session = boto3.Session(profile_name='kg-dev', region_name='us-east-1')
    users = session.client('cognito-idp').list_users(UserPoolId=outputs['UserPoolId'], Limit=10)['Users']
    user = next(item for item in users if item['Username'] == 'demo-a')
    sub = next(item['Value'] for item in user['Attributes'] if item['Name'] == 'sub')
    claims = {'iss': config['issuer'], 'client_id': config['client_id'], 'sub': sub,
              'token_use': 'access', 'scope': 'openid kg/read',
              'exp': str(int(datetime.now(UTC).timestamp()) + 300)}
    client = session.client('lambda', config=Config(retries={'total_max_attempts': 1}, read_timeout=10))
    for label, changed, expected in [('active_member', {}, 200), ('id_token_rejected', {'token_use': 'id'}, 401),
                                      ('scope_missing', {'scope': 'openid'}, 403),
                                      ('nonmember_rejected', {'sub': str(uuid.uuid4())}, 403)]:
        event = {'rawPath': '/v1/status', 'requestContext': {'http': {'method': 'GET'},
                 'authorizer': {'jwt': {'claims': claims | changed}}}}
        result = client.invoke(FunctionName='kg-r72-query', InvocationType='RequestResponse',
                                Payload=json.dumps(event).encode())
        with result['Payload'] as stream:
            payload = json.loads(stream.read())
        assert not result.get('FunctionError'), label
        assert payload['statusCode'] == expected, (label, payload['statusCode'])
        checks.append({'runtime_check': label, 'status': payload['statusCode'], 'expected': expected})
    result = {'status': 'PASS', 'checked_at': datetime.now(UTC).isoformat(), 'public_url': origin + '/',
              'checks': checks, 'inference_calls': 0, 'runtime_membership_read_verified': True,
              'jwt_browser_login_verified': False,
              'runtime_probe_auth': 'Admin SDK invocation with synthetic claims; not an OAuth/JWT end-to-end test'}
    Path('results/tracer/cloud_smoke.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        result = {'status': 'INCOMPLETE', 'category': type(error).__name__,
                  'detail': str(error) if isinstance(error, AssertionError) else 'Check cloud configuration',
                  'inference_calls': 0}
        Path('results/tracer/cloud_smoke_failure.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(result, indent=2))
        raise SystemExit(1) from None
