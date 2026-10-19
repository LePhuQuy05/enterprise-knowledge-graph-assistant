"""Create two invite-only accounts, suppress all delivery, keep temporary secrets local."""

import json
import os
import secrets
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from botocore.config import Config

from scripts.foundation import _session


def private_directory(path):
    path.mkdir(parents=True, exist_ok=True)
    if os.name == 'nt':
        sid = subprocess.run([
            'powershell', '-NoProfile', '-NonInteractive', '-Command',
            '[System.Security.Principal.WindowsIdentity]::GetCurrent().User.Value',
        ], capture_output=True, text=True, check=True).stdout.strip()
        if not sid.startswith('S-1-'):
            raise ValueError('Cannot establish current Windows identity for private file ACL')
        subprocess.run(['icacls', str(path.resolve()), '/inheritance:r', '/grant:r',
                        f'*{sid}:(OI)(CI)(F)'], capture_output=True, check=True)
    else:
        path.chmod(0o700)


def main():
    path = Path('results/private/reviewer_credentials.json')
    private_directory(path.parent)
    records = [{'username': name, 'temporary_password': secrets.token_urlsafe(24) + 'Aa7!',
                'creation_status': 'NOT_ATTEMPTED', 'membership_created': False}
               for name in ['demo-a', 'demo-b']]
    # Exclusive create: never reset existing users' passwords or overwrite prior secrets.
    with path.open('x', encoding='utf-8') as file:
        json.dump({'secret': True, 'accounts': records}, file, indent=2)
    if os.name != 'nt':
        path.chmod(0o600)
    session = _session('kg-dev')
    pool_id = json.loads(Path('results/tracer/deployment.json').read_bytes())['outputs']['UserPoolId']
    client = session.client('cognito-idp', config=Config(retries={'total_max_attempts': 1}))
    state = session.client('dynamodb', config=Config(retries={'total_max_attempts': 1}))
    try:
        for item in records:
            result = client.admin_create_user(
                UserPoolId=pool_id, Username=item['username'],
                TemporaryPassword=item['temporary_password'], MessageAction='SUPPRESS',
            )
            sub = next(attribute['Value'] for attribute in result['User']['Attributes']
                       if attribute['Name'] == 'sub')
            item['creation_status'] = result['User']['UserStatus']
            item['user_sub'] = sub
            stamp = datetime.now(UTC).isoformat()
            state.put_item(TableName='kg-r72-metadata', ConditionExpression='attribute_not_exists(PK)', Item={
                'PK': {'S': 'USER#' + sub}, 'SK': {'S': 'MEMBERSHIP'},
                'record_type': {'S': 'MEMBERSHIP'}, 'schema_version': {'N': '1'},
                'tenant_id': {'S': 'demo'}, 'status': {'S': 'ACTIVE'},
                'roles': {'SS': ['reviewer']}, 'created_at': {'S': stamp}, 'updated_at': {'S': stamp},
            })
            item['membership_created'] = True
    finally:
        path.write_text(json.dumps({'secret': True, 'accounts': records}, indent=2) + '\n', encoding='utf-8')
    summary = {'status': 'DEMO_USERS_CREATED', 'users': [item['username'] for item in records],
               'message_delivery': 'SUPPRESSED', 'passwords_printed': False,
               'private_file': str(path), 'browser_login_verified': False}
    Path('results/tracer/demo_users.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'INCOMPLETE', 'category': type(error).__name__,
                          'passwords_printed': False}))
        raise SystemExit(1) from None
