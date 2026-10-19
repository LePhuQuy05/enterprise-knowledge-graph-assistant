import json
from types import SimpleNamespace

import pytest

from scripts import create_demo_users


def test_bootstrap_suppresses_messages_preserves_secrets_and_never_resets_users(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    folder = tmp_path / 'results/tracer'
    folder.mkdir(parents=True)
    (folder / 'deployment.json').write_text(json.dumps({'outputs': {'UserPoolId': 'fixture-pool'}}))
    created, memberships = [], []
    class Cognito:
        def admin_create_user(self, **kwargs):
            assert kwargs['MessageAction'] == 'SUPPRESS'
            created.append(kwargs)
            return {'User': {'UserStatus': 'FORCE_CHANGE_PASSWORD', 'Attributes': [
                {'Name': 'sub', 'Value': f'fixture-sub-{len(created)}'},
            ]}}
    class Dynamo:
        def put_item(self, **kwargs):
            assert kwargs['ConditionExpression'] == 'attribute_not_exists(PK)'
            memberships.append(kwargs['Item'])
    session = SimpleNamespace(client=lambda service, **kwargs: Cognito() if service == 'cognito-idp' else Dynamo())
    monkeypatch.setattr(create_demo_users, '_session', lambda profile: session)
    monkeypatch.setattr(create_demo_users, 'private_directory', lambda path: path.mkdir(parents=True, exist_ok=True))
    create_demo_users.main()
    output = capsys.readouterr().out
    assert len(created) == len(memberships) == 2
    for item in created:
        assert item['TemporaryPassword'] not in output
    saved = json.loads((tmp_path / 'results/private/reviewer_credentials.json').read_text())
    assert saved['secret'] is True and len(saved['accounts']) == 2
    with pytest.raises(FileExistsError):
        create_demo_users.main()
    assert len(created) == 2
