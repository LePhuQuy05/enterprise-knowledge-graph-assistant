import json
from functools import partial

import pytest

from app.cli import main
from app.model_smoke import run_smoke


def test_smoke_default_is_offline_forecast(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Forecast must not create an AWS session')
    monkeypatch.setattr('app.model_smoke.boto3.Session', forbidden)
    result = run_smoke()
    assert result['status'] == 'FORECAST_ONLY'
    assert result['maximum_two_call_reservations_micro_usd'] == 7275
    assert result['inference_calls'] == 0
    embedding = run_smoke(embedding_only=True)
    assert embedding['maximum_two_call_reservations_micro_usd'] == 164
    assert embedding['max_inference_calls'] == 1


def test_unconfirmed_credit_and_zero_personal_allowance_block_before_network(monkeypatch, tmp_path):
    def forbidden(*args, **kwargs):
        pytest.fail('No AWS client before funding gate')
    monkeypatch.setattr('app.model_smoke.boto3.Session', forbidden)
    funding = tmp_path / 'funding.json'
    funding.write_text(json.dumps({'anthropic_credit_coverage': 'UNCONFIRMED',
                                   'personal_budget_for_initial_smoke_micro_usd': 0}))
    with pytest.raises(ValueError, match='funding is unconfirmed'):
        run_smoke(execute=True, funding_path=funding)


def test_model_smoke_cli_exposes_forecast_and_rejects_unfunded_execution(capsys, monkeypatch, tmp_path):
    funding = tmp_path / 'funding.json'
    funding.write_text('{}')
    monkeypatch.setattr('app.model_smoke.run_smoke', partial(run_smoke, funding_path=funding))
    assert main(['model-smoke']) == 0
    assert json.loads(capsys.readouterr().out)['inference_calls'] == 0
    assert main(['model-smoke', '--execute']) == 1
    assert json.loads(capsys.readouterr().out)['error']['code'] == 'ValueError'
