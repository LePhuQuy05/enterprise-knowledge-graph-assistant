import json
from datetime import UTC, datetime, timedelta
from functools import partial
from pathlib import Path

import pytest

from app.cli import main
from app.cost import PriceSheet
from app.model_smoke import run_smoke


@pytest.fixture
def smoke_clock(monkeypatch):
    """Historical price fixtures need an explicit clock, never extended validity."""
    def set_clock(instant):
        class ForecastDateTime(datetime):
            @classmethod
            def now(cls, tz=None):
                return instant.astimezone(tz) if tz is not None else instant.replace(tzinfo=None)

        monkeypatch.setattr('app.model_smoke.datetime', ForecastDateTime)

    set_clock(datetime(2026, 10, 17, 4, tzinfo=UTC))
    return set_clock


def test_smoke_default_is_offline_forecast(monkeypatch, smoke_clock):
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


def test_unconfirmed_credit_and_zero_personal_allowance_block_before_network(monkeypatch, tmp_path,
                                                                           smoke_clock):
    def forbidden(*args, **kwargs):
        pytest.fail('No AWS client before funding gate')
    monkeypatch.setattr('app.model_smoke.boto3.Session', forbidden)
    funding = tmp_path / 'funding.json'
    funding.write_text(json.dumps({'anthropic_credit_coverage': 'UNCONFIRMED',
                                   'personal_budget_for_initial_smoke_micro_usd': 0}))
    with pytest.raises(ValueError, match='funding is unconfirmed'):
        run_smoke(execute=True, funding_path=funding)


def test_model_smoke_cli_exposes_forecast_and_rejects_unfunded_execution(capsys, monkeypatch, tmp_path,
                                                                      smoke_clock):
    funding = tmp_path / 'funding.json'
    funding.write_text('{}')
    monkeypatch.setattr('app.model_smoke.run_smoke', partial(run_smoke, funding_path=funding))
    assert main(['model-smoke']) == 0
    assert json.loads(capsys.readouterr().out)['inference_calls'] == 0
    assert main(['model-smoke', '--execute']) == 1
    assert json.loads(capsys.readouterr().out)['error']['code'] == 'ValueError'


@pytest.mark.parametrize('execute', [False, True])
@pytest.mark.parametrize('boundary', ['before_checked', 'at_expiry'])
def test_invalid_price_clock_still_blocks_forecast_and_execution_before_network(
    monkeypatch, tmp_path, smoke_clock, execute, boundary,
):
    prices = PriceSheet.model_validate_json(
        Path('configs/prices/aws-us-east-1-2026-10-16.json').read_bytes())
    instant = (prices.checked_at - timedelta(microseconds=1)
               if boundary == 'before_checked' else prices.expires_at)
    smoke_clock(instant)

    def forbidden(*args, **kwargs):
        pytest.fail('Invalid prices must block before any AWS session')

    monkeypatch.setattr('app.model_smoke.boto3.Session', forbidden)
    # A missing funding file additionally proves stale prices fail before the
    # execution branch reads funding or can initialize a provider/ledger.
    with pytest.raises(ValueError, match='Price sheet is stale or not yet valid'):
        run_smoke(execute=execute, funding_path=tmp_path / 'must-not-be-read.json')
