from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.cost import PriceSheet, TokenUsage, usd_to_micro


@pytest.fixture
def prices():
    return PriceSheet.model_validate_json(
        Path("configs/prices/aws-us-east-1-2026-10-16.json").read_bytes()
    )


@pytest.mark.parametrize("value,expected", [("1", 1_000_000), ("0.10", 100_000),
                                           ("0.0000001", 1), ("0", 0)])
def test_decimal_money_rounds_up_once(value, expected):
    assert usd_to_micro(value) == expected


@pytest.mark.parametrize("value", [0.1, True, "NaN", "Infinity", "-1", "invalid", "1e100"])
def test_money_rejects_float_nonfinite_and_invalid(value):
    with pytest.raises(ValueError):
        usd_to_micro(value)


def test_geo_route_prices_and_exact_token_arithmetic(prices):
    rate = prices.rates[0]
    assert (rate.input_micro_usd_per_unit, rate.output_micro_usd_per_unit) == (1_100_000, 5_500_000)
    assert rate.quote(TokenUsage(input_tokens=2000, output_tokens=500)) == 4950
    assert rate.quote(TokenUsage(input_tokens=1, output_tokens=0)) == 2
    assert prices.rates[1].quote(TokenUsage(input_tokens=8192, output_tokens=0)) == 164
    with pytest.raises(ValueError):
        prices.rates[1].quote(TokenUsage(input_tokens=1, output_tokens=1))


def test_missing_stale_future_or_fixture_price_denies_live(prices):
    rate = prices.rates[0]
    kwargs = dict(model_id=rate.model_id, invocation_id=rate.invocation_id,
                  region=rate.source_region, now=prices.checked_at, live=True)
    assert prices.rate_for(**kwargs) == rate
    for changes in ({"now": prices.expires_at}, {"now": prices.checked_at - timedelta(seconds=1)},
                    {"invocation_id": "global." + rate.model_id}, {"region": "us-west-2"}):
        with pytest.raises(ValueError):
            prices.rate_for(**(kwargs | changes))
    fixture = PriceSheet(**(prices.model_dump() | {"purpose": "FIXTURE"}))
    with pytest.raises(ValueError):
        fixture.rate_for(**kwargs)
    with pytest.raises(ValueError):
        prices.rate_for(**(kwargs | {"now": datetime(2026, 10, 16)}))


def test_usage_and_price_types_are_strict(prices):
    with pytest.raises(ValidationError):
        TokenUsage(input_tokens=True, output_tokens=1)
    with pytest.raises(ValidationError):
        PriceSheet(**(prices.model_dump() | {"expires_at": prices.checked_at + timedelta(days=8)}))
    with pytest.raises(ValidationError):
        PriceSheet(**(prices.model_dump() | {"checked_at": datetime(2026, 10, 16, tzinfo=UTC)}))
