"""Integer token-cost accounting with explicit route and price-sheet validity."""

from datetime import datetime, timedelta
from decimal import ROUND_CEILING, Decimal, InvalidOperation
from typing import Annotated, Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from app.contracts import Nonempty, Record, Sha256

MICRO_USD = 1_000_000


def usd_to_micro(value: str) -> int:
    if not isinstance(value, str) or len(value) > 64:
        raise ValueError("Money must be a bounded decimal string")
    try:
        amount = Decimal(value)
        if not amount.is_finite() or not 0 <= amount <= Decimal("1000000"):
            raise ValueError("Money is negative, nonfinite or outside supported range")
        return int((amount * MICRO_USD).to_integral_value(rounding=ROUND_CEILING))
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal money") from exc


class TokenUsage(Record):
    input_tokens: Annotated[int, Field(ge=0)]
    output_tokens: Annotated[int, Field(ge=0)]


class PriceSource(Record):
    url: Nonempty
    sha256: Sha256
    published_at: AwareDatetime | None


class TokenRate(Record):
    model_id: Nonempty
    invocation_id: Nonempty
    source_region: Literal["us-east-1"]
    kind: Literal["GENERATION", "EMBEDDING"]
    performance_tier: Literal["standard"]
    tokens_per_unit: Annotated[int, Field(gt=0)]
    input_micro_usd_per_unit: Annotated[int, Field(gt=0)]
    output_micro_usd_per_unit: Annotated[int, Field(ge=0)]
    input_rate_code: Nonempty
    output_rate_code: str | None

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.kind == "EMBEDDING" and self.output_micro_usd_per_unit != 0:
            raise ValueError("Embedding rate must not charge output tokens")
        if self.kind == "GENERATION" and (
            self.output_micro_usd_per_unit == 0 or not self.output_rate_code
        ):
            raise ValueError("Generation output rate is required")
        return self

    def quote(self, usage: TokenUsage) -> int:
        if self.kind == "EMBEDDING" and usage.output_tokens:
            raise ValueError("Embedding output-token usage is unsupported")
        numerator = (
            usage.input_tokens * self.input_micro_usd_per_unit
            + usage.output_tokens * self.output_micro_usd_per_unit
        )
        return (numerator + self.tokens_per_unit - 1) // self.tokens_per_unit


class PriceSheet(Record):
    version: Nonempty
    checked_at: AwareDatetime
    expires_at: AwareDatetime
    currency: Literal["USD"]
    purpose: Literal["LIVE", "FIXTURE"]
    sources: Annotated[tuple[PriceSource, ...], Field(min_length=1)]
    rates: Annotated[tuple[TokenRate, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.checked_at.utcoffset() != timedelta(0) or self.expires_at.utcoffset() != timedelta(0):
            raise ValueError("Price timestamps must be UTC")
        if not timedelta(0) < self.expires_at - self.checked_at <= timedelta(days=7):
            raise ValueError("Price validity must be positive and at most seven days")
        keys = [(r.model_id, r.invocation_id, r.source_region) for r in self.rates]
        if len(set(keys)) != len(keys):
            raise ValueError("Duplicate price routes")
        return self

    def rate_for(
        self, model_id: str, invocation_id: str, region: str, *, now: datetime, live: bool,
    ) -> TokenRate:
        if now.utcoffset() != timedelta(0):
            raise ValueError("Price check clock must be UTC-aware")
        if live and self.purpose != "LIVE":
            raise ValueError("Fixture prices cannot authorize live calls")
        if not self.checked_at <= now < self.expires_at:
            raise ValueError("Price sheet is stale or not yet valid")
        for rate in self.rates:
            if (rate.model_id, rate.invocation_id, rate.source_region) == (
                model_id, invocation_id, region,
            ):
                return rate
        raise ValueError("Approved route has no verified price")
