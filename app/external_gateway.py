"""DeepSeek/Voyage boundary: approved routes, conservative costing, one dispatch.

Settlement is a published-rate upper bound on reported tokens, NOT an invoice:
DeepSeek peak/cache-miss rates and Voyage paid rates even if allowances apply.
"""

import math
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, model_validator

from app.contracts import Nonempty, Record, Sha256, canonical_json, sha256_bytes
from app.cost import PriceSheet, TokenRate, TokenUsage
from app.external_http import ExternalHttpError, OfficialHttpsTransport
from app.ledger import DynamoLedgerStore, ExternalOperationSpec, Ledger
from app.model_gateway import CallContext, GatewayError, ModelResult, UncertainModelCall

DEEPSEEK = "deepseek-flash"
VOYAGE = "voyage-4-lite"
COST_BASIS = "PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND"


class ExternalRoutes(Record):
    version: Literal["external-direct-v1"] = "external-direct-v1"
    source_region: Literal["external-default"] = "external-default"
    generation_model_id: Literal["deepseek-flash"] = DEEPSEEK
    generation_invocation_id: Literal["deepseek-flash"] = DEEPSEEK
    generation_context_ceiling: Literal[1048576] = 1048576
    generation_endpoint: Literal["https://api.deepseek.com/chat/completions"] = (
        "https://api.deepseek.com/chat/completions"
    )
    generation_protocol: Literal["CHAT_COMPLETIONS_TEXT_JSON_NONTHINKING_V1"] = (
        "CHAT_COMPLETIONS_TEXT_JSON_NONTHINKING_V1"
    )
    account_model_metadata_sha256: Sha256
    embedding_model_id: Literal["voyage-4-lite"] = VOYAGE
    embedding_invocation_id: Literal["voyage-4-lite"] = VOYAGE
    embedding_endpoint: Literal["https://api.voyageai.com/v1/embeddings"] = (
        "https://api.voyageai.com/v1/embeddings"
    )
    embedding_context_ceiling: Literal[32000] = 32000
    embedding_dimension: Literal[1024] = 1024
    embedding_normalized: Literal[True] = True
    live_inference_enabled: bool = False
    live_inference_gate: Nonempty = "ACCOUNT_RATES_FUNDING_AND_TESTS_REQUIRED"


class ExternalTokenRate(TokenRate):
    provider: Literal["DEEPSEEK", "VOYAGE"]
    source_region: Literal["external-default"] = "external-default"
    billing_source: Literal["DIRECT_API"] = "DIRECT_API"

    @model_validator(mode="after")
    def approved_identity(self) -> Self:
        model, kind = (DEEPSEEK, "GENERATION") if self.provider == "DEEPSEEK" else (VOYAGE, "EMBEDDING")
        if (self.model_id, self.invocation_id, self.kind) != (model, model, kind):
            raise ValueError("External rate provider/model/kind mismatch")
        return self


class ExternalPriceSheet(PriceSheet):
    cost_basis: Literal["PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND"] = COST_BASIS
    rates: Annotated[tuple[ExternalTokenRate, ...], Field(min_length=2, max_length=2)]

    @model_validator(mode="after")
    def both_providers(self) -> Self:
        if {rate.provider for rate in self.rates} != {"DEEPSEEK", "VOYAGE"}:
            raise ValueError("External sheet requires both provider rates")
        return self


@dataclass(frozen=True)
class ExternalModelResult(ModelResult):
    cost_basis: str = COST_BASIS
    provider: str = ""
    observed_model: str = ""
    input_role: str | None = None


class ExternalCallUncertain(UncertainModelCall):
    def __init__(self, operation_id: str, category: str):
        self.category = category
        super().__init__(operation_id)


def _integer(value, *, positive=False) -> int:
    if type(value) is not int or value < (1 if positive else 0):
        raise GatewayError("Invalid provider usage integer")
    return value


def _deepseek_usage(payload: dict) -> TokenUsage:
    raw = payload["usage"]
    usage = TokenUsage(input_tokens=_integer(raw["prompt_tokens"], positive=True),
                       output_tokens=_integer(raw["completion_tokens"]))
    if _integer(raw["total_tokens"]) != usage.input_tokens + usage.output_tokens:
        raise GatewayError("Inconsistent provider total tokens")
    has_hit, has_miss = "prompt_cache_hit_tokens" in raw, "prompt_cache_miss_tokens" in raw
    if has_hit != has_miss:
        raise GatewayError("Incomplete provider cache breakdown")
    if has_hit:
        hit, miss = _integer(raw["prompt_cache_hit_tokens"]), _integer(raw["prompt_cache_miss_tokens"])
        if hit + miss != usage.input_tokens:
            raise GatewayError("Inconsistent provider cache breakdown")
        details = raw.get("prompt_tokens_details", {})
        if "cached_tokens" in details and _integer(details["cached_tokens"]) != hit:
            raise GatewayError("Conflicting provider cache details")
    reasoning = raw.get("completion_tokens_details", {}).get("reasoning_tokens", 0)
    if _integer(reasoning) > usage.output_tokens:
        raise GatewayError("Inconsistent reasoning token usage")
    # All input is charged at cache-miss upper bound, including valid cache hits.
    return usage


class ExternalModelGateway:
    def __init__(self, ledger: Ledger, provider, routes: ExternalRoutes, prices: ExternalPriceSheet):
        self.ledger, self.provider, self.routes, self.prices = ledger, provider, routes, prices
        if provider.live:
            if not routes.live_inference_enabled or type(provider) is not OfficialHttpsTransport:
                raise GatewayError("External live inference is disabled or transport is unapproved")
            if not isinstance(ledger.store, DynamoLedgerStore) or (
                ledger.store.client.meta.endpoint_url != "https://dynamodb.us-east-1.amazonaws.com"
                or ledger.store.table_name != "kg-r72-query-state"
            ):
                raise GatewayError("External live calls require the managed shared ledger")
        elif prices.purpose != "FIXTURE":
            raise GatewayError("Fixture transport requires fixture prices")

    def generate(
        self, context: CallContext, *, system: str, prompt: str, max_output_tokens: int = 800,
    ) -> ExternalModelResult:
        caps = {"SMOKE": 64, "GENERATE": 800, "EXTRACT": 2048, "EVALUATE": 800, "REWRITE": 256}
        if context.stage not in caps or type(max_output_tokens) is not int or not (
            1 <= max_output_tokens <= caps[context.stage]
        ) or not system.strip() or not prompt.strip():
            raise GatewayError("Invalid generation stage, content or output cap")
        body = canonical_json({
            "model": DEEPSEEK, "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": max_output_tokens, "stream": False,
            "response_format": {"type": "json_object"},
        })
        # No proxy tokenizer is described as exact. Reserve the accepted context ceiling.
        return self._call(context, "DEEPSEEK", body, self.routes.generation_context_ceiling,
                          max_output_tokens, None)

    def embed(
        self, context: CallContext, *, text: str, input_role: Literal["document", "query"] = "document",
    ) -> ExternalModelResult:
        if context.stage != "EMBED" or input_role not in {"document", "query"} or (
            not text.strip() or len(text) > 50_000
        ):
            raise GatewayError("Invalid embedding stage, text or input role")
        body = canonical_json({
            "input": [text], "model": VOYAGE, "input_type": input_role,
            "truncation": False, "output_dimension": 1024, "output_dtype": "float",
        })
        return self._call(context, "VOYAGE", body, self.routes.embedding_context_ceiling, 0, input_role)

    def _call(self, context, provider, body, input_bound, output_bound, input_role):
        if len(body) > 100_000:
            raise GatewayError("External input payload exceeds bounded text request size")
        model, path = (DEEPSEEK, "/chat/completions") if provider == "DEEPSEEK" else (
            VOYAGE, "/v1/embeddings",
        )
        rate = self.prices.rate_for(
            model, model, "external-default", now=self.ledger._now(), live=self.provider.live,
        )
        bound = rate.quote(TokenUsage(input_tokens=input_bound, output_tokens=output_bound))
        spec = ExternalOperationSpec(
            **context.model_dump(), provider=provider, input_sha256=sha256_bytes(body),
            model_id=model, invocation_id=model, source_region="external-default",
            price_sheet_version=self.prices.version, max_cost_micro_usd=bound,
        )
        self.ledger.reserve(spec)
        self.ledger.begin_call(spec.operation_id)
        try:
            payload = self.provider.request(provider, "POST", path, body)
            usage = _deepseek_usage(payload) if provider == "DEEPSEEK" else TokenUsage(
                input_tokens=_integer(payload["usage"]["total_tokens"], positive=True), output_tokens=0,
            )
            budget_cost = rate.quote(usage)
            self.ledger.settle(spec.operation_id, budget_cost, usage)
        except Exception as exc:
            try:
                self.ledger.mark_unknown(spec.operation_id)
            except Exception:
                pass  # Never refund or retry an ambiguous provider/settlement result.
            category = exc.category if isinstance(exc, ExternalHttpError) else "USAGE_OR_SETTLEMENT_UNKNOWN"
            raise ExternalCallUncertain(spec.operation_id, category) from None
        if usage.input_tokens > input_bound or usage.output_tokens > output_bound:
            self.ledger.halt_operation_budgets(spec.operation_id)
            raise GatewayError("External token bound exceeded; usage settled and budgets halted")
        try:
            if payload.get("model") != model:
                raise GatewayError("Unexpected observed external model")
            if provider == "VOYAGE":
                data = payload["data"]
                if type(data) is not list or len(data) != 1 or type(data[0].get("index")) is not int or (
                    data[0]["index"] != 0
                ):
                    raise GatewayError("Invalid embedding count/index mapping")
                values = data[0]["embedding"]
                if type(values) is not list or len(values) != 1024 or any(
                    type(x) not in (float, int) or not math.isfinite(x) for x in values
                ):
                    raise GatewayError("Invalid embedding shape or finite values")
                vector = np.asarray(values, dtype=np.float32)
                norm = float(np.linalg.norm(vector))
                if not np.isfinite(vector).all() or not math.isfinite(norm) or norm <= 0:
                    raise GatewayError("Embedding has invalid norm")
                # Pin local float32 L2 normalization as part of the provider recipe.
                vector = vector / norm
                return ExternalModelResult(
                    spec.operation_id, usage, budget_cost, embedding=tuple(float(x) for x in vector),
                    provider=provider, observed_model=model, input_role=input_role,
                )
            choices = payload["choices"]
            if type(choices) is not list or len(choices) != 1 or choices[0].get("index") != 0:
                raise GatewayError("Expected one text completion")
            message = choices[0]["message"]
            if message.get("role") != "assistant" or message.get("tool_calls") or message.get("reasoning_content"):
                raise GatewayError("Unsupported reasoning/tool output")
            if payload["usage"].get("completion_tokens_details", {}).get("reasoning_tokens", 0):
                raise GatewayError("Unexpected reasoning usage in non-thinking request")
            text, finish = message["content"], choices[0]["finish_reason"]
            if type(text) is not str or not text.strip() or finish not in {"stop", "length"}:
                raise GatewayError("Invalid text output or finish reason")
            return ExternalModelResult(
                spec.operation_id, usage, budget_cost, text=text,
                stop_reason="end_turn" if finish == "stop" else "max_tokens",
                provider=provider, observed_model=model,
            )
        except (KeyError, TypeError, ValueError, AttributeError):
            raise GatewayError("External model output invalid; reported usage already settled") from None
