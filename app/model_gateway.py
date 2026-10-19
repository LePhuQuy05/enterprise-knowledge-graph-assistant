"""The only paid-model boundary: pinned routes, a shared reservation, one dispatch.

Input counting is mandatory for Haiku. Unsupported CountTokens fails closed, rather
than silently replacing provider tokens with a local tokenizer estimate.
"""

import json
import math
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Protocol

import numpy as np
from botocore.config import Config
from pydantic import Field

from app.contracts import Nonempty, Record, canonical_json, sha256_bytes
from app.cost import PriceSheet, TokenUsage
from app.ledger import DynamoLedgerStore, Identifier, Ledger, OperationSpec

HAIKU = "anthropic.claude-haiku-4-5-20251001-v1:0"
HAIKU_ROUTE = "us." + HAIKU
TITAN = "amazon.titan-embed-text-v2:0"
MAX_BODY_BYTES = 1_048_576


class ModelRoutes(Record):
    version: Literal["bedrock-us-east-1-v2"]
    source_region: Literal["us-east-1"]
    generation_model_id: Literal["anthropic.claude-haiku-4-5-20251001-v1:0"]
    generation_invocation_id: Literal["us.anthropic.claude-haiku-4-5-20251001-v1:0"]
    generation_destination_regions: tuple[Literal["us-east-1", "us-east-2", "us-west-2"], ...]
    embedding_model_id: Literal["amazon.titan-embed-text-v2:0"]
    embedding_invocation_id: Literal["amazon.titan-embed-text-v2:0"]
    embedding_dimension: Literal[1024]
    embedding_normalized: Literal[True]
    performance_tier: Literal["standard"]
    prompt_caching_enabled: Literal[False]
    token_count_model_id: Literal["anthropic.claude-haiku-4-5-20251001-v1:0"]
    token_count_api: Literal["BEDROCK_RUNTIME_COUNT_TOKENS"]
    live_inference_enabled: bool
    live_inference_gate: Nonempty


class CallContext(Record):
    project_id: Literal["kg-r72"] = "kg-r72"
    tenant_id: Identifier
    run_id: Identifier
    request_id: Identifier
    stage: Literal["EMBED", "EXTRACT", "GENERATE", "EVALUATE", "SMOKE", "REWRITE"]
    attempt: Annotated[int, Field(ge=1, le=10)] = 1


class GatewayError(ValueError):
    """Safe category; never include prompt, credentials or raw SDK error strings."""


class UncertainModelCall(GatewayError):
    def __init__(self, operation_id: str):
        self.operation_id = operation_id
        super().__init__("Dispatched operation needs reconciliation; do not retry this operation")


def safe_failure_code(exc: Exception) -> str:
    """Persist a bounded diagnostic category, never raw provider/body/credential text."""
    detail = getattr(exc, "category", None)
    allowed = {"AUTHENTICATION", "BALANCE", "PERMISSION", "RATE_LIMIT", "TIMEOUT",
               "TRANSPORT_UNKNOWN", "INVALID_JSON", "RESPONSE_TOO_LARGE", "REDIRECT_BLOCKED",
               "PROVIDER_ERROR", "UNAPPROVED_ROUTE", "USAGE_OR_SETTLEMENT_UNKNOWN"}
    return type(exc).__name__ + (":" + detail if isinstance(detail, str) and detail in allowed else "")


@dataclass(frozen=True)
class ModelResult:
    operation_id: str
    usage: TokenUsage
    actual_micro_usd: int
    text: str | None = None
    embedding: tuple[float, ...] | None = None
    stop_reason: str | None = None


class Provider(Protocol):
    live: bool

    def count_input(self, route: str, body: bytes) -> int: ...

    def invoke(self, route: str, body: bytes) -> dict: ...


def _positive_integer(value: Any) -> int:
    if type(value) is not int or value <= 0:
        raise GatewayError("Provider token count must be a positive integer")
    return value


class BedrockProvider:
    live = True

    def __init__(self, session: Any):
        self.client = session.client(
            "bedrock-runtime", region_name="us-east-1",
            config=Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=12),
        )
        if self.client.meta.endpoint_url != "https://bedrock-runtime.us-east-1.amazonaws.com":
            raise GatewayError("Only the approved Bedrock runtime endpoint is allowed")

    def count_input(self, route: str, body: bytes) -> int:
        if route != HAIKU_ROUTE:
            raise GatewayError("Token counter only supports the approved Haiku route")
        try:
            # CountTokens accepts the base model ID in this account; the profile ID
            # returned ValidationException. Inference still uses the approved US profile.
            response = self.client.count_tokens(modelId=HAIKU, input={"invokeModel": {"body": body}})
            return _positive_integer(response["inputTokens"])
        except Exception as exc:
            raise GatewayError("Provider CountTokens failed; no inference permitted") from exc

    def invoke(self, route: str, body: bytes) -> dict:
        response = self.client.invoke_model(
            modelId=route, body=body, accept="application/json", contentType="application/json",
            performanceConfigLatency="standard",
        )
        stream = response["body"]
        try:
            raw = stream.read(MAX_BODY_BYTES + 1)
        finally:
            stream.close()
        if len(raw) > MAX_BODY_BYTES:
            raise GatewayError("Provider response exceeds supported size")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise GatewayError("Provider response must be an object")
        return payload


class FixtureProvider:
    """Deterministic local provider, never an alternative route for a live call."""

    live = False

    def count_input(self, route: str, body: bytes) -> int:
        return max(1, len(body) // 4)  # Toy fixture count, deliberately not a billing estimate.

    def invoke(self, route: str, body: bytes) -> dict:
        if route == TITAN:
            return {"inputTextTokenCount": 8, "embedding": [1.0] + [0.0] * 1023}
        return {"usage": {"input_tokens": self.count_input(route, body), "output_tokens": 3},
                "content": [{"type": "text", "text": "FIXTURE_ONLY_NO_MODEL"}],
                "stop_reason": "end_turn"}


class ModelGateway:
    def __init__(self, ledger: Ledger, provider: Provider, routes: ModelRoutes, prices: PriceSheet):
        self.ledger, self.provider, self.routes, self.prices = ledger, provider, routes, prices
        if set(routes.generation_destination_regions) != {"us-east-1", "us-east-2", "us-west-2"}:
            raise GatewayError("Generation destinations differ from the verified profile")
        if provider.live:
            if not routes.live_inference_enabled or not isinstance(provider, BedrockProvider):
                raise GatewayError("Live inference is disabled or provider is not approved")
            if not isinstance(ledger.store, DynamoLedgerStore) or (
                ledger.store.client.meta.endpoint_url != "https://dynamodb.us-east-1.amazonaws.com"
                or ledger.store.table_name != "kg-r72-query-state"
            ):
                raise GatewayError("Live calls require the managed shared DynamoDB ledger")
        elif prices.purpose != "FIXTURE":
            raise GatewayError("Fixture provider requires explicitly marked fixture prices")

    def generate(
        self, context: CallContext, *, system: str, prompt: str, max_output_tokens: int = 800,
    ) -> ModelResult:
        caps = {"SMOKE": 64, "GENERATE": 800, "EXTRACT": 2048, "EVALUATE": 800}
        if context.stage not in caps or type(max_output_tokens) is not int or not (
            1 <= max_output_tokens <= caps[context.stage]
        ):
            raise GatewayError("Invalid stage or output-token cap")
        if not system.strip() or not prompt.strip():
            raise GatewayError("System and prompt must be nonempty")
        body = canonical_json({
            "anthropic_version": "bedrock-2023-05-31", "max_tokens": max_output_tokens,
            "temperature": 0, "system": system,
            "messages": [{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        })
        return self._call(context, HAIKU, HAIKU_ROUTE, body, None, max_output_tokens)

    def embed(self, context: CallContext, *, text: str) -> ModelResult:
        if context.stage != "EMBED" or not text.strip() or len(text) > 50_000:
            raise GatewayError("Embedding needs EMBED stage and 1..50000 characters")
        body = canonical_json({"inputText": text, "dimensions": 1024, "normalize": True})
        # Titan V2 rejects inputs above 8192 tokens. Reserve its complete accepted input
        # ceiling (164 micro-USD), not a guess from characters or a different tokenizer.
        return self._call(context, TITAN, TITAN, body, 8192, 0)

    def _call(self, context, model, route, body, input_bound, output_bound) -> ModelResult:
        if len(body) > 100_000:
            raise GatewayError("Input payload exceeds bounded text-only request size")
        rate = self.prices.rate_for(
            model, route, self.routes.source_region, now=self.ledger._now(), live=self.provider.live,
        )
        if input_bound is None:
            input_bound = self.provider.count_input(route, body)
            if not 1 <= input_bound <= 6144:
                raise GatewayError("Counted input is outside the 6144-token serving limit")
        bound = rate.quote(TokenUsage(input_tokens=input_bound, output_tokens=output_bound))
        spec = OperationSpec(
            **context.model_dump(), input_sha256=sha256_bytes(body), model_id=model,
            invocation_id=route, source_region=self.routes.source_region,
            price_sheet_version=self.prices.version, max_cost_micro_usd=bound,
        )
        self.ledger.reserve(spec)
        self.ledger.begin_call(spec.operation_id)
        try:
            payload = self.provider.invoke(route, body)
            if model == TITAN:
                usage = TokenUsage(input_tokens=_positive_integer(payload["inputTextTokenCount"]),
                                   output_tokens=0)
            else:
                raw_usage = payload["usage"]
                # Cache/tool/image pricing is deliberately not authorized by this gateway.
                if any(raw_usage.get(k, 0) != 0 for k in (
                    "cache_creation_input_tokens", "cache_read_input_tokens",
                )):
                    raise GatewayError("Unsupported cached-token usage needs reconciliation")
                usage = TokenUsage(input_tokens=_positive_integer(raw_usage["input_tokens"]),
                                   output_tokens=raw_usage["output_tokens"])
            actual = rate.quote(usage)
            self.ledger.settle(spec.operation_id, actual, usage)
        except Exception as exc:
            try:
                self.ledger.mark_unknown(spec.operation_id)
            except Exception:
                pass  # Already settled, or DB unavailable: never refund or invoke again.
            raise UncertainModelCall(spec.operation_id) from exc
        if usage.input_tokens > input_bound or usage.output_tokens > output_bound:
            self.ledger.halt_operation_budgets(spec.operation_id)
            raise GatewayError("Provider usage exceeded token bound; budgets halted after settlement")
        # Validate output after accounting. Invalid model output is still chargeable work.
        try:
            if model == TITAN:
                raw_vector = payload["embedding"]
                if not isinstance(raw_vector, list) or len(raw_vector) != 1024 or any(
                    type(x) not in (int, float) or not math.isfinite(x) for x in raw_vector
                ):
                    raise GatewayError("Invalid embedding shape or values")
                vector = np.asarray(raw_vector, dtype=np.float32)
                if not np.isfinite(vector).all() or abs(float(np.linalg.norm(vector)) - 1) > 0.001:
                    raise GatewayError("Embedding must be a finite normalized vector")
                return ModelResult(spec.operation_id, usage, actual,
                                   embedding=tuple(float(x) for x in vector))
            content = payload["content"]
            if not isinstance(content, list) or not content or any(
                not isinstance(block, dict) or block.get("type") != "text"
                or not isinstance(block.get("text"), str) for block in content
            ):
                raise GatewayError("Only text output is supported")
            text = "".join(block["text"] for block in content)
            stop = payload["stop_reason"]
            if not text.strip() or stop not in {"end_turn", "max_tokens", "stop_sequence"}:
                raise GatewayError("Missing text or unsupported stop reason")
            return ModelResult(spec.operation_id, usage, actual, text=text, stop_reason=stop)
        except (KeyError, TypeError, ValueError) as exc:
            raise GatewayError("Model output is invalid; usage has already been settled") from exc
