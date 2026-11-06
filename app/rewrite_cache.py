"""Durable, request-bound search rewrites; rewrites are never answer evidence.

Settled provider receipts are committed before rewrite JSON/identifier checks.
Failed or interrupted requests require reconciliation rather than another call.
An offline replay uses the pinned route and the same validation function as a miss.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, StringConstraints, ValidationError

from app.contracts import Record, Sha256, canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.external_gateway import ExternalRoutes
from app.model_gateway import CallContext, ModelResult, UncertainModelCall, safe_failure_code
from app.query_language import RewriteError, rewrite_for_search

RECIPE = "durable-vi-en-aws-search-v1"
PROMPT_VERSION = "vi-en-aws-search-v1"
MAX_RECEIPT_BYTES = 64 * 1024


class RewriteCacheError(RewriteError):
    """Bounded rewrite-cache integrity or configuration error."""


class RewriteCacheBlocked(RewriteCacheError):
    """An existing intent may not be dispatched or repaired automatically."""


class _ProviderReceipt(Record):
    operation_id: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    raw_output: Annotated[str, StringConstraints(max_length=6000)]
    usage: TokenUsage
    cost_micro_usd: Annotated[int, Field(ge=0)]
    stop_reason: Annotated[str, StringConstraints(max_length=100)] | None
    prompt_version: Literal["vi-en-aws-search-v1"]


class _Receipt(Record):
    schema_version: Literal[1] = 1
    namespace: Literal["LIVE", "FIXTURE"]
    cache_key: str
    binding_sha256: Sha256
    recipe: Literal["durable-vi-en-aws-search-v1"] = RECIPE
    provider_receipt: _ProviderReceipt


@dataclass(frozen=True)
class RewriteLookup:
    search_query: str
    cache_hit: bool
    receipt: dict | None
    new_cost_micro_usd: int


def _route_identity(routes: ExternalRoutes) -> dict:
    if not isinstance(routes, ExternalRoutes):
        raise RewriteCacheError("REWRITE_ROUTES_REQUIRED")
    try:
        # Revalidate even model_copy(update=...) values, without operational flags.
        routes = ExternalRoutes.model_validate_json(canonical_json(routes.model_dump(mode="json")))
    except (ValueError, TypeError) as exc:
        raise RewriteCacheError("REWRITE_ROUTES_INVALID") from exc
    return routes.model_dump(mode="json", include={
        "generation_model_id", "generation_invocation_id", "generation_context_ceiling",
        "generation_endpoint", "generation_protocol", "source_region", "account_model_metadata_sha256",
    })


class _SavedGateway:
    """Pure receipt adapter for existing rewrite validation, without dispatch."""

    def __init__(self, receipt: _ProviderReceipt):
        self.receipt = receipt

    def generate(self, *_args, **_kwargs):
        value = self.receipt
        return ModelResult(
            operation_id=value.operation_id, usage=value.usage,
            actual_micro_usd=value.cost_micro_usd, text=value.raw_output,
            stop_reason=value.stop_reason,
        )


class RewriteCache:
    def __init__(self, directory: Path, *, namespace: Literal["LIVE", "FIXTURE"] = "LIVE"):
        if namespace not in {"LIVE", "FIXTURE"}:
            raise RewriteCacheError("REWRITE_CACHE_NAMESPACE")
        self.namespace = namespace
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "rewrites.sqlite3"
        if self.path.is_symlink():
            raise RewriteCacheError("REWRITE_CACHE_UNSAFE_PATH")
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS rewrites ("
                "namespace TEXT NOT NULL,cache_key TEXT NOT NULL,context_key TEXT NOT NULL,"
                "state TEXT NOT NULL CHECK(state IN ('IN_PROGRESS','FAILED','READY')),"
                "owner TEXT NOT NULL,intent BLOB NOT NULL,payload BLOB,payload_sha256 TEXT,"
                "error_category TEXT,PRIMARY KEY(namespace,cache_key),UNIQUE(namespace,context_key))"
            )

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _get(self, key: str, context_key: str, binding_sha: str):
        with self._connect() as db:
            row = db.execute(
                "SELECT cache_key,state,length(payload),payload_sha256 FROM rewrites "
                "WHERE namespace=? AND context_key=?", (self.namespace, context_key),
            ).fetchone()
            if row is None:
                return None
            if row[0] != key:
                raise RewriteCacheBlocked("REWRITE_CACHE_IDEMPOTENCY_CONFLICT")
            if row[1] != "READY":
                raise RewriteCacheBlocked("REWRITE_CACHE_RECONCILIATION_REQUIRED")
            if row[2] is None or not 1 <= row[2] <= MAX_RECEIPT_BYTES:
                raise RewriteCacheError("REWRITE_CACHE_PAYLOAD_SIZE")
            raw = db.execute(
                "SELECT payload FROM rewrites WHERE namespace=? AND cache_key=?",
                (self.namespace, key),
            ).fetchone()[0]
        if not isinstance(raw, bytes) or sha256_bytes(raw) != row[3]:
            raise RewriteCacheError("REWRITE_CACHE_CHECKSUM")
        try:
            receipt = _Receipt.model_validate_json(raw)
        except (ValueError, ValidationError) as exc:
            raise RewriteCacheError("REWRITE_CACHE_PAYLOAD_INVALID") from exc
        if (receipt.namespace != self.namespace or receipt.cache_key != key
                or receipt.binding_sha256 != binding_sha):
            raise RewriteCacheError("REWRITE_CACHE_BINDING_MISMATCH")
        return receipt

    @staticmethod
    def _validate(receipt, context, question):
        if receipt.operation_id != stable_id("op", **context.model_dump(mode="json")):
            raise RewriteCacheError("REWRITE_RECEIPT_OPERATION_MISMATCH")
        query, _ = rewrite_for_search(
            question, gateway=_SavedGateway(receipt), tenant_id=context.tenant_id,
            request_id=context.request_id, run_id=context.run_id,
        )
        return query

    def resolve(self, context: CallContext, *, question: str, routes: ExternalRoutes,
                gateway=None) -> RewriteLookup:
        if (not isinstance(context, CallContext) or context.stage != "REWRITE" or context.attempt != 1
                or type(question) is not str or not 1 <= len(question) <= 2000 or not question.strip()):
            raise RewriteCacheError("REWRITE_CACHE_INPUT_INVALID")
        route = _route_identity(routes)
        binding = {"context": context.model_dump(mode="json"), "question_sha256": sha256_bytes(
            question.encode("utf-8")), "route": route, "recipe": RECIPE,
            "prompt_version": PROMPT_VERSION}
        binding_sha = sha256_bytes(canonical_json(binding))
        key = stable_id("rewrite", binding_sha256=binding_sha)
        context_key = stable_id("rewrite-context", **context.model_dump(mode="json"))
        receipt = self._get(key, context_key, binding_sha)
        if receipt is not None:
            query = self._validate(receipt.provider_receipt, context, question)
            return RewriteLookup(query, True, receipt.provider_receipt.model_dump(mode="json"), 0)
        if gateway is None:
            raise RewriteCacheError("REWRITE_CACHE_MISS_NO_GATEWAY")
        if (getattr(getattr(gateway, "provider", None), "live", None)
                is not (self.namespace == "LIVE")):
            raise RewriteCacheError("REWRITE_GATEWAY_NAMESPACE_MISMATCH")
        if _route_identity(getattr(gateway, "routes", None)) != route:
            raise RewriteCacheError("REWRITE_GATEWAY_ROUTE_MISMATCH")
        owner = uuid4().hex
        with self._connect() as db:
            changed = db.execute(
                "INSERT OR IGNORE INTO rewrites(namespace,cache_key,context_key,state,owner,intent) "
                "VALUES (?,?,?,'IN_PROGRESS',?,?)",
                (self.namespace, key, context_key, owner, canonical_json(binding)),
            ).rowcount
            if changed != 1:
                raise RewriteCacheBlocked("REWRITE_CACHE_CLAIM_EXISTS")
        saved = None

        def persist_settled(value):
            nonlocal saved
            if saved is not None:
                raise RewriteCacheError("REWRITE_MULTIPLE_RECEIPTS")
            try:
                provider_receipt = _ProviderReceipt.model_validate_json(canonical_json(value))
                payload = _Receipt(
                    namespace=self.namespace, cache_key=key, binding_sha256=binding_sha,
                    provider_receipt=provider_receipt,
                )
                raw = canonical_json(payload.model_dump(mode="json"))
            except (ValueError, TypeError) as exc:
                raise RewriteCacheError("REWRITE_RECEIPT_INVALID") from exc
            if len(raw) > MAX_RECEIPT_BYTES:
                raise RewriteCacheError("REWRITE_RECEIPT_SIZE")
            with self._connect() as db:
                changed = db.execute(
                    "UPDATE rewrites SET payload=?,payload_sha256=? WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'",
                    (raw, sha256_bytes(raw), self.namespace, key, owner),
                ).rowcount
                if changed != 1:
                    raise RewriteCacheBlocked("REWRITE_CACHE_OWNER_LOST")
            saved = payload

        try:
            query, _ = rewrite_for_search(
                question, gateway=gateway, tenant_id=context.tenant_id,
                request_id=context.request_id, run_id=context.run_id, receipt_sink=persist_settled,
            )
            if saved is None:
                raise RewriteCacheError("REWRITE_RECEIPT_MISSING")
            if self._validate(saved.provider_receipt, context, question) != query:
                raise RewriteCacheError("REWRITE_VALIDATION_MISMATCH")
            with self._connect() as db:
                changed = db.execute(
                    "UPDATE rewrites SET state='READY' WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'", (self.namespace, key, owner),
                ).rowcount
                if changed != 1:
                    raise RewriteCacheBlocked("REWRITE_CACHE_OWNER_LOST")
            return RewriteLookup(query, False, saved.provider_receipt.model_dump(mode="json"),
                                 saved.provider_receipt.cost_micro_usd)
        except Exception as exc:
            with self._connect() as db:
                db.execute(
                    "UPDATE rewrites SET state='FAILED',error_category=? WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'",
                    (safe_failure_code(exc), self.namespace, key, owner),
                )
            if isinstance(exc, (RewriteError, UncertainModelCall)):
                raise
            raise RewriteCacheError("REWRITE_DISPATCH_FAILED") from None
