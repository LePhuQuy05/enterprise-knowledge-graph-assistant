"""Durable per-request answer receipts, no automatic repair or uncertain-call retry."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field

from app.contracts import QueryResponse, Record, canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.generation import (
    AnswerBinding,
    GenerationError,
    GenerationRecipe,
    empty_answer,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.model_gateway import safe_failure_code


class AnswerCacheBlocked(GenerationError):
    pass


class AnswerReceipt(Record):
    schema_version: Literal[1] = 1
    namespace: Literal["LIVE", "FIXTURE"]
    cache_key: str
    operation_id: str
    recipe_sha256: str
    binding_sha256: str
    raw_output: str
    stop_reason: str
    usage: TokenUsage
    original_cost_micro_usd: Annotated[int, Field(ge=0)]


@dataclass(frozen=True)
class AnswerLookup:
    response: QueryResponse
    receipt: AnswerReceipt | None
    cache_hit: bool
    deterministic_no_evidence: bool = False

    @property
    def new_cost_micro_usd(self):
        return 0 if self.cache_hit or self.receipt is None else self.receipt.original_cost_micro_usd


class AnswerCache:
    def __init__(self, directory: Path, *, namespace: Literal["LIVE", "FIXTURE"] = "LIVE"):
        if namespace not in {"LIVE", "FIXTURE"}:
            raise GenerationError("ANSWER_CACHE_NAMESPACE")
        self.namespace = namespace
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "answers.sqlite3"
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS answers (namespace TEXT NOT NULL,cache_key TEXT NOT NULL,"
                "state TEXT NOT NULL CHECK(state IN ('IN_PROGRESS','FAILED','READY')),owner TEXT NOT NULL,"
                "intent BLOB NOT NULL,payload BLOB,payload_sha256 TEXT,error_category TEXT,"
                "PRIMARY KEY(namespace,cache_key))")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def _get(self, key):
        with self._connect() as db:
            row = db.execute("SELECT state,length(payload),payload_sha256 FROM answers WHERE namespace=? AND cache_key=?",
                             (self.namespace, key)).fetchone()
            if row is None:
                return None
            if row[0] != "READY":
                raise AnswerCacheBlocked("ANSWER_CACHE_RECONCILIATION_REQUIRED")
            if row[1] is None or not 1 <= row[1] <= 128 * 1024:
                raise GenerationError("ANSWER_CACHE_PAYLOAD_SIZE")
            raw = db.execute("SELECT payload FROM answers WHERE namespace=? AND cache_key=?", (self.namespace, key)).fetchone()[0]
        if not isinstance(raw, bytes) or sha256_bytes(raw) != row[2]:
            raise GenerationError("ANSWER_CACHE_CHECKSUM")
        receipt = AnswerReceipt.model_validate_json(raw)
        if receipt.namespace != self.namespace or receipt.cache_key != key:
            raise GenerationError("ANSWER_CACHE_IDENTITY")
        return receipt

    def resolve(self, context, *, question, snapshot, packed, binding: AnswerBinding,
                recipe: GenerationRecipe, gateway=None) -> AnswerLookup:
        if (context.stage != "GENERATE" or context.attempt != 1 or context.tenant_id != binding.tenant_id
                or context.request_id != binding.request_id):
            raise GenerationError("GENERATION_CONTEXT_BINDING_MISMATCH")
        system, prompt = generation_request(question, snapshot, packed, binding.tenant_id, recipe)
        if not packed.chunk_ids:
            return AnswerLookup(empty_answer(snapshot=snapshot, packed=packed, binding=binding), None, False, True)
        binding_sha = sha256_bytes(canonical_json(binding.model_dump(mode="json")))
        key = stable_id("answer", context=context.model_dump(mode="json"), binding_sha=binding_sha,
            request_sha=sha256_bytes(canonical_json([system, prompt])), recipe=recipe.fingerprint,
            snapshot_sha=snapshot.manifest_sha256)
        receipt = self._get(key)

        def validate(receipt):
            if (receipt.recipe_sha256 != recipe.fingerprint or receipt.binding_sha256 != binding_sha
                    or receipt.operation_id != stable_id("op", **context.model_dump())):
                raise GenerationError("ANSWER_RECEIPT_BINDING_MISMATCH")
            return resolve_answer(parse_generated(receipt.raw_output, receipt.stop_reason),
                                  snapshot=snapshot, packed=packed, binding=binding)

        if receipt is not None:
            return AnswerLookup(validate(receipt), receipt, True)
        if gateway is None:
            raise GenerationError("ANSWER_CACHE_MISS_NO_GATEWAY")
        if (self.namespace != ("LIVE" if gateway.provider.live else "FIXTURE")
                or gateway.routes.generation_model_id != recipe.model_id
                or gateway.routes.generation_invocation_id != recipe.invocation_id
                or gateway.routes.source_region != recipe.source_region
                or gateway.routes.account_model_metadata_sha256 != recipe.model_metadata_sha256):
            raise GenerationError("GENERATION_GATEWAY_RECIPE_MISMATCH")
        owner = uuid4().hex
        intent = canonical_json({"context": context.model_dump(mode="json"),
                                 "binding_sha256": binding_sha, "recipe_sha256": recipe.fingerprint})
        with self._connect() as db:
            changed = db.execute("INSERT OR IGNORE INTO answers(namespace,cache_key,state,owner,intent) "
                                 "VALUES (?,?,'IN_PROGRESS',?,?)", (self.namespace, key, owner, intent)).rowcount
            if changed != 1:
                raise AnswerCacheBlocked("ANSWER_CACHE_CLAIM_EXISTS")
        try:
            result = gateway.generate(context, system=system, prompt=prompt, max_output_tokens=recipe.max_output_tokens)
            receipt = AnswerReceipt(namespace=self.namespace, cache_key=key, operation_id=result.operation_id,
                recipe_sha256=recipe.fingerprint, binding_sha256=binding_sha, raw_output=result.text or "",
                stop_reason=result.stop_reason or "MISSING", usage=result.usage,
                original_cost_micro_usd=result.actual_micro_usd)
            raw = receipt.model_dump_json().encode()
            if len(raw) > 128 * 1024:
                raise GenerationError("ANSWER_RECEIPT_SIZE")
            with self._connect() as db:
                changed = db.execute("UPDATE answers SET payload=?,payload_sha256=? WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'", (raw, sha256_bytes(raw), self.namespace, key, owner)).rowcount
                if changed != 1:
                    raise AnswerCacheBlocked("ANSWER_CACHE_OWNER_LOST")
            response = validate(receipt)  # Settled output is already durable even when invalid.
            with self._connect() as db:
                changed = db.execute("UPDATE answers SET state='READY' WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'", (self.namespace, key, owner)).rowcount
                if changed != 1:
                    raise AnswerCacheBlocked("ANSWER_CACHE_OWNER_LOST")
            return AnswerLookup(response, receipt, False)
        except Exception as exc:
            with self._connect() as db:
                db.execute("UPDATE answers SET state='FAILED',error_category=? WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'", (safe_failure_code(exc), self.namespace, key, owner))
            raise
