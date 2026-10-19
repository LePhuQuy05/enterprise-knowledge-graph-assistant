"""Local cache around ModelGateway; cache misses never create an implicit retry.

LIVE and FIXTURE occupy separate namespaces. A durable claim is committed before
dispatch, so a crash or ambiguous failure needs reconciliation rather than a new
paid call. This local cache does not replace the shared cost ledger.
"""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self
from uuid import uuid4

import numpy as np
from pydantic import Field, StringConstraints, TypeAdapter, model_validator

from app.contracts import (
    EmbeddingRecipe,
    ExternalEmbeddingRecipe,
    LocalBgeRecipe,
    Record,
    Sha256,
    TenantId,
    canonical_json,
    embedding_cache_key,
    embedding_input,
    sha256_bytes,
    stable_id,
    text_hash,
)
from app.corpus import LocalCorpus
from app.cost import TokenUsage
from app.model_gateway import TITAN, CallContext, ModelGateway, safe_failure_code

MAX_CACHE_RECORD_BYTES = 256 * 1024
Namespace = Literal["FIXTURE", "LIVE"]


class EmbeddingCacheError(ValueError):
    pass


class EmbeddingCacheBlocked(EmbeddingCacheError):
    pass


def query_embedding_input(question: str) -> str:
    value = question.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not 1 <= len(value) <= 2000:
        raise EmbeddingCacheError("QUERY_INPUT_LIMIT")
    return value


class CachedEmbedding(Record):
    schema_version: Literal[1] = 1
    namespace: Namespace
    tenant_id: TenantId
    cache_key: str
    input_text: Annotated[str, StringConstraints(min_length=1, max_length=50_000)]
    input_sha256: Sha256
    recipe: EmbeddingRecipe | ExternalEmbeddingRecipe | LocalBgeRecipe
    operation_id: str
    usage: TokenUsage
    original_cost_micro_usd: Annotated[int, Field(ge=0)]
    vector: Annotated[tuple[float, ...], Field(min_length=384, max_length=1024)]

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.input_sha256 != text_hash(self.input_text) or self.cache_key != embedding_cache_key(
            self.tenant_id, self.input_text, self.recipe,
        ):
            raise EmbeddingCacheError("CACHE_INPUT_IDENTITY_MISMATCH")
        vector = np.asarray(self.vector, dtype=np.float32)
        if vector.shape != (self.recipe.dimension,) or not np.isfinite(vector).all() or not np.isclose(
            np.linalg.norm(vector), 1, atol=1e-5, rtol=0,
        ):
            raise EmbeddingCacheError("CACHE_VECTOR_INVALID")
        if not self.operation_id or self.usage.input_tokens < 1 or self.usage.output_tokens != 0:
            raise EmbeddingCacheError("CACHE_RECEIPT_INVALID")
        if isinstance(self.recipe, LocalBgeRecipe) and (self.original_cost_micro_usd != 0 or not self.operation_id.startswith("localembed_")):
            raise EmbeddingCacheError("LOCAL_RECEIPT_MUST_BE_UNBILLED")
        return self


@dataclass(frozen=True)
class EmbeddingLookup:
    record: CachedEmbedding
    cache_hit: bool

    @property
    def new_cost_micro_usd(self) -> int:
        return 0 if self.cache_hit else self.record.original_cost_micro_usd


def titan_recipe() -> EmbeddingRecipe:
    return EmbeddingRecipe(version="titan-v2-title-section-text-v1", model_id=TITAN,
                           invocation_id=TITAN, source_region="us-east-1")


class EmbeddingCache:
    def __init__(self, directory: Path, *, namespace: Namespace = "LIVE"):
        self.namespace = TypeAdapter(Namespace).validate_python(namespace, strict=True)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "embeddings.sqlite3"
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS embeddings (
                namespace TEXT NOT NULL, cache_key TEXT NOT NULL,
                state TEXT NOT NULL CHECK (state IN ('IN_PROGRESS','FAILED','READY')),
                owner TEXT NOT NULL, claim BLOB NOT NULL,
                payload BLOB, payload_sha256 TEXT, error_category TEXT,
                PRIMARY KEY(namespace, cache_key))""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def _key(tenant_id, text, recipe):
        TypeAdapter(TenantId).validate_python(tenant_id, strict=True)
        if not text.strip() or len(text) > 50_000:
            raise EmbeddingCacheError("EMBEDDING_INPUT_LIMIT")
        return embedding_cache_key(tenant_id, text, recipe)

    def get(self, tenant_id: str, text: str, recipe: EmbeddingRecipe) -> CachedEmbedding | None:
        key = self._key(tenant_id, text, recipe)
        with self._connect() as db:
            row = db.execute("SELECT state,length(payload),payload_sha256 FROM embeddings "
                             "WHERE namespace=? AND cache_key=?", (self.namespace, key)).fetchone()
            if row is None:
                return None
            if row[0] != "READY":
                raise EmbeddingCacheBlocked("CACHE_RECONCILIATION_REQUIRED")
            if row[1] is None or not 1 <= row[1] <= MAX_CACHE_RECORD_BYTES:
                raise EmbeddingCacheError("CACHE_RECORD_SIZE_INVALID")
            raw = db.execute("SELECT payload FROM embeddings WHERE namespace=? AND cache_key=?",
                             (self.namespace, key)).fetchone()[0]
        if not isinstance(raw, bytes) or sha256_bytes(raw) != row[2]:
            raise EmbeddingCacheError("CACHE_CHECKSUM_MISMATCH")
        record = CachedEmbedding.model_validate_json(raw)
        if record.namespace != self.namespace or record.tenant_id != tenant_id or (
            record.cache_key != key or record.input_text != text
            or record.recipe.fingerprint != recipe.fingerprint
        ):
            raise EmbeddingCacheError("CACHE_RECIPE_OR_TENANT_MISMATCH")
        return record

    def resolve(self, context: CallContext, *, text: str, recipe: EmbeddingRecipe,
                gateway: ModelGateway | None = None) -> EmbeddingLookup:
        if isinstance(recipe, LocalBgeRecipe):
            raise EmbeddingCacheError("LOCAL_BGE_REQUIRES_LOCAL_ENCODER")
        if context.stage != "EMBED":
            raise EmbeddingCacheError("CACHE_REQUIRES_EMBED_STAGE")
        key = self._key(context.tenant_id, text, recipe)
        cached = self.get(context.tenant_id, text, recipe)
        if cached is not None:
            return EmbeddingLookup(cached, True)
        if gateway is None:
            raise EmbeddingCacheError("CACHE_MISS_NO_GATEWAY")
        if gateway.routes.source_region == "external-default" and not isinstance(recipe, ExternalEmbeddingRecipe):
            raise EmbeddingCacheError("CACHE_EXTERNAL_ROLE_RECIPE_REQUIRED")
        expected = "LIVE" if gateway.provider.live else "FIXTURE"
        if self.namespace != expected or recipe.model_id != gateway.routes.embedding_model_id or (
            recipe.invocation_id != gateway.routes.embedding_invocation_id
            or recipe.source_region != gateway.routes.source_region
            or recipe.dimension != gateway.routes.embedding_dimension
            or recipe.normalized is not True
        ):
            raise EmbeddingCacheError("CACHE_GATEWAY_RECIPE_MISMATCH")
        owner = uuid4().hex
        claim = canonical_json({"context": context.model_dump(mode="json"),
                                "operation_id": stable_id("op", **context.model_dump()),
                                "input_sha256": text_hash(text),
                                "recipe_sha256": recipe.fingerprint})
        with self._connect() as db:
            changed = db.execute("INSERT OR IGNORE INTO embeddings "
                                 "(namespace,cache_key,state,owner,claim) VALUES (?,?,'IN_PROGRESS',?,?)",
                                 (self.namespace, key, owner, claim)).rowcount
        if changed != 1:
            # A competing winner may have finished. No takeover of a pending owner.
            cached = self.get(context.tenant_id, text, recipe)
            if cached is None:
                raise EmbeddingCacheBlocked("CACHE_RECONCILIATION_REQUIRED")
            return EmbeddingLookup(cached, True)
        try:
            if isinstance(recipe, ExternalEmbeddingRecipe):
                result = gateway.embed(context, text=text, input_role=recipe.input_role)
            else:
                result = gateway.embed(context, text=text)
            vector = np.asarray(result.embedding, dtype=np.float64)
            if vector.shape != (1024,) or not np.isfinite(vector).all():
                raise EmbeddingCacheError("PROVIDER_VECTOR_INVALID")
            norm = float(np.linalg.norm(vector))
            if abs(norm - 1) > 0.001:
                raise EmbeddingCacheError("PROVIDER_VECTOR_NORM_INVALID")
            # Tighten provider tolerance once for the float32 snapshot contract.
            normalized = (vector / norm).astype(np.float32)
            record = CachedEmbedding(
                namespace=self.namespace, tenant_id=context.tenant_id, cache_key=key,
                input_text=text, input_sha256=text_hash(text), recipe=recipe,
                operation_id=result.operation_id, usage=result.usage,
                original_cost_micro_usd=result.actual_micro_usd,
                vector=tuple(float(v) for v in normalized),
            )
            raw = record.model_dump_json().encode("utf-8")
            if len(raw) > MAX_CACHE_RECORD_BYTES:
                raise EmbeddingCacheError("CACHE_RECORD_SIZE_INVALID")
            with self._connect() as db:
                changed = db.execute("UPDATE embeddings SET state='READY',payload=?,payload_sha256=? "
                                     "WHERE namespace=? AND cache_key=? AND owner=? "
                                     "AND state='IN_PROGRESS'",
                                     (raw, sha256_bytes(raw), self.namespace, key, owner)).rowcount
                if changed != 1:
                    raise EmbeddingCacheBlocked("CACHE_OWNER_LOST")
        except Exception as exc:
            with self._connect() as db:
                db.execute("UPDATE embeddings SET state='FAILED',error_category=? "
                           "WHERE namespace=? AND cache_key=? AND owner=? AND state='IN_PROGRESS'",
                           (safe_failure_code(exc), self.namespace, key, owner))
            raise
        return EmbeddingLookup(record, False)

    def resolve_local(self, tenant_id: str, *, text: str, recipe: LocalBgeRecipe, encoder=None) -> EmbeddingLookup:
        """Actual CPU receipts use LIVE; fake encoders must stay in FIXTURE. No paid ledger operation."""
        if not isinstance(recipe, LocalBgeRecipe):
            raise EmbeddingCacheError("LOCAL_BGE_RECIPE_REQUIRED")
        key = self._key(tenant_id, text, recipe)
        cached = self.get(tenant_id, text, recipe)
        if cached is not None:
            return EmbeddingLookup(cached, True)
        if encoder is None:
            raise EmbeddingCacheError("LOCAL_CACHE_MISS_NO_ENCODER")
        if self.namespace != ("FIXTURE" if encoder.fixture_only else "LIVE") or encoder.recipe(recipe.input_role) != recipe:
            raise EmbeddingCacheError("LOCAL_ENCODER_RECIPE_OR_NAMESPACE_MISMATCH")
        owner = uuid4().hex
        execution_id = stable_id("localembed", tenant_id=tenant_id, cache_key=key)
        claim = canonical_json({"execution_id": execution_id, "recipe_sha256": recipe.fingerprint})
        with self._connect() as db:
            changed = db.execute("INSERT OR IGNORE INTO embeddings (namespace,cache_key,state,owner,claim) "
                                 "VALUES (?,?,'IN_PROGRESS',?,?)", (self.namespace, key, owner, claim)).rowcount
        if changed != 1:
            raise EmbeddingCacheBlocked("LOCAL_CACHE_CLAIM_EXISTS")
        try:
            vector = encoder.encode([text], role=recipe.input_role)[0]
            record = CachedEmbedding(namespace=self.namespace, tenant_id=tenant_id, cache_key=key,
                input_text=text, input_sha256=text_hash(text), recipe=recipe, operation_id=execution_id,
                usage=TokenUsage(input_tokens=encoder.count(text, role=recipe.input_role), output_tokens=0),
                original_cost_micro_usd=0, vector=tuple(float(v) for v in vector))
            raw = record.model_dump_json().encode("utf-8")
            if len(raw) > MAX_CACHE_RECORD_BYTES:
                raise EmbeddingCacheError("CACHE_RECORD_SIZE_INVALID")
            with self._connect() as db:
                changed = db.execute("UPDATE embeddings SET state='READY',payload=?,payload_sha256=? "
                                     "WHERE namespace=? AND cache_key=? AND owner=? AND state='IN_PROGRESS'",
                                     (raw, sha256_bytes(raw), self.namespace, key, owner)).rowcount
                if changed != 1:
                    raise EmbeddingCacheBlocked("LOCAL_CACHE_OWNER_LOST")
        except Exception as exc:
            with self._connect() as db:
                db.execute("UPDATE embeddings SET state='FAILED',error_category=? WHERE namespace=? AND cache_key=? AND owner=?",
                           (type(exc).__name__, self.namespace, key, owner))
            raise
        return EmbeddingLookup(record, False)


def plan_embeddings(corpus: LocalCorpus, cache: EmbeddingCache, recipe: EmbeddingRecipe) -> dict:
    docs = {d.document_version_id: d for d in corpus.documents}
    inputs = {}
    for chunk in corpus.chunks:
        text = embedding_input(docs[chunk.document_version_id], chunk)
        key = embedding_cache_key(corpus.tenant_id, text, recipe)
        inputs.setdefault(key, text)
    hits, misses, blocked = [], [], []
    for key, text in inputs.items():
        try:
            record = cache.get(corpus.tenant_id, text, recipe)
            (hits if record is not None else misses).append(key)
        except EmbeddingCacheBlocked:
            blocked.append(key)
    return {"status": "PLANNED_ONLY", "corpus_version": corpus.version,
            "tenant_id": corpus.tenant_id, "namespace": cache.namespace,
            "embedding_recipe": recipe.model_dump(mode="json"), "chunks": len(corpus.chunks),
            "unique_inputs": len(inputs), "cache_hits": len(hits), "cache_misses": len(misses),
            "blocked_inputs": len(blocked), "missing_keys": misses, "blocked_keys": blocked,
            "inference_calls": 0,
            "notice": "No vectors created. A plan is not a model invocation or batch authorization."}
