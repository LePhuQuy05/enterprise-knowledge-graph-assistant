"""Durable local extraction claims and receipts around the shared model ledger."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field, TypeAdapter

from app.contracts import (
    Chunk,
    DocumentVersion,
    ExtractionRecipe,
    Record,
    Sha256,
    TenantId,
    canonical_json,
    sha256_bytes,
    stable_id,
    text_hash,
)
from app.corpus import LocalCorpus
from app.cost import PriceSheet, TokenUsage
from app.extraction import (
    MAX_OUTPUT_BYTES,
    AliasDictionary,
    CanonicalExtraction,
    ExtractionError,
    canonicalize,
    check_source,
    extraction_recipe,
    extraction_request,
    raw_cache_key,
    raw_recipe,
    validate_output,
)
from app.model_gateway import CallContext, ModelGateway, safe_failure_code

Namespace = Literal["LIVE", "FIXTURE"]
MAX_RECORD_BYTES = 512 * 1024


class ExtractionCacheBlocked(ExtractionError):
    pass


class ExtractionAttempt(Record):
    operation_id: str
    attempt: Annotated[int, Field(ge=1, le=2)]
    text: Annotated[str, Field(max_length=MAX_OUTPUT_BYTES)]
    stop_reason: str
    usage: TokenUsage
    original_cost_micro_usd: Annotated[int, Field(ge=0)]


class RawExtractionRecord(Record):
    schema_version: Literal[1] = 1
    namespace: Namespace
    tenant_id: TenantId
    cache_key: str
    input_sha256: Sha256
    raw_recipe_sha256: Sha256
    attempts: Annotated[tuple[ExtractionAttempt, ...], Field(min_length=1, max_length=2)]


@dataclass(frozen=True)
class ExtractionLookup:
    record: RawExtractionRecord
    canonical: CanonicalExtraction
    cache_hit: bool
    offset_repairs: tuple[str, ...]

    @property
    def new_cost_micro_usd(self) -> int:
        return 0 if self.cache_hit else sum(a.original_cost_micro_usd for a in self.record.attempts)


class ExtractionCache:
    def __init__(self, directory: Path, *, namespace: Namespace = "LIVE"):
        self.namespace = TypeAdapter(Namespace).validate_python(namespace, strict=True)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / "extraction.sqlite3"
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS extraction (
                namespace TEXT NOT NULL, cache_key TEXT NOT NULL,
                state TEXT NOT NULL CHECK(state IN ('IN_PROGRESS','FAILED','READY')),
                owner TEXT NOT NULL, claim BLOB NOT NULL, receipts BLOB NOT NULL,
                payload BLOB, payload_sha256 TEXT, error_category TEXT,
                PRIMARY KEY(namespace,cache_key))""")

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def get(self, tenant_id: str, text: str, recipe: ExtractionRecipe) -> RawExtractionRecord | None:
        TypeAdapter(TenantId).validate_python(tenant_id, strict=True)
        key = raw_cache_key(tenant_id, text, recipe)
        with self._connect() as db:
            row = db.execute("SELECT state,length(payload),payload_sha256 FROM extraction "
                             "WHERE namespace=? AND cache_key=?", (self.namespace, key)).fetchone()
            if row is None:
                return None
            if row[0] != "READY":
                raise ExtractionCacheBlocked("EXTRACTION_CACHE_RECONCILIATION_REQUIRED")
            if row[1] is None or not 1 <= row[1] <= MAX_RECORD_BYTES:
                raise ExtractionError("EXTRACTION_CACHE_RECORD_SIZE")
            raw = db.execute("SELECT payload FROM extraction WHERE namespace=? AND cache_key=?",
                             (self.namespace, key)).fetchone()[0]
        if not isinstance(raw, bytes) or sha256_bytes(raw) != row[2]:
            raise ExtractionError("EXTRACTION_CACHE_CHECKSUM")
        record = RawExtractionRecord.model_validate_json(raw)
        if (record.namespace != self.namespace or record.tenant_id != tenant_id
                or record.cache_key != key or record.input_sha256 != text_hash(text)
                or record.raw_recipe_sha256 != sha256_bytes(canonical_json(raw_recipe(recipe)))
                or tuple(a.attempt for a in record.attempts) != tuple(range(1, len(record.attempts) + 1))
                or len({a.operation_id for a in record.attempts}) != len(record.attempts)):
            raise ExtractionError("EXTRACTION_CACHE_IDENTITY")
        return record

    def _write(self, key: str, owner: str, *, receipts: list[ExtractionAttempt],
               record: RawExtractionRecord | None = None) -> None:
        raw = None if record is None else record.model_dump_json().encode("utf-8")
        if raw is not None and len(raw) > MAX_RECORD_BYTES:
            raise ExtractionError("EXTRACTION_CACHE_RECORD_SIZE")
        with self._connect() as db:
            changed = db.execute(
                "UPDATE extraction SET state=?,receipts=?,payload=?,payload_sha256=? "
                "WHERE namespace=? AND cache_key=? AND owner=? AND state='IN_PROGRESS'",
                ("READY" if record else "IN_PROGRESS",
                 canonical_json([r.model_dump(mode="json") for r in receipts]), raw,
                 None if raw is None else sha256_bytes(raw), self.namespace, key, owner),
            ).rowcount
            if changed != 1:
                raise ExtractionCacheBlocked("EXTRACTION_CACHE_OWNER_LOST")

    def resolve(self, context: CallContext, *, chunk: Chunk, document: DocumentVersion,
                recipe: ExtractionRecipe, aliases: AliasDictionary,
                gateway: ModelGateway | None = None, allow_repair: bool = False) -> ExtractionLookup:
        check_source(chunk, document)
        if context.stage != "EXTRACT" or context.attempt != 1 or context.tenant_id != chunk.tenant_id:
            raise ExtractionError("EXTRACTION_CONTEXT_MISMATCH")
        if recipe.canonicalization_version != aliases.identity:
            raise ExtractionError("EXTRACTION_ALIAS_RECIPE_MISMATCH")
        key = raw_cache_key(context.tenant_id, chunk.text, recipe)
        cached = self.get(context.tenant_id, chunk.text, recipe)
        if cached is not None:
            return self._lookup(cached, chunk, document, recipe, aliases, hit=True)
        if gateway is None:
            raise ExtractionError("EXTRACTION_CACHE_MISS_NO_GATEWAY")
        expected_namespace = "LIVE" if gateway.provider.live else "FIXTURE"
        if (self.namespace != expected_namespace
                or recipe.model_id != gateway.routes.generation_model_id
                or recipe.invocation_id != gateway.routes.generation_invocation_id
                or recipe.source_region != gateway.routes.source_region):
            raise ExtractionError("EXTRACTION_GATEWAY_RECIPE_MISMATCH")
        if recipe.source_region == "external-default" and recipe.parameters.get("model_metadata_sha256") != gateway.routes.account_model_metadata_sha256:
            raise ExtractionError("EXTRACTION_MODEL_METADATA_MISMATCH")
        owner = uuid4().hex
        claim = canonical_json({"context": context.model_dump(mode="json"),
                                "operation_id": stable_id("op", **context.model_dump()),
                                "input_sha256": text_hash(chunk.text), "allow_repair": allow_repair})
        with self._connect() as db:
            changed = db.execute("INSERT OR IGNORE INTO extraction "
                                 "(namespace,cache_key,state,owner,claim,receipts) "
                                 "VALUES (?,?,'IN_PROGRESS',?,?,'[]')",
                                 (self.namespace, key, owner, claim)).rowcount
            if changed != 1:
                raise ExtractionCacheBlocked("EXTRACTION_CACHE_CLAIM_EXISTS")
        system, prompt = extraction_request(chunk.text, recipe)
        receipts = []
        try:
            for attempt in range(1, 3 if allow_repair else 2):
                call = context.model_copy(update={"attempt": attempt})
                intent = canonical_json({
                    "initial_context": context.model_dump(mode="json"),
                    "latest_dispatch_intent": call.model_dump(mode="json"),
                    "operation_id": stable_id("op", **call.model_dump()),
                    "input_sha256": text_hash(chunk.text), "allow_repair": allow_repair,
                })
                with self._connect() as db:
                    changed = db.execute("UPDATE extraction SET claim=? WHERE namespace=? AND cache_key=? "
                                         "AND owner=? AND state='IN_PROGRESS'",
                                         (intent, self.namespace, key, owner)).rowcount
                    if changed != 1:
                        raise ExtractionCacheBlocked("EXTRACTION_CACHE_OWNER_LOST")
                result = gateway.generate(call, system=system, prompt=prompt, max_output_tokens=2048)
                receipt = ExtractionAttempt(
                    operation_id=result.operation_id, attempt=attempt, text=result.text or "",
                    stop_reason=result.stop_reason or "MISSING", usage=result.usage,
                    original_cost_micro_usd=result.actual_micro_usd,
                )
                receipts.append(receipt)
                # Persist settled output before parse/repair, so a crash is never a new miss.
                self._write(key, owner, receipts=receipts)
                try:
                    validate_output(receipt.text, chunk=chunk, document=document,
                                    stop_reason=receipt.stop_reason, recipe=recipe)
                except ExtractionError as exc:
                    if not allow_repair or attempt == 2:
                        raise
                    # Only settled format/source errors can repair. UncertainModelCall
                    # leaves this inner branch immediately and never dispatches attempt2.
                    prompt = canonical_json({
                        "original_request": extraction_request(chunk.text, recipe)[1],
                        "validation_error": str(exc), "untrusted_previous_output": receipt.text[:8192],
                        "instruction": "Return one complete corrected JSON object. Prefer fewer records.",
                    }).decode("utf-8")
                    continue
                record = RawExtractionRecord(
                    namespace=self.namespace, tenant_id=context.tenant_id, cache_key=key,
                    input_sha256=text_hash(chunk.text),
                    raw_recipe_sha256=sha256_bytes(canonical_json(raw_recipe(recipe))),
                    attempts=tuple(receipts),
                )
                self._write(key, owner, receipts=receipts, record=record)
                return self._lookup(record, chunk, document, recipe, aliases, hit=False)
        except Exception as exc:
            with self._connect() as db:
                db.execute("UPDATE extraction SET state='FAILED',error_category=? "
                           "WHERE namespace=? AND cache_key=? AND owner=? AND state='IN_PROGRESS'",
                           (safe_failure_code(exc), self.namespace, key, owner))
            raise
        raise ExtractionError("EXTRACTION_NO_VALID_OUTCOME")

    @staticmethod
    def _lookup(record, chunk, document, recipe, aliases, *, hit):
        last = record.attempts[-1]
        validated = validate_output(last.text, chunk=chunk, document=document, stop_reason=last.stop_reason, recipe=recipe)
        return ExtractionLookup(record, canonicalize(validated, chunk=chunk, document=document,
                                                     recipe=recipe, aliases=aliases), hit,
                                validated.offset_repairs)


def plan_extraction(corpus: LocalCorpus, cache: ExtractionCache, aliases: AliasDictionary,
                    prices: PriceSheet, *, now) -> dict:
    recipe = extraction_recipe(aliases)
    keys = {raw_cache_key(corpus.tenant_id, c.text, recipe): c.text for c in corpus.chunks}
    hits, misses, blocked = 0, 0, 0
    for text in keys.values():
        try:
            if cache.get(corpus.tenant_id, text, recipe) is None:
                misses += 1
            else:
                hits += 1
        except ExtractionCacheBlocked:
            blocked += 1
    rate = prices.rate_for(recipe.model_id, recipe.invocation_id, recipe.source_region,
                          now=now, live=False)
    bound = rate.quote(TokenUsage(input_tokens=6144, output_tokens=2048))
    return {
        "status": "PLANNED_ONLY", "purpose": "NO_MODEL_CALLS_OR_FUNDING_AUTHORIZATION",
        "corpus_version": corpus.version, "chunks": len(corpus.chunks), "unique_inputs": len(keys),
        "namespace": cache.namespace, "cache_hits": hits, "cache_misses": misses,
        "blocked_inputs": blocked, "extraction_recipe": recipe.model_dump(mode="json"),
        "per_attempt_bound_micro_usd": bound,
        "all_misses_one_attempt_bound_micro_usd": misses * bound,
        "all_misses_with_one_repair_bound_micro_usd": misses * 2 * bound,
        "inference_calls": 0, "semantic_review": "NOT_STARTED",
        "notice": "Bounds use gateway ceilings, not measured tokens. Repair optional, maximum one. "
                  "No repair after ambiguous dispatch. This corpus is a serving candidate, not the full archive.",
    }
