"""Versioned, database-independent records and deterministic identities."""

import hashlib
import json
import re
from datetime import timedelta
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlsplit, urlunsplit

from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    StringConstraints,
    field_validator,
    model_validator,
)

IDENTITY_RECIPE = "identity-v2"
SNAPSHOT_SCHEMA = 2
MAX_SNAPSHOT_BYTES = 100 * 1024 * 1024
Nonempty = Annotated[str, StringConstraints(min_length=1)]
TenantId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")]
Sha256 = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$", min_length=64, max_length=64)]
Nonnegative = Annotated[int, Field(ge=0)]
EntityType = Literal["AWS_SERVICE", "FEATURE", "PROTOCOL", "RESOURCE", "CONCEPT"]
Predicate = Literal["INTEGRATES_WITH", "DEPENDS_ON", "PART_OF", "SUPPORTS", "USES", "REPLACES"]
Mode = Literal["vector", "hybrid", "graph_1hop", "graph_2hop"]
DATA_FILES = (
    "assertions.jsonl",
    "chunks.jsonl",
    "documents.jsonl",
    "entities.jsonl",
    "mentions.jsonl",
    "vectors.npz",
)


def canonical_json(value: Any) -> bytes:
    """Project canonicalization, not an implementation of all of RFC 8785."""
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def text_hash(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def stable_id(prefix: str, **fields: Any) -> str:
    return f"{prefix}_{sha256_bytes(canonical_json({'identity_recipe': IDENTITY_RECIPE, **fields}))}"


def canonical_uri(uri: str) -> str:
    if re.search(r"[\x00-\x20\\]", uri):
        raise ValueError("Source URI contains whitespace, controls or backslashes")
    parsed = urlsplit(uri)
    if parsed.scheme.lower() != "https" or not parsed.hostname or parsed.username is not None:
        raise ValueError("Source URI must be HTTPS without user credentials")
    host = parsed.hostname.encode("idna").decode("ascii").lower()
    if ":" in host:
        host = f"[{host}]"
    port = parsed.port
    if port is not None and port != 443:
        host = f"{host}:{port}"
    return urlunsplit(("https", host, parsed.path or "/", parsed.query, ""))


class Record(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", frozen=True, allow_inf_nan=False)


class DocumentVersion(Record):
    tenant_id: TenantId
    document_id: Nonempty
    document_version_id: Nonempty
    source_uri: Nonempty
    title: Nonempty
    fetched_at: AwareDatetime
    raw_sha256: Sha256
    normalized_sha256: Sha256
    normalizer_version: Nonempty
    normalized_text_path: Nonempty
    normalized_text: Annotated[str, StringConstraints(min_length=1, max_length=1_000_000)]
    source_kind: Literal["HTML", "MARKDOWN", "TEXT"]

    @classmethod
    def create(cls, **data: Any) -> Self:
        data = dict(data)
        data["source_uri"] = canonical_uri(data["source_uri"])
        data["document_id"] = stable_id(
            "doc", tenant_id=data["tenant_id"], canonical_source_uri=data["source_uri"]
        )
        data["document_version_id"] = stable_id(
            "docv",
            document_id=data["document_id"],
            raw_sha256=data["raw_sha256"],
            normalizer_version=data["normalizer_version"],
        )
        data["normalized_sha256"] = text_hash(data["normalized_text"])
        return cls(**data)

    @model_validator(mode="after")
    def integrity(self) -> Self:
        expected_doc = stable_id("doc", tenant_id=self.tenant_id, canonical_source_uri=self.source_uri)
        expected_version = stable_id(
            "docv", document_id=self.document_id, raw_sha256=self.raw_sha256,
            normalizer_version=self.normalizer_version,
        )
        if self.source_uri != canonical_uri(self.source_uri):
            raise ValueError("Source URI is not canonical")
        if (self.document_id, self.document_version_id) != (expected_doc, expected_version):
            raise ValueError("Document identity mismatch")
        if text_hash(self.normalized_text) != self.normalized_sha256 or "\r" in self.normalized_text:
            raise ValueError("Normalized document hash or line endings mismatch")
        if self.fetched_at.utcoffset() != timedelta(0):
            raise ValueError("fetched_at must be UTC")
        return self


class Chunk(Record):
    chunk_id: Nonempty
    tenant_id: TenantId
    document_version_id: Nonempty
    chunking_version: Nonempty
    section_path: tuple[Nonempty, ...]
    position: Nonnegative
    start_char: Nonnegative
    end_char: Annotated[int, Field(gt=0)]
    text: Nonempty
    text_sha256: Sha256
    token_count: Annotated[int, Field(gt=0)]
    tokenizer_version: Nonempty

    @classmethod
    def create(cls, **data: Any) -> Self:
        data = dict(data)
        data["text_sha256"] = text_hash(data["text"])
        data["chunk_id"] = stable_id(
            "chk", **{key: data[key] for key in (
                "tenant_id", "document_version_id", "chunking_version", "start_char", "end_char",
                "text_sha256",
            )}
        )
        return cls(**data)

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.end_char - self.start_char != len(self.text) or not self.text.strip():
            raise ValueError("Chunk span must match Unicode text length")
        if text_hash(self.text) != self.text_sha256:
            raise ValueError("Chunk text hash mismatch")
        expected = stable_id(
            "chk", **self.model_dump(include={
                "tenant_id", "document_version_id", "chunking_version", "start_char", "end_char",
                "text_sha256",
            })
        )
        if self.chunk_id != expected:
            raise ValueError("Chunk identity mismatch")
        return self


class Entity(Record):
    entity_id: Nonempty
    tenant_id: TenantId
    canonical_key: Nonempty
    canonical_name: Nonempty
    entity_type: EntityType
    aliases: tuple[Nonempty, ...]
    canonicalization_version: Nonempty

    @classmethod
    def create(cls, **data: Any) -> Self:
        data = dict(data)
        data["aliases"] = tuple(sorted(set(data.get("aliases", ()))))
        data["entity_id"] = stable_id(
            "ent", **{key: data[key] for key in (
                "tenant_id", "canonical_key", "entity_type", "canonicalization_version",
            )}
        )
        return cls(**data)

    @model_validator(mode="after")
    def integrity(self) -> Self:
        expected = stable_id(
            "ent", **self.model_dump(include={
                "tenant_id", "canonical_key", "entity_type", "canonicalization_version",
            })
        )
        if self.entity_id != expected or self.aliases != tuple(sorted(set(self.aliases))):
            raise ValueError("Entity identity or alias ordering mismatch")
        return self


class EvidenceSpan(Record):
    start_char: Nonnegative
    end_char: Annotated[int, Field(gt=0)]
    quote: Nonempty

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.end_char - self.start_char != len(self.quote):
            raise ValueError("Evidence length must use Unicode code points")
        return self


class Mention(Record):
    mention_id: Nonempty
    tenant_id: TenantId
    entity_id: Nonempty
    chunk_id: Nonempty
    document_version_id: Nonempty
    start_char: Nonnegative
    end_char: Annotated[int, Field(gt=0)]
    quote: Nonempty
    canonicalization_version: Nonempty

    @classmethod
    def create(cls, **data: Any) -> Self:
        data = dict(data)
        data["mention_id"] = cls.expected_id(data)
        return cls(**data)

    @staticmethod
    def expected_id(data: dict[str, Any]) -> str:
        return stable_id(
            "men", **{key: data[key] for key in (
                "tenant_id", "entity_id", "chunk_id", "document_version_id", "start_char",
                "end_char", "quote", "canonicalization_version",
            )}
        )

    @model_validator(mode="after")
    def integrity(self) -> Self:
        EvidenceSpan(start_char=self.start_char, end_char=self.end_char, quote=self.quote)
        if self.mention_id != self.expected_id(self.model_dump()):
            raise ValueError("Mention identity mismatch")
        return self


class Qualifier(Record):
    kind: Literal["REGION", "CONFIGURATION", "VERSION", "LIMITATION", "OTHER"]
    text: Nonempty


class ExtractionRecipe(Record):
    version: Nonempty
    model_id: Nonempty
    invocation_id: Nonempty
    source_region: Nonempty
    parameters: dict[str, JsonValue]
    prompt_version: Nonempty
    output_schema_version: Nonempty
    ontology_version: Nonempty
    canonicalization_version: Nonempty

    @field_validator("parameters")
    @classmethod
    def finite_json(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        canonical_json(value)
        return value

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


class RelationAssertion(Record):
    assertion_id: Nonempty
    tenant_id: TenantId
    subject_id: Nonempty
    predicate: Predicate
    object_id: Nonempty
    chunk_id: Nonempty
    document_version_id: Nonempty
    evidence: EvidenceSpan
    polarity: Literal["POSITIVE", "NEGATIVE"]
    qualifiers: tuple[Qualifier, ...]
    ontology_version: Nonempty
    extraction_prompt_version: Nonempty
    extraction_model_id: Nonempty
    extraction_recipe_sha256: Sha256

    @classmethod
    def create(cls, *, recipe: ExtractionRecipe, **data: Any) -> Self:
        data = dict(data)
        data.update(
            ontology_version=recipe.ontology_version,
            extraction_prompt_version=recipe.prompt_version,
            extraction_model_id=recipe.model_id,
            extraction_recipe_sha256=recipe.fingerprint,
        )
        data["qualifiers"] = cls.sorted_qualifiers(tuple(data.get("qualifiers", ())))
        if data["predicate"] == "INTEGRATES_WITH":
            data["subject_id"], data["object_id"] = sorted((data["subject_id"], data["object_id"]))
        data["assertion_id"] = cls.expected_id(data)
        return cls(**data)

    @staticmethod
    def sorted_qualifiers(value: tuple[Qualifier, ...]) -> tuple[Qualifier, ...]:
        return tuple(Qualifier(kind=k, text=t) for k, t in sorted({(q.kind, q.text) for q in value}))

    @staticmethod
    def expected_id(data: dict[str, Any]) -> str:
        evidence = data["evidence"]
        if isinstance(evidence, EvidenceSpan):
            evidence = evidence.model_dump(mode="json")
        qualifiers = data["qualifiers"]
        return stable_id(
            "ast", **{key: data[key] for key in (
                "tenant_id", "document_version_id", "chunk_id", "subject_id", "predicate",
                "object_id", "polarity", "extraction_recipe_sha256",
            )},
            evidence=evidence,
            qualifiers=[q.model_dump(mode="json") if isinstance(q, Qualifier) else q for q in qualifiers],
        )

    @field_validator("qualifiers")
    @classmethod
    def canonical_qualifiers(cls, value: tuple[Qualifier, ...]) -> tuple[Qualifier, ...]:
        return cls.sorted_qualifiers(value)

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.subject_id == self.object_id:
            raise ValueError("Self-loop assertions are not supported")
        if self.predicate == "INTEGRATES_WITH" and self.subject_id > self.object_id:
            raise ValueError("Symmetric predicate endpoints must be sorted")
        if self.assertion_id != self.expected_id(self.model_dump(mode="json")):
            raise ValueError("Assertion identity mismatch")
        return self


class EmbeddingRecipe(Record):
    version: Nonempty
    model_id: Nonempty
    invocation_id: Nonempty
    source_region: Nonempty
    dimension: Annotated[int, Field(ge=1024, le=1024)] = 1024
    normalized: bool = True
    input_recipe_version: Literal["title-section-text-v1"] = "title-section-text-v1"

    @field_validator("normalized")
    @classmethod
    def must_normalize(cls, value: bool) -> bool:
        if not value:
            raise ValueError("Only normalized vectors are supported")
        return value

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


class ExternalEmbeddingRecipe(EmbeddingRecipe):
    """Versioned role-aware branch; legacy recipes acquire no new default fields."""

    schema_version: Literal[2] = 2
    version: Literal["voyage-role-aware-v1"] = "voyage-role-aware-v1"
    provider: Literal["VOYAGE"] = "VOYAGE"
    model_id: Literal["voyage-4-lite"] = "voyage-4-lite"
    invocation_id: Literal["voyage-4-lite"] = "voyage-4-lite"
    source_region: Literal["external-default"] = "external-default"
    endpoint: Literal["https://api.voyageai.com/v1/embeddings"] = "https://api.voyageai.com/v1/embeddings"
    input_role: Literal["document", "query"]
    input_recipe_version: Literal["title-section-text-v1", "question-text-v1"]
    truncation: Literal[False] = False
    output_dtype: Literal["float"] = "float"
    normalization_version: Literal["local-float32-l2-v1"] = "local-float32-l2-v1"

    @model_validator(mode="after")
    def role_recipe(self) -> Self:
        expected = "title-section-text-v1" if self.input_role == "document" else "question-text-v1"
        if self.input_recipe_version != expected:
            raise ValueError("Embedding role/input recipe mismatch")
        return self


class EmbeddingRecord(Record):
    chunk_id: Nonempty
    embedding_input_sha256: Sha256
    model_id: Nonempty
    dimension: Literal[384, 1024]
    normalized: Literal[True]
    embedding_recipe_version: Nonempty
    vector_row: Nonnegative

    @model_validator(mode="after")
    def model_dimension(self) -> Self:
        expected = 384 if self.model_id == "BAAI/bge-small-en-v1.5" else 1024
        if self.dimension != expected:
            raise ValueError("Embedding record model/dimension mismatch")
        return self


class LocalBgeRecipe(EmbeddingRecipe):
    """Explicit local-encoder variant; existing 1024-dimensional recipes stay unchanged."""

    schema_version: Literal[3] = 3
    version: Literal["bge-onnx-cls-v1"] = "bge-onnx-cls-v1"
    provider: Literal["LOCAL"] = "LOCAL"
    model_id: Literal["BAAI/bge-small-en-v1.5"] = "BAAI/bge-small-en-v1.5"
    invocation_id: Literal["bge-small-en-v1.5-onnx-fp32"] = "bge-small-en-v1.5-onnx-fp32"
    source_region: Literal["local-cpu"] = "local-cpu"
    dimension: Literal[384] = 384
    model_revision: Literal["5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"] = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
    artifact_manifest_sha256: Sha256
    onnxruntime_version: Nonempty
    tokenizers_version: Nonempty
    pooling: Literal["CLS"] = "CLS"
    normalization_version: Literal["float32-l2-v1"] = "float32-l2-v1"
    max_tokens: Literal[512] = 512
    truncation: Literal[False] = False
    input_role: Literal["document", "query"]
    input_recipe_version: Literal["title-section-text-v1", "question-text-v1"]
    query_prefix: Literal["Represent this sentence for searching relevant passages: "] = "Represent this sentence for searching relevant passages: "

    @model_validator(mode="after")
    def role_input(self) -> Self:
        expected = "title-section-text-v1" if self.input_role == "document" else "question-text-v1"
        if self.input_recipe_version != expected:
            raise ValueError("Local BGE role/input mismatch")
        return self


class SnapshotCounts(Record):
    documents: Annotated[int, Field(ge=1, le=100)]
    chunks: Annotated[int, Field(ge=1, le=5000)]
    entities: Annotated[int, Field(ge=0, le=100000)]
    mentions: Annotated[int, Field(ge=0, le=100000)]
    assertions: Annotated[int, Field(ge=0, le=150000)]


class SnapshotFile(Record):
    name: Literal[
        "documents.jsonl", "chunks.jsonl", "entities.jsonl", "mentions.jsonl", "assertions.jsonl",
        "vectors.npz",
    ]
    sha256: Sha256
    byte_size: Annotated[int, Field(ge=0, le=MAX_SNAPSHOT_BYTES)]
    uncompressed_size: Annotated[int, Field(ge=0, le=MAX_SNAPSHOT_BYTES)]
    record_count: Nonnegative


class SnapshotManifest(Record):
    schema_version: Annotated[int, Field(ge=2, le=2)] = SNAPSHOT_SCHEMA
    identity_recipe: Literal["identity-v2"] = IDENTITY_RECIPE
    snapshot_version: Nonempty
    tenant_id: TenantId
    purpose: Literal["FIXTURE", "DEMO"]
    created_at: AwareDatetime
    git_commit: str | None
    git_dirty: bool
    corpus_version: Nonempty
    normalizer_version: Nonempty
    chunking_version: Nonempty
    tokenizer_version: Nonempty
    extraction_recipe: ExtractionRecipe
    embedding_recipe: EmbeddingRecipe | ExternalEmbeddingRecipe | LocalBgeRecipe
    counts: SnapshotCounts
    files: tuple[SnapshotFile, ...]

    @staticmethod
    def version_for(payload: dict[str, Any]) -> str:
        payload = {k: v for k, v in payload.items() if k not in {"snapshot_version", "created_at"}}
        return "snap_" + sha256_bytes(canonical_json(payload))

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.created_at.utcoffset() != timedelta(0):
            raise ValueError("Manifest creation time must be UTC")
        if isinstance(self.embedding_recipe, (ExternalEmbeddingRecipe, LocalBgeRecipe)) and self.embedding_recipe.input_role != "document":
            raise ValueError("Snapshot vectors require a document embedding recipe")
        if tuple(f.name for f in self.files) != DATA_FILES:
            raise ValueError("Manifest must list the six data files exactly once in sorted order")
        if sum(f.uncompressed_size for f in self.files) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Snapshot exceeds uncompressed size limit")
        if sum(f.byte_size for f in self.files) > MAX_SNAPSHOT_BYTES:
            raise ValueError("Snapshot exceeds stored size limit")
        if self.snapshot_version != self.version_for(self.model_dump(mode="json")):
            raise ValueError("Snapshot identity mismatch")
        return self


class QueryRequest(Record):
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    mode: Mode | None = None
    corpus_scope: Literal["snapshot", "aws_official"] = "snapshot"
    answer_language: Literal["auto", "en", "vi"] = "auto"


class Citation(Record):
    citation_id: Nonempty
    chunk_id: Nonempty
    document_version_id: Nonempty
    title: Nonempty
    source_uri: Nonempty
    section_path: tuple[Nonempty, ...]
    quote: Annotated[str, StringConstraints(min_length=1, max_length=600)]


class Claim(Record):
    claim_id: Nonempty
    text: Nonempty
    citation_ids: Annotated[tuple[Nonempty, ...], Field(min_length=1, max_length=8)]


class QueryResponse(Record):
    request_id: Nonempty
    answer: Nonempty
    claims: tuple[Claim, ...]
    citations: tuple[Citation, ...]
    answer_status: Literal["ANSWERED", "PARTIAL", "ABSTAINED"]
    mode_requested: Mode
    mode_used: Mode
    used_graph: bool
    retrieval_backend: Literal["snapshot_numpy_adjacency", "aws_knowledge_mcp"] = "snapshot_numpy_adjacency"
    snapshot_version: Nonempty
    warnings: tuple[Nonempty, ...] = ()
    answer_language: Literal["en", "vi"] = "en"
    evidence_bundle_sha256: Sha256 | None = None

    @model_validator(mode="after")
    def integrity(self) -> Self:
        ids = [c.citation_id for c in self.citations]
        claim_ids = [c.claim_id for c in self.claims]
        referenced = {cid for claim in self.claims for cid in claim.citation_ids}
        if len(set(ids)) != len(ids) or len(set(claim_ids)) != len(claim_ids):
            raise ValueError("Duplicate citation or claim IDs")
        if referenced != set(ids):
            raise ValueError("Claims/citations must resolve without unused or unknown citations")
        if self.answer_status == "ABSTAINED":
            if self.claims or self.citations:
                raise ValueError("Abstention must not contain factual claims")
        elif not self.claims or self.answer != "\n".join(c.text for c in self.claims):
            raise ValueError("Answer must be rendered from cited claims")
        if self.used_graph and self.mode_used not in {"graph_1hop", "graph_2hop"}:
            raise ValueError("Graph contribution conflicts with effective mode")
        return self


def embedding_input(document: DocumentVersion, chunk: Chunk) -> str:
    return document.title + "\n" + " > ".join(chunk.section_path) + "\n\n" + chunk.text


def embedding_cache_key(tenant_id: str, input_text: str, recipe: EmbeddingRecipe) -> str:
    return stable_id(
        "emb", tenant_id=tenant_id, input_sha256=text_hash(input_text), recipe=recipe.fingerprint
    )


def extraction_cache_key(tenant_id: str, input_text: str, recipe: ExtractionRecipe) -> str:
    return stable_id(
        "ext", tenant_id=tenant_id, input_sha256=text_hash(input_text), recipe=recipe.fingerprint
    )
