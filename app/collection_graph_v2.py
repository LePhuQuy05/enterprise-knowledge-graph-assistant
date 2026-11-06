"""Bounded collection graph V2 with indexed, source-exact validation.

Increasing capacity is separate from increasing trust: serving still requires a
trusted release checksum, opt-in, and a separately source-reviewed edge set.
"""

from bisect import bisect_left
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from app.collection_graph import graph_fingerprint
from app.contracts import (
    Entity,
    Mention,
    Record,
    RelationAssertion,
    Sha256,
    canonical_json,
    sha256_bytes,
)
from app.external_http import strict_json
from app.extraction import normalize_alias
from app.generation import GenerationError

MAX_GRAPH_BYTES_V2 = 64 * 1024 * 1024


class CollectionGraphOverlayV2(Record):
    schema_version: Literal[2] = 2
    kind: Literal["COLLECTION_GRAPH_SOURCE_REVIEWED_EXPERIMENTAL_V2"] = "COLLECTION_GRAPH_SOURCE_REVIEWED_EXPERIMENTAL_V2"
    tenant_id: str
    collection_version: str
    collection_manifest_sha256: Sha256
    graph_sha256: Sha256
    source_inventory_sha256: Sha256
    extraction_recipe_sha256: Sha256
    aliases_sha256: Sha256
    curation_sha256: Sha256
    upstream_review_queue_sha256: Sha256
    review_status: Literal["AI_SOURCE_REVIEWED_EXPERIMENTAL"] = "AI_SOURCE_REVIEWED_EXPERIMENTAL"
    human_reviewed: Literal[False] = False
    full_relation_extraction: Literal[False] = False
    complete_source_processing: bool = False
    inventory_units: Annotated[int, Field(ge=1, le=5000)]
    processed_units: Annotated[int, Field(ge=1, le=5000)]
    quality_metrics: None = None
    entities: Annotated[tuple[Entity, ...], Field(max_length=20000)]
    mentions: Annotated[tuple[Mention, ...], Field(max_length=150000)]
    assertions: Annotated[tuple[RelationAssertion, ...], Field(max_length=20000)]

    @model_validator(mode="after")
    def coverage_consistency(self) -> Self:
        if self.processed_units > self.inventory_units or self.complete_source_processing != (self.processed_units == self.inventory_units):
            raise ValueError("GRAPH_V2_SOURCE_COVERAGE_COUNTS")
        return self


@dataclass(frozen=True)
class ValidatedCollectionGraphV2:
    overlay: CollectionGraphOverlayV2
    file_sha256: str


def validate_graph_sources_v2(overlay, collection):
    if (overlay.tenant_id != collection.manifest.tenant_id
            or overlay.collection_manifest_sha256 != collection.manifest_sha256
            or overlay.collection_version != collection.manifest.collection_version):
        raise GenerationError("GRAPH_V2_COLLECTION_BINDING")
    if overlay.graph_sha256 != graph_fingerprint(collection.manifest_sha256,
            overlay.entities, overlay.mentions, overlay.assertions):
        raise GenerationError("GRAPH_V2_CONTENT_FINGERPRINT")
    for rows, field in ((overlay.entities, "entity_id"), (overlay.mentions, "mention_id"), (overlay.assertions, "assertion_id")):
        if len({getattr(row, field) for row in rows}) != len(rows) or any(r.tenant_id != overlay.tenant_id for r in rows):
            raise GenerationError("GRAPH_V2_RECORD_ID_OR_TENANT")
    entities = {e.entity_id: e for e in overlay.entities}
    aliases = {key: {normalize_alias(v) for v in (*e.aliases, e.canonical_name)} for key, e in entities.items()}
    docs, chunks = {}, {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        docs.update(shard.documents)
        chunks.update(shard.chunks)
    mentions = defaultdict(list)
    for m in overlay.mentions:
        chunk = chunks.get(m.chunk_id)
        if chunk is None or m.entity_id not in entities or chunk.document_version_id != m.document_version_id:
            raise GenerationError("GRAPH_V2_MENTION_BINDING")
        if (not chunk.start_char <= m.start_char < m.end_char <= chunk.end_char
                or docs[m.document_version_id].normalized_text[m.start_char:m.end_char] != m.quote
                or normalize_alias(m.quote) not in aliases[m.entity_id]):
            raise GenerationError("GRAPH_V2_MENTION_SOURCE_OR_ALIAS")
        mentions[(m.chunk_id, m.entity_id)].append((m.start_char, m.end_char))
    for spans in mentions.values():
        spans.sort()
    # Index by BOTH source chunk and endpoint. V1 scanned all mentions twice
    # per assertion; V2 first narrows the source and bisects to the evidence.
    for a in overlay.assertions:
        chunk, evidence = chunks.get(a.chunk_id), a.evidence
        if (chunk is None or chunk.document_version_id != a.document_version_id
                or {a.subject_id, a.object_id} - entities.keys()):
            raise GenerationError("GRAPH_V2_ASSERTION_BINDING")
        if (not chunk.start_char <= evidence.start_char < evidence.end_char <= chunk.end_char
                or docs[a.document_version_id].normalized_text[evidence.start_char:evidence.end_char] != evidence.quote
                or any(q.text not in evidence.quote for q in a.qualifiers)):
            raise GenerationError("GRAPH_V2_ASSERTION_SOURCE")
        for endpoint in (a.subject_id, a.object_id):
            spans = mentions.get((a.chunk_id, endpoint), ())
            start = bisect_left(spans, (evidence.start_char, -1))
            end = bisect_left(spans, (evidence.end_char, -1))
            if not any(right <= evidence.end_char for left, right in spans[start:end]):
                raise GenerationError("GRAPH_V2_ENDPOINT_SOURCE")


def load_collection_graph_v2(path: Path, collection, *, trusted_sha256: str, allow_experimental=False):
    if allow_experimental is not True:
        raise GenerationError("GRAPH_V2_EXPERIMENTAL_OPT_IN_REQUIRED")
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= MAX_GRAPH_BYTES_V2:
        raise GenerationError("GRAPH_V2_FILE_BOUND")
    with path.open("rb") as handle:
        raw = handle.read(MAX_GRAPH_BYTES_V2 + 1)
    if len(raw) > MAX_GRAPH_BYTES_V2 or sha256_bytes(raw) != trusted_sha256:
        raise GenerationError("GRAPH_V2_FILE_SIZE_OR_CHECKSUM")
    overlay = CollectionGraphOverlayV2.model_validate_json(canonical_json(strict_json(raw)))
    validate_graph_sources_v2(overlay, collection)
    return ValidatedCollectionGraphV2(overlay, trusted_sha256)
