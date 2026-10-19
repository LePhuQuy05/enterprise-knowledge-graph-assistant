"""Deterministic service/chunk mention graph; no inferred service relationships."""

import re
from dataclasses import dataclass

from app.contracts import Entity, ExtractionRecipe, Mention
from app.extraction import AliasDictionary
from app.snapshot import LoadedSnapshot, SnapshotContent

MATCHER_VERSION = "approved-service-mentions-v1"


@dataclass(frozen=True)
class ServiceMentionBuild:
    content: SnapshotContent
    recipe: ExtractionRecipe
    covered_chunk_ids: tuple[str, ...]
    inspected_chunks: int


def derive_service_mentions(snapshot: LoadedSnapshot, aliases: AliasDictionary) -> ServiceMentionBuild:
    """Match literal aliases in source text, preserving Unicode character offsets.

    A lexical service occurrence is an index for finding a passage, not evidence
    that two services integrate or that a function resource equals its service.
    Graph-bearing input is rejected rather than silently stripping its records.
    """
    if snapshot.content.entities or snapshot.content.mentions or snapshot.content.assertions:
        raise ValueError("SERVICE_MENTIONS_REQUIRE_GRAPH_FREE_BASELINE")
    entries = sorted((e for e in aliases.entries if e.entity_type == "AWS_SERVICE"), key=lambda e: e.canonical_key)
    if not entries or len(entries) > 64 or any(not e.canonical_key.startswith("aws/") for e in entries):
        raise ValueError("SERVICE_DICTIONARY_SCOPE")
    entities = {e.canonical_key: Entity.create(tenant_id=snapshot.manifest.tenant_id,
        canonical_key=e.canonical_key, canonical_name=e.canonical_name, entity_type="AWS_SERVICE",
        aliases=e.aliases, canonicalization_version=aliases.identity) for e in entries}
    patterns = []
    for entry in entries:
        surfaces = sorted(set((*entry.aliases, entry.canonical_name)))
        for surface in surfaces:
            patterns.append((entry.canonical_key, re.compile(r"(?<!\w)(?:" + re.escape(surface) + r")(?!\w)", re.IGNORECASE)))
    if len(patterns) > 512:
        raise ValueError("SERVICE_ALIAS_PATTERN_CAP")
    mentions, covered, present = [], set(), set()
    for chunk in sorted(snapshot.content.chunks, key=lambda c: c.chunk_id):
        document = snapshot.documents.get(chunk.document_version_id)
        if document is None or chunk.tenant_id != snapshot.manifest.tenant_id or (
            document.tenant_id != snapshot.manifest.tenant_id
            or chunk.text != document.normalized_text[chunk.start_char:chunk.end_char]
        ):
            raise ValueError("SERVICE_MENTION_SOURCE_OR_TENANT_MISMATCH")
        hits = sorted((match.start(), match.end(), key) for key, pattern in patterns
                      for match in pattern.finditer(chunk.text))
        # Prefer the longest alias at a position; omit nested overlaps, never fuzzy-merge.
        hits.sort(key=lambda h: (h[0], -(h[1] - h[0]), h[2]))
        previous_end = -1
        for start, end, key in hits:
            if start < previous_end:
                continue
            if len(mentions) >= 50_000:
                raise ValueError("SERVICE_MENTION_RECORD_CAP")
            mentions.append(Mention.create(tenant_id=chunk.tenant_id, entity_id=entities[key].entity_id,
                chunk_id=chunk.chunk_id, document_version_id=chunk.document_version_id,
                start_char=chunk.start_char + start, end_char=chunk.start_char + end,
                quote=chunk.text[start:end], canonicalization_version=aliases.identity))
            previous_end = end
            covered.add(chunk.chunk_id)
            present.add(key)
    content = SnapshotContent(snapshot.content.documents, snapshot.content.chunks,
        tuple(sorted((entities[key] for key in present), key=lambda e: e.entity_id)),
        tuple(sorted(mentions, key=lambda m: m.mention_id)), ())
    recipe = ExtractionRecipe(version=MATCHER_VERSION, model_id="deterministic-literal-alias-matcher",
        invocation_id="local-no-model", source_region="local-cpu", prompt_version="no-prompt",
        output_schema_version="existing-entity-mention-records-v2", ontology_version="service-mentions-only-v1",
        canonicalization_version=aliases.identity, parameters={
            "matcher": "unicode-boundaries-case-insensitive-literal-longest-overlap-v1",
            "max_alias_patterns": 512, "max_mention_records": 50_000, "assertions_enabled": False})
    return ServiceMentionBuild(content, recipe, tuple(sorted(covered)), len(snapshot.content.chunks))
