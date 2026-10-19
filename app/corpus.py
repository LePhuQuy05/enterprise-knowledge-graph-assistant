"""Read the local ingestion corpus without models, graph data or source requests."""

import json
from dataclasses import dataclass
from pathlib import Path

from app.contracts import (
    MAX_SNAPSHOT_BYTES,
    Chunk,
    DocumentVersion,
    Record,
    sha256_bytes,
    stable_id,
)


@dataclass(frozen=True)
class LocalCorpus:
    version: str
    tenant_id: str
    documents: tuple[DocumentVersion, ...]
    chunks: tuple[Chunk, ...]
    recipe: dict


def _records(model: type[Record], raw: bytes, limit: int) -> tuple:
    # Split on LF bytes; Unicode line separators inside JSON strings are content.
    records = []
    for line in raw.split(b"\n"):
        if not line:
            continue
        if len(records) >= limit:
            raise ValueError("CORPUS_RECORD_LIMIT")
        records.append(model.model_validate_json(line))
    return tuple(records)


def load_corpus(directory: Path, *, tenant_id: str) -> LocalCorpus:
    """Self-consistency check, not publisher authentication or semantic review."""
    names = ("documents.jsonl", "chunks.jsonl", "sources.jsonl", "recipe.json")
    payloads, total = {}, 0
    for name in names:
        path = directory / name
        size = path.stat().st_size
        total += size
        limit = 65536 if name == "recipe.json" else MAX_SNAPSHOT_BYTES
        if path.is_symlink() or not path.is_file() or total > MAX_SNAPSHOT_BYTES or size > limit:
            raise ValueError("CORPUS_FILE_LIMIT")
        with path.open("rb") as stream:
            raw = stream.read(size + 1)
        if len(raw) != size:
            raise ValueError("CORPUS_CHANGED_DURING_READ")
        payloads[name] = raw
    version = stable_id("corpus", files={n: sha256_bytes(b) for n, b in payloads.items()})
    if directory.name != version:
        raise ValueError("CORPUS_VERSION_MISMATCH")
    docs = _records(DocumentVersion, payloads["documents.jsonl"], 100)
    chunks = _records(Chunk, payloads["chunks.jsonl"], 5000)
    if not 1 <= len(docs) <= 100 or not 1 <= len(chunks) <= 5000:
        raise ValueError("CORPUS_RECORD_LIMIT")
    doc_ids = [d.document_version_id for d in docs]
    chunk_ids = [c.chunk_id for c in chunks]
    if doc_ids != sorted(set(doc_ids)) or chunk_ids != sorted(set(chunk_ids)):
        raise ValueError("CORPUS_ORDER_OR_DUPLICATES")
    if any(r.tenant_id != tenant_id for r in (*docs, *chunks)):
        raise ValueError("CORPUS_TENANT_MISMATCH")
    by_id = {d.document_version_id: d for d in docs}
    groups: dict[str, list[Chunk]] = {key: [] for key in by_id}
    for chunk in chunks:
        doc = by_id.get(chunk.document_version_id)
        if doc is None or chunk.text != doc.normalized_text[chunk.start_char:chunk.end_char]:
            raise ValueError("CORPUS_SOURCE_SLICE_FAILED")
        if chunk.token_count > 768:
            raise ValueError("CORPUS_TOKEN_BOUND_FAILED")
        groups[chunk.document_version_id].append(chunk)
    for key, own in groups.items():
        cursor = 0
        for position, chunk in enumerate(sorted(own, key=lambda c: c.position)):
            if chunk.position != position or chunk.start_char > cursor and (
                by_id[key].normalized_text[cursor:chunk.start_char].strip()
            ):
                raise ValueError("CORPUS_CHUNK_GAP")
            cursor = max(cursor, chunk.end_char)
        if by_id[key].normalized_text[cursor:].strip():
            raise ValueError("CORPUS_TAIL_MISSING")
    recipe = json.loads(payloads["recipe.json"])
    if not isinstance(recipe, dict):
        raise ValueError("CORPUS_RECIPE_INVALID")
    return LocalCorpus(version, tenant_id, docs, chunks, recipe)
