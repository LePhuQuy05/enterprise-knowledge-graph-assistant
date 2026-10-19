"""Immutable snapshot writer and bounded, independently reading loader."""

import io
import json
import time
import uuid
import zipfile
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
from pydantic import ValidationError

from app.contracts import (
    DATA_FILES,
    Chunk,
    DocumentVersion,
    EmbeddingRecipe,
    Entity,
    ExtractionRecipe,
    Mention,
    Record,
    RelationAssertion,
    SnapshotCounts,
    SnapshotFile,
    SnapshotManifest,
    canonical_json,
    sha256_bytes,
)

MAX_MANIFEST_BYTES = 64 * 1024
RECORD_TYPES = {
    "documents.jsonl": (DocumentVersion, "document_version_id", "documents"),
    "chunks.jsonl": (Chunk, "chunk_id", "chunks"),
    "entities.jsonl": (Entity, "entity_id", "entities"),
    "mentions.jsonl": (Mention, "mention_id", "mentions"),
    "assertions.jsonl": (RelationAssertion, "assertion_id", "assertions"),
}


class SnapshotError(ValueError):
    """Safe validation message; raw source/provider payloads are not exposed."""


@dataclass(frozen=True)
class SnapshotContent:
    documents: tuple[DocumentVersion, ...]
    chunks: tuple[Chunk, ...]
    entities: tuple[Entity, ...] = ()
    mentions: tuple[Mention, ...] = ()
    assertions: tuple[RelationAssertion, ...] = ()


@dataclass(frozen=True)
class LoadedSnapshot:
    manifest: SnapshotManifest
    manifest_sha256: str
    content: SnapshotContent
    vectors: np.ndarray
    documents: Mapping[str, DocumentVersion]
    chunks: Mapping[str, Chunk]
    entities: Mapping[str, Entity]

    @property
    def chunk_ids(self) -> tuple[str, ...]:
        return tuple(c.chunk_id for c in self.content.chunks)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError("Non-finite JSON constant")


def _parse(model: type[Record], raw: bytes, location: str) -> Any:
    try:
        text = raw.decode("utf-8", errors="strict")
        json.loads(text, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        return model.model_validate_json(text)
    except (UnicodeError, ValueError, ValidationError) as exc:
        raise SnapshotError(f"Invalid schema or JSON in {location}") from exc


def _read_bounded(path: Path, limit: int) -> bytes:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
        raise SnapshotError(f"Missing, unsafe or oversized artifact: {path.name}")
    with path.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise SnapshotError(f"Artifact grew beyond limit: {path.name}")
    return raw


def _jsonl(model: type[Record], raw: bytes, name: str, count: int, key: str) -> tuple:
    # splitlines() would incorrectly split a literal Unicode U+2028 inside a JSON string.
    lines = raw.split(b"\n")
    if lines[-1] == b"":
        lines.pop()
    if len(lines) != count:
        raise SnapshotError(f"Record count mismatch in {name}")
    records = tuple(_parse(model, line, f"{name}:{n}") for n, line in enumerate(lines, 1))
    ids = [getattr(record, key) for record in records]
    if ids != sorted(set(ids)):
        raise SnapshotError(f"IDs must be unique and sorted in {name}")
    return records


def _vectors(raw: bytes, rows: int, uncompressed_size: int, dimension: int = 1024) -> np.ndarray:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            infos = archive.infolist()
            if len(infos) != 1 or infos[0].filename != "embeddings.npy":
                raise SnapshotError("NPZ must contain only embeddings.npy")
            info = infos[0]
            max_npy_bytes = rows * dimension * 4 + 4096
            if info.flag_bits & 1 or info.compress_type not in {
                zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED,
            }:
                raise SnapshotError("Unsupported NPZ member encoding")
            if info.file_size != uncompressed_size or info.file_size > max_npy_bytes:
                raise SnapshotError("NPZ expansion exceeds declared array size")
            with archive.open(info) as stream:
                version = np.lib.format.read_magic(stream)
                readers = {
                    (1, 0): np.lib.format.read_array_header_1_0,
                    (2, 0): np.lib.format.read_array_header_2_0,
                }
                if version not in readers:
                    raise SnapshotError("Unsupported NPY format")
                shape, fortran, dtype = readers[version](stream, max_header_size=4096)
                if shape != (rows, dimension) or dtype != np.dtype("float32") or fortran:
                    raise SnapshotError("NPY shape, dtype or layout mismatch")
                if stream.tell() + rows * dimension * 4 != info.file_size:
                    raise SnapshotError("NPY payload length mismatch")
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            vectors = archive["embeddings"]
    except (OSError, ValueError, EOFError, zipfile.BadZipFile, RuntimeError) as exc:
        if isinstance(exc, SnapshotError):
            raise
        raise SnapshotError("Invalid NPZ artifact") from exc
    if not np.isfinite(vectors).all():
        raise SnapshotError("Vectors contain non-finite values")
    if not np.allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5, rtol=0):
        raise SnapshotError("Vectors must have unit L2 norm")
    vectors.setflags(write=False)
    return vectors


def _validate_links(content: SnapshotContent, manifest: SnapshotManifest) -> None:
    documents = {d.document_version_id: d for d in content.documents}
    chunks = {c.chunk_id: c for c in content.chunks}
    entities = {e.entity_id: e for e in content.entities}
    groups = defaultdict(list)
    if len({d.document_id for d in content.documents}) != len(content.documents):
        raise SnapshotError("Only one document version per document is allowed")
    for category in (content.documents, content.chunks, content.entities,
                     content.mentions, content.assertions):
        if any(record.tenant_id != manifest.tenant_id for record in category):
            raise SnapshotError("Cross-tenant record")
    for document in content.documents:
        if document.normalizer_version != manifest.normalizer_version:
            raise SnapshotError("Normalizer recipe mismatch")
    for chunk in content.chunks:
        document = documents.get(chunk.document_version_id)
        if document is None or document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text:
            raise SnapshotError("Chunk source foreign key or slice mismatch")
        if (chunk.chunking_version, chunk.tokenizer_version) != (
            manifest.chunking_version, manifest.tokenizer_version,
        ):
            raise SnapshotError("Chunk recipe mismatch")
        groups[chunk.document_version_id].append(chunk.position)
    if set(groups) != set(documents):
        raise SnapshotError("A document has no indexed chunks")
    for positions in groups.values():
        if sorted(positions) != list(range(len(positions))):
            raise SnapshotError("Chunk positions must be contiguous and unique within a document")
    version = manifest.extraction_recipe.canonicalization_version
    if any(entity.canonicalization_version != version for entity in content.entities):
        raise SnapshotError("Entity canonicalization version mismatch")

    def span_matches(chunk: Chunk, doc_id: str, start: int, end: int, quote: str) -> bool:
        return (
            chunk.document_version_id == doc_id and chunk.start_char <= start < end <= chunk.end_char
            and documents[doc_id].normalized_text[start:end] == quote
            and chunk.text[start - chunk.start_char:end - chunk.start_char] == quote
        )

    for mention in content.mentions:
        chunk = chunks.get(mention.chunk_id)
        if chunk is None or mention.entity_id not in entities:
            raise SnapshotError("Mention has dangling references")
        if mention.canonicalization_version != version or not span_matches(
            chunk, mention.document_version_id, mention.start_char, mention.end_char, mention.quote
        ):
            raise SnapshotError("Mention recipe or source evidence mismatch")
    recipe = manifest.extraction_recipe
    for assertion in content.assertions:
        chunk = chunks.get(assertion.chunk_id)
        if chunk is None or any(eid not in entities for eid in (
            assertion.subject_id, assertion.object_id,
        )):
            raise SnapshotError("Assertion has dangling references")
        if (assertion.extraction_recipe_sha256, assertion.extraction_model_id,
            assertion.extraction_prompt_version, assertion.ontology_version) != (
            recipe.fingerprint, recipe.model_id, recipe.prompt_version, recipe.ontology_version,
        ):
            raise SnapshotError("Assertion extraction recipe mismatch")
        evidence = assertion.evidence
        if not span_matches(
            chunk, assertion.document_version_id, evidence.start_char, evidence.end_char,
            evidence.quote,
        ) or any(q.text not in chunk.text for q in assertion.qualifiers):
            raise SnapshotError("Assertion source or qualifier slice mismatch")


def load_snapshot(
    directory: Path, *, tenant_id: str, embedding_model_id: str, allow_fixture: bool = False,
    trusted_manifest_sha256: str | None = None,
) -> LoadedSnapshot:
    directory = directory.resolve()
    if not directory.is_dir():
        raise SnapshotError("Snapshot directory unavailable")
    if {p.name for p in directory.iterdir()} != set(DATA_FILES) | {"manifest.json"}:
        raise SnapshotError("Snapshot must contain exactly the seven approved files")
    manifest_raw = _read_bounded(directory / "manifest.json", MAX_MANIFEST_BYTES)
    manifest_hash = sha256_bytes(manifest_raw)
    if trusted_manifest_sha256 is not None and manifest_hash != trusted_manifest_sha256:
        raise SnapshotError("Trusted manifest checksum mismatch")
    manifest = _parse(SnapshotManifest, manifest_raw, "manifest.json")
    if manifest.tenant_id != tenant_id or manifest.embedding_recipe.model_id != embedding_model_id:
        raise SnapshotError("Snapshot tenant or approved embedding model mismatch")
    if manifest.purpose == "FIXTURE" and not allow_fixture:
        raise SnapshotError("Fixture snapshots are disabled in this serving context")
    records = {}
    vector_raw = b""
    vector_size = 0
    for artifact in manifest.files:
        raw = _read_bounded(directory / artifact.name, artifact.byte_size)
        if len(raw) != artifact.byte_size or sha256_bytes(raw) != artifact.sha256:
            raise SnapshotError(f"Artifact checksum/size mismatch: {artifact.name}")
        if artifact.name == "vectors.npz":
            if artifact.record_count != manifest.counts.chunks:
                raise SnapshotError("Vector row count metadata mismatch")
            vector_raw, vector_size = raw, artifact.uncompressed_size
        else:
            model, key, category = RECORD_TYPES[artifact.name]
            if artifact.uncompressed_size != len(raw):
                raise SnapshotError("JSONL byte size mismatch")
            expected = getattr(manifest.counts, category)
            if artifact.record_count != expected:
                raise SnapshotError("Manifest count mismatch")
            records[category] = _jsonl(model, raw, artifact.name, expected, key)
    content = SnapshotContent(**records)
    _validate_links(content, manifest)
    vectors = _vectors(vector_raw, manifest.counts.chunks, vector_size, manifest.embedding_recipe.dimension)
    return LoadedSnapshot(
        manifest, manifest_hash, content, vectors,
        MappingProxyType({d.document_version_id: d for d in content.documents}),
        MappingProxyType({c.chunk_id: c for c in content.chunks}),
        MappingProxyType({e.entity_id: e for e in content.entities}),
    )


def _encode_vectors(matrix: np.ndarray) -> tuple[bytes, int]:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, matrix, allow_pickle=False, version=(1, 0))
    raw = stream.getvalue()
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        info = zipfile.ZipInfo("embeddings.npy", date_time=(1980, 1, 1, 0, 0, 0))
        # ZipInfo defaults create_system to 0 on Windows and 3 on Unix.
        # Pin metadata so identical NPY bytes do not acquire platform-specific IDs.
        info.create_system = 3
        info.external_attr = 0o600 << 16
        info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(info, raw, compresslevel=6)
    return output.getvalue(), len(raw)


def _rename_validated_snapshot(staging: Path, destination: Path, output_root: Path) -> None:
    """Bound transient Windows sharing/access locks; never replace an existing release."""
    for attempt in range(5):
        if staging.resolve().parent != output_root or destination.resolve().parent != output_root:
            raise SnapshotError("Snapshot rename escaped its output root")
        if destination.exists():
            raise SnapshotError("Snapshot target appeared during publication; validate it before retry")
        try:
            staging.rename(destination)
            return
        except PermissionError as exc:
            if getattr(exc, "winerror", None) not in {5, 32} or attempt == 4:
                raise
            time.sleep(0.05 * (2 ** attempt))


def build_snapshot(
    output_root: Path, content: SnapshotContent, vectors: Mapping[str, np.ndarray], *,
    tenant_id: str, purpose: str, extraction_recipe: ExtractionRecipe,
    embedding_recipe: EmbeddingRecipe, corpus_version: str, normalizer_version: str,
    chunking_version: str, tokenizer_version: str, git_commit: str | None = None,
    git_dirty: bool = True,
) -> Path:
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    chunk_ids = sorted(c.chunk_id for c in content.chunks)
    if set(chunk_ids) != set(vectors) or not chunk_ids:
        raise SnapshotError("Vector keys must match the non-empty chunk set")
    arrays = [vectors[key] for key in chunk_ids]
    if any(a.dtype != np.dtype("float32") or a.shape != (embedding_recipe.dimension,) for a in arrays):
        raise SnapshotError("Writer requires float32 vectors matching recipe dimension")
    matrix = np.stack(arrays)
    vector_raw, uncompressed = _encode_vectors(matrix)
    payloads = {"vectors.npz": vector_raw}
    files = []
    counts = {category: len(getattr(content, category)) for _, _, category in RECORD_TYPES.values()}
    for name, (_, key, category) in RECORD_TYPES.items():
        rows = sorted(getattr(content, category), key=lambda r: getattr(r, key))
        payloads[name] = b"".join(canonical_json(row.model_dump(mode="json")) + b"\n" for row in rows)
    for name in DATA_FILES:
        raw = payloads[name]
        count = counts[RECORD_TYPES[name][2]] if name in RECORD_TYPES else len(chunk_ids)
        files.append(SnapshotFile(
            name=name, sha256=sha256_bytes(raw), byte_size=len(raw), record_count=count,
            uncompressed_size=uncompressed if name == "vectors.npz" else len(raw),
        ))
    metadata = dict(
        schema_version=2, identity_recipe="identity-v2", tenant_id=tenant_id, purpose=purpose,
        created_at=datetime.now(UTC), git_commit=git_commit, git_dirty=git_dirty,
        corpus_version=corpus_version, normalizer_version=normalizer_version,
        chunking_version=chunking_version, tokenizer_version=tokenizer_version,
        extraction_recipe=extraction_recipe, embedding_recipe=embedding_recipe,
        counts=SnapshotCounts(**counts), files=tuple(files),
    )
    json_payload = {k: v.model_dump(mode="json") if isinstance(v, Record) else v
                    for k, v in metadata.items() if k != "created_at"}
    json_payload["files"] = [f.model_dump(mode="json") for f in files]
    version = SnapshotManifest.version_for(json_payload)
    manifest = SnapshotManifest(snapshot_version=version, **metadata)
    staging = output_root / f".staging-{uuid.uuid4().hex}"
    staging.mkdir()
    try:
        for name, raw in payloads.items():
            (staging / name).write_bytes(raw)
        (staging / "manifest.json").write_bytes(canonical_json(manifest.model_dump(mode="json")))
        load_snapshot(
            staging, tenant_id=tenant_id, embedding_model_id=embedding_recipe.model_id,
            allow_fixture=purpose == "FIXTURE",
        )
        destination = output_root / version
        if destination.exists():
            existing = load_snapshot(
                destination, tenant_id=tenant_id, embedding_model_id=embedding_recipe.model_id,
                allow_fixture=purpose == "FIXTURE",
            )
            if existing.manifest.snapshot_version != version:
                raise SnapshotError("Immutable snapshot identity conflict")
            return destination
        _rename_validated_snapshot(staging, destination, output_root)
        return destination
    finally:
        if staging.exists():
            if staging.resolve().parent != output_root or not staging.name.startswith(".staging-"):
                raise SnapshotError("Unsafe staging cleanup target")
            # Delete only the seven known files we created, never recursively traverse user data.
            for name in (*DATA_FILES, "manifest.json"):
                path = staging / name
                if path.exists() and path.is_file() and not path.is_symlink():
                    path.unlink()
            staging.rmdir()
