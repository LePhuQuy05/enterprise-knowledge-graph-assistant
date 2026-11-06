"""Local, versioned collections of unchanged v2 snapshots.

This is an offline scaling spike, not a serving release. Opening a collection
reads only the small shard manifests. Vectors and records load on demand into
an LRU cache bounded by shard count and *serialized uncompressed* bytes. These
are cache accounting limits, not a process-RSS guarantee: Python objects and
references retained by callers need separate memory measurements.

This module provides exact vector retrieval across shards. Global BM25 and
Hybrid live in app.collection_lexical with statistics over the full collection;
independent per-shard BM25 scores must not be merged as global lexical scores.
"""

import json
import os
import re
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from threading import RLock
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import AwareDatetime, Field, StringConstraints, ValidationError, model_validator

from app.contracts import (
    MAX_SNAPSHOT_BYTES,
    EmbeddingRecipe,
    ExternalEmbeddingRecipe,
    ExtractionRecipe,
    LocalBgeRecipe,
    Nonempty,
    Record,
    Sha256,
    SnapshotCounts,
    SnapshotManifest,
    TenantId,
    canonical_json,
    sha256_bytes,
)
from app.snapshot import MAX_MANIFEST_BYTES, LoadedSnapshot, SnapshotError, load_snapshot

MAX_COLLECTION_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_COLLECTION_SHARDS = 128
SnapshotVersion = Annotated[str, StringConstraints(pattern=r"^snap_[0-9a-f]{64}$")]
CollectionVersion = Annotated[str, StringConstraints(pattern=r"^collection_[0-9a-f]{64}$")]


class CollectionError(ValueError):
    """Fail-closed collection validation or unsupported operation."""


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or path.as_posix() != value:
        raise ValueError("Shard path must be a normalized local relative POSIX path")
    for part in path.parts:
        if (not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", part)
                or part in {".", ".."} or part.endswith(".")):
            raise ValueError("Unsafe shard path component")
        if part.split(".")[0].upper() in {
            "CON", "PRN", "AUX", "NUL",
            *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10)),
        }:
            raise ValueError("Reserved shard path component")
    return value


class CollectionCounts(Record):
    """Aggregate counts deliberately do not inherit the per-shard 100-doc cap."""

    documents: Annotated[int, Field(ge=1, le=100 * MAX_COLLECTION_SHARDS)]
    chunks: Annotated[int, Field(ge=1, le=5000 * MAX_COLLECTION_SHARDS)]
    entities: Annotated[int, Field(ge=0)]
    mentions: Annotated[int, Field(ge=0)]
    assertions: Annotated[int, Field(ge=0)]


class CollectionShard(Record):
    shard_id: SnapshotVersion
    relative_path: Nonempty
    manifest_sha256: Sha256
    corpus_version: Nonempty
    counts: SnapshotCounts
    # Small document inventories make duplicate source/version detection possible
    # without reading vectors. Chunk identities include their document version.
    document_ids: tuple[Nonempty, ...]
    document_version_ids: tuple[Nonempty, ...]
    labels: tuple[Nonempty, ...] = ()

    @model_validator(mode="after")
    def integrity(self) -> Self:
        _relative_path(self.relative_path)
        for values in (self.document_ids, self.document_version_ids):
            if len(values) != self.counts.documents or tuple(sorted(set(values))) != values:
                raise ValueError("Shard document inventory must be unique, sorted and complete")
        if tuple(sorted(set(self.labels))) != self.labels:
            raise ValueError("Shard labels must be unique and sorted")
        return self


def _corpus_version_for(shards: tuple[CollectionShard, ...] | list[CollectionShard]) -> str:
    return "corpus_collection_" + sha256_bytes(canonical_json(
        sorted(ref.corpus_version for ref in shards),
    ))


class CollectionManifest(Record):
    schema_version: Literal[1] = 1
    kind: Literal["SNAPSHOT_COLLECTION"] = "SNAPSHOT_COLLECTION"
    collection_version: CollectionVersion
    created_at: AwareDatetime
    tenant_id: TenantId
    purpose: Literal["FIXTURE", "DEMO"]
    corpus_version: Nonempty
    normalizer_version: Nonempty
    chunking_version: Nonempty
    tokenizer_version: Nonempty
    extraction_recipe: ExtractionRecipe
    embedding_recipe: EmbeddingRecipe | ExternalEmbeddingRecipe | LocalBgeRecipe
    counts: CollectionCounts
    shards: Annotated[tuple[CollectionShard, ...], Field(min_length=1, max_length=MAX_COLLECTION_SHARDS)]

    @staticmethod
    def version_for(payload: dict[str, Any]) -> str:
        payload = {k: v for k, v in payload.items() if k not in {"collection_version", "created_at"}}
        return "collection_" + sha256_bytes(canonical_json(payload))

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.created_at.utcoffset() != timedelta(0):
            raise ValueError("Collection time must be UTC")
        if isinstance(self.embedding_recipe, (ExternalEmbeddingRecipe, LocalBgeRecipe)):
            if self.embedding_recipe.input_role != "document":
                raise ValueError("Collection needs a document embedding recipe")
        paths = tuple(shard.relative_path for shard in self.shards)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("Shard paths must be unique and sorted")
        if len({shard.shard_id for shard in self.shards}) != len(self.shards):
            raise ValueError("Duplicate snapshot shard")
        for field in ("document_ids", "document_version_ids"):
            values = [value for shard in self.shards for value in getattr(shard, field)]
            if len(values) != len(set(values)):
                raise ValueError("Duplicate source or document version across shards")
        for field, count in self.counts.model_dump().items():
            if sum(getattr(shard.counts, field) for shard in self.shards) != count:
                raise ValueError("Aggregate collection counts do not match shards")
        if self.corpus_version != _corpus_version_for(self.shards):
            raise ValueError("Aggregate corpus version does not match shard versions")
        if self.collection_version != self.version_for(self.model_dump(mode="json")):
            raise ValueError("Collection identity mismatch")
        return self


_COMMON_FIELDS = (
    "tenant_id", "purpose", "normalizer_version", "chunking_version",
    "tokenizer_version", "extraction_recipe", "embedding_recipe",
)


def _common(manifest: SnapshotManifest | CollectionManifest) -> dict[str, Any]:
    return manifest.model_dump(mode="json", include=set(_COMMON_FIELDS))


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result


def _read_record(path: Path, model: type[Record], limit: int) -> tuple[Any, str]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            raise CollectionError("Missing, unsafe or oversized collection artifact")
        with path.open("rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit:
            raise CollectionError("Collection artifact exceeds limit")
        json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicate_keys)
        return model.model_validate_json(raw), sha256_bytes(raw)
    except (OSError, UnicodeError, ValidationError, ValueError, RecursionError) as exc:
        if isinstance(exc, CollectionError):
            raise
        raise CollectionError("Invalid collection artifact") from exc


def _shard_path(root: Path, relative: str) -> Path:
    _relative_path(relative)
    path = root
    for component in PurePosixPath(relative).parts:
        path = path / component
        if path.is_symlink() or path.is_junction():
            raise CollectionError("Linked shard directories are not permitted")
    resolved = path.resolve()
    if not resolved.is_relative_to(root) or not resolved.is_dir():
        raise CollectionError("Shard directory missing or outside collection root")
    return resolved


def _check_header(root: Path, ref: CollectionShard, collection: CollectionManifest) -> Path:
    directory = _shard_path(root, ref.relative_path)
    header, checksum = _read_record(directory / "manifest.json", SnapshotManifest, MAX_MANIFEST_BYTES)
    if checksum != ref.manifest_sha256 or header.snapshot_version != ref.shard_id:
        raise CollectionError("Pinned shard manifest mismatch")
    if (header.counts != ref.counts or header.corpus_version != ref.corpus_version
            or _common(header) != _common(collection)):
        raise CollectionError("Shard counts, tenant or recipes incompatible with collection")
    return directory


def _check_inventory(snapshot: LoadedSnapshot, ref: CollectionShard) -> None:
    if (tuple(sorted(document.document_id for document in snapshot.content.documents))
            != ref.document_ids or tuple(sorted(snapshot.documents)) != ref.document_version_ids):
        raise CollectionError("Shard document inventory mismatch")


@dataclass(frozen=True)
class CollectionHit:
    shard_id: str
    snapshot_version: str
    chunk_id: str
    document_version_id: str
    score: float


@dataclass(frozen=True)
class CollectionCacheInfo:
    resident_shards: int
    accounted_bytes: int
    loads: int
    hits: int
    evictions: int


class SnapshotCollection:
    """Lazy local collection; ranking is exact only over the selected shards."""

    def __init__(
        self, path: Path, manifest: CollectionManifest, manifest_sha256: str, *,
        allow_fixture: bool, cache_shards: int, cache_bytes: int,
    ):
        if type(cache_shards) is not int or not 0 <= cache_shards <= MAX_COLLECTION_SHARDS:
            raise CollectionError("Invalid cache shard cap")
        if type(cache_bytes) is not int or cache_bytes < 1:
            raise CollectionError("Invalid cache byte cap")
        self.path = path
        self.manifest = manifest
        self.manifest_sha256 = manifest_sha256
        self._allow_fixture = allow_fixture
        self._refs = {ref.shard_id: ref for ref in manifest.shards}
        self._cache: OrderedDict[str, tuple[LoadedSnapshot, int]] = OrderedDict()
        self._cache_shards, self._cache_bytes = cache_shards, cache_bytes
        self._accounted_bytes = self._loads = self._hits = self._evictions = 0
        self._lock = RLock()

    @property
    def cache_info(self) -> CollectionCacheInfo:
        with self._lock:
            return CollectionCacheInfo(
                len(self._cache), self._accounted_bytes, self._loads, self._hits, self._evictions,
            )

    def _evict(self) -> None:
        _, (_, size) = self._cache.popitem(last=False)
        self._accounted_bytes -= size
        self._evictions += 1

    def get_shard(self, shard_id: str) -> LoadedSnapshot:
        with self._lock:
            ref = self._refs.get(shard_id)
            if ref is None:
                raise CollectionError("Unknown collection shard")
            if shard_id in self._cache:
                self._hits += 1
                self._cache.move_to_end(shard_id)
                return self._cache[shard_id][0]
            directory = _check_header(self.path.parent, ref, self.manifest)
            # Admission/eviction happens before allocating the next shard.
            header, _ = _read_record(directory / "manifest.json", SnapshotManifest, MAX_MANIFEST_BYTES)
            size = sum(artifact.uncompressed_size for artifact in header.files)
            while self._cache and (
                len(self._cache) >= self._cache_shards
                or self._accounted_bytes + size > self._cache_bytes
            ):
                self._evict()
            try:
                snapshot = load_snapshot(
                    directory, tenant_id=self.manifest.tenant_id,
                    embedding_model_id=self.manifest.embedding_recipe.model_id,
                    allow_fixture=self._allow_fixture, trusted_manifest_sha256=ref.manifest_sha256,
                )
            except SnapshotError as exc:
                raise CollectionError("Shard content validation failed") from exc
            _check_inventory(snapshot, ref)
            self._loads += 1
            if self._cache_shards > 0 and size <= self._cache_bytes:
                self._cache[shard_id] = (snapshot, size)
                self._accounted_bytes += size
            return snapshot

    def validate_all(self) -> CollectionCounts:
        """Re-read every artifact, rather than treating cache hits as a disk audit."""
        with self._lock:
            while self._cache:
                self._evict()
            for ref in self.manifest.shards:
                self.get_shard(ref.shard_id)
        return self.manifest.counts

    def search_vector(
        self, query_vector: np.ndarray, *, k: int = 20, shard_ids: tuple[str, ...] | None = None,
    ) -> list[CollectionHit]:
        if type(k) is not int or not 1 <= k <= 20:
            raise CollectionError("Vector candidate cap must be 1..20")
        if (not isinstance(query_vector, np.ndarray)
                or query_vector.dtype != np.dtype("float32")
                or query_vector.shape != (self.manifest.embedding_recipe.dimension,)
                or not np.isfinite(query_vector).all()
                or not np.isclose(np.linalg.norm(query_vector), 1, atol=1e-5, rtol=0)):
            raise CollectionError("Query vector must be finite normalized float32 matching recipe")
        selected = tuple(self._refs) if shard_ids is None else shard_ids
        if not selected or len(selected) != len(set(selected)) or set(selected) - self._refs.keys():
            raise CollectionError("Selected shards must be nonempty, unique and known")
        best: list[CollectionHit] = []
        for shard_id in sorted(selected):
            snapshot = self.get_shard(shard_id)
            scores = snapshot.vectors @ query_vector
            ids = snapshot.chunk_ids
            # Keeping the best k in each partition cannot discard a global top-k
            # candidate; deterministic tie breaking is the same as the baseline.
            rows = sorted(range(len(ids)), key=lambda i: (-float(scores[i]), ids[i]))[:k]
            best.extend(CollectionHit(
                shard_id, snapshot.manifest.snapshot_version, ids[row],
                snapshot.chunks[ids[row]].document_version_id, float(scores[row]),
            ) for row in rows)
            best.sort(key=lambda hit: (-hit.score, hit.chunk_id, hit.shard_id))
            del best[k:]
            del snapshot
        return best

    def search(
        self, query_vector: np.ndarray, *, mode: str = "vector", k: int = 20,
        shard_ids: tuple[str, ...] | None = None,
    ) -> list[CollectionHit]:
        if mode != "vector":
            raise CollectionError("Collection spike supports vector only; global BM25 is not built")
        return self.search_vector(query_vector, k=k, shard_ids=shard_ids)


def load_collection(
    path: Path, *, tenant_id: str, embedding_model_id: str, allow_fixture: bool = False,
    trusted_manifest_sha256: str | None = None, cache_shards: int = 2,
    cache_bytes: int = 2 * MAX_SNAPSHOT_BYTES,
) -> SnapshotCollection:
    manifest, checksum = _read_record(path, CollectionManifest, MAX_COLLECTION_MANIFEST_BYTES)
    path = path.resolve()
    if trusted_manifest_sha256 is not None and checksum != trusted_manifest_sha256:
        raise CollectionError("Trusted collection manifest checksum mismatch")
    if manifest.tenant_id != tenant_id or manifest.embedding_recipe.model_id != embedding_model_id:
        raise CollectionError("Collection tenant or approved embedding model mismatch")
    if manifest.purpose == "FIXTURE" and not allow_fixture:
        raise CollectionError("Fixture collections are disabled in this context")
    for ref in manifest.shards:
        _check_header(path.parent, ref, manifest)
    return SnapshotCollection(
        path, manifest, checksum, allow_fixture=allow_fixture,
        cache_shards=cache_shards, cache_bytes=cache_bytes,
    )


def build_collection(
    output_root: Path, shard_directories: tuple[Path, ...] | list[Path], *,
    tenant_id: str, embedding_model_id: str, allow_fixture: bool = False,
    shard_labels: dict[str, tuple[str, ...]] | None = None,
) -> Path:
    """Validate all shards once, then publish a pinned root beside their dirs.

    All shard directories must already live below output_root. Existing v2
    snapshots are never rewritten. Publication uses a same-directory hard link
    so concurrent writers cannot replace an already published root manifest.
    """
    root = output_root.resolve()
    if not root.is_dir() or not 1 <= len(shard_directories) <= MAX_COLLECTION_SHARDS:
        raise CollectionError("Collection needs an existing root and 1..128 shards")
    refs = []
    first: dict[str, Any] | None = None
    for directory in shard_directories:
        try:
            relative = directory.absolute().relative_to(root).as_posix()
            directory = _shard_path(root, relative)
            snapshot = load_snapshot(
                directory, tenant_id=tenant_id, embedding_model_id=embedding_model_id,
                allow_fixture=allow_fixture,
            )
        except (ValueError, OSError) as exc:
            raise CollectionError("Invalid or incompatible collection shard") from exc
        common = _common(snapshot.manifest)
        if first is None:
            first = common
        elif common != first:
            raise CollectionError("All shard recipes, tenant and purpose must match")
        refs.append(CollectionShard(
            shard_id=snapshot.manifest.snapshot_version, relative_path=relative,
            manifest_sha256=snapshot.manifest_sha256, counts=snapshot.manifest.counts,
            corpus_version=snapshot.manifest.corpus_version,
            document_ids=tuple(sorted(document.document_id for document in snapshot.content.documents)),
            document_version_ids=tuple(sorted(snapshot.documents)),
            labels=tuple(sorted(set((shard_labels or {}).get(relative, ())))),
        ))
        del snapshot
    counts = {field: sum(getattr(ref.counts, field) for ref in refs)
              for field in CollectionCounts.model_fields}
    metadata = {
        "schema_version": 1, "kind": "SNAPSHOT_COLLECTION", **first, "counts": counts,
        "corpus_version": _corpus_version_for(refs),
        "shards": [ref.model_dump(mode="json") for ref in sorted(refs, key=lambda r: r.relative_path)],
    }
    try:
        manifest = CollectionManifest.model_validate_json(canonical_json({
            **metadata, "collection_version": CollectionManifest.version_for(metadata),
            "created_at": datetime.now(UTC).isoformat(),
        }))
    except ValidationError as exc:
        raise CollectionError("Invalid collection counts or duplicate shard/source inventory") from exc
    raw = canonical_json(manifest.model_dump(mode="json"))
    if len(raw) > MAX_COLLECTION_MANIFEST_BYTES:
        raise CollectionError("Collection manifest exceeds size cap")
    destination = root / f"{manifest.collection_version}.json"
    if destination.exists():
        existing = load_collection(
            destination, tenant_id=tenant_id, embedding_model_id=embedding_model_id,
            allow_fixture=allow_fixture,
        )
        if existing.manifest.collection_version != manifest.collection_version:
            raise CollectionError("Immutable collection identity conflict")
        return destination
    staging = root / f".collection-{uuid.uuid4().hex}.tmp"
    try:
        with staging.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(staging, destination)
        except FileExistsError:
            existing, _ = _read_record(destination, CollectionManifest, MAX_COLLECTION_MANIFEST_BYTES)
            if existing.collection_version != manifest.collection_version:
                raise CollectionError("Concurrent collection identity conflict") from None
    finally:
        staging.unlink(missing_ok=True)
    return destination
