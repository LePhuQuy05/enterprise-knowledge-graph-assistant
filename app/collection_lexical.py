"""Immutable disk-backed BM25 with corpus statistics over an entire collection.

Uses the same title/section/text input and ASCII tokenization as LexicalIndex.
This is global BM25Okapi (k1=1.5, b=.75, epsilon=.25), including the sometimes
negative epsilon * average-IDF floor used by rank_bm25. No shard-local IDFs are
merged or mislabeled as a global lexical ranking.
"""

import heapq
import json
import math
import os
import sqlite3
import uuid
from collections import Counter
from pathlib import Path
from typing import Annotated, Any, Literal, Self

import numpy as np
from pydantic import Field, StringConstraints, model_validator

from app.collections import SnapshotCollection
from app.contracts import Nonempty, Record, Sha256, canonical_json, embedding_input, sha256_bytes
from app.retrieval import lexical_tokens, reciprocal_rank_fusion

MAX_INDEX_BYTES = 512 * 1024 * 1024
MAX_INDEX_MANIFEST_BYTES = 64 * 1024
APPLICATION_ID = 0x4B474C58
IndexVersion = Annotated[str, StringConstraints(pattern=r"^lexical_[0-9a-f]{64}$")]


class CollectionLexicalError(ValueError):
    """Invalid, incompatible or modified lexical artifact."""


class GlobalLexicalManifest(Record):
    schema_version: Literal[1] = 1
    kind: Literal["COLLECTION_GLOBAL_BM25"] = "COLLECTION_GLOBAL_BM25"
    index_version: IndexVersion
    collection_version: Nonempty
    collection_manifest_sha256: Sha256
    corpus_version: Nonempty
    embedding_recipe_sha256: Sha256
    lexical_input_recipe: Literal["title-section-text-v1"] = "title-section-text-v1"
    tokenizer_recipe: Literal["ascii-alnum-lower-v1"] = "ascii-alnum-lower-v1"
    k1: Literal[1.5] = 1.5
    b: Literal[0.75] = 0.75
    epsilon: Literal[0.25] = 0.25
    sqlite_sha256: Sha256
    sqlite_bytes: Annotated[int, Field(gt=0, le=MAX_INDEX_BYTES)]
    chunks: Annotated[int, Field(gt=0)]
    terms: Annotated[int, Field(ge=0)]
    total_tokens: Annotated[int, Field(ge=0)]
    average_document_length: Annotated[float, Field(ge=0)]
    average_idf: float

    @staticmethod
    def version_for(payload: dict[str, Any]) -> str:
        return "lexical_" + sha256_bytes(canonical_json({
            key: value for key, value in payload.items() if key != "index_version"
        }))

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.average_document_length != self.total_tokens / self.chunks:
            raise ValueError("Global document length statistics mismatch")
        if (self.terms == 0) != (self.total_tokens == 0):
            raise ValueError("Empty global vocabulary/token count mismatch")
        if self.terms == 0 and self.average_idf != 0:
            raise ValueError("Empty vocabulary must have zero average IDF")
        if self.index_version != self.version_for(self.model_dump(mode="json")):
            raise ValueError("Global lexical identity mismatch")
        return self


def _file_hash(path: Path, *, limit: int) -> tuple[str, int]:
    import hashlib

    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
        raise CollectionLexicalError("Missing, linked or oversized lexical artifact")
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            size += len(part)
            if size > limit:
                raise CollectionLexicalError("Lexical artifact grew beyond limit")
            digest.update(part)
    return digest.hexdigest(), size


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate lexical JSON key")
        result[key] = value
    return result


def _metadata(collection: SnapshotCollection) -> dict[str, Any]:
    return {
        "collection_version": collection.manifest.collection_version,
        "collection_manifest_sha256": collection.manifest_sha256,
        "corpus_version": collection.manifest.corpus_version,
        "embedding_recipe_sha256": collection.manifest.embedding_recipe.fingerprint,
        "lexical_input_recipe": "title-section-text-v1", "tokenizer_recipe": "ascii-alnum-lower-v1",
        "k1": 1.5, "b": 0.75, "epsilon": 0.25,
    }


class GlobalLexicalIndex:
    def __init__(self, connection: sqlite3.Connection, manifest: GlobalLexicalManifest):
        self._connection = connection
        self.manifest = manifest

    def close(self) -> None:
        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def search(self, question: str, *, k: int = 20) -> list[tuple[str, float]]:
        if type(k) is not int or not 1 <= k <= 20:
            raise CollectionLexicalError("GLOBAL_LEXICAL_CANDIDATE_CAP")
        if not isinstance(question, str) or len(question) > 2000:
            raise CollectionLexicalError("GLOBAL_LEXICAL_QUERY_LIMIT")
        frequencies = Counter(lexical_tokens(question))
        if not frequencies or not self.manifest.terms:
            return []
        scores: dict[str, float] = {}
        for term, query_tf in frequencies.items():
            row = self._connection.execute("SELECT idf FROM terms WHERE term=?", (term,)).fetchone()
            if row is None:
                continue
            idf = row[0]
            rows = self._connection.execute(
                "SELECT p.chunk_id,p.tf,c.document_length FROM postings p "
                "JOIN chunks c ON c.chunk_id=p.chunk_id WHERE p.term=?", (term,),
            )
            for chunk_id, tf, length in rows:
                denominator = tf + 1.5 * (
                    1 - 0.75 + 0.75 * length / self.manifest.average_document_length
                )
                contribution = idf * (tf * (1.5 + 1) / denominator)
                # Repeated query tokens each contribute, matching BM25Okapi.
                # Multiplication can differ from repeated float additions by an ulp.
                scores[chunk_id] = scores.get(chunk_id, 0.0) + query_tf * contribution
        # A matching chunk remains eligible even when its IDF/score is zero or
        # negative. Filtering positive scores would change baseline semantics.
        ranked = heapq.nsmallest(k, scores, key=lambda key: (-scores[key], key))
        return [(key, scores[key]) for key in ranked]

    def shard_for_chunk(self, chunk_id: str) -> str:
        row = self._connection.execute("SELECT shard_id FROM chunks WHERE chunk_id=?", (chunk_id,)).fetchone()
        if row is None:
            raise CollectionLexicalError("GLOBAL_LEXICAL_UNKNOWN_CHUNK")
        return row[0]


def load_global_lexical_index(
    manifest_path: Path, collection: SnapshotCollection, *, trusted_manifest_sha256: str | None = None,
) -> GlobalLexicalIndex:
    try:
        manifest_hash, size = _file_hash(manifest_path, limit=MAX_INDEX_MANIFEST_BYTES)
        with manifest_path.open("rb") as stream:
            raw = stream.read(MAX_INDEX_MANIFEST_BYTES + 1)
        if len(raw) != size or sha256_bytes(raw) != manifest_hash:
            raise CollectionLexicalError("Lexical manifest changed during read")
        if trusted_manifest_sha256 is not None and manifest_hash != trusted_manifest_sha256:
            raise CollectionLexicalError("Trusted lexical manifest checksum mismatch")
        json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_keys)
        manifest = GlobalLexicalManifest.model_validate_json(raw)
        if any(getattr(manifest, key) != value for key, value in _metadata(collection).items()):
            raise CollectionLexicalError("Lexical index collection or recipes mismatch")
        if manifest.chunks != collection.manifest.counts.chunks:
            raise CollectionLexicalError("Lexical index chunk count mismatch")
        database = manifest_path.parent / "index.sqlite3"
        checksum, size = _file_hash(database, limit=MAX_INDEX_BYTES)
        if checksum != manifest.sqlite_sha256 or size != manifest.sqlite_bytes:
            raise CollectionLexicalError("Lexical database checksum/size mismatch")
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro&immutable=1", uri=True)
        try:
            connection.execute("PRAGMA trusted_schema=OFF")
            connection.execute("PRAGMA query_only=ON")
            connection.execute("PRAGMA cache_size=-8192")
            if connection.execute("PRAGMA application_id").fetchone()[0] != APPLICATION_ID:
                raise CollectionLexicalError("Unexpected lexical database application")
            if connection.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise CollectionLexicalError("Unsupported lexical database schema")
            objects = connection.execute("SELECT type,name FROM sqlite_schema WHERE type!='index'").fetchall()
            if set(objects) != {("table", name) for name in ("metadata", "chunks", "terms", "postings")}:
                raise CollectionLexicalError("Unexpected lexical database objects")
            if connection.execute("PRAGMA quick_check(1)").fetchone()[0] != "ok":
                raise CollectionLexicalError("Invalid lexical database pages")
            actual = {key: json.loads(value) for key, value in connection.execute("SELECT key,value FROM metadata")}
            expected = {**_metadata(collection), "chunks": manifest.chunks, "terms": manifest.terms,
                        "total_tokens": manifest.total_tokens,
                        "average_document_length": manifest.average_document_length,
                        "average_idf": manifest.average_idf}
            if actual != expected:
                raise CollectionLexicalError("Lexical database metadata mismatch")
            if connection.execute("SELECT COUNT(*),COALESCE(SUM(document_length),0) FROM chunks").fetchone() != (
                manifest.chunks, manifest.total_tokens,
            ) or connection.execute("SELECT COUNT(*) FROM terms").fetchone()[0] != manifest.terms:
                raise CollectionLexicalError("Lexical database record counts mismatch")
            return GlobalLexicalIndex(connection, manifest)
        except Exception:
            connection.close()
            raise
    except (OSError, UnicodeError, ValueError, sqlite3.Error, RecursionError) as exc:
        if isinstance(exc, CollectionLexicalError):
            raise
        raise CollectionLexicalError("Invalid global lexical artifact") from exc


def build_global_lexical_index(collection: SnapshotCollection, output_root: Path) -> Path:
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    staging = root / f".lexical-{uuid.uuid4().hex}"
    staging.mkdir()
    database = staging / "index.sqlite3"
    connection = None
    try:
        connection = sqlite3.connect(database)
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA page_size=4096")
        connection.execute(f"PRAGMA max_page_count={MAX_INDEX_BYTES // 4096}")
        connection.execute(f"PRAGMA application_id={APPLICATION_ID}")
        connection.execute("PRAGMA user_version=1")
        connection.executescript("""
            CREATE TABLE metadata(key TEXT PRIMARY KEY,value TEXT NOT NULL) WITHOUT ROWID;
            CREATE TABLE chunks(chunk_id TEXT PRIMARY KEY,shard_id TEXT NOT NULL,
                document_version_id TEXT NOT NULL,document_length INTEGER NOT NULL CHECK(document_length>=0)) WITHOUT ROWID;
            CREATE TABLE terms(term TEXT PRIMARY KEY,df INTEGER NOT NULL CHECK(df>0),idf REAL NOT NULL) WITHOUT ROWID;
            CREATE TABLE postings(term TEXT NOT NULL,chunk_id TEXT NOT NULL,
                tf INTEGER NOT NULL CHECK(tf>0),PRIMARY KEY(term,chunk_id),
                FOREIGN KEY(term) REFERENCES terms(term),FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id)) WITHOUT ROWID;
        """)
        total_tokens = chunks = 0
        for ref in collection.manifest.shards:
            snapshot = collection.get_shard(ref.shard_id)
            for chunk in snapshot.content.chunks:
                tokens = lexical_tokens(embedding_input(snapshot.documents[chunk.document_version_id], chunk))
                counts = Counter(tokens)
                connection.execute("INSERT INTO chunks VALUES(?,?,?,?)", (
                    chunk.chunk_id, ref.shard_id, chunk.document_version_id, len(tokens),
                ))
                connection.executemany(
                    "INSERT INTO terms(term,df,idf) VALUES(?,1,0) "
                    "ON CONFLICT(term) DO UPDATE SET df=df+1", ((term,) for term in counts),
                )
                connection.executemany("INSERT INTO postings VALUES(?,?,?)", (
                    (term, chunk.chunk_id, tf) for term, tf in counts.items()
                ))
                total_tokens += len(tokens)
                chunks += 1
            del snapshot
        if chunks != collection.manifest.counts.chunks:
            raise CollectionLexicalError("Collection chunk count changed during index build")
        terms = connection.execute("SELECT COUNT(*) FROM terms").fetchone()[0]
        idf_sum = 0.0
        # Term order is deterministic; float rounding can differ by a final ulp
        # from rank_bm25's dict-insertion order, not from its BM25 definition.
        for term, df in connection.execute("SELECT term,df FROM terms ORDER BY term"):
            idf = math.log(chunks - df + 0.5) - math.log(df + 0.5)
            idf_sum += idf
            connection.execute("UPDATE terms SET idf=? WHERE term=?", (idf, term))
        average_idf = idf_sum / terms if terms else 0.0
        connection.execute("UPDATE terms SET idf=? WHERE idf<0", (0.25 * average_idf,))
        metadata = {**_metadata(collection), "chunks": chunks, "terms": terms,
                    "total_tokens": total_tokens, "average_document_length": total_tokens / chunks,
                    "average_idf": average_idf}
        connection.executemany("INSERT INTO metadata VALUES(?,?)", (
            (key, canonical_json(value).decode("utf-8")) for key, value in sorted(metadata.items())
        ))
        connection.commit()
        connection.close()
        connection = None
        checksum, size = _file_hash(database, limit=MAX_INDEX_BYTES)
        payload = {
            "schema_version": 1, "kind": "COLLECTION_GLOBAL_BM25", **metadata,
            "sqlite_sha256": checksum, "sqlite_bytes": size,
        }
        manifest = GlobalLexicalManifest.model_validate_json(canonical_json({
            **payload, "index_version": GlobalLexicalManifest.version_for(payload),
        }))
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(canonical_json(manifest.model_dump(mode="json")))
            stream.flush()
            os.fsync(stream.fileno())
        with load_global_lexical_index(staging / "manifest.json", collection):
            pass
        destination = root / manifest.index_version
        if destination.exists():
            with load_global_lexical_index(destination / "manifest.json", collection):
                pass
            return destination / "manifest.json"
        try:
            staging.rename(destination)
        except FileExistsError:
            with load_global_lexical_index(destination / "manifest.json", collection):
                pass
        return destination / "manifest.json"
    except (OSError, sqlite3.Error, ValueError) as exc:
        if isinstance(exc, CollectionLexicalError):
            raise
        raise CollectionLexicalError("Global lexical index build failed") from exc
    finally:
        if connection is not None:
            connection.close()
        if staging.exists():
            if staging.resolve().parent != root or not staging.name.startswith(".lexical-"):
                raise CollectionLexicalError("Unsafe lexical staging cleanup")
            for name in ("manifest.json", "index.sqlite3", "index.sqlite3-journal"):
                path = staging / name
                if path.is_file() and not path.is_symlink():
                    path.unlink()
            staging.rmdir()


def retrieve_collection_hybrid(
    collection: SnapshotCollection, lexical_index: GlobalLexicalIndex, question: str,
    query_vector: np.ndarray, *, vector_k: int = 20, lexical_k: int = 20,
) -> list[str]:
    if any(getattr(lexical_index.manifest, key) != value
           for key, value in _metadata(collection).items()):
        raise CollectionLexicalError("Hybrid lexical/vector collection mismatch")
    vectors = collection.search_vector(query_vector, k=vector_k)
    lexical = lexical_index.search(question, k=lexical_k)
    return reciprocal_rank_fusion([
        [hit.chunk_id for hit in vectors], [chunk_id for chunk_id, _ in lexical],
    ])
