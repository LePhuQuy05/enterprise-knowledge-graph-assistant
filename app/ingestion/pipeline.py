"""Durable local jobs and immutable corpus artifacts; no Bedrock/AWS SDK imports."""

import json
import os
import sqlite3
import time
import uuid
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import Chunk, DocumentVersion, TenantId, canonical_json, sha256_bytes, stable_id
from app.ingestion.collector import Collector
from app.ingestion.models import ChunkRecipe, SourceManifest
from app.ingestion.text import ProxyTokenizer, chunk_document, normalize_html, normalizer_version


def _inside(root: Path, relative: str) -> Path:
    if (not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative
            or Path(relative).is_absolute() or any(p in ("", ".", "..") for p in relative.split("/"))):
        raise ValueError("ARTIFACT_PATH_ESCAPES_ROOT")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("ARTIFACT_PATH_ESCAPES_ROOT")
    return path


def _read(root: Path, relative: str, limit=10 * 1024 * 1024) -> bytes:
    path = _inside(root, relative)
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("ARTIFACT_MISSING_OR_OVERSIZED")
    raw = path.read_bytes()
    if len(raw) > limit:
        raise ValueError("ARTIFACT_OVERSIZED")
    return raw


def _write(root: Path, relative: str, raw: bytes, *, immutable=True) -> str:
    path = _inside(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if immutable and path.exists():
        if _read(root, relative, max(len(raw), 1)) != raw:
            raise ValueError("IMMUTABLE_ARTIFACT_CONFLICT")
        return relative
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return relative


class Registry:
    """An OS lock protects one writer; SQLite checkpoints survive process exit."""

    def __init__(self, root: Path):
        self.root = root
        self.lock = None
        self.db = None
        self.recovered_jobs = 0

    def __enter__(self):
        self.lock = (self.root / "jobs.lock").open("a+b")
        self.lock.seek(0, os.SEEK_END)
        if self.lock.tell() == 0:
            self.lock.write(b"0")
            self.lock.flush()
        self.lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.lock.close()
            raise ValueError("INGESTION_WRITER_ALREADY_ACTIVE") from exc
        try:
            self.db = sqlite3.connect(self.root / "jobs.sqlite3", timeout=5)
            self.db.execute("""CREATE TABLE IF NOT EXISTS jobs (
                job_key TEXT PRIMARY KEY, source_url TEXT NOT NULL,
                status TEXT NOT NULL, stage TEXT NOT NULL, attempts INTEGER NOT NULL,
                updated_at TEXT NOT NULL, error_category TEXT, result TEXT)""")
            # An exclusive OS lock proves there is no live writer to recover.
            self.recovered_jobs = self.db.execute(
                "UPDATE jobs SET status='FAILED', error_category='STALE_PROCESS_RECOVERED' "
                "WHERE status='RUNNING'"
            ).rowcount
            self.db.commit()
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.db:
            self.db.close()
        if self.lock and not self.lock.closed:
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()

    def update(self, key, url, status, stage, result=None, error=None):
        stamp = datetime.now(UTC).isoformat()
        self.db.execute("""INSERT INTO jobs VALUES (?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(job_key) DO UPDATE SET status=excluded.status, stage=excluded.stage,
            attempts=jobs.attempts + CASE WHEN excluded.stage='START' THEN 1 ELSE 0 END,
            updated_at=excluded.updated_at, error_category=excluded.error_category,
            result=COALESCE(excluded.result,jobs.result)""",
            (key, url, status, stage, stamp, error,
             json.dumps(result, sort_keys=True) if result else None))
        self.db.commit()


def _jsonl(records) -> bytes:
    return b"".join(canonical_json(r.model_dump(mode="json") if hasattr(r, "model_dump") else r)
                    + b"\n" for r in records)


def _source(root, source, tenant, recipe, tokenizer, collector, registry, *, refresh, offline):
    fingerprint = sha256_bytes(canonical_json({
        "tenant": tenant, "url": source.url, "normalizer": normalizer_version(),
        "chunking": recipe.version(tokenizer.version),
    }))
    registry.update(fingerprint, source.url, "RUNNING", "START")
    fetched_path = "fetched/" + sha256_bytes(source.url.encode()) + ".json"
    if not refresh and _inside(root, fetched_path).is_file():
        fetched = json.loads(_read(root, fetched_path))
        raw = _read(root, fetched["raw_path"])
        if sha256_bytes(raw) != fetched["raw_sha256"] or fetched["requested_url"] != source.url:
            raise ValueError("RAW_CACHE_INTEGRITY_FAILED")
        source_cache_hit = True
    else:
        if offline:
            raise ValueError("SOURCE_NOT_CACHED")
        response = collector.fetch(source.url)
        raw = response.body
        raw_hash = sha256_bytes(raw)
        raw_path = _write(root, f"raw/{raw_hash}.html", raw)
        fetched = {
            "requested_url": source.url, "effective_url": response.effective_url,
            "fetched_at": response.fetched_at.isoformat(), "raw_sha256": raw_hash,
            "raw_path": raw_path, "content_type": response.content_type,
            "http_status": 200, "redirects": list(response.redirects), "attempts": response.attempts,
        }
        # Preserve fetch observations separately from the mutable latest-cache pointer.
        _write(root, f"fetch-history/{uuid.uuid4().hex}.json", canonical_json(fetched))
        _write(root, fetched_path, canonical_json(fetched), immutable=False)
        source_cache_hit = False
    registry.update(fingerprint, source.url, "RUNNING", "FETCHED", fetched)
    normalized = normalize_html(raw, source.expected_title)
    document = DocumentVersion.create(
        tenant_id=tenant, source_uri=source.url, title=normalized.title,
        fetched_at=datetime.fromisoformat(fetched["fetched_at"]), raw_sha256=fetched["raw_sha256"],
        normalizer_version=normalizer_version(), normalized_text_path="pending",
        normalized_text=normalized.text, source_kind="HTML",
    )
    prefix = "normalized/" + document.document_version_id
    document = document.model_copy(update={"normalized_text_path": prefix + "/source.txt"})
    document_path = prefix + "/document.json"
    if _inside(root, document_path).exists():
        original = DocumentVersion.model_validate_json(_read(root, document_path))
        a, b = original.model_dump(exclude={"fetched_at"}), document.model_dump(exclude={"fetched_at"})
        if a != b:
            raise ValueError("DOCUMENT_VERSION_IMMUTABLE_CONFLICT")
        document = original  # Same bytes/version retain their original first-observed timestamp.
    _write(root, prefix + "/source.txt", document.normalized_text.encode("utf-8"))
    _write(root, document_path, document.model_dump_json().encode())
    _write(root, prefix + "/sections.json", canonical_json([
        s.model_dump(mode="json") for s in normalized.sections
    ]))
    registry.update(fingerprint, source.url, "RUNNING", "NORMALIZED")
    chunks = chunk_document(document, normalized.sections, recipe, tokenizer)
    chunk_path = f"chunks/{recipe.version(tokenizer.version)}/{document.document_version_id}.jsonl"
    _write(root, chunk_path, _jsonl(chunks))
    registry.update(fingerprint, source.url, "RUNNING", "CHUNKED")
    # Re-read the actual serialized artifacts before calling the job complete.
    checked_doc = DocumentVersion.model_validate_json(_read(root, document_path))
    checked_chunks = tuple(Chunk.model_validate_json(line) for line in _read(root, chunk_path).split(b"\n") if line)
    if checked_doc != document or checked_chunks != chunks:
        raise ValueError("SERIALIZED_SOURCE_VALIDATION_FAILED")
    result = {
        "document_path": document_path, "chunks_path": chunk_path,
        "raw_path": fetched["raw_path"], "source_cache_hit": source_cache_hit,
        "document_version_id": document.document_version_id,
        "warnings": list(normalized.warnings),
    }
    registry.update(fingerprint, source.url, "COMPLETE", "VALIDATED", result)
    return document, chunks, result


def ingest(manifest_path: Path, output_root: Path, *, tenant_id: TenantId = "demo",
           limit: int | None = None, offline=False, refresh=False, collector=None,
           recipe: ChunkRecipe | None = None) -> dict:
    from pydantic import TypeAdapter

    tenant_id = TypeAdapter(TenantId).validate_python(tenant_id, strict=True)
    if offline and refresh:
        raise ValueError("OFFLINE_REFRESH_CONFLICT")
    if manifest_path.stat().st_size > 1024 * 1024:
        raise ValueError("MANIFEST_SIZE_LIMIT")
    manifest = SourceManifest.model_validate_json(manifest_path.read_bytes())
    if limit is not None and (type(limit) is not int or not 1 <= limit <= len(manifest.sources)):
        raise ValueError("INVALID_INGEST_LIMIT")
    selected = manifest.sources[:limit]
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    collector = collector or Collector()
    recipe = recipe or ChunkRecipe()
    tokenizer = ProxyTokenizer()  # Requires explicit preparation; never downloads implicitly.
    start = time.monotonic()
    documents, all_chunks, inventory, failures, duplicates = [], [], [], [], []
    observations = []
    seen = {}
    with Registry(root) as registry:
        for source in selected:
            key = sha256_bytes(canonical_json({
                "tenant": tenant_id, "url": source.url, "normalizer": normalizer_version(),
                "chunking": recipe.version(tokenizer.version),
            }))
            try:
                doc, chunks, result = _source(
                    root, source, tenant_id, recipe, tokenizer, collector, registry,
                    refresh=refresh, offline=offline,
                )
                observations.append({"source_id": source.source_id, **result})
                item = {**source.model_dump(mode="json"), "document_version_id": doc.document_version_id,
                        "raw_sha256": doc.raw_sha256, "normalized_sha256": doc.normalized_sha256}
                if doc.normalized_sha256 in seen:
                    duplicates.append({"source_id": source.source_id,
                                       "duplicate_of": seen[doc.normalized_sha256]})
                    item["duplicate_of"] = seen[doc.normalized_sha256]
                else:
                    seen[doc.normalized_sha256] = doc.document_version_id
                    documents.append(doc)
                    all_chunks.extend(chunks)
                inventory.append(item)
            except (ValueError, OSError, sqlite3.Error) as exc:
                # Pydantic errors may contain source text; persist only controlled categories.
                category = type(exc).__name__
                safe = str(exc) if re_safe_error(exc) else category
                failures.append({"source_id": source.source_id, "error_category": safe})
                registry.update(key, source.url, "FAILED", "FAILED", error=safe)
        recovered = registry.recovered_jobs
    if collector.robots_bytes:
        _write(root, "robots/" + sha256_bytes(collector.robots_bytes) + ".txt", collector.robots_bytes)
    ordered_docs = sorted(documents, key=lambda d: d.document_version_id)
    ordered_chunks = sorted(all_chunks, key=lambda c: c.chunk_id)
    artifacts = {
        "documents.jsonl": _jsonl(ordered_docs), "chunks.jsonl": _jsonl(ordered_chunks),
        "sources.jsonl": _jsonl(sorted(inventory, key=lambda x: x["source_id"])),
        "recipe.json": canonical_json({
            "manifest_version": manifest.version, "normalizer_version": normalizer_version(),
            "chunk_recipe": recipe.model_dump(), "tokenizer_version": tokenizer.version,
            "tokenizer_asset_sha256": tokenizer.version.rsplit("-", 1)[-1],
        }),
    }
    corpus_version = stable_id("corpus", files={n: sha256_bytes(b) for n, b in artifacts.items()})
    corpus_dir = "corpora/" + corpus_version
    if sum(map(len, artifacts.values())) > 100 * 1024 * 1024 or len(all_chunks) > 5000:
        raise ValueError("CORPUS_HARD_SIZE_LIMIT")
    for name, raw in artifacts.items():
        _write(root, corpus_dir + "/" + name, raw)
    report = {
        "status": "INGESTED" if not failures and len(all_chunks) <= 1000 else "INCOMPLETE",
        "recorded_at": datetime.now(UTC).isoformat(), "tenant_id": tenant_id,
        "corpus_version": corpus_version, "corpus_directory": str(root / corpus_dir),
        "manifest_version": manifest.version, "normalizer_version": normalizer_version(),
        "chunking_version": recipe.version(tokenizer.version),
        "sources_requested": len(selected), "sources_validated": len(inventory),
        "unique_documents": len(documents), "chunks": len(all_chunks),
        "service_counts": dict(Counter(s.service for s in selected)),
        "total_proxy_tokens": sum(c.token_count for c in all_chunks),
        "max_chunk_proxy_tokens": max((c.token_count for c in all_chunks), default=0),
        "token_histogram": {str(a)+"-"+str(b): sum(a <= c.token_count <= b for c in all_chunks)
                            for a, b in [(1, 255), (256, 511), (512, 768)]},
        "operating_target_exceeded": len(all_chunks) > 1000,
        "duplicates": duplicates, "failures": failures, "sources": observations,
        "source_cache_hits": sum(o["source_cache_hit"] for o in observations),
        "http_requests_this_run": collector.request_count,
        "collector_delay_seconds": collector.delay_seconds,
        "recovered_jobs": recovered, "duration_seconds": round(time.monotonic()-start, 3),
        "inference_calls": 0, "semantic_source_review": "NOT_REVIEWED",
        "notice": "Ingestion integrity only; intent labels are not gold evidence or answer-quality results.",
        "artifact_sha256": {name: sha256_bytes(raw) for name, raw in artifacts.items()},
    }
    report_path = "runs/" + uuid.uuid4().hex + "/ingestion_report.json"
    _write(root, report_path, canonical_json(report))
    report["report_path"] = str(root / report_path)
    return report


def re_safe_error(exc) -> bool:
    import re

    return isinstance(exc, ValueError) and bool(re.fullmatch(r"[A-Z][A-Z0-9_]{2,80}", str(exc)))
