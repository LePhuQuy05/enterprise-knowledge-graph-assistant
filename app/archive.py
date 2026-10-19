"""Resumable, disk-backed archive of official AWS documentation, independent of MVP caps.

Sitemap/link inventory is not a claim that all AWS knowledge was downloaded. No
model client, embedding, cloud upload or inference is created by this module.
"""

import gzip
import io
import json
import os
import re
import shutil
import sqlite3
import time
import xml.etree.ElementTree as ET
from collections import deque
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import quote, unquote, urljoin, urlsplit, urlunsplit
from uuid import uuid4

from pydantic import Field, field_validator

from app.contracts import Record, canonical_json, canonical_uri, sha256_bytes, stable_id, text_hash
from app.ingestion.collector import Collector
from app.ingestion.text import normalize_html, normalizer_version

MIB = 1024 * 1024
XML_LIMIT = 50 * MIB
PAGE_LIMIT = 10 * MIB
NORMALIZED_LIMIT = 48 * MIB
ROOT_SITEMAP = "https://docs.aws.amazon.com/sitemap_index.xml"
ROOT_LLMS = "https://docs.aws.amazon.com/llms.txt"
FORMATS = Literal["html", "markdown", "text", "pdf"]


def archive_url(value: str) -> str:
    value = canonical_uri(value)
    parsed = urlsplit(value)
    if parsed.hostname != "docs.aws.amazon.com" or parsed.port not in (None, 443):
        raise ValueError("ARCHIVE_HOST_NOT_ALLOWED")
    if parsed.query or len(value) > 2048 or re.search(r"%(?![a-fA-F0-9]{2})", parsed.path):
        raise ValueError("ARCHIVE_QUERY_OR_URL_LIMIT")
    path = unquote(parsed.path, errors="strict")
    if re.search(r"[\x00-\x20\\]", path) or any(p in (".", "..") for p in path.split("/")):
        raise ValueError("ARCHIVE_UNSAFE_PATH")
    # Inspect decoded characters, but preserve encoded reserved characters: %2F
    # and / need not identify the same HTTP resource.
    path = quote(parsed.path, safe="/:@-._~!$&'()*+,;=%")
    path = re.sub(r"%[a-fA-F0-9]{2}", lambda m: m.group().upper(), path)
    return urlunsplit(("https", "docs.aws.amazon.com", path or "/", "", ""))


def location(url: str) -> tuple[str, str]:
    parts = urlsplit(url).path.strip("/").split("/")
    if parts[0] == "pdfs" and len(parts) > 1:
        parts.pop(0)
    language = "en"
    if re.fullmatch(r"[a-zA-Z]{2}_[a-zA-Z]{2}", parts[0]):
        language = parts.pop(0).split("_")[0].lower()
    return language, parts[0] if parts and parts[0] else "_root"


def resource_format(url: str) -> str:
    path = urlsplit(url).path.lower()
    for suffix, kind in ((".pdf", "pdf"), (".md", "markdown"), (".markdown", "markdown"),
                         (".txt", "text")):
        if path.endswith(suffix):
            return kind
    return "html" if path.endswith((".html", ".htm", "/")) else "other"


def llms_links(raw: bytes, base: str) -> tuple[str, ...]:
    """A directory of official sources, not a complete content dump or instructions."""
    if len(raw) > PAGE_LIMIT:
        raise ValueError("LLMS_DIRECTORY_SIZE_LIMIT")
    text = raw.decode("utf-8", errors="strict")
    if "\x00" in text:
        raise ValueError("LLMS_DIRECTORY_INVALID")
    found = set()
    for candidate in re.findall(r"\[[^\]\n]*\]\(([^\s)]+)(?:\s+[^)]*)?\)", text):
        try:
            target = archive_url(urljoin(base, candidate.strip("<>")))
        except ValueError:
            continue
        if target != base and (resource_format(target) == "markdown" or target.endswith("/llms.txt")):
            found.add(target)
        if len(found) > 50_000:
            raise ValueError("LLMS_DIRECTORY_ENTRY_LIMIT")
    if not found:
        raise ValueError("LLMS_DIRECTORY_EMPTY")
    return tuple(sorted(found))


class ArchivePolicy(Record):
    version: str
    root_sitemap: str = ROOT_SITEMAP
    languages: Annotated[tuple[str, ...], Field(min_length=1)] = ("en",)
    formats: Annotated[tuple[FORMATS, ...], Field(min_length=1)] = ("html", "markdown", "text", "pdf")
    follow_document_links: bool = True
    max_storage_bytes: Annotated[int, Field(ge=64 * MIB)] = 8 * 1024 * MIB
    min_free_disk_bytes: Annotated[int, Field(ge=0)] = 2 * 1024 * MIB
    max_page_bytes: Annotated[int, Field(ge=MIB, le=100 * MIB)] = PAGE_LIMIT
    max_pdf_bytes: Annotated[int, Field(ge=MIB, le=512 * MIB)] = 100 * MIB
    priority_sitemaps: tuple[str, ...] = ()

    @field_validator("root_sitemap")
    @classmethod
    def valid_root(cls, value):
        return archive_url(value)

    @field_validator("languages")
    @classmethod
    def valid_languages(cls, value):
        if any(v != "*" and not re.fullmatch(r"[a-z]{2}", v) for v in value):
            raise ValueError("ARCHIVE_LANGUAGE_INVALID")
        return value

    @field_validator("priority_sitemaps")
    @classmethod
    def valid_priorities(cls, value):
        return tuple(archive_url(v) for v in value)

    @property
    def scope_json(self):
        return canonical_json(self.model_dump(mode="json", include={
            "version", "root_sitemap", "languages", "formats", "follow_document_links",
        })).decode("utf-8")

    @property
    def fingerprint(self):
        return sha256_bytes(self.scope_json.encode("utf-8"))


def parse_sitemap(raw: bytes) -> tuple[str, list[tuple[str, str | None]]]:
    if raw.startswith(b"\x1f\x8b"):
        try:
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                raw = stream.read(XML_LIMIT + 1)
        except (OSError, EOFError) as exc:
            raise ValueError("SITEMAP_GZIP_INVALID") from exc
    if len(raw) > XML_LIMIT:
        raise ValueError("SITEMAP_EXPANSION_LIMIT")
    text = raw.decode("utf-8", errors="strict")
    if "\x00" in text or "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        raise ValueError("SITEMAP_DTD_OR_ENCODING_REJECTED")
    entries, root_kind = [], None
    for event, element in ET.iterparse(io.BytesIO(raw), events=("start", "end")):
        tag = element.tag.rsplit("}", 1)[-1]
        if root_kind is None:
            root_kind = tag
            if root_kind not in {"sitemapindex", "urlset"}:
                raise ValueError("SITEMAP_ROOT_INVALID")
        if event == "end" and tag in {"sitemap", "url"}:
            values = {e.tag.rsplit("}", 1)[-1]: e.text for e in element}
            loc = values.get("loc")
            if not loc or len(entries) >= 50_000:
                raise ValueError("SITEMAP_ENTRY_LIMIT_OR_MISSING_LOC")
            entries.append((loc.strip(), values.get("lastmod")))
            element.clear()
    return root_kind, entries


class Archive:
    def __init__(self, root: Path, policy: ArchivePolicy, *, collector=None, read_only=False):
        self.root, self.policy = root.resolve(), policy
        self.collector = collector or Collector()
        self.read_only = read_only
        self.lock = self.db = None
        self.original_transport = None
        self.robots_recorded = False
        self.blob_bytes = 0

    def __enter__(self):
        if not self.read_only:
            self.root.mkdir(parents=True, exist_ok=True)
            self.lock = (self.root / "archive.lock").open("a+b")
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
                raise ValueError("ARCHIVE_WRITER_ALREADY_ACTIVE") from exc
        try:
            path = self.root / "archive.sqlite3"
            self.db = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=5) if (
                self.read_only) else sqlite3.connect(path, timeout=5)
            self.db.row_factory = sqlite3.Row
            if not self.read_only:
                self._create()
            row = self.db.execute("SELECT value FROM metadata WHERE key='policy'").fetchone()
            if row is None or row[0] != self.policy.scope_json:
                raise ValueError("ARCHIVE_POLICY_MISMATCH_USE_SEPARATE_ROOT")
            if not self.read_only:
                self.db.execute("UPDATE resources SET state='QUEUED' WHERE state='FETCHING'")
                self.db.commit()
                self.blob_bytes = sum(p.stat().st_size for p in (self.root / "blobs").glob("*/*.gz"))
                marker = self.root / "request-clock.json"
                if marker.exists() and self.collector.last_start is None:
                    if marker.stat().st_size > 4096:
                        raise ValueError("ARCHIVE_REQUEST_CLOCK_INVALID")
                    try:
                        value = json.loads(marker.read_bytes())
                        previous = datetime.fromisoformat(value["at"])
                        delay = value["delay_seconds"]
                    except (KeyError, TypeError, ValueError) as exc:
                        raise ValueError("ARCHIVE_REQUEST_CLOCK_INVALID") from exc
                    if previous.utcoffset() is None or type(delay) not in (int, float) or not 5 <= delay <= 60:
                        raise ValueError("ARCHIVE_REQUEST_CLOCK_INVALID")
                    elapsed = max(0.0, (datetime.now(UTC) - previous).total_seconds())
                    self.collector.delay_seconds = float(delay)
                    self.collector.last_start = self.collector.clock() - min(elapsed, 60.0)
                self.original_transport = self.collector.transport

                def transport(url, limit, timeout):
                    temporary = marker.with_suffix(".tmp")
                    temporary.write_bytes(canonical_json({"at": datetime.now(UTC).isoformat(),
                                                          "delay_seconds": self.collector.delay_seconds}))
                    temporary.replace(marker)
                    return self.original_transport(url, limit, timeout)

                self.collector.transport = transport
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *args):
        if self.original_transport is not None:
            self.collector.transport = self.original_transport
        if self.db is not None:
            self.db.close()
        if self.lock is not None and not self.lock.closed:
            self.lock.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.lock.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.lock.fileno(), fcntl.LOCK_UN)
            self.lock.close()

    def _create(self):
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS resources (
                url TEXT PRIMARY KEY, kind TEXT NOT NULL, family TEXT NOT NULL,
                language TEXT NOT NULL, format TEXT NOT NULL, priority INTEGER NOT NULL,
                state TEXT NOT NULL, reason TEXT, discovered_from TEXT, lastmod TEXT,
                raw_sha256 TEXT, document_sha256 TEXT, effective_url TEXT,
                fetched_at TEXT, content_type TEXT, attempts INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS resource_queue ON resources(kind,state,priority,url);
            CREATE INDEX IF NOT EXISTS page_queue ON resources(kind,state,family,url);
            CREATE TABLE IF NOT EXISTS versions (
                version_id TEXT PRIMARY KEY, url TEXT NOT NULL, document_sha256 TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS fetches (
                id INTEGER PRIMARY KEY, url TEXT, effective_url TEXT, fetched_at TEXT,
                raw_sha256 TEXT, content_type TEXT, attempts INTEGER);
            CREATE TABLE IF NOT EXISTS rejections (
                source TEXT, candidate_sha256 TEXT, reason TEXT,
                PRIMARY KEY(source,candidate_sha256));
            CREATE TABLE IF NOT EXISTS discovery_edges (
                parent TEXT, url TEXT, lastmod TEXT, PRIMARY KEY(parent,url));
            CREATE TABLE IF NOT EXISTS robots_snapshots (
                checked_at TEXT, raw_sha256 TEXT, delay_seconds REAL,
                PRIMARY KEY(checked_at,raw_sha256));
            CREATE TABLE IF NOT EXISTS html_redirects (url TEXT PRIMARY KEY, target TEXT NOT NULL);
            CREATE VIRTUAL TABLE IF NOT EXISTS page_text USING fts5(title,body,tokenize='unicode61');
        """)
        current = self.db.execute("SELECT value FROM metadata WHERE key='policy'").fetchone()
        if current is not None and current[0] != self.policy.scope_json:
            raise ValueError("ARCHIVE_POLICY_MISMATCH_USE_SEPARATE_ROOT")
        self.db.execute("INSERT OR IGNORE INTO metadata VALUES ('policy',?)", (self.policy.scope_json,))
        self.db.execute("INSERT OR REPLACE INTO metadata VALUES ('operational_policy',?)", (self.policy.model_dump_json(),))
        self._add(self.policy.root_sitemap, "SITEMAP", "ROOT", None)
        self.db.commit()

    def _add(self, candidate, kind, source, lastmod):
        try:
            url = archive_url(candidate)
        except ValueError as exc:
            self.db.execute("INSERT OR IGNORE INTO rejections VALUES (?,?,?)",
                            (source, text_hash(candidate), str(exc)))
            return
        language, family = location(url)
        fmt = "sitemap" if kind == "SITEMAP" else resource_format(url)
        reason = None
        if "*" not in self.policy.languages and language not in self.policy.languages:
            reason = "LANGUAGE_EXCLUDED"
        elif kind == "PAGE" and fmt not in self.policy.formats:
            reason = "FORMAT_EXCLUDED"
        elif kind == "SITEMAP" and not urlsplit(url).path.endswith((".xml", ".xml.gz")):
            reason = "SITEMAP_FORMAT_EXCLUDED"
        priority = self.policy.priority_sitemaps.index(url) if url in self.policy.priority_sitemaps else 1000
        if url == ROOT_LLMS or url.endswith("/llms.txt"):
            priority = -100
        if url == self.policy.root_sitemap:
            priority = -1
        previous = self.db.execute("SELECT lastmod,discovered_from,state FROM resources WHERE url=?", (url,)).fetchone()
        self.db.execute("""INSERT OR IGNORE INTO resources
            (url,kind,family,language,format,priority,state,reason,discovered_from,lastmod)
            VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (url, kind, family, language, fmt, priority, "EXCLUDED" if reason else "QUEUED",
             reason, source, lastmod))
        self.db.execute("INSERT OR IGNORE INTO discovery_edges VALUES (?,?,?)", (source, url, lastmod))
        # Only the first discovery parent can update the lastmod hint. Otherwise
        # cycles/inconsistent hints from two indexes could requeue each other forever.
        if previous is not None and lastmod is not None and source == previous["discovered_from"]:
            if previous["lastmod"] is not None and previous["lastmod"] != lastmod and (
                previous["state"] not in {"EXCLUDED", "FETCHING"}
            ):
                self.db.execute("UPDATE resources SET lastmod=?,state='QUEUED',reason=NULL,raw_sha256=NULL WHERE url=?",
                                (lastmod, url))
            elif previous["lastmod"] is None:
                self.db.execute("UPDATE resources SET lastmod=? WHERE url=?", (lastmod, url))

    def _guard(self, extra=0):
        db_bytes = sum(p.stat().st_size for p in self.root.glob("archive.sqlite3*"))
        if self.blob_bytes + db_bytes + extra > self.policy.max_storage_bytes:
            raise ValueError("ARCHIVE_STORAGE_BUDGET_REACHED")
        if shutil.disk_usage(self.root).free < self.policy.min_free_disk_bytes + extra:
            raise ValueError("ARCHIVE_FREE_DISK_GUARD")

    def _blob_path(self, digest):
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError("ARCHIVE_BLOB_HASH_INVALID")
        path = (self.root / "blobs" / digest[:2] / (digest + ".gz")).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("ARCHIVE_BLOB_PATH_ESCAPE")
        return path

    def read_blob(self, digest, limit=NORMALIZED_LIMIT):
        path = self._blob_path(digest)
        with gzip.open(path, "rb") as stream:
            raw = stream.read(limit + 1)
        if len(raw) > limit or sha256_bytes(raw) != digest:
            raise ValueError("ARCHIVE_BLOB_INTEGRITY_FAILED")
        return raw

    def _put(self, raw):
        digest = sha256_bytes(raw)
        path = self._blob_path(digest)
        if path.exists():
            if self.read_blob(digest, max(len(raw), 1)) != raw:
                raise ValueError("ARCHIVE_BLOB_CONFLICT")
            return digest
        compressed = gzip.compress(raw, mtime=0)
        self._guard(len(compressed) + MIB)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(compressed)
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        self.blob_bytes += len(compressed)
        return digest

    def _fetch(self, row):
        limit = XML_LIMIT if row["kind"] == "SITEMAP" else (
            self.policy.max_pdf_bytes if row["format"] == "pdf" else self.policy.max_page_bytes)
        if row["raw_sha256"] is not None:
            return self.read_blob(row["raw_sha256"], limit), row
        formats = {"sitemap": ("text/xml", "application/xml", "application/gzip", "application/octet-stream"),
                   "html": ("text/html",), "pdf": ("application/pdf",),
                   "markdown": ("text/plain", "text/markdown", "application/octet-stream"),
                   "text": ("text/plain",)}
        def validate(value):
            value = archive_url(value)
            language, _ = location(value)
            if "*" not in self.policy.languages and language not in self.policy.languages:
                raise ValueError("ARCHIVE_REDIRECT_LANGUAGE_EXCLUDED")
            return value

        result = self.collector.fetch_resource(
            row["url"], validator=validate, content_types=formats[row["format"]],
            max_bytes=limit,
        )
        if not self.robots_recorded and self.collector.robots_bytes is not None:
            digest = self._put(self.collector.robots_bytes)
            self.db.execute("INSERT OR IGNORE INTO robots_snapshots VALUES (?,?,?)",
                            (self.collector.robots_fetched_at.isoformat(), digest, self.collector.delay_seconds))
            self.robots_recorded = True
        raw_hash = self._put(result.body)
        stamp = result.fetched_at.isoformat()
        self.db.execute("""INSERT INTO fetches
            (url,effective_url,fetched_at,raw_sha256,content_type,attempts) VALUES (?,?,?,?,?,?)""",
            (row["url"], result.effective_url, stamp, raw_hash, result.content_type, result.attempts))
        self.db.execute("""UPDATE resources SET raw_sha256=?,effective_url=?,fetched_at=?,
            content_type=?,attempts=attempts+? WHERE url=?""",
            (raw_hash, result.effective_url, stamp, result.content_type, result.attempts, row["url"]))
        self.db.commit()
        return result.body, self.db.execute("SELECT rowid,* FROM resources WHERE url=?", (row["url"],)).fetchone()

    def _page(self, row, raw):
        if row["url"] == ROOT_LLMS or row["url"].endswith("/llms.txt"):
            links = llms_links(raw, row["effective_url"])
            self._guard(sum(len(link.encode("utf-8")) + 1024 for link in links) * 2)
            for link in links:
                self._add(link, "PAGE", row["url"], None)
            self.db.execute("UPDATE resources SET state='DIRECTORY',reason='LLMS_SOURCE_DIRECTORY' WHERE url=?", (row["url"],))
            return
        fmt = row["format"]
        if fmt == "html" and self.policy.follow_document_links:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(raw.decode("utf-8", errors="strict"), "html.parser")
            links = {urljoin(row["effective_url"], a["href"]) for a in soup.find_all("a", href=True)}
            if len(links) > 50_000:
                raise ValueError("ARCHIVE_LINK_COUNT_LIMIT")
            for link in sorted(links):
                self._add(link, "PAGE", row["url"], None)
            has_article = soup.find(id="main-col-body") or soup.find("main") or soup.find("article")
            refresh = next((m for m in soup.find_all("meta") if
                            m.get("http-equiv", "").lower() == "refresh"), None)
            if not has_article and refresh is not None:
                match = re.fullmatch(r"\s*\d+(?:\.\d+)?\s*;\s*url\s*=\s*(.+?)\s*",
                                     refresh.get("content", ""), flags=re.I)
                if match is None:
                    raise ValueError("ARCHIVE_HTML_REFRESH_INVALID")
                target = archive_url(urljoin(row["effective_url"], match[1].strip("'\"")))
                if target in {row["url"], row["effective_url"]}:
                    raise ValueError("ARCHIVE_HTML_REFRESH_LOOP")
                visited, next_target = {row["url"]}, target
                while True:
                    if next_target in visited or len(visited) > 32:
                        raise ValueError("ARCHIVE_HTML_REFRESH_LOOP")
                    visited.add(next_target)
                    previous = self.db.execute("SELECT h.target FROM html_redirects AS h "
                                               "JOIN resources AS r ON r.url=h.url "
                                               "WHERE h.url=? AND r.state='REDIRECTED'", (next_target,)).fetchone()
                    if previous is None:
                        break
                    next_target = previous[0]
                self._add(target, "PAGE", row["url"], None)
                self.db.execute("INSERT OR REPLACE INTO html_redirects VALUES (?,?)", (row["url"], target))
                self.db.execute("UPDATE resources SET state='REDIRECTED',reason='HTML_META_REFRESH_STUB' WHERE url=?",
                                (row["url"],))
                return
            self.db.commit()
        if fmt == "pdf":
            self.db.execute("UPDATE resources SET state='RAW_ONLY',reason='PDF_TEXT_NOT_EXTRACTED' WHERE url=?",
                            (row["url"],))
            return
        if fmt == "html":
            parsed = normalize_html(raw, row["effective_url"])
            text, title, sections, warnings = (parsed.text, parsed.title, parsed.sections, parsed.warnings)
            version = normalizer_version()
        else:
            text = raw.decode("utf-8", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
            if not text.strip() or re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", text):
                raise ValueError("ARCHIVE_TEXT_INVALID")
            title, sections, warnings, version = row["effective_url"], (), (), "utf8-source-text-v1"
        version_id = stable_id("archdoc", source_uri=row["url"], raw_sha256=row["raw_sha256"], normalizer=version)
        existing = self.db.execute("SELECT document_sha256 FROM versions WHERE version_id=?", (version_id,)).fetchone()
        record = {"schema_version": 1, "archive_document_version_id": version_id,
                  "source_uri": row["url"], "effective_uri": row["effective_url"],
                  "family": row["family"], "language": row["language"], "format": fmt,
                  "version_hint": "latest" if "/latest/" in row["url"] else "versioned_or_unspecified",
                  "fetched_at": row["fetched_at"], "raw_sha256": row["raw_sha256"],
                  "normalized_sha256": text_hash(text), "normalizer_version": version,
                  "title": title, "text": text, "sections": [s.model_dump() for s in sections],
                  "warnings": list(warnings), "semantic_review": "NOT_REVIEWED"}
        document_hash = existing[0] if existing else self._put(canonical_json(record))
        self._guard(len(text.encode("utf-8")) * 6 + MIB)
        self.db.execute("INSERT OR IGNORE INTO versions VALUES (?,?,?)", (version_id, row["url"], document_hash))
        self.db.execute("DELETE FROM page_text WHERE rowid=?", (row["rowid"],))
        self.db.execute("INSERT INTO page_text(rowid,title,body) VALUES (?,?,?)", (row["rowid"], title, text))
        self.db.execute("UPDATE resources SET state='DONE',reason=NULL,document_sha256=? WHERE url=?",
                        (document_hash, row["url"]))

    def run(self, kind: Literal["SITEMAP", "PAGE"], *, max_resources=25, max_seconds=600,
            progress=None, cached_only=False):
        if self.read_only or kind not in {"SITEMAP", "PAGE"}:
            raise ValueError("ARCHIVE_WRITE_MODE_REQUIRED")
        if type(max_resources) is not int or max_resources < 0 or type(max_seconds) is not int or max_seconds < 1:
            raise ValueError("ARCHIVE_RUN_LIMIT_INVALID")
        started, processed, cache_hits = time.monotonic(), 0, 0
        request_start = self.collector.request_count
        cached_clause = " AND raw_sha256 IS NOT NULL" if cached_only else ""
        families = deque(r[0] for r in self.db.execute(
            "SELECT DISTINCT family FROM resources WHERE kind='PAGE' AND state='QUEUED'" + cached_clause + " ORDER BY family"))
        stop = "QUEUE_DRAINED"
        while True:
            if (self.root / "STOP").exists():
                stop = "USER_STOP_REQUESTED"
                break
            if max_resources and processed >= max_resources:
                stop = "RUN_BATCH_LIMIT"
                break
            if time.monotonic() - started >= max_seconds:
                stop = "RUN_TIME_LIMIT"
                break
            if kind == "PAGE":
                # Catalogs have priority, but ordinary content retains family fairness.
                row = self.db.execute("SELECT rowid,* FROM resources WHERE kind='PAGE' AND state='QUEUED' "
                    "AND priority=-100" + cached_clause + " ORDER BY url LIMIT 1").fetchone()
                if not families:
                    families.extend(r[0] for r in self.db.execute(
                        "SELECT DISTINCT family FROM resources WHERE kind='PAGE' AND state='QUEUED'" + cached_clause + " ORDER BY family"))
                while families and row is None:
                    family = families.popleft()
                    row = self.db.execute("SELECT rowid,* FROM resources WHERE kind='PAGE' AND state='QUEUED' "
                                          "AND family=?" + cached_clause + " ORDER BY url LIMIT 1", (family,)).fetchone()
                    if row is not None:
                        families.append(family)
            else:
                row = self.db.execute("SELECT rowid,* FROM resources WHERE kind=? AND state='QUEUED' "
                                      + cached_clause + " ORDER BY priority,url LIMIT 1", (kind,)).fetchone()
            if row is None:
                if cached_only:
                    stop = "CACHED_QUEUE_DRAINED"
                break
            try:
                limit = XML_LIMIT if kind == "SITEMAP" else (
                    self.policy.max_pdf_bytes if row["format"] == "pdf" else self.policy.max_page_bytes)
                self._guard(limit + MIB)
                self.db.execute("UPDATE resources SET state='FETCHING' WHERE url=?", (row["url"],))
                self.db.commit()
                cache_hits += int(row["raw_sha256"] is not None)
                raw, row = self._fetch(row)
                if kind == "SITEMAP":
                    sitemap_kind, entries = parse_sitemap(raw)
                    self._guard(sum(len(url.encode("utf-8")) + 1024 for url, _ in entries) * 2)
                    for url, modified in entries:
                        self._add(url, "SITEMAP" if sitemap_kind == "sitemapindex" else "PAGE", row["url"], modified)
                    self.db.execute("UPDATE resources SET state='DONE',reason=NULL WHERE url=?", (row["url"],))
                else:
                    self._page(row, raw)
                self.db.commit()
            except (ValueError, OSError, ET.ParseError) as exc:
                self.db.rollback()
                category = str(exc) if re.fullmatch(r"[A-Z][A-Z0-9_]{0,95}", str(exc)) else type(exc).__name__
                if category in {"ARCHIVE_STORAGE_BUDGET_REACHED", "ARCHIVE_FREE_DISK_GUARD"}:
                    self.db.execute("UPDATE resources SET state='QUEUED',reason=? WHERE url=?", (category, row["url"]))
                    self.db.commit()
                    stop = category
                    break
                state = "EXCLUDED" if category == "ROBOTS_DENIED" else "FAILED"
                self.db.execute("UPDATE resources SET state=?,reason=? WHERE url=?", (state, category, row["url"]))
                self.db.commit()
                if category in {"ROBOTS_UNAVAILABLE", "ROBOTS_DELAY_REQUIRES_REVIEW"}:
                    stop = category
                    processed += 1
                    break
            processed += 1
            if progress is not None:
                progress({"kind": kind, "processed": processed,
                          "requests": self.collector.request_count - request_start})
        return {**self.status(), "run_kind": kind, "processed_this_run": processed,
                "cached_only": cached_only,
                "cache_hits_this_run": cache_hits, "stop_reason": stop,
                "http_requests_this_run": self.collector.request_count - request_start,
                "duration_seconds": round(time.monotonic() - started, 3)}

    def status(self):
        counts = {"SITEMAP": {}, "PAGE": {}}
        for row in self.db.execute("SELECT kind,state,count(*) AS n FROM resources GROUP BY kind,state"):
            counts[row["kind"]][row["state"]] = row["n"]
        reasons = {r[0]: r[1] for r in self.db.execute(
            "SELECT reason,count(*) FROM resources WHERE reason IS NOT NULL GROUP BY reason")}
        families = [dict(r) for r in self.db.execute("""SELECT family,count(*) AS discovered,
            sum(state='DONE') AS normalized,sum(state='RAW_ONLY') AS raw_only,
            sum(state='FAILED') AS failed FROM resources WHERE kind='PAGE' AND state!='EXCLUDED'
            GROUP BY family ORDER BY family""")]
        pending_maps = sum(counts["SITEMAP"].get(s, 0) for s in ("QUEUED", "FETCHING", "FAILED"))
        pending_pages = sum(counts["PAGE"].get(s, 0) for s in ("QUEUED", "FETCHING", "FAILED", "RAW_ONLY"))
        unfinished_downloads = sum(counts["PAGE"].get(s, 0) for s in ("QUEUED", "FETCHING", "FAILED"))
        root_row = self.db.execute("SELECT state FROM resources WHERE url=?", (self.policy.root_sitemap,)).fetchone()
        root_done = root_row is not None and root_row[0] == "DONE"
        pending_directories = self.db.execute("SELECT count(*) FROM resources WHERE priority=-100 "
            "AND state IN ('QUEUED','FETCHING','FAILED')").fetchone()[0]
        return {"status": "ARCHIVE_STATUS", "checked_at": datetime.now(UTC).isoformat(),
                "policy_version": self.policy.version, "policy_sha256": self.policy.fingerprint,
                "counts": counts, "reasons": reasons, "families": families,
                "rejected_candidates": self.db.execute("SELECT count(*) FROM rejections").fetchone()[0],
                "unique_raw_payloads": self.db.execute("SELECT count(DISTINCT raw_sha256) FROM resources").fetchone()[0],
                "catalog_complete_for_known_scope": root_done and pending_maps == 0 and pending_directories == 0,
                "llms_directories_pending": pending_directories,
                "download_complete_for_known_scope": root_done and pending_maps == 0 and unfinished_downloads == 0,
                "text_complete_for_known_scope": root_done and pending_maps == 0 and pending_pages == 0,
                "absolute_aws_completeness_claimed": False, "inference_calls": 0,
                "archive_root": str(self.root),
                "notice": "Sitemap/link scope only; excluded languages/formats/robots and raw-only sources are explicit. Not a serving snapshot."}

    def search(self, question: str, *, limit=5):
        if type(limit) is not int or not 1 <= limit <= 20 or not 1 <= len(question) <= 2000:
            raise ValueError("ARCHIVE_SEARCH_LIMIT")
        terms = re.findall(r"[^\W_]+", question, flags=re.UNICODE)[:32]
        if not terms:
            return []
        expression = " OR ".join('"' + term[:64] + '"' for term in terms)
        return [dict(r) for r in self.db.execute("""SELECT r.url,r.family,r.fetched_at,
            page_text.title,snippet(page_text,1,'[',']',' ... ',40) AS excerpt,bm25(page_text) AS score
            FROM page_text JOIN resources AS r ON r.rowid=page_text.rowid
            WHERE page_text MATCH ? AND r.state='DONE' ORDER BY score,r.url LIMIT ?""", (expression, limit))]

    def retry_failed(self, kind, *, refetch=False):
        if self.read_only or kind not in {"SITEMAP", "PAGE"}:
            raise ValueError("ARCHIVE_RETRY_MODE_INVALID")
        update = ",raw_sha256=NULL" if refetch else ""
        changed = self.db.execute("UPDATE resources SET state='QUEUED',reason=NULL" + update +
                                  " WHERE kind=? AND state='FAILED'", (kind,)).rowcount
        self.db.commit()
        return changed

    def refresh(self, kind):
        if self.read_only or kind not in {"SITEMAP", "PAGE"}:
            raise ValueError("ARCHIVE_REFRESH_MODE_INVALID")
        changed = self.db.execute("UPDATE resources SET state='QUEUED',reason=NULL,raw_sha256=NULL "
                                  "WHERE kind=? AND state!='EXCLUDED'", (kind,)).rowcount
        self.db.commit()
        return changed

    def export(self, output: Path, *, inventory=False):
        if output.exists():
            raise ValueError("ARCHIVE_EXPORT_ALREADY_EXISTS")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + "." + uuid4().hex + ".tmp")
        count = 0
        try:
            with temporary.open("xb") as file, gzip.GzipFile(filename="", mode="wb", fileobj=file, mtime=0) as stream:
                query = "SELECT * FROM resources ORDER BY url" if inventory else (
                    "SELECT document_sha256 FROM resources WHERE kind='PAGE' AND state='DONE' ORDER BY url")
                for row in self.db.execute(query):
                    raw = canonical_json(dict(row)) if inventory else self.read_blob(row[0])
                    stream.write(raw + b"\n")
                    count += 1
                    if count % 100 == 0 and shutil.disk_usage(output.parent).free < self.policy.min_free_disk_bytes:
                        raise ValueError("ARCHIVE_EXPORT_FREE_DISK_GUARD")
            temporary.replace(output)
        finally:
            temporary.unlink(missing_ok=True)
        return {"status": "EXPORTED_ARCHIVE_INVENTORY" if inventory else "EXPORTED_NORMALIZED_ARCHIVE",
                "records": count, "output": str(output), "inference_calls": 0,
                "format": "AWSArchiveResource-v1 JSONL gzip" if inventory else "AWSArchiveDocument-v1 JSONL gzip",
                "notice": "All known URL states, including unfinished/excluded/raw-only; raw blobs stay in the archive." if inventory else
                    "No MVP document-count cap; excludes unfinished/raw-only resources. Not a graph snapshot."}
