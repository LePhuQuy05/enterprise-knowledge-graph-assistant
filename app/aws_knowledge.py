"""Read-only official AWS discovery; no credentials, arbitrary URLs or agent tools.

Search excerpts are pinned per request. They are not a downloaded full AWS corpus,
a graph snapshot, or an assurance that every source supports a generated claim.
"""

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from urllib.parse import unquote, urlsplit

from app.context import PackedContext
from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes
from app.external_http import strict_json
from app.ingestion.text import ProxyTokenizer

ENDPOINT = "https://knowledge-mcp.global.api.aws"
MAX_RESPONSE = 512 * 1024
BUNDLE_LIMIT = 64 * 1024
LEGACY_RECIPE = "aws-knowledge-search-excerpts-v1"
RECIPE = "aws-knowledge-search-excerpts-v2"
NORMALIZER = "markdown-prose-soft-wrap-v1"


def normalize_excerpt(text):
    """Join only plain prose paragraphs, before creating any offsets or prompts.

    Structural Markdown blocks remain byte-for-byte intact (apart from newline
    encoding). This is not a quote repair or a general Markdown renderer.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines, output, paragraph = text.split("\n"), [], []
    fence = None

    def flush():
        if paragraph:
            # Preserve the whole block if any line may carry Markdown/code
            # semantics. A false negative merely keeps the original wrapping.
            structural = any(re.match(r"^(?:\s|[#>*+\-|`~{\[\]}]|\d+[.)]\s)", line)
                or line.endswith(("  ", "\\")) or "|" in line
                or re.match(r"^[=_-]{3,}\s*$", line) for line in paragraph)
            output.extend(paragraph if structural else [" ".join(paragraph)])
            paragraph.clear()

    for line in lines:
        marker = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence:
            output.append(line)
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                fence = None
        elif marker:
            flush()
            fence = marker[1]
            output.append(line)
        elif not line.strip():
            flush()
            output.append(line)
        else:
            paragraph.append(line)
    flush()
    return "\n".join(output)


class KnowledgeError(ValueError):
    pass


def official_url(value):
    if not isinstance(value, str) or not 1 <= len(value) <= 2048:
        raise KnowledgeError("KNOWLEDGE_SOURCE_URL_INVALID")
    try:
        url = urlsplit(value)
        decoded = unquote(url.path, errors="strict")
        if (url.scheme != "https" or url.hostname not in {"docs.aws.amazon.com", "aws.amazon.com"}
                or url.port not in {None, 443} or url.username or url.password or url.query
                or re.search(r"[\x00-\x20\\]", value) or re.search(r"[\x00-\x20\\]", decoded)
                or any(part in {".", ".."} for part in decoded.split("/"))):
            raise ValueError()
    except (ValueError, UnicodeError):
        raise KnowledgeError("KNOWLEDGE_SOURCE_URL_INVALID") from None
    return value.split("#", 1)[0]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise KnowledgeError("KNOWLEDGE_REDIRECT_BLOCKED")


class AwsKnowledgeClient:
    def __init__(self, *, timeout=6, deadline=None, transport=None):
        self.timeout, self.deadline = timeout, deadline
        self.transport = transport or self._post

    def _post(self, body, timeout):
        request = urllib.request.Request(ENDPOINT, data=body, headers={
            "Content-Type": "application/json", "Accept": "application/json, text/event-stream",
            "User-Agent": "aws-evidence-notebook/1.0",
        }, method="POST")
        try:
            with urllib.request.build_opener(NoRedirect()).open(request, timeout=timeout) as response:
                raw = response.read(MAX_RESPONSE + 1)
                kind = response.headers.get("Content-Type", "").split(";", 1)[0]
        except urllib.error.HTTPError as exc:
            raise KnowledgeError("KNOWLEDGE_RATE_LIMIT" if exc.code == 429 else "KNOWLEDGE_UNAVAILABLE") from None
        except OSError:
            raise KnowledgeError("KNOWLEDGE_UNAVAILABLE") from None
        if len(raw) > MAX_RESPONSE:
            raise KnowledgeError("KNOWLEDGE_RESPONSE_TOO_LARGE")
        if kind == "text/event-stream":
            events = [part for part in raw.decode("utf-8").replace("\r\n", "\n").split("\n\n") if part.strip()]
            values = ["\n".join(line[5:].lstrip() for line in event.splitlines() if line.startswith("data:")) for event in events]
            values = [value for value in values if value]
            if len(values) != 1:
                raise KnowledgeError("KNOWLEDGE_PROTOCOL_INVALID")
            raw = values[0].encode("utf-8")
        elif kind != "application/json":
            raise KnowledgeError("KNOWLEDGE_PROTOCOL_INVALID")
        return raw

    def search(self, phrase):
        if not isinstance(phrase, str) or not 1 <= len(phrase.strip()) <= 2000:
            raise KnowledgeError("KNOWLEDGE_QUERY_INVALID")
        timeout = min(self.timeout, self.deadline - time.monotonic()) if self.deadline else self.timeout
        if timeout < 0.2:
            raise KnowledgeError("KNOWLEDGE_DEADLINE")
        body = canonical_json({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
            "name": "aws___search_documentation", "arguments": {
                "search_phrase": phrase.strip(), "limit": 8, "topics": ["general"],
            }}})
        try:
            raw = self.transport(body, timeout)
            if len(raw) > MAX_RESPONSE:
                raise KnowledgeError("KNOWLEDGE_RESPONSE_TOO_LARGE")
            envelope = strict_json(raw)
            if envelope.get("jsonrpc") != "2.0" or envelope.get("id") != 1 or "error" in envelope:
                raise ValueError()
            result = envelope["result"]
            if result.get("isError") or len(result["content"]) != 1 or result["content"][0]["type"] != "text":
                raise ValueError()
            items = strict_json(result["content"][0]["text"].encode("utf-8"))["content"]["result"]
            if not isinstance(items, list) or len(items) > 30:
                raise ValueError()
        except (KeyError, TypeError, ValueError, UnicodeError):
            raise KnowledgeError("KNOWLEDGE_PROTOCOL_INVALID") from None
        selected, seen = [], set()
        for item in items:
            if not isinstance(item, dict):
                raise KnowledgeError("KNOWLEDGE_PROTOCOL_INVALID")
            try:
                url = official_url(item.get("url"))
            except KnowledgeError:
                continue
            title, context = item.get("title"), item.get("context")
            if (url in seen or not isinstance(title, str) or not 1 <= len(title) <= 2000
                    or not isinstance(context, str) or not 20 <= len(context) <= 50_000):
                continue
            context = context.replace("\r\n", "\n").replace("\r", "\n")
            context = re.split(r"\n\s*\*?\[truncated", context, maxsplit=1)[0]
            if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", context):
                continue
            # Keep a contiguous prefix, never splice separated source spans.
            text = context[:2400]
            if len(context) > len(text) and " " in text:
                text = text.rsplit(" ", 1)[0]
            if not text.strip():
                continue
            selected.append({"url": url, "title": title, "text": text,
                             "excerpt_sha256": sha256_bytes(text.encode("utf-8"))})
            seen.add(url)
            if len(selected) == 6:
                break
        return selected


@dataclass(frozen=True)
class EvidencePolicy:
    tenant_id: str
    snapshot_version: str  # Existing publication policy, not a full-corpus snapshot.


@dataclass(frozen=True)
class GroundingBundle:
    manifest: EvidencePolicy
    documents: object
    chunks: object
    record: dict
    sha256: str


def create_bundle(items, *, tenant_id, snapshot_version, request_id, owner_id, search_phrase, fetched_at=None):
    fetched_at = fetched_at or datetime.now(UTC)
    tokenizer = ProxyTokenizer()
    documents, chunks, excerpts = [], [], []
    record = {"version": RECIPE, "normalizer": NORMALIZER, "tenant_id": tenant_id,
        "owner_id": owner_id, "request_id": request_id, "snapshot_version": snapshot_version,
        "endpoint": ENDPOINT, "search_phrase": search_phrase, "fetched_at": fetched_at.isoformat(),
        "documents": documents, "chunks": chunks, "raw_excerpts": excerpts,
        "source_kind": "OFFICIAL_SEARCH_EXCERPT", "full_corpus_downloaded": False}
    for item in items:
        if len(chunks) == 6:
            break
        raw_text = item["text"]
        if not isinstance(raw_text, str) or not 1 <= len(raw_text) <= 2400:
            raise KnowledgeError("KNOWLEDGE_EXCERPT_INVALID")
        raw_hash = sha256_bytes(raw_text.encode("utf-8"))
        text = normalize_excerpt(raw_text)
        document = DocumentVersion.create(tenant_id=tenant_id, source_uri=official_url(item["url"]),
            title=item["title"], fetched_at=fetched_at, raw_sha256=raw_hash,
            normalizer_version=RECIPE, normalized_text_path="request-evidence/" + request_id,
            normalized_text=text, source_kind="TEXT")
        chunk = Chunk.create(tenant_id=tenant_id, document_version_id=document.document_version_id,
            chunking_version=RECIPE, section_path=("Official AWS search excerpt",), position=0,
            start_char=0, end_char=len(text), text=text, token_count=tokenizer.count(text),
            tokenizer_version="cl100k_base")
        documents.append(document.model_dump(mode="json"))
        chunks.append(chunk.model_dump(mode="json"))
        excerpts.append({"document_version_id": document.document_version_id,
                         "raw_text": raw_text, "raw_sha256": raw_hash})
        if len(canonical_json(record)) > BUNDLE_LIMIT:
            documents.pop()
            chunks.pop()
            excerpts.pop()
            break
    return load_bundle(record, sha256_bytes(canonical_json(record)), tenant_id=tenant_id,
                       owner_id=owner_id, request_id=request_id, snapshot_version=snapshot_version)


def load_bundle(record, digest, *, tenant_id, owner_id, request_id, snapshot_version):
    raw = canonical_json(record)
    version = record.get("version")
    if version not in {LEGACY_RECIPE, RECIPE} or len(raw) > BUNDLE_LIMIT or sha256_bytes(raw) != digest or any(record.get(k) != v for k, v in {
        "tenant_id": tenant_id, "owner_id": owner_id, "request_id": request_id,
        "snapshot_version": snapshot_version, "endpoint": ENDPOINT, "source_kind": "OFFICIAL_SEARCH_EXCERPT",
        "full_corpus_downloaded": False,
    }.items()):
        raise KnowledgeError("KNOWLEDGE_BUNDLE_INVALID")
    try:
        documents = [DocumentVersion.model_validate_json(canonical_json(r)) for r in record["documents"]]
        chunks = [Chunk.model_validate_json(canonical_json(r)) for r in record["chunks"]]
        if len(documents) != len(chunks) or len(chunks) > 6:
            raise ValueError()
        docs = {d.document_version_id: d for d in documents}
        if len(docs) != len(documents) or len({c.chunk_id for c in chunks}) != len(chunks):
            raise ValueError()
        for chunk in chunks:
            doc = docs[chunk.document_version_id]
            if (chunk.tenant_id != tenant_id or doc.tenant_id != tenant_id or chunk.start_char != 0
                    or chunk.text != doc.normalized_text or doc.normalizer_version != version
                    or chunk.chunking_version != version):
                raise ValueError()
            official_url(doc.source_uri)
        if version == RECIPE:
            excerpts = record["raw_excerpts"]
            if record.get("normalizer") != NORMALIZER or len(excerpts) != len(documents):
                raise ValueError()
            if len({e["document_version_id"] for e in excerpts}) != len(excerpts):
                raise ValueError()
            for excerpt in excerpts:
                doc = docs[excerpt["document_version_id"]]
                text = excerpt["raw_text"]
                if (not isinstance(text, str) or not 1 <= len(text) <= 2400
                        or sha256_bytes(text.encode("utf-8")) != excerpt["raw_sha256"]
                        or doc.raw_sha256 != excerpt["raw_sha256"]
                        or normalize_excerpt(text) != doc.normalized_text):
                    raise ValueError()
    except (ValueError, TypeError, KeyError):
        raise KnowledgeError("KNOWLEDGE_BUNDLE_INVALID") from None
    return GroundingBundle(EvidencePolicy(tenant_id, snapshot_version), MappingProxyType(docs),
        MappingProxyType({c.chunk_id: c for c in chunks}), record, digest)


def pack_bundle(bundle):
    tokenizer = ProxyTokenizer()
    selected, source_rows = [], []
    for key, chunk in bundle.chunks.items():
        proposed = [*source_rows, {"source_id": "s" + str(len(selected) + 1), "untrusted_text": chunk.text}]
        if tokenizer.count(canonical_json({"untrusted_sources": proposed}).decode()) > 4096:
            continue
        selected.append(key)
        source_rows = proposed
    serialized = canonical_json({"untrusted_sources": source_rows}).decode()
    return PackedContext(tuple(selected), tuple("s" + str(i + 1) for i in range(len(selected))), (),
        serialized, tokenizer.count(serialized), ("AWS_OFFICIAL_LIVE_SEARCH_EXCERPTS", "FULL_AWS_CORPUS_NOT_DOWNLOADED"))
