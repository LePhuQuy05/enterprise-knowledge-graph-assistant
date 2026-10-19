"""Source-preserving HTML normalization and bounded Unicode chunk spans."""

import hashlib
import importlib.metadata
import os
import platform
import re
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import tiktoken

from app.contracts import Chunk, DocumentVersion, sha256_bytes
from app.ingestion.models import ChunkRecipe, Section

TOKENIZER_URL = "https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken"
TOKENIZER_SHA256 = "223921b76ee99bde995b7ff738513eef100fb51d18c93597a113bcffe865b2a7"
TOKENIZER_CACHE = Path(__file__).resolve().parents[2] / "data" / "tokenizer-cache"
_TOKENIZER_LOCK = threading.Lock()


def prepare_tokenizer() -> dict:
    """Explicit one-time public asset download, never implicit in tests/ingest."""
    TOKENIZER_CACHE.mkdir(parents=True, exist_ok=True)
    path = TOKENIZER_CACHE / hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()
    if path.exists():
        raw = path.read_bytes()
    else:
        # This fixed asset is distinct from the AWS source collector allowlist.
        with urllib.request.urlopen(TOKENIZER_URL, timeout=15) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
        if len(raw) > 4 * 1024 * 1024 or sha256_bytes(raw) != TOKENIZER_SHA256:
            raise ValueError("TOKENIZER_ASSET_INTEGRITY_FAILED")
        temporary = path.with_suffix(".tmp")
        temporary.write_bytes(raw)
        temporary.replace(path)
    if sha256_bytes(raw) != TOKENIZER_SHA256:
        raise ValueError("TOKENIZER_ASSET_INTEGRITY_FAILED")
    return {"status": "TOKENIZER_READY", "encoding": "cl100k_base",
            "asset_sha256": TOKENIZER_SHA256, "bytes": len(raw), "inference_calls": 0}


class ProxyTokenizer:
    def __init__(self):
        path = TOKENIZER_CACHE / hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()
        if not path.is_file() or sha256_bytes(path.read_bytes()) != TOKENIZER_SHA256:
            raise ValueError("TOKENIZER_NOT_PREPARED_RUN_PREPARE_TOKENIZER")
        # Keep ingestion offline once the explicit asset preparation has finished.
        with _TOKENIZER_LOCK:
            old = os.environ.get("TIKTOKEN_CACHE_DIR")
            os.environ["TIKTOKEN_CACHE_DIR"] = str(TOKENIZER_CACHE)
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            finally:
                if old is None:
                    os.environ.pop("TIKTOKEN_CACHE_DIR", None)
                else:
                    os.environ["TIKTOKEN_CACHE_DIR"] = old
        self.version = (
            "tiktoken-" + importlib.metadata.version("tiktoken")
            + "-cl100k_base-" + TOKENIZER_SHA256
        )

    def count(self, text: str) -> int:
        # Source text that resembles a special token is ordinary untrusted data.
        return len(self.encoding.encode(text, disallowed_special=()))


def normalizer_version() -> str:
    return (
        "aws-main-html-v2-bs4-" + importlib.metadata.version("beautifulsoup4")
        + "-html.parser-python-" + platform.python_version()
    )


@dataclass(frozen=True)
class Normalized:
    text: str
    title: str
    sections: tuple[Section, ...]
    warnings: tuple[str, ...]


def normalize_html(raw: bytes, fallback_title: str) -> Normalized:
    from bs4 import BeautifulSoup, Comment, NavigableString, Tag

    html = raw.decode("utf-8-sig", errors="strict").replace("\r\n", "\n").replace("\r", "\n")
    if re.search(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", html):
        raise ValueError("BINARY_OR_CONTROL_SOURCE")
    soup = BeautifulSoup(html, "html.parser")
    article = soup.find(id="main-col-body") or soup.find("main") or soup.find("article")
    if article is None:
        raise ValueError("MAIN_ARTICLE_NOT_FOUND")
    warnings = set()
    if article.find("img"):
        warnings.add("IMAGES_NOT_TRANSCRIBED")
    for tag in article.find_all([
        "script", "style", "nav", "footer", "noscript", "form", "svg", "button", "aside",
        "awsdocs-copyright", "awsdocs-thumb-feedback", "awsdocs-page-header",
        "awsdocs-language-banner",
    ]):
        tag.decompose()
    for tag in article.select(".awsdocs-page-header-container, #inline-topiclist, .highlights"):
        tag.decompose()
    blocks: list[tuple[str, int, str | None]] = []

    def raw_inline(node) -> str:
        # Preserve adjacency around inline identifiers, instead of inserting spaces
        # around every span/code/link. Only normalize source whitespace.
        return "".join(str(n) if isinstance(n, NavigableString) else "\n" if n.name == "br"
                       else raw_inline(n) for n in node.children if not isinstance(n, Comment))

    def inline(node) -> str:
        return re.sub(r"\s+", " ", raw_inline(node)).strip()

    def emit(text: str, level: int = 0, anchor=None):
        if text.strip():
            blocks.append((text.strip("\n"), level, str(anchor) if anchor else None))

    def walk(node):
        if isinstance(node, Comment):
            return
        if isinstance(node, NavigableString):
            emit(re.sub(r"\s+", " ", str(node)).strip())
            return
        if not isinstance(node, Tag):
            return
        if re.fullmatch(r"h[1-6]", node.name):
            emit(inline(node), int(node.name[1]), node.get("id"))
            return
        if node.name == "pre":
            emit("```\n" + node.get_text("", strip=False).strip("\n") + "\n```")
            return
        if node.name == "table":
            if node.find(attrs={"rowspan": True}) or node.find(attrs={"colspan": True}):
                warnings.add("COMPLEX_TABLE_SPANS_REVIEW")
            rows = [row.find_all(["th", "td"], recursive=False) for row in node.find_all("tr")]
            rows = [row for row in rows if row]
            headers = [inline(cell) for cell in rows[0]] if rows and all(
                cell.name == "th" for cell in rows[0]
            ) else []
            lines = []
            caption = node.find("caption")
            if caption:
                lines.append(inline(caption))
            if headers:
                lines.append(" | ".join(headers))
            for row in rows[1:] if headers else rows:
                cells = [inline(cell) for cell in row]
                if headers and len(headers) == len(cells):
                    lines.append(" | ".join(f"{h}: {v}" for h, v in zip(headers, cells)))
                else:
                    lines.append(" | ".join(cells))
            emit("\n".join(lines))
            return
        if node.name == "p":
            emit(inline(node))
            return
        block_names = {
            "p", "pre", "table", "ul", "ol", "li", "div", "section", "article", "main",
            "blockquote", "h1", "h2", "h3", "h4", "h5", "h6",
            "dl", "dt", "dd", "figure", "figcaption",
        }
        pending = []

        def flush():
            text = re.sub(r"\s+", " ", "".join(pending)).strip()
            if text:
                emit(("- " if node.name == "li" else "") + text)
            pending.clear()

        for child in node.children:
            if isinstance(child, Comment):
                continue
            if isinstance(child, Tag) and (
                child.name in block_names or child.find(list(block_names))
            ):
                flush()
                walk(child)
            elif isinstance(child, NavigableString):
                pending.append(str(child))
            elif isinstance(child, Tag):
                pending.append("\n" if child.name == "br" else raw_inline(child))
        flush()

    walk(article)
    if not blocks:
        raise ValueError("EMPTY_MAIN_ARTICLE")
    parts = []
    sections = []
    path: list[tuple[int, str]] = []
    cursor = 0
    section_start = 0
    section_path: tuple[str, ...] = ()
    section_anchor = None
    title = next((text for text, level, _ in blocks if level == 1), fallback_title)
    for text, level, anchor in blocks:
        if parts:
            parts.append("\n\n")
            cursor += 2
        if level:
            if cursor > section_start and "".join(parts)[section_start:cursor].strip():
                sections.append(Section(start_char=section_start, end_char=cursor,
                                        section_path=section_path, anchor=section_anchor))
            path = [(n, label) for n, label in path if n < level] + [(level, text)]
            section_start, section_path, section_anchor = cursor, tuple(t for _, t in path), anchor
        parts.append(text)
        cursor += len(text)
    text = "".join(parts)
    sections.append(Section(start_char=section_start, end_char=len(text),
                            section_path=section_path, anchor=section_anchor))
    return Normalized(text, title, tuple(sections), tuple(sorted(warnings)))


def chunk_document(document: DocumentVersion, sections: tuple[Section, ...],
                   recipe: ChunkRecipe, tokenizer: ProxyTokenizer) -> tuple[Chunk, ...]:
    text = document.normalized_text
    if not sections or sections[0].start_char != 0 or sections[-1].end_char != len(text):
        raise ValueError("SECTION_COVERAGE_FAILED")
    if any(a.end_char != b.start_char for a, b in zip(sections, sections[1:])):
        raise ValueError("SECTION_COVERAGE_FAILED")
    chunks = []
    for section in sections:
        start = section.start_char
        while start < section.end_char:
            remaining = text[start:section.end_char]
            if not remaining.strip():
                break
            if tokenizer.count(remaining) <= recipe.max_tokens:
                end = section.end_char
            else:
                # Character-prefix search never decodes an incomplete UTF-8 token.
                # Prefix BPE counts need not be monotonic; every selected final
                # substring is independently checked against the hard cap below.
                low, high = start + 1, min(section.end_char, start + recipe.target_tokens * 8)
                end = low
                while low <= high:
                    mid = (low + high) // 2
                    if tokenizer.count(text[start:mid]) <= recipe.target_tokens:
                        end, low = mid, mid + 1
                    else:
                        high = mid - 1
                boundary = max(text.rfind("\n", start + (end-start)//2, end),
                               text.rfind(" ", start + (end-start)//2, end))
                if boundary > start:
                    end = boundary + 1
            count = tokenizer.count(text[start:end])
            if count <= 0 or count > recipe.max_tokens:
                raise ValueError("CHUNK_TOKEN_BOUND_FAILED")
            chunks.append(Chunk.create(
                tenant_id=document.tenant_id, document_version_id=document.document_version_id,
                chunking_version=recipe.version(tokenizer.version), section_path=section.section_path,
                position=len(chunks), start_char=start, end_char=end, text=text[start:end],
                token_count=count, tokenizer_version=tokenizer.version,
            ))
            if end == section.end_char:
                break
            next_start = end
            while next_start > start + 1 and recipe.overlap_tokens and (
                tokenizer.count(text[next_start-1:end]) <= recipe.overlap_tokens
            ):
                next_start -= 1
            start = next_start
    if not chunks or len(chunks) > 5000:
        raise ValueError("CHUNK_COUNT_FAILED")
    # Union coverage must retain every non-whitespace source character, including
    # the tail and negations near a chunk boundary.
    coverage_end = 0
    for chunk in chunks:
        if chunk.text != text[chunk.start_char:chunk.end_char]:
            raise ValueError("CHUNK_SOURCE_SLICE_FAILED")
        if text[coverage_end:chunk.start_char].strip():
            raise ValueError("CHUNK_COVERAGE_FAILED")
        coverage_end = max(coverage_end, chunk.end_char)
    if text[coverage_end:].strip():
        raise ValueError("CHUNK_COVERAGE_FAILED")
    return tuple(chunks)
