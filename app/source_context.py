"""Contiguous document windows for source review; never fabricated snapshot chunks."""

import re
from typing import Literal

from pydantic import Field, model_validator

from app.contracts import Record, Sha256, sha256_bytes, stable_id
from app.extraction import ExtractionError

POLICY = "CONTIGUOUS_DOCUMENT_CONTEXT_REVIEW_ONLY_V1"


def classify_source(text, title=""):
    """Heuristic scheduling tags, not semantic labels or blanket rejection rules."""
    flags = []
    if re.search(r"\b(?:error codes?|troubleshoot(?:ing)?|errors?)\b", title, re.I) or re.search(r"HTTP status code:|errorCode:", text):
        flags.append("DIAGNOSTIC_REVIEW_REQUIRED")
    if "```" in text:
        flags.append("CONTAINS_CODE_REVIEW_REQUIRED")
    if any(line.count("|") >= 2 for line in text.splitlines()):
        flags.append("CONTAINS_TABLE_REVIEW_REQUIRED")
    if re.search(r"\b(?:the following|these|those|this role|that role|the target)\b", text, re.I):
        flags.append("REFERENCE_OR_LIST_REVIEW_REQUIRED")
    if text and (text[0].islower() or text.lstrip().startswith((",", ";", ")"))):
        flags.append("POSSIBLE_PARTIAL_START")
    return tuple(flags) or ("PROSE_REVIEW_REQUIRED",)


class SourceContextWindow(Record):
    version: Literal["source-context-window-v1"] = "source-context-window-v1"
    policy: Literal[POLICY] = POLICY
    window_id: str
    tenant_id: str
    collection_manifest_sha256: Sha256
    document_version_id: str
    document_normalized_sha256: Sha256
    source_uri: str
    source_title: str
    primary_chunk_id: str
    primary_chunk_sha256: Sha256
    start_char: int = Field(ge=0)
    end_char: int = Field(gt=0)
    text: str = Field(min_length=1, max_length=8000)
    text_sha256: Sha256
    overlapping_chunk_ids: tuple[str, ...]
    review_flags: tuple[str, ...]
    context_before_chars: int = Field(ge=0, le=1500)
    context_after_chars: int = Field(ge=0, le=1500)
    supplemental_context_only: Literal[True] = True
    native_extraction_completed: Literal[False] = False
    serving_published: Literal[False] = False

    @model_validator(mode="after")
    def identity_and_extent(self):
        expected = stable_id("source-window-v1", tenant=self.tenant_id, collection=self.collection_manifest_sha256,
            document=self.document_version_id, primary=self.primary_chunk_id, start=self.start_char,
            end=self.end_char, text=self.text_sha256, before=self.context_before_chars, after=self.context_after_chars)
        if self.end_char - self.start_char != len(self.text) or self.text_sha256 != sha256_bytes(self.text.encode()) or self.window_id != expected:
            raise ValueError("SOURCE_WINDOW_ID_OR_TEXT")
        return self


def build_source_window(document, primary, chunks, collection_manifest_sha256, *, before=1000, after=1000):
    if (type(before) is not int or type(after) is not int or not 0 <= before <= 1500 or not 0 <= after <= 1500
            or document.tenant_id != primary.tenant_id or document.document_version_id != primary.document_version_id
            or document.normalized_text[primary.start_char:primary.end_char] != primary.text
            or primary.chunk_id not in chunks or chunks[primary.chunk_id] != primary):
        raise ExtractionError("SOURCE_WINDOW_PRIMARY_OR_BOUND")
    text = document.normalized_text
    desired_start, desired_end = max(0, primary.start_char - before), min(len(text), primary.end_char + after)
    left = text.rfind("\n\n", 0, desired_start)
    start = 0 if left < 0 else left + 2
    right = text.find("\n\n", desired_end)
    end = len(text) if right < 0 else right
    # Fall back to bounded literal offsets if a paragraph is unusually large.
    if end - start > 8000:
        start, end = desired_start, desired_end
    if end - start > 8000 or not start <= primary.start_char < primary.end_char <= end:
        raise ExtractionError("SOURCE_WINDOW_TOO_LARGE_FOR_PRIMARY")
    own = [c for c in chunks.values() if c.document_version_id == document.document_version_id
        and c.tenant_id == document.tenant_id and c.start_char < end and c.end_char > start]
    if any(text[c.start_char:c.end_char] != c.text for c in own):
        raise ExtractionError("SOURCE_WINDOW_OVERLAP_SOURCE_CHANGED")
    overlap = tuple(c.chunk_id for c in sorted(own, key=lambda c: (c.start_char, c.end_char, c.chunk_id)))
    window_text = text[start:end]
    digest = sha256_bytes(window_text.encode())
    data = dict(tenant_id=document.tenant_id, collection_manifest_sha256=collection_manifest_sha256,
        document_version_id=document.document_version_id, document_normalized_sha256=document.normalized_sha256,
        source_uri=document.source_uri, source_title=document.title, primary_chunk_id=primary.chunk_id,
        primary_chunk_sha256=sha256_bytes(primary.text.encode()), start_char=start, end_char=end,
        text=window_text, text_sha256=digest, overlapping_chunk_ids=overlap,
        review_flags=classify_source(primary.text, document.title), context_before_chars=before, context_after_chars=after)
    identity = stable_id("source-window-v1", tenant=document.tenant_id, collection=collection_manifest_sha256,
        document=document.document_version_id, primary=primary.chunk_id, start=start, end=end, text=digest, before=before, after=after)
    return SourceContextWindow(window_id=identity, **data)


def validate_source_window(window, document, chunks, *, collection_manifest_sha256):
    primary = chunks.get(window.primary_chunk_id)
    if primary is None or window != build_source_window(document, primary, chunks, collection_manifest_sha256,
            before=window.context_before_chars, after=window.context_after_chars):
        raise ExtractionError("SOURCE_WINDOW_RELEASE_OR_DOCUMENT_CHANGED")
    return window
