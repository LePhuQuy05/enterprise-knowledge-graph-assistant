"""Whole-chunk paging: read long routed documents without truncating sources."""

from app.contracts import canonical_json, sha256_bytes
from app.generation import GenerationError
from app.literal_source_catalog import SourceCatalog
from app.routed_reader_json_v2 import READ_SYSTEM
from app.routed_source_reader import DocumentDirectory

PAGE_SYSTEM = READ_SYSTEM.replace("TWELVE", "FOUR") + (
    " This is ONE PAGE of a bounded document-reading job. Select evidence useful for any requested part "
    "even if other pages are needed for a complete answer. Do not claim the question is unanswerable just "
    "because this page lacks the rest. Return at most four IDs. No summaries or factual conclusions."
)


def source_message(directory, keys, documents, chunks):
    rows = []
    for i, key in enumerate(keys, 1):
        c = chunks[key]
        d = documents[c.document_version_id]
        if (c.tenant_id != d.tenant_id or d.normalized_text[c.start_char:c.end_char] != c.text
                or sha256_bytes(c.text.encode()) != c.text_sha256):
            raise GenerationError("PAGED_READER_SOURCE_BINDING")
        rows.append({"source_id": f"r{i}", "untrusted_title": d.title,
            "untrusted_section": c.section_path, "untrusted_text": c.text})
    return canonical_json({"untrusted_question": directory.question, "untrusted_english_search": directory.search,
        "language": directory.language, "untrusted_sources": rows}).decode()


def catalog_pages(directory, selected_documents, documents, chunks, *, max_bytes=45000, max_pages=8):
    if (type(directory) is not DocumentDirectory or not 1000 <= max_bytes <= 45000 or not 1 <= max_pages <= 8
            or len(selected_documents) > 6 or len(set(selected_documents)) != len(selected_documents)
            or not set(selected_documents) <= set(directory.document_ids)):
        raise GenerationError("PAGED_READER_DOCUMENTS_OR_BOUNDS")
    ids = tuple(k for d in selected_documents for k in sorted(
        (k for k, c in chunks.items() if c.document_version_id == d), key=lambda k: (chunks[k].start_char, k)))
    pages, current = [], []
    for key in ids:
        if len(source_message(directory, [*current, key], documents, chunks).encode()) > max_bytes:
            if not current:
                raise GenerationError("PAGED_READER_SINGLE_CHUNK_EXCEEDS_BOUND")
            pages.append(tuple(current))
            current = []
            if len(source_message(directory, [key], documents, chunks).encode()) > max_bytes:
                raise GenerationError("PAGED_READER_SINGLE_CHUNK_EXCEEDS_BOUND")
        current.append(key)
    if current:
        pages.append(tuple(current))
    if len(pages) > max_pages:
        raise GenerationError("PAGED_READER_PAGE_BUDGET_NO_TRUNCATION")
    if tuple(key for page in pages for key in page) != ids:
        raise GenerationError("PAGED_READER_SOURCE_LOSS")
    return tuple(SourceCatalog(directory.collection_sha256, directory.question, directory.search,
        directory.language, page, source_message(directory, page, documents, chunks),
        {"policy": "PAGED_WHOLE_ROUTED_DOCUMENTS_V3", "directory": directory.fingerprint,
            "document_ids": tuple(selected_documents), "page_index": i, "page_count": len(pages),
            "source_annotations_used": False, "whole_original_chunks": True}) for i, page in enumerate(pages))


def consolidation_catalog(directory, selected_by_page, pages, documents, chunks):
    if len(selected_by_page) != len(pages):
        raise GenerationError("PAGED_READER_PAGE_SELECTION_COUNT")
    for selected, page in zip(selected_by_page, pages, strict=True):
        if len(selected) > 4 or len(set(selected)) != len(selected) or not set(selected) <= set(page.source_ids):
            raise GenerationError("PAGED_READER_UNSUPPLIED_SELECTION")
    keys = tuple(dict.fromkeys(k for selected in selected_by_page for k in selected))
    message = source_message(directory, keys, documents, chunks)
    if len(keys) > 32 or len(message.encode()) > 85000:
        raise GenerationError("PAGED_READER_CONSOLIDATION_BOUND")
    return SourceCatalog(directory.collection_sha256, directory.question, directory.search, directory.language,
        keys, message, {"policy": "NATIVE_PAGE_SELECTION_CONSOLIDATION_V3",
            "directory": directory.fingerprint, "page_fingerprints": tuple(p.fingerprint for p in pages),
            "source_annotations_used": False, "whole_original_chunks": True})
