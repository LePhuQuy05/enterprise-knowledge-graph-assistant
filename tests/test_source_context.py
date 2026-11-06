
import pytest

from app.contracts import Chunk, canonical_json, sha256_bytes
from app.source_context import (
    SourceContextWindow,
    build_source_window,
    classify_source,
    validate_source_window,
)
from scripts.prepare_graph_source_context import source_priority
from tests.test_extraction import source


def window_case():
    prefix = "Original heading\n\nCondition in account A. Unicode tiếng Việt 🚀.\n\n"
    text = "Lambda assumes your execution role when you invoke your function."
    doc, chunk = source(text, prefix=prefix)
    return doc, chunk, {chunk.chunk_id: chunk}


def test_contiguous_document_context_keeps_preceding_condition_unicode_and_raw_chunk():
    doc, chunk, chunks = window_case()
    window = build_source_window(doc, chunk, chunks, "1" * 64)
    assert window.text == doc.normalized_text[window.start_char:window.end_char]
    assert "account A" in window.text and "tiếng Việt 🚀" in window.text
    assert window.primary_chunk_sha256 == sha256_bytes(chunk.text.encode())
    assert window.primary_chunk_id in window.overlapping_chunk_ids
    assert window.supplemental_context_only and not window.native_extraction_completed
    assert validate_source_window(window, doc, chunks, collection_manifest_sha256="1" * 64) == window


@pytest.mark.parametrize("change", ["source", "document", "collection", "quote", "overlap", "window-id"])
def test_window_rejects_changed_source_scope_and_artifact(change):
    doc, chunk, chunks = window_case()
    window = build_source_window(doc, chunk, chunks, "1" * 64)
    digest = "1" * 64
    if change == "source":
        chunks = {chunk.chunk_id: chunk.model_copy(update={"text": "X" + chunk.text[1:]})}
    elif change == "document":
        doc, _ = source(chunk.text, suffix="foreign")
    elif change == "collection":
        digest = "f" * 64
    elif change == "overlap":
        window = window.model_copy(update={"overlapping_chunk_ids": ()})
    else:
        data = window.model_dump(mode="json")
        data["text" if change == "quote" else "window_id"] = "changed"
        with pytest.raises(ValueError, match="SOURCE_WINDOW"):
            SourceContextWindow.model_validate_json(canonical_json(data))
        return
    with pytest.raises(ValueError, match="SOURCE_WINDOW"):
        validate_source_window(window, doc, chunks, collection_manifest_sha256=digest)


@pytest.mark.parametrize("text,title,flag", [
    ("HTTP status code: 403 | AccessDenied | role could not be assumed", "Example", "DIAGNOSTIC_REVIEW_REQUIRED"),
    ('```\n{"Action":"sts:AssumeRole"}\n```', "Example", "CONTAINS_CODE_REVIEW_REQUIRED"),
    ("A | B | C", "Example", "CONTAINS_TABLE_REVIEW_REQUIRED"),
    ("The following destinations are supported:", "Example", "REFERENCE_OR_LIST_REVIEW_REQUIRED"),
    ("nt delivery fails, send to another target", "Example", "POSSIBLE_PARTIAL_START"),
    ("A role could not be assumed.", "Troubleshooting IAM", "DIAGNOSTIC_REVIEW_REQUIRED"),
])
def test_scheduler_flags_do_not_promote_code_errors_lists_to_facts(text, title, flag):
    assert flag in classify_source(text, title)


def test_context_does_not_invent_a_collection_chunk_or_merge_source_identities():
    doc, chunk, chunks = window_case()
    one = build_source_window(doc, chunk, chunks, "1" * 64)
    other_doc, other_chunk = source(chunk.text, prefix=doc.normalized_text[:chunk.start_char], suffix="other")
    two = build_source_window(other_doc, other_chunk, {other_chunk.chunk_id: other_chunk}, "1" * 64)
    assert one.text == two.text and one.window_id != two.window_id
    assert one.primary_chunk_id != two.primary_chunk_id


def test_large_paragraph_and_context_limits_remain_bounded():
    doc, chunk = source("A" * 8001, prefix="")
    with pytest.raises(ValueError, match="TOO_LARGE"):
        build_source_window(doc, chunk, {chunk.chunk_id: chunk}, "1" * 64)


def test_all_actual_overlapping_chunks_are_reported_without_stitching():
    doc, chunk, chunks = window_case()
    previous = Chunk.create(tenant_id=chunk.tenant_id, document_version_id=doc.document_version_id,
        chunking_version="fixture-v1", section_path=("Fixture",), position=1, start_char=0, end_char=20,
        text=doc.normalized_text[:20], token_count=4, tokenizer_version="fixture-v1")
    chunks[previous.chunk_id] = previous
    window = build_source_window(doc, chunk, chunks, "1" * 64)
    assert set(window.overlapping_chunk_ids) == set(chunks)
    assert window.text == doc.normalized_text[window.start_char:window.end_char]


def test_source_selection_prefers_explicit_role_action_to_error_and_generic_procedure():
    flags = ("PROSE_REVIEW_REQUIRED",)
    positive = source_priority("ASSUME_ROLE", flags, "Lambda automatically assumes your execution role when you invoke your function.")
    generic = source_priority("ASSUME_ROLE", flags, "You can assume an IAM role.")
    diagnostic = source_priority("ASSUME_ROLE", ("DIAGNOSTIC_REVIEW_REQUIRED",), "Lambda assumes a role but an error occurs.")
    assert positive < generic < diagnostic
