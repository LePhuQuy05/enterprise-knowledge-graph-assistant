from datetime import UTC, datetime

import pytest

from app.citation_antecedents import audit_claim_antecedents
from app.contracts import Chunk, Citation, DocumentVersion, sha256_bytes


def fixture():
    first = "When you upload an object, this example denies PutObject without the configured key. "
    second = "Regardless, the key must match the policy or the request is denied."
    text = first + second
    doc = DocumentVersion.create(tenant_id="demo", source_uri="https://docs.aws.amazon.com/fixture/antecedent.html",
        title="Synthetic upload example", fetched_at=datetime(2026, 11, 5, tzinfo=UTC), raw_sha256=sha256_bytes(text.encode()),
        normalizer_version="fixture", normalized_text_path="fixture.txt", normalized_text=text, source_kind="TEXT")
    pieces = [Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id, position=i,
        section_path=("Upload example",), chunking_version="fixture", tokenizer_version="fixture", start_char=start,
        end_char=start+len(body), text=body, token_count=20) for i, (body, start) in enumerate(((first, 0), (second, len(first))))]
    citations = [Citation(citation_id=f"c{i}", chunk_id=c.chunk_id, document_version_id=doc.document_version_id,
        title=doc.title, source_uri=doc.source_uri, section_path=c.section_path, quote=c.text) for i, c in enumerate(pieces)]
    return {doc.document_version_id: doc}, {c.chunk_id: c for c in pieces}, pieces, citations


def test_anaphoric_tail_cannot_hide_that_upload_condition_is_missing_from_context():
    docs, chunks, pieces, cites = fixture()
    row = audit_claim_antecedents(cites[1:], docs, chunks, (pieces[1].chunk_id,))[0]
    assert row["status"] == "MISSING_ANTECEDENT_IN_CONTEXT"
    assert pieces[0].chunk_id in row["candidate_support_chunk_ids"]


def test_context_presence_is_not_citation_completeness_or_semantic_entailment():
    docs, chunks, pieces, cites = fixture()
    keys = tuple(c.chunk_id for c in pieces)
    assert audit_claim_antecedents(cites[1:], docs, chunks, keys)[0]["status"] == "ANTECEDENT_AVAILABLE_BUT_NOT_CITED"
    assert audit_claim_antecedents(cites, docs, chunks, keys)[0]["status"] == "PARAGRAPH_PREFIX_PRESENT_AND_CITED"


def test_explicit_source_without_anaphoric_prefix_does_not_get_an_invented_dependency():
    docs, chunks, pieces, cites = fixture()
    assert audit_claim_antecedents(cites[:1], docs, chunks, tuple(c.chunk_id for c in pieces)) == []


def test_a_mutated_antecedent_outside_the_cited_chunk_is_not_accepted():
    docs, chunks, pieces, cites = fixture()
    key = next(iter(docs))
    docs[key] = docs[key].model_copy(update={"normalized_text": "x" * len(pieces[0].text) + pieces[1].text})
    with pytest.raises(ValueError, match="SOURCE_BINDING"):
        audit_claim_antecedents(cites[1:], docs, chunks, (pieces[1].chunk_id,))
