import json
from datetime import timedelta, timezone

import pytest
from pydantic import ValidationError

from app.contracts import (
    Chunk,
    EvidenceSpan,
    ExtractionRecipe,
    Qualifier,
    QueryRequest,
    QueryResponse,
    RelationAssertion,
    canonical_uri,
    embedding_cache_key,
    embedding_input,
    extraction_cache_key,
)
from app.fixtures import fixture_content, fixture_recipes


@pytest.mark.parametrize("question", ["", "  ", "x" * 2001, 123, None])
def test_bad_question(question):
    with pytest.raises(ValidationError):
        QueryRequest(question=question)


def test_request_strict_no_client_tenant():
    assert QueryRequest(question="  why? ").question == "why?"
    with pytest.raises(ValidationError):
        QueryRequest(question="why?", tenant_id="attacker")


def test_unicode_offsets_are_code_points():
    content = fixture_content()
    chunk = next(c for c in content.chunks if "\U0001f680" in c.text)
    doc = next(d for d in content.documents if d.document_version_id == chunk.document_version_id)
    assert doc.normalized_text[chunk.start_char:chunk.end_char] == chunk.text
    assert len(chunk.text.encode("utf-8")) > len(chunk.text)
    data = chunk.model_dump()
    data["end_char"] = chunk.start_char + len(chunk.text.encode("utf-8"))
    with pytest.raises(ValidationError):
        Chunk(**data)
    with pytest.raises(ValidationError):
        EvidenceSpan(start_char=0, end_char=2, quote="\U0001f680")
    assert EvidenceSpan(start_char=0, end_char=1, quote="\U0001f680")


def test_strict_int_and_integrity_checks():
    chunk = fixture_content().chunks[0]
    for changes in ({"position": "0"}, {"token_count": True}, {"text": "spoofed"}):
        with pytest.raises(ValidationError):
            Chunk(**(chunk.model_dump() | changes))


def test_document_hash_and_utc():
    doc = fixture_content().documents[0]
    for changes in (
        {"normalized_text": doc.normalized_text + "tamper"},
        {"fetched_at": doc.fetched_at.astimezone(timezone(timedelta(hours=7)))},
        {"document_id": "doc_bad"},
    ):
        with pytest.raises(ValidationError):
            type(doc)(**(doc.model_dump() | changes))


@pytest.mark.parametrize("uri", [
    "http://example.com/a", "https://user:secret@example.com/", "https://example.com/a b",
    "https://example.com/\npath", "https://example.com\\other", "file:///etc/passwd",
])
def test_source_uri_rejects_unsafe_forms(uri):
    with pytest.raises(ValueError):
        canonical_uri(uri)


def test_canonical_uri_preserves_query_and_removes_anchor():
    assert canonical_uri("HTTPS://EXAMPLE.COM:443/path?b=2&a=1#section") == (
        "https://example.com/path?b=2&a=1"
    )


def assertion_variant(**changes):
    extraction, _ = fixture_recipes()
    assertion = fixture_content().assertions[0]
    data = assertion.model_dump(exclude={
        "assertion_id", "ontology_version", "extraction_prompt_version", "extraction_model_id",
        "extraction_recipe_sha256",
    })
    data["evidence"] = assertion.evidence
    data["qualifiers"] = assertion.qualifiers
    data.update(changes)
    return RelationAssertion.create(recipe=extraction, **data)


def test_assertion_identity_includes_semantics_and_full_recipe():
    extraction, _ = fixture_recipes()
    base = assertion_variant()
    assert base == assertion_variant()
    assert base.assertion_id != assertion_variant(polarity="NEGATIVE").assertion_id
    qualifier = Qualifier(kind="REGION", text="demo-1")
    assert base.assertion_id != assertion_variant(qualifiers=(qualifier,)).assertion_id
    data = extraction.model_dump() | {"parameters": {"temperature": 0.1, "max_tokens": 2048}}
    changed = ExtractionRecipe(**data)
    original = base.model_dump(exclude={
        "assertion_id", "ontology_version", "extraction_prompt_version", "extraction_model_id",
        "extraction_recipe_sha256",
    }) | {"evidence": base.evidence, "qualifiers": base.qualifiers}
    assert base.assertion_id != RelationAssertion.create(recipe=changed, **original).assertion_id


def test_qualifier_set_and_symmetric_predicate_are_canonical():
    q1, q2 = Qualifier(kind="REGION", text="r1"), Qualifier(kind="VERSION", text="v1")
    assert assertion_variant(qualifiers=(q2, q1, q1)) == assertion_variant(qualifiers=(q1, q2))
    a = assertion_variant(predicate="INTEGRATES_WITH")
    b = assertion_variant(predicate="INTEGRATES_WITH", subject_id=a.object_id, object_id=a.subject_id)
    assert a == b
    with pytest.raises(ValidationError):
        assertion_variant(subject_id=a.object_id, object_id=a.object_id)


def test_cache_identity_separates_tenants_inputs_and_recipe():
    extraction, embedding = fixture_recipes()
    doc, chunk = fixture_content().documents[0], fixture_content().chunks[0]
    source = embedding_input(doc, chunk)
    assert source == doc.title + "\nSynthetic fixture\n\n" + chunk.text
    for key_function, recipe in ((embedding_cache_key, embedding), (extraction_cache_key, extraction)):
        key = key_function("tenant_a", source, recipe)
        assert key != key_function("tenant_b", source, recipe)
        assert key != key_function("tenant_a", source + "changed", recipe)
        updated = type(recipe)(**(recipe.model_dump() | {"invocation_id": "other-profile"}))
        assert key != key_function("tenant_a", source, updated)


def test_json_roundtrip_and_unsupported_fields():
    doc = fixture_content().documents[0]
    assert type(doc).model_validate_json(doc.model_dump_json()) == doc
    payload = json.loads(doc.model_dump_json()) | {"unexpected": "private"}
    with pytest.raises(ValidationError):
        type(doc).model_validate_json(json.dumps(payload))


def response_data():
    return dict(
        request_id="request", answer="Supported fact.",
        claims=({"claim_id": "c1", "text": "Supported fact.", "citation_ids": ["s1"]},),
        citations=({"citation_id": "s1", "chunk_id": "chk1", "document_version_id": "docv1",
                    "title": "Source", "source_uri": "https://example.invalid/",
                    "section_path": ["section"], "quote": "Supported fact."},),
        answer_status="ANSWERED", mode_requested="hybrid", mode_used="hybrid", used_graph=False,
        snapshot_version="snap1",
    )


@pytest.mark.parametrize("changes", [
    {"answer": "Supported fact. Plus an unsupported claim."},
    {"answer_status": "ABSTAINED"}, {"used_graph": True},
    {"claims": [{"claim_id": "c1", "text": "Supported fact.", "citation_ids": ["unknown"]}]},
    {"claims": []},
])
def test_response_rejects_uncited_prose_and_inconsistent_references(changes):
    with pytest.raises(ValidationError):
        QueryResponse.model_validate_json(json.dumps(response_data() | changes))


def test_answer_and_abstention_shape():
    assert QueryResponse.model_validate_json(json.dumps(response_data())).answer_status == "ANSWERED"
    data = response_data() | {"answer_status": "ABSTAINED", "answer": "Insufficient evidence.",
                              "claims": [], "citations": []}
    assert not QueryResponse.model_validate_json(json.dumps(data)).claims
