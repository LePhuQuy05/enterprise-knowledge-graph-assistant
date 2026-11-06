import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.answer_qualification import qualify_answer
from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes
from app.generation import AnswerBinding
from app.profiled_generation import (
    ProfiledGenerationRecipe,
    parse_profiled_answer,
    resolve_profiled_answer,
)
from app.profiled_source_context import SourceContextProfile, render_profiled_context
from tests.test_citation_antecedents import fixture as source_fixture


def setup(*, raw=None, question="Which Lambda permissions are needed?", owner="owner-a", mode="hybrid", used_graph=False):
    docs, chunks, pieces, _ = source_fixture()
    context = render_profiled_context(docs, chunks, tuple(c.chunk_id for c in pieces), question=question, language="en",
        tenant_id="demo", collection_sha256="a" * 64, profile=SourceContextProfile(max_chunks=12))
    binding = AnswerBinding(request_id="qualified-fixture", tenant_id="demo", owner_id=owner,
        mode_requested=mode, mode_used=mode, used_graph=used_graph)
    recipe = ProfiledGenerationRecipe(model_metadata_sha256="b" * 64)
    payload = raw or {"answer_status": "ANSWERED", "reason": "NONE", "limitations": [],
        "claims": [{"text": "For uploads, the policy requires the configured key.",
            "evidence": [{"source_id": "s1", "span_id": "s1p1"}, {"source_id": "s2", "span_id": "s2p1"}]}]}
    response = resolve_profiled_answer(parse_profiled_answer(json.dumps(payload), "end_turn"), context, docs, chunks,
        binding=binding, recipe=recipe)
    return docs, chunks, context, binding, response


def test_cited_prefix_is_eligible_but_not_called_semantically_verified():
    docs, chunks, context, binding, response = setup()
    q = qualify_answer(response, context, docs, chunks, binding=binding)
    assert q.status == "NO_MECHANICAL_FLAGS" and q.display_response == response
    assert not q.public_payload()["semantic_verified"]


def test_uncited_antecedent_holds_entire_response_without_editing_native_claims_or_leaking_text():
    raw = {"answer_status": "ANSWERED", "reason": "NONE", "limitations": [],
        "claims": [{"text": "Unsafe generic request assertion.", "evidence": [{"source_id": "s2", "span_id": "s2p1"}]}]}
    docs, chunks, context, binding, response = setup(raw=raw)
    before = response.model_dump_json()
    q = qualify_answer(response, context, docs, chunks, binding=binding)
    assert q.status == "REVIEW_REQUIRED" and q.display_response is None
    assert q.flags[0].rule_id == "ANTECEDENT_AVAILABLE_BUT_NOT_CITED"
    public = q.public_payload()
    assert public["response"] is None and "Unsafe generic" not in canonical_json(public).decode()
    assert response.model_dump_json() == before
    assert q.review_record()["native_response_for_review"]["claims"] == response.model_dump(mode="json")["claims"]


def test_invocation_clause_needs_explicit_operation_wording_in_its_own_citations():
    raw = {"answer_status": "ANSWERED", "reason": "NONE", "limitations": [],
        "claims": [{"text": "Quyền của execution role tách biệt với quyền gọi Lambda.",
            "evidence": [{"source_id": "s1", "span_id": "s1p1"}]}]}
    docs, chunks, context, binding, response = setup(raw=raw)
    q = qualify_answer(response, context, docs, chunks, binding=binding)
    assert q.flags[0].rule_id == "INVOCATION_PREMISE_NOT_EXPLICIT_IN_CITATIONS"
    assert q.public_payload()["mode_used"] == "hybrid"


@pytest.mark.parametrize("mutation", ["owner", "question", "title", "method"])
def test_response_cannot_be_requalified_under_a_different_owner_task_source_or_method(mutation):
    docs, chunks, context, binding, response = setup()
    if mutation == "owner":
        binding = binding.model_copy(update={"owner_id": "owner-b"})
    elif mutation == "question":
        context = render_profiled_context(docs, chunks, context.chunk_ids, question="Different original task?", language="en",
            tenant_id="demo", collection_sha256=context.collection_sha256, profile=context.profile)
    elif mutation == "title":
        cites = list(response.citations)
        cites[0] = cites[0].model_copy(update={"title": "Fake source title"})
        response = response.model_copy(update={"citations": tuple(cites)})
    else:
        binding = binding.model_copy(update={"mode_used": "vector"})
    with pytest.raises(ValueError):
        qualify_answer(response, context, docs, chunks, binding=binding)


def test_graph_mode_without_actual_graph_contribution_does_not_become_display_eligible():
    docs, chunks, context, binding, response = setup(mode="graph_1hop")
    with pytest.raises(ValueError, match="NO_FALLBACK"):
        qualify_answer(response, context, docs, chunks, binding=binding)


def test_tampered_context_is_rejected_before_claim_checks():
    docs, chunks, context, binding, response = setup()
    with pytest.raises(ValueError):
        qualify_answer(response, replace(context, user_message="altered"), docs, chunks, binding=binding)


def single_statement(source, claim):
    doc = DocumentVersion.create(tenant_id="demo", source_uri="https://docs.aws.amazon.com/fixture/operation.html",
        title="Synthetic operation fixture", fetched_at=datetime(2026, 11, 5, tzinfo=UTC),
        raw_sha256=sha256_bytes(source.encode()), normalized_text_path="fixture.txt", normalizer_version="fixture",
        normalized_text=source, source_kind="TEXT")
    chunk = Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id, position=0,
        section_path=("Operation",), chunking_version="fixture", tokenizer_version="fixture", start_char=0,
        end_char=len(source), text=source, token_count=30)
    docs, chunks = {doc.document_version_id: doc}, {chunk.chunk_id: chunk}
    context = render_profiled_context(docs, chunks, (chunk.chunk_id,), question="How does Lambda use permissions?",
        language="en", tenant_id="demo", collection_sha256="a" * 64, profile=SourceContextProfile(max_chunks=12))
    binding = AnswerBinding(request_id="operation-fixture", tenant_id="demo", owner_id="owner-a",
        mode_requested="hybrid", mode_used="hybrid", used_graph=False)
    raw = {"answer_status": "ANSWERED", "reason": "NONE", "limitations": [],
        "claims": [{"text": claim, "evidence": [{"source_id": "s1", "span_id": "s1p1"}]}]}
    response = resolve_profiled_answer(parse_profiled_answer(json.dumps(raw), "end_turn"), context, docs, chunks,
        binding=binding, recipe=ProfiledGenerationRecipe(model_metadata_sha256="b" * 64))
    return qualify_answer(response, context, docs, chunks, binding=binding)


def test_explicitly_cited_invocation_is_not_blocked_by_the_literal_witness_check():
    q = single_statement("A resource-based policy allows the S3 service to invoke the Lambda function.",
        "The policy allows S3 to invoke the Lambda function.")
    assert q.status == "NO_MECHANICAL_FLAGS"


def test_lambda_calling_another_service_is_not_mistaken_for_permission_to_invoke_lambda():
    q = single_statement("The function code calls DynamoDB GetItem using its execution role.",
        "Lambda code invokes the DynamoDB GetItem API using its execution role.")
    assert q.status == "NO_MECHANICAL_FLAGS"
