from datetime import UTC, datetime

import pytest

from app.contracts import Chunk, DocumentVersion, sha256_bytes
from app.profiled_generation import ProfiledGenerationRecipe, profiled_generation_request
from app.profiled_source_context import SourceContextProfile, render_profiled_context
from app.task_scoped_generation_v13 import TaskScopedRecipeV13
from app.task_scoped_generation_v13 import task_scoped_request as v13_request
from app.task_scoped_generation_v14 import (
    TaskScopedRecipeV14,
    parse_v14_answer,
    task_scoped_request,
)
from tests.test_profiled_source_context import render
from tests.test_task_scoped_generation_v13 import output


def test_v14_preserves_exact_v12_user_message_but_versions_the_constant_scope_policy():
    docs, chunks, source = render(12)
    recipe = TaskScopedRecipeV14(model_metadata_sha256="b" * 64)
    original, body = profiled_generation_request(source, docs, chunks,
        ProfiledGenerationRecipe(model_metadata_sha256="b" * 64))
    system, prompt = task_scoped_request(source, docs, chunks, recipe, search=source.question)
    assert prompt == body and system.startswith(original)
    assert "TASK-GRANULARITY RULES V13" in system  # shared policy; V14 changes the source-message binding
    assert recipe.version.endswith("v14")


def test_nearly_full_context_reproduces_v13_overflow_but_v14_fits_without_private_artifacts():
    question = "Which policy types are required for this operation and which operations support the documented permission model?"

    def candidate(n):
        text = "Synthetic source. " + "word " * n
        doc = DocumentVersion.create(tenant_id="demo", source_uri="https://docs.aws.amazon.com/fixture/budget.html",
            title="Synthetic near-boundary fixture", fetched_at=datetime(2026, 11, 5, tzinfo=UTC),
            raw_sha256=sha256_bytes(text.encode()), normalizer_version="fixture", normalized_text_path="fixture.txt",
            normalized_text=text, source_kind="TEXT")
        chunk = Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id, position=0,
            section_path=("Fixture",), chunking_version="fixture", tokenizer_version="fixture", start_char=0,
            end_char=len(text), text=text, token_count=max(1, n))
        docs, chunks = {doc.document_version_id: doc}, {chunk.chunk_id: chunk}
        source = render_profiled_context(docs, chunks, (chunk.chunk_id,), tenant_id="demo", question=question,
            language="en", collection_sha256="a" * 64, profile=SourceContextProfile(max_chunks=12), enforce_budget=False)
        return docs, chunks, source

    low, high = 1, 4000
    while low < high:
        mid = (low + high + 1) // 2
        if candidate(mid)[2].proxy_tokens <= 4090:
            low = mid
        else:
            high = mid - 1
    docs, chunks, source = candidate(low)
    assert 4080 <= source.proxy_tokens <= 4090
    with pytest.raises(ValueError, match="NO_TRUNCATION"):
        v13_request(source, docs, chunks, TaskScopedRecipeV13(model_metadata_sha256="b" * 64), search=question)
    _, prompt = task_scoped_request(source, docs, chunks, TaskScopedRecipeV14(model_metadata_sha256="b" * 64), search=question)
    assert prompt == source.user_message


def test_unasked_policy_arn_is_still_rejected_but_an_original_literal_gap_is_allowed():
    with pytest.raises(ValueError, match="NOT_LITERAL"):
        parse_v14_answer(output("full policy ARN"), "end_turn", question="Which policies are required?", search="Which policies are required?")
    assert parse_v14_answer(output("Which policies"), "end_turn", question="Which policies are required?", search="Which policies are required?")
