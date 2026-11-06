"""Auditor rejects omitted text, forged titles, misbound questions and budgets."""

import json
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from app.contracts import sha256_bytes
from scripts.audit_facet_recall_v5 import algorithm_equivalence, rendered_sources


@pytest.fixture
def source_case():
    text = "Actor reads resource.\nOnly under this condition."
    chunk = NS(document_version_id="doc", start_char=0, end_char=len(text), text=text,
        text_sha256=sha256_bytes(text.encode()))
    doc = NS(normalized_text=text, title="Synthetic source", source_uri="https://docs.aws.amazon.com/example")
    prompt = {"untrusted_question": "Question?", "untrusted_sources": [{"source_id": "s1",
        "untrusted_document_title": doc.title, "spans": [{"text": text}]}]}
    row = {"question": "Question?", "context_chunk_ids": ["chunk"], "pool_ids": ["chunk"],
        "source_message": json.dumps(prompt), "user_message_proxy_tokens": 100}
    tokenizer = NS(count=lambda _: 100)
    return row, prompt, {"doc": doc}, {"chunk": chunk}, tokenizer


def test_auditor_accepts_full_original_source(source_case):
    row, _, docs, chunks, tokenizer = source_case
    assert rendered_sources(row, docs, chunks, tokenizer) == (1, 1)


@pytest.mark.parametrize("change", ["prefix_only", "suffix_only", "invented", "title", "question", "source_id"])
def test_auditor_detects_rendering_and_binding_errors(source_case, change):
    row, prompt, docs, chunks, tokenizer = deepcopy(source_case)
    source = prompt["untrusted_sources"][0]
    if change == "prefix_only":
        source["spans"][0]["text"] = "Actor reads resource."
    elif change == "suffix_only":
        source["spans"][0]["text"] = "Only under this condition."
    elif change == "invented":
        source["spans"][0]["text"] = "The actor has permission."
    elif change == "title":
        source["untrusted_document_title"] = "Other title"
    elif change == "question":
        prompt["untrusted_question"] = "Another question?"
    else:
        source["source_id"] = "s2"
    row["source_message"] = json.dumps(prompt)
    with pytest.raises(ValueError):
        rendered_sources(row, docs, chunks, tokenizer)


@pytest.mark.parametrize("change", [{"pool_ids": []}, {"user_message_proxy_tokens": 101},
    {"context_chunk_ids": ["chunk", "chunk"]}])
def test_auditor_rejects_pool_and_budget_mismatch(source_case, change):
    row, _, docs, chunks, tokenizer = source_case
    with pytest.raises(ValueError):
        rendered_sources(row | change, docs, chunks, tokenizer)


def test_document_recall_versions_differ_only_in_declared_plan_binding_identity():
    assert algorithm_equivalence()
