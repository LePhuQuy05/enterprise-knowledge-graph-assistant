"""Whole-source profile boundaries and V12 ID/source resolution controls."""

import json
from dataclasses import replace
from datetime import UTC, datetime

import pytest

from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes
from app.generation import AnswerBinding, SelectedAnswerV4
from app.profiled_generation import (
    ProfiledGenerationRecipe,
    parse_profiled_answer,
    profiled_generation_request,
    resolve_profiled_answer,
)
from app.profiled_source_context import (
    SourceContextProfile,
    pack_profiled_context,
    render_profiled_context,
    validate_profiled_context,
)


def fixture_sources(*, long=False):
    texts = [(f"Synthetic source {i}. " + ("word " * 700 if long else "Reader needs permission for the named resource only."))
        for i in range(1, 13)]
    text = "\n\n".join(texts)
    doc = DocumentVersion.create(tenant_id="demo", source_uri="https://docs.aws.amazon.com/fixture/profile.html",
        title="Synthetic profile fixture", fetched_at=datetime(2026, 10, 28, tzinfo=UTC), raw_sha256=sha256_bytes(text.encode()),
        normalizer_version="fixture", normalized_text_path="fixture.txt", normalized_text=text, source_kind="TEXT")
    chunks, start = {}, 0
    for i, body in enumerate(texts):
        chunk = Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id, position=i,
            section_path=("Synthetic",), chunking_version="fixture", tokenizer_version="fixture",
            start_char=start, end_char=start+len(body), text=body, token_count=20)
        chunks[chunk.chunk_id] = chunk
        start += len(body)+2
    return {doc.document_version_id: doc}, chunks


def render(n=12, cap=12, *, long=False):
    docs, chunks = fixture_sources(long=long)
    return docs, chunks, render_profiled_context(docs, chunks, tuple(chunks)[:n], question="Synthetic permission question?",
        language="en", tenant_id="demo", collection_sha256="a"*64, profile=SourceContextProfile(max_chunks=cap))


def test_profile12_retains_complete_original_sources_and_canonical_ids():
    docs, chunks, context = render()
    payload = json.loads(context.user_message)
    assert [s["source_id"] for s in payload["untrusted_sources"]] == [f"s{i}" for i in range(1, 13)]
    assert "s12p1" in context.span_lookup_json
    for key, source in zip(context.chunk_ids, payload["untrusted_sources"], strict=True):
        assert "".join(s["text"] for s in source["spans"]) == chunks[key].text
    assert validate_profiled_context(context, docs, chunks) == context


def test_profiles_share_identical_renderer_and_generation_prompt_on_same_eight_sources():
    docs, chunks, eight = render(8, 8)
    _, _, twelve = render(8, 12)
    assert eight.user_message == twelve.user_message and eight.proxy_tokens == twelve.proxy_tokens
    assert eight.profile.fingerprint != twelve.profile.fingerprint
    recipe = ProfiledGenerationRecipe(model_metadata_sha256="a"*64)
    assert profiled_generation_request(eight, docs, chunks, recipe) == profiled_generation_request(twelve, docs, chunks, recipe)


def test_cap8_is_still_strict_and_not_relaxed_by_new_profile():
    with pytest.raises(ValueError, match="CONTEXT_BOUND"):
        render(9, 8)
    with pytest.raises(ValueError):
        SourceContextProfile(max_chunks=9)


@pytest.mark.parametrize("kind", ["message", "lookup", "tokens", "language", "profile"])
def test_tampered_context_cannot_reach_generation(kind):
    docs, chunks, c = render()
    changes = {"message": {"user_message": c.user_message+"x"}, "lookup": {"span_lookup_json": "[]"},
        "tokens": {"proxy_tokens": 0}, "language": {"language": "ja"}, "profile": {"profile": SourceContextProfile(max_chunks=8)}}
    with pytest.raises(ValueError):
        validate_profiled_context(replace(c, **changes[kind]), docs, chunks)


def test_valid_language_change_needs_a_new_identity_and_can_be_bound_to_trusted_request():
    docs, chunks, c = render()
    vi = replace(c, language="vi")
    assert vi.fingerprint != c.fingerprint
    assert validate_profiled_context(vi, docs, chunks) == vi
    with pytest.raises(ValueError, match="TRUSTED_TASK_BINDING"):
        validate_profiled_context(vi, docs, chunks, expected_language="en")


def test_unused_unknown_pool_candidate_is_rejected_before_packing():
    docs, chunks = fixture_sources()
    with pytest.raises(ValueError, match="SOURCE_TENANT_OR_ID"):
        pack_profiled_context(docs, chunks, (next(iter(chunks)), "unknown"), [], question="Test?", language="en",
            tenant_id="demo", collection_sha256="a"*64, profile=SourceContextProfile(max_chunks=12))


@pytest.mark.parametrize("kind", ["tenant", "quote", "document", "unknown", "duplicate"])
def test_source_provenance_and_tenant_tampering_fail_closed(kind):
    docs, chunks = fixture_sources()
    keys = tuple(chunks)
    if kind == "tenant":
        chunks[keys[0]] = chunks[keys[0]].model_copy(update={"tenant_id": "foreign"})
    elif kind == "quote":
        chunks[keys[0]] = chunks[keys[0]].model_copy(update={"text": "x"})
    elif kind == "document":
        did = next(iter(docs))
        docs[did] = docs[did].model_copy(update={"normalized_text": "x"})
    elif kind == "unknown":
        keys = ("unknown",)
    else:
        keys = (keys[0], keys[0])
    with pytest.raises(ValueError):
        render_profiled_context(docs, chunks, keys, question="Test?", language="en", tenant_id="demo",
            collection_sha256="a"*64, profile=SourceContextProfile(max_chunks=12))


def pack(cap=12, *, long=False, graph=False, balanced=False):
    docs, chunks = fixture_sources(long=long)
    keys = tuple(chunks)
    bundles = [{"bundle_id": f"b{i}", "chunk_ids": (key,), "query_index": i % 2} for i, key in enumerate(keys)]
    records = {}
    if graph:
        bundles = [{"bundle_id": "path", "chunk_ids": keys[:2], "assertion_ids": ("f1", "f2"), "query_index": 0}, *bundles[2:]]
        records = {f"f{i+1}": {"assertion_id": f"f{i+1}", "chunk_id": key, "untrusted_evidence_quote": chunks[key].text}
            for i, key in enumerate(keys[:2])}
    context, audit = pack_profiled_context(docs, chunks, keys, bundles, question="Synthetic permission question?",
        language="en", tenant_id="demo", collection_sha256="a"*64, profile=SourceContextProfile(max_chunks=cap),
        balanced=balanced, query_order=(0, 1), graph_records=records)
    return context, audit


def test_packer_increases_only_slot_cap_and_keeps_token_cap_and_full_sources():
    a, old = pack(8)
    b, new = pack(12)
    assert len(a.chunk_ids) == 8 and len(b.chunk_ids) == 12
    assert b.proxy_tokens <= 4096 and a.chunk_ids == b.chunk_ids[:8]
    assert old["hidden_fallbacks"] == new["partial_source_truncations"] == 0


def test_token_overflow_rejects_bundles_instead_of_truncating_source_text():
    context, audit = pack(long=True)
    assert len(context.chunk_ids) < 12 and context.proxy_tokens <= 4096
    assert any(d["reason"] == "TOKEN_BUDGET" for d in audit["decisions"])
    docs, chunks = fixture_sources(long=True)
    validate_profiled_context(context, docs, chunks)


def test_graph_path_support_remains_atomic_and_qb_changes_only_order():
    q, a = pack(graph=True)
    qb, b = pack(graph=True, balanced=True)
    assert a["context_assertion_ids"] == b["context_assertion_ids"] == ["f1", "f2"]
    assert a["used_graph"] and b["used_graph"]
    assert q.chunk_ids != qb.chunk_ids and set(q.chunk_ids) == set(qb.chunk_ids)


def test_graph_incomplete_support_is_error_not_hybrid_fallback():
    docs, chunks = fixture_sources()
    keys = tuple(chunks)
    records = {"f1": {"assertion_id": "f1", "chunk_id": keys[0], "untrusted_evidence_quote": chunks[keys[0]].text}}
    with pytest.raises(ValueError, match="GRAPH_ATOMIC_SUPPORT"):
        pack_profiled_context(docs, chunks, keys, [{"bundle_id": "bad", "chunk_ids": keys[:2], "assertion_ids": ("f1",), "query_index": 0}],
            question="Test?", language="en", tenant_id="demo", collection_sha256="a"*64,
            profile=SourceContextProfile(max_chunks=12), graph_records=records)


def answer(n=8, source="s12", span="s12p1"):
    return {"answer_status": "ANSWERED", "claims": [{"text": f"Synthetic claim {i}.",
        "evidence": [{"source_id": source, "span_id": span}]} for i in range(n)], "reason": "NONE", "limitations": []}


def test_v12_accepts_eight_claims_and_source12_and_resolves_original_citations():
    docs, chunks, context = render()
    raw = canonical_json(answer()).decode()
    out = parse_profiled_answer(raw, "end_turn")
    with pytest.raises(ValueError):
        SelectedAnswerV4.model_validate_json(raw)
    binding = AnswerBinding(request_id="fixture", tenant_id="demo", owner_id="reviewer", mode_requested="hybrid", mode_used="hybrid", used_graph=False)
    recipe = ProfiledGenerationRecipe(model_metadata_sha256="a"*64)
    response = resolve_profiled_answer(out, context, docs, chunks, binding=binding, recipe=recipe)
    assert len(response.claims) == 8 and len(response.citations) == 1
    assert response.citations[0].chunk_id == context.chunk_ids[-1]
    assert response.citations[0].quote == chunks[context.chunk_ids[-1]].text
    other = resolve_profiled_answer(out, context, docs, chunks, binding=binding.model_copy(update={"owner_id": "other"}), recipe=recipe)
    assert response.citations[0].citation_id != other.citations[0].citation_id


@pytest.mark.parametrize("kind", ["source13", "span100", "nine_claims", "unknown_key", "duplicate_json", "null_extra", "truncated", "too_many_citations"])
def test_v12_strict_protocol_failures_are_preserved(kind):
    data = answer()
    stop = "end_turn"
    if kind == "source13":
        data = answer(source="s13")
    elif kind == "span100":
        data = answer(span="s12p100")
    elif kind == "nine_claims":
        data = answer(9)
    elif kind in {"unknown_key", "null_extra"}:
        data["extra"] = None if kind == "null_extra" else "x"
    elif kind == "truncated":
        stop = "max_tokens"
    elif kind == "too_many_citations":
        for i, c in enumerate(data["claims"]):
            c["evidence"] = [{"source_id": f"s{i+1}", "span_id": f"s{i+1}p{p}"} for p in (1, 2, 3)]
    raw = canonical_json(data).decode()
    if kind == "duplicate_json":
        raw = raw.replace('"reason":"NONE"', '"reason":"NONE","reason":"NONE"')
    with pytest.raises(ValueError):
        parse_profiled_answer(raw, stop)


@pytest.mark.parametrize("kind", ["absent_source", "wrong_pair", "wrong_tenant"])
def test_schema_valid_ids_still_need_actual_supplied_context_and_tenant(kind):
    docs, chunks, c = render(8, 8)
    data = answer(1, source="s9", span="s9p1") if kind == "absent_source" else answer(1, source="s1", span="s2p1")
    if kind == "wrong_tenant":
        data = answer(1, source="s1", span="s1p1")
    out = parse_profiled_answer(canonical_json(data).decode(), "end_turn")
    binding = AnswerBinding(request_id="fixture", tenant_id="foreign" if kind == "wrong_tenant" else "demo",
        owner_id="reviewer", mode_requested="hybrid", mode_used="hybrid", used_graph=False)
    with pytest.raises(ValueError):
        resolve_profiled_answer(out, c, docs, chunks, binding=binding, recipe=ProfiledGenerationRecipe(model_metadata_sha256="a"*64))
