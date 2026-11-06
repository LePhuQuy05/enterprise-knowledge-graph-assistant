"""Counterexamples for AND/OR equivalence, provenance and non-semantic scoring."""

from datetime import UTC, datetime

import pytest

from app.authorization_evidence import LiteralSpan
from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes
from app.question_coverage_review import (
    ORIGIN,
    QuestionRubric,
    ReviewedSpan,
    audit_question_coverage,
    load_rubric,
    make_witness,
)


def fixtures():
    docs, chunks = {}, {}
    for name, text in (("a", "ACTOR invokes FUNCTION. CONDITION: owner A only."),
            ("b", "ALTERNATIVE: ACTOR invokes FUNCTION only for owner A."),
            ("c", "ACTOR invokes FUNCTION. CONDITION: owner A only.")):
        doc = DocumentVersion.create(tenant_id="demo", source_uri=f"https://docs.aws.amazon.com/fixture/{name}.html",
            title="Synthetic review fixture", fetched_at=datetime(2026, 10, 28, tzinfo=UTC),
            raw_sha256=sha256_bytes(text.encode()), normalizer_version="fixture-review-v1",
            normalized_text_path=f"fixture-{name}.txt", normalized_text=text, source_kind="TEXT")
        docs[name] = doc

        def chunk(start, end):
            return Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id, position=start,
                section_path=("Synthetic",), chunking_version="fixture-review-v1", tokenizer_version="fixture",
                start_char=start, end_char=end, text=text[start:end], token_count=20)

        chunks[name] = chunk(0, len(text))
        if name == "a":
            split = text.index("CONDITION")
            chunks["a1"], chunks["a2"] = chunk(0, split), chunk(split, len(text))
    req = {"id": "R1", "intent": "task", "claim": "Synthetic invocation requirement",
        "actor": "ACTOR", "action": "invoke", "resource": "FUNCTION", "condition": "owner A only", "kind": "EXPLICIT_SUPPORT"}
    data = {"version": "question-coverage-rubric-v1", "origin": ORIGIN, "human_reviewers": 0,
        "exposure": "Synthetic mechanics fixture", "scope": "Synthetic only", "missing_semantics": "Unreviewed, not absent",
        "requirements": [req, req | {"id": "R2", "kind": "QUALIFIED_INFERENCE", "requires": ["R1"]}],
        "controls_not_yet_retrieved": []}
    rubric = QuestionRubric.model_validate_json(canonical_json(data))

    def span(name, start, end):
        doc = docs[name]
        return ReviewedSpan(document_version_id=doc.document_version_id, document_normalized_sha256=doc.normalized_sha256,
            source_uri=doc.source_uri, span=LiteralSpan(start_char=start, end_char=end, quote=doc.normalized_text[start:end]))

    split = docs["a"].normalized_text.index("CONDITION")
    witness = make_witness("R1", "EXPLICIT_SUPPORT", (span("a", 0, split), span("a", split, len(docs["a"].normalized_text))), "Synthetic AND condition.")
    alternative = make_witness("R1", "EXPLICIT_SUPPORT", (span("b", 0, len(docs["b"].normalized_text)),), "Synthetic alternate document.")
    return docs, chunks, rubric, witness, alternative


def run(keys, *, alter=None, witnesses=None):
    docs, chunks, rubric, witness, alternative = fixtures()
    selected = tuple(chunks[k].chunk_id for k in keys)
    ds = {d.document_version_id: d for d in docs.values()}
    cs = {c.chunk_id: c for c in chunks.values()}
    ws = (witness, alternative) if witnesses is None else witnesses
    if alter:
        rubric, ws, ds, cs, selected = alter(rubric, ws, ds, cs, selected)
    return audit_question_coverage(rubric, ws, ds, cs, selected, intent="task", tenant_id="demo")


@pytest.mark.parametrize("keys", [("a",), ("a1", "a2"), ("b",)])
def test_alternative_documents_and_adjacent_chunk_unions_satisfy_review_without_fixed_chunk_id(keys):
    result = run(keys)
    assert result["requirements"][0]["status"] == "EXPLICIT_REVIEWED_WITNESS_PRESENT"
    assert result["requirements"][1]["status"] == "QUALIFIED_SYNTHESIS_PREMISES_PRESENT"
    assert result["reviewed_requirements_present"] == 2 and result["all_reviewed_requirements_present"]
    assert not result["answer_quality_measured"] and not result["semantic_entailment_automatically_verified"]
    assert not result["permission_or_configuration_verified"] and not result["serving_gate_enabled"]


def test_partial_witness_does_not_satisfy_requirement_or_synthesis():
    result = run(("a1",))
    assert result["reviewed_requirements_present"] == 0
    assert result["requirements"][0]["witness_checks"][0]["any_overlap"]
    assert result["requirements"][0]["witness_checks"][0]["spans"][1]["missing_intervals"]


def test_equal_text_in_unreviewed_document_is_not_silently_approved():
    result = run(("c",))
    assert result["reviewed_requirements_present"] == 0
    assert not result["all_equivalent_sources_discovered"]


def test_empty_context_keeps_missing_as_unreviewed_and_never_inserts_sources():
    result = run(())
    assert result["selected_chunk_ids_unchanged"] == [] and result["reviewed_requirements_present"] == 0
    assert result["requirements"][0]["status"] == "NO_COMPLETE_REVIEWED_WITNESS"


def test_partial_alternatives_are_not_joined_into_a_new_approved_witness():
    docs, chunks, rubric, witness, _ = fixtures()
    # Two AND alternatives have different second premises. Neither is complete.
    different = make_witness("R1", "EXPLICIT_SUPPORT", (witness.spans[1],
        ReviewedSpan(document_version_id=docs["b"].document_version_id,
            document_normalized_sha256=docs["b"].normalized_sha256, source_uri=docs["b"].source_uri,
            span=LiteralSpan(start_char=0, end_char=11, quote=docs["b"].normalized_text[:11]))), "Synthetic second alternative.")
    result = run(("a1", "b"), witnesses=(witness, different))
    assert result["reviewed_requirements_present"] == 0


@pytest.mark.parametrize("kind,status", [("MIS_SCOPED_IF_GENERALIZED", "NO_COMPLETE_REVIEWED_WITNESS"),
    ("CONTRADICTORY", "CONFLICTING_REVIEW_ANNOTATIONS_REQUIRE_ADJUDICATION"),
    ("QUALIFIED_INFERENCE", "QUALIFIED_REVIEWED_WITNESS_PRESENT")])
def test_scope_warning_conflict_and_qualified_support_remain_distinct(kind, status):
    _, _, _, w, _ = fixtures()
    other = make_witness("R1", kind, w.spans, "Synthetic explicit review category.")
    result = run(("a",), witnesses=(other,))
    assert result["requirements"][0]["status"] == status
    if kind == "MIS_SCOPED_IF_GENERALIZED":
        assert result["requirements"][0]["scope_cautions"] == [other.witness_id]


def test_support_and_contradiction_require_adjudication_instead_of_cherry_picking():
    _, _, _, w, _ = fixtures()
    other = make_witness("R1", "CONTRADICTORY", w.spans, "Synthetic conflict.")
    assert run(("a",), witnesses=(w, other))["reviewed_requirements_present"] == 0


@pytest.mark.parametrize("kind", ["uri", "hash", "quote", "tenant", "document_text", "chunk_text", "chunk_map_key", "witness_identity", "unknown_requirement"])
def test_tampered_bindings_fail_closed(kind):
    def alter(rubric, ws, ds, cs, selected):
        w, alternate = ws
        if kind in {"uri", "hash", "quote"}:
            s = w.spans[0]
            if kind == "uri":
                s = s.model_copy(update={"source_uri": "https://example.com/wrong"})
            elif kind == "hash":
                s = s.model_copy(update={"document_normalized_sha256": "0" * 64})
            else:
                s = s.model_copy(update={"span": s.span.model_copy(update={"quote": "x" * len(s.span.quote)})})
            w = make_witness("R1", w.evidence_kind, (s, w.spans[1]), w.rationale)
        elif kind in {"tenant", "document_text"}:
            key = w.spans[0].document_version_id
            ds[key] = ds[key].model_copy(update={"tenant_id": "other"} if kind == "tenant"
                else {"normalized_text": "x" * len(ds[key].normalized_text)})
        elif kind == "chunk_text":
            key = selected[0]
            cs[key] = cs[key].model_copy(update={"text": "x" * len(cs[key].text)})
        elif kind == "chunk_map_key":
            cs[selected[0]] = next(c for k, c in cs.items() if k != selected[0])
        elif kind == "witness_identity":
            w = w.model_copy(update={"rationale": "tampered without identity change"})
        else:
            w = make_witness("UNKNOWN", w.evidence_kind, w.spans, w.rationale)
        return rubric, (w, alternate), ds, cs, selected
    with pytest.raises(ValueError):
        run(("a",), alter=alter)


@pytest.mark.parametrize("kind", ["duplicate_chunk", "too_many_chunks", "duplicate_witness", "wrong_intent"])
def test_duplicate_bounds_and_intent_are_rejected(kind):
    docs, chunks, rubric, w, alternative = fixtures()
    selected = [chunks["a"].chunk_id]
    ws = (w, alternative)
    if kind == "duplicate_chunk":
        selected *= 2
    elif kind == "too_many_chunks":
        selected = [str(i) for i in range(9)]
    elif kind == "duplicate_witness":
        ws = (w, w)
    with pytest.raises(ValueError, match="BOUNDS_OR_INTENT"):
        audit_question_coverage(rubric, ws, {d.document_version_id: d for d in docs.values()},
            {c.chunk_id: c for c in chunks.values()}, tuple(selected), intent="wrong" if kind == "wrong_intent" else "task", tenant_id="demo")


def test_rubric_hash_and_dependency_order_are_trusted_inputs():
    _, _, rubric, _, _ = fixtures()
    raw = rubric.model_dump_json().encode()
    assert load_rubric(raw, trusted_sha256=sha256_bytes(raw)) == rubric
    with pytest.raises(ValueError, match="TRUST_PIN"):
        load_rubric(raw, trusted_sha256="0" * 64)
    data = rubric.model_dump(mode="json")
    data["requirements"].reverse()
    with pytest.raises(ValueError, match="CYCLIC_REQUIREMENT"):
        QuestionRubric.model_validate_json(canonical_json(data))


def test_frozen_project_rubric_has_explicit_review_origin_and_eight_new_control_groups():
    from scripts.review_question_coverage import ROOT, RUBRIC, RUBRIC_SHA
    rubric = load_rubric((ROOT / RUBRIC).read_bytes(), trusted_sha256=RUBRIC_SHA)
    assert len(rubric.requirements) == 11 and len(rubric.controls_not_yet_retrieved) == 8
    assert rubric.human_reviewers == 0 and "not independent gold" in rubric.exposure
    assert len({c.group for c in rubric.controls_not_yet_retrieved}) == 8


def test_source_review_amendment_changes_only_k2_annotations_and_preserves_all_original_inputs(tmp_path, monkeypatch):
    import socket

    from scripts import amend_question_coverage as amendment
    from scripts import review_question_coverage as parent

    if not all((parent.OUTPUT / name).is_file() for name in (parent.SCOPE, "receipts.json")):
        pytest.skip("Requires the local source-review scope and recorded collection319 receipts")

    def blocked(*args, **kwargs):
        raise AssertionError("NETWORK_FORBIDDEN")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(amendment, "OUTPUT", tmp_path)
    pins, _, _ = parent.read_rows()
    original = (parent.OUTPUT / "receipts.json").read_bytes()
    amendment.freeze()
    payload = amendment.evaluate()
    assert len(payload["cells"]) == 216 and payload["annotation_changes"]
    assert {c["requirement_id"] for c in payload["annotation_changes"]} == {"K2"}
    assert all(c["after"] == "QUALIFIED_REVIEWED_WITNESS_PRESENT" for c in payload["annotation_changes"])
    assert (parent.OUTPUT / "receipts.json").read_bytes() == original
    assert parent.read_rows()[0] == pins
    assert payload["new_model_calls"] == payload["new_aws_calls"] == 0
