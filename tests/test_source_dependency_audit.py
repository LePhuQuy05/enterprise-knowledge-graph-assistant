"""Qualifier loss and alternate overlapping-source coverage counterexamples."""

import pytest

from app.authorization_evidence import LiteralSpan
from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes
from app.graph_document_allocation import document_diverse_tail
from app.source_dependency_audit import (
    SourceDependency,
    audit_source_dependencies,
    make_source_dependency,
    missing_intervals,
)


def fixture_case():
    from datetime import UTC, datetime

    text = "EXAMPLE: The policy permits the operation.\n\nCONDITION: Only for the named account, never every account."
    doc = DocumentVersion.create(tenant_id="demo", source_uri="https://docs.aws.amazon.com/lambda/latest/dg/example.html",
        title="Synthetic dependency fixture", fetched_at=datetime(2026, 10, 27, tzinfo=UTC), raw_sha256=sha256_bytes(text.encode()),
        normalizer_version="fixture", normalized_text_path="fixture.txt", normalized_text=text, source_kind="TEXT")
    split = text.index("CONDITION")

    def chunk(start, end, label):
        return Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id,
            position=start, section_path=("Synthetic example",), chunking_version="fixture", tokenizer_version="fixture",
            start_char=start, end_char=end, text=text[start:end], token_count=30)

    primary, qualifier, alternate = chunk(0, split, "a"), chunk(split, len(text), "a2"), chunk(0, len(text), "a-whole")
    anchor = LiteralSpan(start_char=0, end_char=split, quote=text[:split])
    required = LiteralSpan(start_char=split, end_char=len(text), quote=text[split:])
    dep = make_source_dependency(doc, anchor, (required,), review_reason="Synthetic example requires its following condition.", fixture=True)
    return doc, dict(zip(("a", "a2", "a-whole"), (primary, qualifier, alternate), strict=True)), dep


def audit(keys, *, change=None):
    doc, chunks, dependency = fixture_case()
    if change:
        doc, chunks, dependency = change(doc, chunks, dependency)
    return audit_source_dependencies({doc.document_version_id: doc}, {c.chunk_id: c for c in chunks.values()},
        tuple(chunks[k].chunk_id for k in keys), (dependency,), tenant_id="demo")


def test_missing_qualifier_is_reported_without_inserting_sources_or_changing_status():
    result = audit(("a",))
    assert result["incomplete_count"] == 1
    assert result["checks"][0]["status"] == "INCOMPLETE_REVIEWED_SOURCE_CONTEXT"
    assert result["selected_chunk_ids_unchanged"] == (fixture_case()[1]["a"].chunk_id,) and not result["serving_gate_enabled"]


@pytest.mark.parametrize("keys", [("a", "a2"), ("a-whole",)])
def test_any_selected_interval_union_can_supply_the_qualifier_not_a_gold_chunk_id(keys):
    assert audit(keys)["checks"][0]["status"] == "COMPLETE_REVIEWED_SOURCE_CONTEXT"


def test_without_anchor_no_claim_of_missing_question_evidence_is_made():
    assert audit(("a2",))["checks"][0]["status"] == "NOT_TRIGGERED"


def test_diversity_can_defer_a_necessary_qualifier_beyond_eight_slots():
    pool = ("a", "a2", *(f"d{i}" for i in range(7)))
    documents = {k: "one" if k in {"a", "a2"} else k for k in pool}
    order, _ = document_diverse_tail(pool, ("a",), documents)
    assert order[:8] == ("a", *(f"d{i}" for i in range(7))) and order[-1] == "a2"
    # Other documents do not provide an interval in the anchor's document.
    assert audit(("a",))["incomplete_count"] == 1


@pytest.mark.parametrize("covered,expected", [([(0, 4), (3, 9)], ()), ([(0, 4), (4, 9)], ()),
    ([(0, 3), (5, 9)], ((3, 5),)), ([], ((0, 9),)), ([(2, 7)], ((0, 2), (7, 9)))])
def test_interval_coverage_handles_overlap_adjacency_and_holes(covered, expected):
    assert missing_intervals(0, 9, covered) == expected


@pytest.mark.parametrize("kind", ["tenant", "source_uri", "document_hash", "quote"])
def test_changed_dependency_source_bindings_fail_closed(kind):
    def change(doc, chunks, dep):
        if kind == "tenant":
            dep = dep.model_copy(update={"tenant_id": "other"})
        elif kind == "source_uri":
            dep = dep.model_copy(update={"source_uri": "https://example.com/fake"})
        elif kind == "document_hash":
            dep = dep.model_copy(update={"document_normalized_sha256": "0" * 64})
        else:
            dep = dep.model_copy(update={"anchor": dep.anchor.model_copy(update={"quote": "X" * len(dep.anchor.quote)})})
        return doc, chunks, dep
    with pytest.raises(ValueError, match="SOURCE_DEPENDENCY_REVIEW_"):
        audit(("a",), change=change)


def test_duplicate_selected_source_ids_fail_closed():
    with pytest.raises(ValueError, match="BOUNDS"):
        audit(("a", "a"))


def test_dependency_identity_is_bound_to_the_reviewed_span_and_reason():
    _, _, dependency = fixture_case()
    data = dependency.model_dump(mode="json") | {"review_reason": "changed"}
    with pytest.raises(ValueError, match="IDENTITY"):
        SourceDependency.model_validate_json(canonical_json(data))
