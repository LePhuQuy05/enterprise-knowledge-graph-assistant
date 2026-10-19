import dataclasses

import pytest
from pydantic import ValidationError

from app.contracts import Chunk, canonical_json
from app.evaluation import (
    DevelopmentDataset,
    EvalItem,
    EvidenceGroup,
    GoldSpan,
    aggregate_retrieval,
    score_retrieval,
    span_covered,
    validate_sources,
)


def span(chunk):
    return GoldSpan(document_version_id=chunk.document_version_id, start_char=chunk.start_char,
                    end_char=chunk.end_char, quote=chunk.text)


def item(*groups, **changes):
    value = dict(question_id="dev-01", intent_group_id="chain-a", category="ENTITY_RELATION",
                 question="What does Service A use?", answerable=True, reference_answer="Service B.",
                 evidence_groups=tuple(EvidenceGroup(group_id=f"g{i}", required_fact="A required fact",
                     alternatives=tuple(g)) for i, g in enumerate(groups)),
                 review_status="AI_DRAFT", review_note="Synthetic fixture, not human reviewed.")
    value.update(changes)
    return EvalItem(**value)


def dataset(loaded, *items):
    return DevelopmentDataset(schema_version=1, dataset_id="test-dev-v1", split="DEVELOPMENT", language="en",
        source_snapshot_version=loaded.manifest.snapshot_version, source_manifest_sha256=loaded.manifest_sha256,
        annotation_policy="SOURCE_FIRST_AI_DRAFT_REQUIRES_HUMAN_REVIEW_V1", items=items)


def test_source_offsets_and_unicode_are_validated_before_scoring(loaded):
    chunk = next(c for c in loaded.content.chunks if "tiếng Việt" in c.text)
    value = dataset(loaded, item((span(chunk),)))
    validate_sources(value, loaded)
    corrupted = span(chunk).model_copy(update={"quote": "X" + chunk.text[1:]})
    with pytest.raises(ValueError, match="GOLD_SOURCE_QUOTE_MISMATCH"):
        validate_sources(dataset(loaded, item((corrupted,))), loaded)
    with pytest.raises(ValueError, match="EVALUATION_SNAPSHOT_MISMATCH"):
        validate_sources(value.model_copy(update={"source_manifest_sha256": "0" * 64}), loaded)


def test_boundary_spanning_gold_uses_gap_free_union_without_double_counting(loaded):
    original = loaded.content.chunks[0]
    target = span(original)

    def piece(start, end):
        doc = loaded.documents[original.document_version_id]
        return Chunk.create(tenant_id=original.tenant_id, document_version_id=doc.document_version_id,
            chunking_version="fixture-cut-v1", section_path=("fixture",), position=start,
            start_char=start, end_char=end, text=doc.normalized_text[start:end],
            token_count=1, tokenizer_version="fixture-words-v1")

    midpoint = (target.start_char + target.end_char) // 2
    left = piece(target.start_char, midpoint)
    right = piece(midpoint, target.end_char)
    gap = piece(midpoint + 1, target.end_char)
    assert not span_covered(target, (left,))
    assert span_covered(target, (right, left, left))
    assert not span_covered(target, (left, gap))
    other_doc = dataclasses.replace(loaded, chunks={left.chunk_id: left, right.chunk_id: right})
    result = score_retrieval(item((target,)), [left.chunk_id, right.chunk_id], [], other_doc)
    assert result["recall_at_k"] == result["complete_evidence_at_k"] == 1
    assert result["context_complete"] == 0


def test_macro_recall_complete_evidence_and_overlap_mrr_are_distinct(loaded):
    a, b, c = loaded.content.chunks[:3]
    value = item((span(a),), (span(b), span(c)))
    result = score_retrieval(value, [a.chunk_id], [a.chunk_id, c.chunk_id], loaded)
    assert result["recall_at_k"] == 0.5
    assert result["complete_evidence_at_k"] == 0
    assert result["mrr_overlap_at_k"] == result["context_complete"] == 1
    assert result["covered_groups_at_k"] == [True, False]


def test_wrong_document_same_quote_does_not_match(loaded):
    c = loaded.content.chunks[0]
    wrong = span(c).model_copy(update={"document_version_id": "another-document"})
    result = score_retrieval(item((wrong,)), [c.chunk_id], [c.chunk_id], loaded)
    assert result["complete_evidence_at_k"] == result["mrr_overlap_at_k"] == 0


def test_k_cutoff_and_context_are_separate(loaded):
    a, b = loaded.content.chunks[:2]
    result = score_retrieval(item((span(b),)), [a.chunk_id, b.chunk_id], [b.chunk_id], loaded, k=1)
    assert result["complete_evidence_at_k"] == 0
    assert result["context_complete"] == 1


def test_unanswerable_and_unreviewed_items_cannot_inflate_approved_metrics(loaded):
    a = loaded.content.chunks[0]
    draft = item((span(a),))
    unknown = item(question_id="dev-u", intent_group_id="unknown", category="UNANSWERABLE", answerable=False,
        evidence_groups=(), unanswerable_reason="No project-specific records in this source corpus.")
    rows = [{"question_id":i.question_id,"metrics":score_retrieval(i,[a.chunk_id],[],loaded)} for i in (draft, unknown)]
    assert all(rows[1]["metrics"][k] is None for k in ("recall_at_k", "complete_evidence_at_k", "context_complete"))
    provisional = aggregate_retrieval(rows, (draft, unknown), approved_only=False)
    approved = aggregate_retrieval(rows, (draft, unknown), approved_only=True)
    assert provisional["answerable_denominator"] == provisional["unanswerable_excluded"] == 1
    assert provisional["complete_evidence_at_k"] == 1
    assert approved["answerable_denominator"] == 0 and approved["complete_evidence_at_k"] is None


def test_human_approved_macro_average_is_per_question_not_pooled_groups(loaded):
    a, b, c = loaded.content.chunks[:3]
    first = item((span(a),), review_status="HUMAN_APPROVED")
    second = item((span(a),), (span(b),), (span(c),), question_id="dev-02", review_status="HUMAN_APPROVED")
    rows = [{"question_id":i.question_id,"metrics":score_retrieval(i,[a.chunk_id],[],loaded)} for i in (first, second)]
    result = aggregate_retrieval(rows, (first, second), approved_only=True)
    assert result["recall_at_k"] == pytest.approx(2 / 3)
    assert result["complete_evidence_at_k"] == 0.5
    assert result["intent_groups"] == 1  # Two paraphrases are not two independent groups.


def test_multi_hop_requires_no_single_document_covering_all_approved_groups(loaded):
    a, b = loaded.content.chunks[:2]
    other = next(c for c in loaded.content.chunks if c.document_version_id != a.document_version_id)
    with pytest.raises(ValidationError, match="ONE_DOCUMENT"):
        item((span(a),), (span(b),), category="MULTI_HOP", multi_hop_rationale="Two required facts.")
    value = item((span(a),), (span(other),), category="MULTI_HOP", multi_hop_rationale="Bridge requires both documents.")
    assert value.category == "MULTI_HOP"
    with pytest.raises(ValidationError, match="ONE_DOCUMENT"):
        item((span(a),), (span(b), span(other)), category="MULTI_HOP", multi_hop_rationale="Alternatives share one doc.")


@pytest.mark.parametrize("change", [{"schema_version": 2}, {"split": "HOLDOUT"}, {"split": "PILOT_EVAL"}, {"unknown": "x"}])
def test_development_contract_rejects_holdout_and_unknown_schema(loaded, change):
    data = dataset(loaded, item((span(loaded.content.chunks[0]),))).model_dump(mode="json")
    data.update(change)
    with pytest.raises(ValidationError):
        DevelopmentDataset.model_validate_json(canonical_json(data))


def test_duplicate_questions_ids_and_group_ids_are_rejected(loaded):
    a = span(loaded.content.chunks[0])
    first = item((a,))
    for second in (first, first.model_copy(update={"question_id":"other"})):
        with pytest.raises(ValidationError, match="DUPLICATE_EVALUATION_ITEM"):
            dataset(loaded, first, second)
    with pytest.raises(ValidationError, match="DUPLICATE_EVIDENCE_GROUP"):
        item(evidence_groups=(first.evidence_groups[0], first.evidence_groups[0]))


@pytest.mark.parametrize("ids", [["unknown"], ["duplicate", "duplicate"]])
def test_invalid_rankings_are_errors_not_zero_scores(loaded, ids):
    with pytest.raises(ValueError, match="INVALID_EVALUATION_CHUNK_IDS"):
        score_retrieval(item((span(loaded.content.chunks[0]),)), ids, [], loaded)


def test_missing_or_duplicate_result_rows_fail_instead_of_shrinking_denominator(loaded):
    value = item((span(loaded.content.chunks[0]),))
    row = {"question_id":value.question_id, "metrics":score_retrieval(value, [], [], loaded)}
    for rows in ([], [row, row]):
        with pytest.raises(ValueError, match="INCOMPLETE_OR_DUPLICATE_EVALUATION_ROWS"):
            aggregate_retrieval(rows, (value,), approved_only=False)
