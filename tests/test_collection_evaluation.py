from dataclasses import replace
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.collection_evaluation import (
    AnnotatedScenario,
    CollectionEvaluationDataset,
    CollectionEvidenceIndex,
    PairedEvidenceGroup,
    score_collection_retrieval,
    validate_annotation_sources,
    validate_control_scope_review,
    validate_scenario_binding,
)
from app.contracts import canonical_json, sha256_bytes
from app.evaluation import GoldSpan


@pytest.fixture
def index(loaded):
    return CollectionEvidenceIndex("collection_" + "a" * 64, "b" * 64,
                                   loaded.documents, loaded.chunks)


def group(chunk, number=1):
    return PairedEvidenceGroup(group_id=f"fact{number}", required_fact="Fixture required fact",
                              required_fact_vi="Ý bắt buộc của fixture", alternatives=(GoldSpan(
                                  document_version_id=chunk.document_version_id,
                                  start_char=chunk.start_char, end_char=chunk.end_char,
                                  quote=chunk.text),))


def scenario(groups=(), **changes):
    values = dict(scenario_id="s1", intent_group_id="intent1", split="UNASSIGNED",
                  kind="workflow", services=("Fixture",), question_en="What does this fixture do?",
                  question_vi="Fixture này làm gì?", annotation_status="DRAFT_COMPLETE",
                  review_status="AI_DRAFT", reviewer_type="AI", review_note="Synthetic test only.",
                  answerable=True, reference_answer_en="A fixture response.",
                  reference_answer_vi="Câu trả lời fixture.", evidence_groups=groups,
                  expected_behavior="ANSWER_WITH_EVIDENCE")
    values.update(changes)
    return AnnotatedScenario(**values)


def dataset(index, cases, **changes):
    values = dict(schema_version=1, dataset_id="fixture", source_collection_version=index.collection_version,
                  source_manifest_sha256=index.manifest_sha256, scenario_set_sha256="c" * 64,
                  annotation_policy="SOURCE_FIRST_PAIRED_AI_DRAFT_V1", split_status="UNASSIGNED",
                  source_policy="PINNED_CORPUS_ONLY", scenarios=cases)
    values.update(changes)
    return CollectionEvaluationDataset(**values)


def test_union_coverage_source_validation_and_context_are_separate(index):
    chunks = list(index.chunks.values())[:2]
    case = scenario(tuple(group(c, n) for n, c in enumerate(chunks, 1)))
    audit = validate_annotation_sources(dataset(index, (case,)), index)
    assert audit["evidence_groups"] == 2
    result = score_collection_retrieval(case, [c.chunk_id for c in chunks],
                                        [chunks[0].chunk_id], index, k=1)
    assert result["recall_at_k"] == .5
    assert result["complete_evidence_at_k"] == result["context_complete"] == 0
    result = score_collection_retrieval(case, [c.chunk_id for c in chunks],
                                        [c.chunk_id for c in chunks], index)
    assert result["complete_evidence_at_k"] == result["context_complete"] == 1


@pytest.mark.parametrize("fault", ["quote", "offset", "unknown_document"])
def test_bad_source_is_rejected_even_when_scoring_directly(index, fault):
    chunk = next(iter(index.chunks.values()))
    fact = group(chunk)
    span = fact.alternatives[0]
    changes = {"quote": {"quote": "X" + span.quote[1:]},
               "offset": {"start_char": span.start_char + 1, "end_char": span.end_char + 1},
               "unknown_document": {"document_version_id": "missing"}}[fault]
    fact = fact.model_copy(update={"alternatives": (span.model_copy(update=changes),)})
    case = scenario((fact,))
    with pytest.raises(ValueError, match="SOURCE_QUOTE_MISMATCH"):
        validate_annotation_sources(dataset(index, (case,)), index)
    with pytest.raises(ValueError, match="SOURCE_QUOTE_MISMATCH"):
        score_collection_retrieval(case, [chunk.chunk_id], [], index)


def test_unrecoverable_gold_is_not_silently_a_retriever_loss(index):
    chunk = next(iter(index.chunks.values()))
    case = scenario((group(chunk),))
    broken = replace(index, chunks={})
    with pytest.raises(ValueError, match="SPAN_NOT_RECOVERABLE"):
        validate_annotation_sources(dataset(index, (case,)), broken)


def test_collection_hash_and_version_mismatch(index):
    case = scenario((group(next(iter(index.chunks.values()))),))
    for change in ({"source_manifest_sha256": "0" * 64},
                   {"source_collection_version": "collection_" + "0" * 64}):
        with pytest.raises(ValueError, match="COLLECTION_MISMATCH"):
            validate_annotation_sources(dataset(index, (case,), **change), index)


def test_pending_annotation_cannot_score_or_freeze(index):
    case = scenario(annotation_status="NEEDS_EVIDENCE")
    assert validate_annotation_sources(dataset(index, (case,)), index)["evidence_groups"] == 0
    with pytest.raises(ValueError, match="NOT_READY"):
        score_collection_retrieval(case, [], [], index)
    assigned = case.model_copy(update={"split": "EVALUATION"})
    with pytest.raises(ValidationError, match="FROZEN_DATASET_INCOMPLETE"):
        dataset(index, (assigned,), split_status="GROUPED_FROZEN")


@pytest.mark.parametrize("kind,behavior", [
    ("account_unavailable", "ACKNOWLEDGE_ACCOUNT_LIMITATION"),
    ("ambiguous", "CLARIFY_OR_CONDITIONAL"),
    ("corpus_gap", "ABSTAIN_IF_NO_CORPUS_EVIDENCE"),
])
def test_controls_have_distinct_rubrics_and_null_retrieval(index, kind, behavior):
    case = scenario(kind=kind, answerable=False, control_reason="Fixture missing inputs.",
                    expected_behavior=behavior,
                    control_scope_audit_sha256="d" * 64 if kind == "corpus_gap" else None)
    result = score_collection_retrieval(case, [], [], index)
    assert result["required_groups"] == 0
    assert all(value is None for key, value in result.items() if key != "required_groups")
    with pytest.raises(ValidationError, match="CONTROL_RUBRIC_MISMATCH"):
        scenario(kind=kind, answerable=False, control_reason="Fixture missing inputs.",
                 expected_behavior="ANSWER_WITH_EVIDENCE")


def test_intent_group_cannot_cross_splits_and_unassigned_is_not_frozen(index):
    first = scenario((group(next(iter(index.chunks.values()))),), split="DEVELOPMENT")
    other = first.model_copy(update={"scenario_id": "s2", "question_en": "Another English question?",
                                    "question_vi": "Một câu hỏi khác?", "split": "EVALUATION"})
    with pytest.raises(ValidationError, match="INTENT_GROUP_SPLIT_LEAKAGE"):
        dataset(index, (first, other), split_status="GROUPED_FROZEN")
    with pytest.raises(ValidationError, match="UNASSIGNED_SPLIT_HAS_ASSIGNMENTS"):
        dataset(index, (first,))


def test_complete_empty_facts_and_ai_human_promotion_are_rejected(index):
    with pytest.raises(ValidationError, match="COMPLETE_ANSWERABLE_NEEDS_EVIDENCE"):
        scenario()
    facts = (group(next(iter(index.chunks.values()))),)
    with pytest.raises(ValidationError, match="HUMAN_REVIEW_REQUIRED"):
        scenario(facts, review_status="HUMAN_APPROVED")
    with pytest.raises(ValidationError, match="HUMAN_REVIEW_REQUIRED"):
        scenario(facts, review_status="HUMAN_APPROVED", reviewer_type="HUMAN")
    with pytest.raises(ValidationError, match="BLANK_ANNOTATION_TEXT"):
        scenario(facts, question_en="     ")


def test_question_and_artifact_binding(index):
    case = scenario((group(next(iter(index.chunks.values()))),))
    source = {"schema_version": 1, "version": "test", "purpose": "DEVELOPMENT_WORKFLOW_FIXTURES_NOT_GOLD_OR_HOLDOUT",
              "semantic_review": "NOT_REVIEWED", "quality_metrics_available": False,
              "cases": [{"id": case.scenario_id, "kind": case.kind, "services": ["Fixture"],
                         "en": case.question_en, "vi": case.question_vi}], "notice": "Synthetic test."}
    raw = canonical_json(source)
    value = dataset(index, (case,), scenario_set_sha256=sha256_bytes(raw))
    validate_scenario_binding(value, raw)
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_scenario_binding(value, raw + b"\n")
    changed = case.model_copy(update={"question_vi": "Câu hỏi bị đổi?"})
    with pytest.raises(ValueError, match="QUESTION_MISMATCH"):
        validate_scenario_binding(value.model_copy(update={"scenarios": (changed,)}), raw)


def test_unknown_and_duplicate_rankings_and_invalid_k(index):
    chunk = next(iter(index.chunks.values()))
    case = scenario((group(chunk),))
    for ranking in (["unknown"], [chunk.chunk_id, chunk.chunk_id]):
        with pytest.raises(ValueError, match="INVALID_COLLECTION_EVALUATION_CHUNK_IDS"):
            score_collection_retrieval(case, ranking, [], index)
    for k in (0, 21, True):
        with pytest.raises(ValueError, match="INVALID_EVALUATION_K"):
            score_collection_retrieval(case, [], [], index, k=k)


def test_index_refuses_colliding_ids_across_shards(loaded):
    collection = SimpleNamespace(manifest=SimpleNamespace(shards=(SimpleNamespace(shard_id="a"),
                                                                  SimpleNamespace(shard_id="b"))),
                                 get_shard=lambda _: loaded)
    with pytest.raises(ValueError, match="DUPLICATE_ID"):
        CollectionEvidenceIndex.from_collection(collection)


def test_gap_cannot_be_completed_without_a_scope_review():
    with pytest.raises(ValidationError, match="CORPUS_GAP_SCOPE_REVIEW_REQUIRED"):
        scenario(kind="corpus_gap", answerable=False, control_reason="Corpus-specific gap.",
                 expected_behavior="ABSTAIN_IF_NO_CORPUS_EVIDENCE")


def test_gap_review_is_bound_to_questions_collection_and_complete_inventory(index):
    source = {"schema_version": 1, "review_id": "fixture", "source_manifest_sha256": index.manifest_sha256,
              "scenario_set_sha256": "c" * 64, "probe_artifact_sha256": "d" * 64,
              "documents_scanned": len(index.documents), "reviewer_type": "AI",
              "semantic_unanswerability_proven": False,
              "cases": [{"scenario_id": "s1", "conclusion": "INSUFFICIENT_TASK_EVIDENCE_AI_DRAFT",
                         "mention_document_version_ids": [], "rationale": "Synthetic scope review."}]}
    raw = canonical_json(source)
    case = scenario(kind="corpus_gap", answerable=False, control_reason="Corpus-specific gap.",
                    expected_behavior="ABSTAIN_IF_NO_CORPUS_EVIDENCE",
                    control_scope_audit_sha256=sha256_bytes(raw))
    data = dataset(index, (case,))
    validate_control_scope_review(data, index, raw)
    for update in ({"documents_scanned": 1 + len(index.documents)},
                   {"source_manifest_sha256": "0" * 64},
                   {"scenario_set_sha256": "0" * 64}):
        with pytest.raises(ValueError, match="SCOPE_REVIEW_MISMATCH"):
            validate_control_scope_review(data, index, canonical_json(source | update))
    with pytest.raises(ValueError, match="SCOPE_REVIEW_MISMATCH"):
        validate_control_scope_review(data, index, raw + b"\n")


def test_span_union_rejects_a_one_character_gap(index):
    chunk = next(iter(index.chunks.values()))
    case = scenario((group(chunk),))
    middle = chunk.start_char + len(chunk.text) // 2
    left = chunk.model_copy(update={"end_char": middle, "text": chunk.text[:middle-chunk.start_char]})
    right = chunk.model_copy(update={"chunk_id": "right", "start_char": middle,
                                     "text": chunk.text[middle-chunk.start_char:]})
    contiguous = replace(index, chunks={left.chunk_id: left, right.chunk_id: right})
    assert score_collection_retrieval(case, list(contiguous.chunks), [], contiguous)["complete_evidence_at_k"] == 1
    gap = right.model_copy(update={"start_char": middle+1, "text": right.text[1:]})
    broken = replace(index, chunks={left.chunk_id: left, gap.chunk_id: gap})
    with pytest.raises(ValueError, match="SPAN_NOT_RECOVERABLE"):
        validate_annotation_sources(dataset(index, (case,)), broken)


def test_cli_cannot_overwrite_scope_review_input(tmp_path):
    from scripts.audit_collection_annotations import main

    target = tmp_path / "review.json"
    with pytest.raises(SystemExit) as error:
        main(["--scope-review", str(target), "--output", str(target)])
    assert error.value.code == 2
    assert not target.exists()
