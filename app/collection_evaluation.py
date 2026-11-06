"""Pinned, paired-language annotations and source validation for collection QA.

This module never retrieves or calls a provider. Labels are external to runtime
queries, and UNASSIGNED authoring cases cannot be reported as held-out results.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

from pydantic import Field, field_validator, model_validator

from app.collection_review import ReviewScenarioSet
from app.collections import CollectionVersion, SnapshotCollection
from app.contracts import Chunk, DocumentVersion, Nonempty, Record, Sha256, sha256_bytes
from app.evaluation import EvidenceGroup, span_covered


class PairedEvidenceGroup(EvidenceGroup):
    required_fact_vi: Nonempty


class AnnotatedScenario(Record):
    scenario_id: Nonempty
    intent_group_id: Nonempty
    split: Literal["UNASSIGNED", "DEVELOPMENT", "EVALUATION"]
    kind: Literal["troubleshooting", "comparison", "workflow", "cross_service",
                  "account_unavailable", "ambiguous", "corpus_gap"]
    services: Annotated[tuple[Nonempty, ...], Field(min_length=1, max_length=10)]
    question_en: Annotated[str, Field(min_length=5, max_length=2000)]
    question_vi: Annotated[str, Field(min_length=5, max_length=2000)]
    annotation_status: Literal["DRAFT_COMPLETE", "NEEDS_EVIDENCE"]
    review_status: Literal["AI_DRAFT", "HUMAN_APPROVED", "REJECTED"]
    reviewer_type: Literal["AI", "HUMAN"]
    human_review_reference: Nonempty | None = None
    review_note: Nonempty
    answerable: bool
    reference_answer_en: Nonempty
    reference_answer_vi: Nonempty
    evidence_groups: Annotated[tuple[PairedEvidenceGroup, ...], Field(max_length=12)]
    control_reason: Nonempty | None = None
    control_scope_audit_sha256: Sha256 | None = None
    expected_behavior: Literal["ANSWER_WITH_EVIDENCE", "ACKNOWLEDGE_ACCOUNT_LIMITATION",
                               "CLARIFY_OR_CONDITIONAL", "ABSTAIN_IF_NO_CORPUS_EVIDENCE"]

    @field_validator("question_en", "question_vi", "reference_answer_en", "reference_answer_vi",
                     "review_note")
    @classmethod
    def not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("BLANK_ANNOTATION_TEXT")
        return value

    @model_validator(mode="after")
    def coherent(self) -> Self:
        ids = [g.group_id for g in self.evidence_groups]
        if len(ids) != len(set(ids)):
            raise ValueError("DUPLICATE_SCENARIO_EVIDENCE_GROUP")
        control = self.kind in {"account_unavailable", "ambiguous", "corpus_gap"}
        if self.answerable == control:
            raise ValueError("CONTROL_ANSWERABILITY_MISMATCH")
        if self.answerable:
            if self.control_reason or self.expected_behavior != "ANSWER_WITH_EVIDENCE":
                raise ValueError("ANSWERABLE_CONTROL_FIELDS")
            if self.annotation_status == "DRAFT_COMPLETE" and not self.evidence_groups:
                raise ValueError("COMPLETE_ANSWERABLE_NEEDS_EVIDENCE")
        else:
            behavior = {"account_unavailable": "ACKNOWLEDGE_ACCOUNT_LIMITATION",
                        "ambiguous": "CLARIFY_OR_CONDITIONAL",
                        "corpus_gap": "ABSTAIN_IF_NO_CORPUS_EVIDENCE"}[self.kind]
            if not self.control_reason or self.expected_behavior != behavior or self.evidence_groups:
                raise ValueError("CONTROL_RUBRIC_MISMATCH")
        if self.review_status == "HUMAN_APPROVED" and (
            self.reviewer_type != "HUMAN" or self.annotation_status != "DRAFT_COMPLETE"
            or not self.human_review_reference
        ):
            raise ValueError("HUMAN_REVIEW_REQUIRED")
        if self.kind == "corpus_gap" and self.annotation_status == "DRAFT_COMPLETE":
            if self.control_scope_audit_sha256 is None:
                raise ValueError("CORPUS_GAP_SCOPE_REVIEW_REQUIRED")
        return self


class CollectionEvaluationDataset(Record):
    schema_version: Literal[1]
    dataset_id: Nonempty
    source_collection_version: CollectionVersion
    source_manifest_sha256: Sha256
    scenario_set_sha256: Sha256
    annotation_policy: Literal["SOURCE_FIRST_PAIRED_AI_DRAFT_V1"]
    split_status: Literal["UNASSIGNED", "GROUPED_FROZEN"]
    source_policy: Literal["PINNED_CORPUS_ONLY"]
    scenarios: Annotated[tuple[AnnotatedScenario, ...], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        for values in ([s.scenario_id for s in self.scenarios],
                       [s.question_en.strip().casefold() for s in self.scenarios],
                       [s.question_vi.strip().casefold() for s in self.scenarios]):
            if len(set(values)) != len(values):
                raise ValueError("DUPLICATE_COLLECTION_SCENARIO")
        groups = {}
        for scenario in self.scenarios:
            groups.setdefault(scenario.intent_group_id, set()).add(scenario.split)
        if any(len(values) != 1 for values in groups.values()):
            raise ValueError("INTENT_GROUP_SPLIT_LEAKAGE")
        if self.split_status == "UNASSIGNED":
            if any(s.split != "UNASSIGNED" for s in self.scenarios):
                raise ValueError("UNASSIGNED_SPLIT_HAS_ASSIGNMENTS")
        elif any(s.split == "UNASSIGNED" or s.annotation_status != "DRAFT_COMPLETE"
                 for s in self.scenarios):
            raise ValueError("FROZEN_DATASET_INCOMPLETE")
        return self


@dataclass(frozen=True)
class CollectionEvidenceIndex:
    collection_version: str
    manifest_sha256: str
    documents: Mapping[str, DocumentVersion]
    chunks: Mapping[str, Chunk]

    @classmethod
    def from_collection(cls, collection: SnapshotCollection) -> "CollectionEvidenceIndex":
        documents, chunks = {}, {}
        for ref in collection.manifest.shards:
            snapshot = collection.get_shard(ref.shard_id)
            for source, target in ((snapshot.documents, documents), (snapshot.chunks, chunks)):
                if target.keys() & source.keys():
                    raise ValueError("COLLECTION_EVIDENCE_DUPLICATE_ID")
                target.update(source)
        return cls(collection.manifest.collection_version, collection.manifest_sha256,
                   documents, chunks)


class CorpusGapReviewCase(Record):
    scenario_id: Nonempty
    conclusion: Literal["INSUFFICIENT_TASK_EVIDENCE_AI_DRAFT"]
    mention_document_version_ids: tuple[Nonempty, ...]
    rationale: Nonempty


class CorpusGapReview(Record):
    schema_version: Literal[1]
    review_id: Nonempty
    source_manifest_sha256: Sha256
    scenario_set_sha256: Sha256
    probe_artifact_sha256: Sha256
    documents_scanned: Annotated[int, Field(ge=1)]
    reviewer_type: Literal["AI"]
    semantic_unanswerability_proven: Literal[False]
    cases: Annotated[tuple[CorpusGapReviewCase, ...], Field(min_length=1, max_length=100)]


def validate_control_scope_review(
    dataset: CollectionEvaluationDataset, index: CollectionEvidenceIndex, raw: bytes,
) -> None:
    review = CorpusGapReview.model_validate_json(raw)
    expected = {s.scenario_id: s for s in dataset.scenarios if s.kind == "corpus_gap"
                and s.annotation_status == "DRAFT_COMPLETE"}
    ids = [c.scenario_id for c in review.cases]
    if (len(set(ids)) != len(ids) or set(ids) != set(expected)
            or review.source_manifest_sha256 != index.manifest_sha256
            or review.scenario_set_sha256 != dataset.scenario_set_sha256
            or review.documents_scanned != len(index.documents)
            or any(s.control_scope_audit_sha256 != sha256_bytes(raw) for s in expected.values())
            or any(doc not in index.documents for c in review.cases for doc in c.mention_document_version_ids)):
        raise ValueError("CORPUS_GAP_SCOPE_REVIEW_MISMATCH")


def validate_scenario_binding(dataset: CollectionEvaluationDataset, scenario_bytes: bytes) -> None:
    if sha256_bytes(scenario_bytes) != dataset.scenario_set_sha256:
        raise ValueError("ANNOTATION_SCENARIO_SET_HASH_MISMATCH")
    original = ReviewScenarioSet.model_validate_json(scenario_bytes)
    by_id = {s.id: s for s in original.cases}
    if set(by_id) != {s.scenario_id for s in dataset.scenarios}:
        raise ValueError("ANNOTATION_SCENARIO_SET_IDS_MISMATCH")
    for scenario in dataset.scenarios:
        source = by_id[scenario.scenario_id]
        if (source.en, source.vi, source.kind, source.services) != (
            scenario.question_en, scenario.question_vi, scenario.kind, scenario.services
        ):
            raise ValueError("ANNOTATION_SCENARIO_QUESTION_MISMATCH")


def _validate_spans(scenario: AnnotatedScenario, index: CollectionEvidenceIndex) -> set:
    by_document = {}
    for chunk in index.chunks.values():
        by_document.setdefault(chunk.document_version_id, []).append(chunk)
    spans = set()
    for group in scenario.evidence_groups:
        for span in group.alternatives:
            doc = index.documents.get(span.document_version_id)
            if doc is None or doc.normalized_text[span.start_char:span.end_char] != span.quote:
                raise ValueError("COLLECTION_GOLD_SOURCE_QUOTE_MISMATCH")
            if not span_covered(span, by_document.get(span.document_version_id, ())):
                raise ValueError("COLLECTION_GOLD_SPAN_NOT_RECOVERABLE")
            spans.add((span.document_version_id, span.start_char, span.end_char))
    return spans


def validate_annotation_sources(
    dataset: CollectionEvaluationDataset, index: CollectionEvidenceIndex,
) -> dict:
    if (dataset.source_collection_version != index.collection_version
            or dataset.source_manifest_sha256 != index.manifest_sha256):
        raise ValueError("EVALUATION_COLLECTION_MISMATCH")
    spans = set()
    for scenario in dataset.scenarios:
        spans.update(_validate_spans(scenario, index))
    return {
        "scenario_count": len(dataset.scenarios), "language_inputs": 2 * len(dataset.scenarios),
        "annotation_counts": dict(Counter(s.annotation_status for s in dataset.scenarios)),
        "review_counts": dict(Counter(s.review_status for s in dataset.scenarios)),
        "answerable_count": sum(s.answerable for s in dataset.scenarios),
        "control_count": sum(not s.answerable for s in dataset.scenarios),
        "evidence_groups": sum(len(s.evidence_groups) for s in dataset.scenarios),
        "unique_source_spans": len(spans),
        "source_documents": len({s[0] for s in spans}),
        "split_status": dataset.split_status, "semantic_correctness_measured": False,
    }


def score_collection_retrieval(
    scenario: AnnotatedScenario, ranking: Sequence[str], context_ids: Sequence[str],
    index: CollectionEvidenceIndex, *, k: int = 5,
) -> dict:
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError("INVALID_EVALUATION_K")
    if scenario.review_status == "REJECTED" or scenario.annotation_status != "DRAFT_COMPLETE":
        raise ValueError("SCENARIO_NOT_READY_FOR_PROVISIONAL_SCORING")
    _validate_spans(scenario, index)
    for ids in (ranking, context_ids):
        if len(ids) != len(set(ids)) or any(c not in index.chunks for c in ids):
            raise ValueError("INVALID_COLLECTION_EVALUATION_CHUNK_IDS")
    if not scenario.answerable:
        return {"required_groups": 0, "covered_groups_at_k": None, "recall_at_k": None,
                "complete_evidence_at_k": None, "context_complete": None}
    top, context = ([index.chunks[c] for c in ids] for ids in (ranking[:k], context_ids))
    covered = [any(span_covered(a, top) for a in g.alternatives) for g in scenario.evidence_groups]
    packed = [any(span_covered(a, context) for a in g.alternatives) for g in scenario.evidence_groups]
    return {"required_groups": len(covered), "covered_groups_at_k": covered,
            "recall_at_k": sum(covered) / len(covered),
            "complete_evidence_at_k": int(all(covered)), "context_complete": int(all(packed))}
