"""Source-anchored development evaluation; no provider, graph labels or answer judge."""

from collections import Counter
from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from app.contracts import Chunk, Nonempty, Record, Sha256
from app.snapshot import LoadedSnapshot


class GoldSpan(Record):
    document_version_id: Nonempty
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(gt=0)]
    quote: Annotated[str, Field(min_length=1, max_length=4000)]

    @model_validator(mode="after")
    def valid_range(self) -> Self:
        if self.end_char - self.start_char != len(self.quote):
            raise ValueError("GOLD_RANGE_LENGTH_MISMATCH")
        return self


class EvidenceGroup(Record):
    group_id: Nonempty
    required_fact: Nonempty
    alternatives: Annotated[tuple[GoldSpan, ...], Field(min_length=1, max_length=8)]


class EvalItem(Record):
    question_id: Nonempty
    intent_group_id: Nonempty
    category: Literal["FACTUAL", "ENTITY_RELATION", "MULTI_HOP", "UNANSWERABLE"]
    question: Annotated[str, Field(min_length=5, max_length=2000)]
    answerable: bool
    reference_answer: Nonempty
    evidence_groups: Annotated[tuple[EvidenceGroup, ...], Field(max_length=12)]
    review_status: Literal["AI_DRAFT", "HUMAN_APPROVED", "REJECTED"]
    review_note: Nonempty
    multi_hop_rationale: Nonempty | None = None
    unanswerable_reason: Nonempty | None = None

    @model_validator(mode="after")
    def coherent_label(self) -> Self:
        ids = [g.group_id for g in self.evidence_groups]
        if len(set(ids)) != len(ids):
            raise ValueError("DUPLICATE_EVIDENCE_GROUP")
        if self.answerable != bool(self.evidence_groups) or (
            (self.category == "UNANSWERABLE") == self.answerable
        ):
            raise ValueError("ANSWERABILITY_EVIDENCE_MISMATCH")
        if not self.answerable and not self.unanswerable_reason:
            raise ValueError("UNANSWERABLE_REASON_REQUIRED")
        if self.answerable and self.unanswerable_reason is not None:
            raise ValueError("ANSWERABLE_HAS_UNANSWERABLE_REASON")
        if self.category == "MULTI_HOP":
            if len(ids) < 2 or not self.multi_hop_rationale:
                raise ValueError("MULTI_HOP_REQUIRES_GROUPS_AND_RATIONALE")
            # Necessary structural condition, not a proof of semantic necessity.
            documents = [{a.document_version_id for a in g.alternatives} for g in self.evidence_groups]
            if set.intersection(*documents):
                raise ValueError("MULTI_HOP_ONE_DOCUMENT_COVERS_ALL_APPROVED_GROUPS")
        elif self.multi_hop_rationale is not None:
            raise ValueError("NON_MULTI_HOP_HAS_RATIONALE")
        return self


class DevelopmentDataset(Record):
    schema_version: Literal[1]
    dataset_id: Nonempty
    split: Literal["DEVELOPMENT"]
    language: Literal["en", "vi"]
    source_snapshot_version: Nonempty
    source_manifest_sha256: Sha256
    annotation_policy: Literal["SOURCE_FIRST_AI_DRAFT_REQUIRES_HUMAN_REVIEW_V1"]
    items: Annotated[tuple[EvalItem, ...], Field(min_length=1, max_length=50)]

    @model_validator(mode="after")
    def unique_items(self) -> Self:
        for values in ([i.question_id for i in self.items], [i.question.strip().casefold() for i in self.items]):
            if len(values) != len(set(values)):
                raise ValueError("DUPLICATE_EVALUATION_ITEM")
        return self


def validate_sources(dataset: DevelopmentDataset, snapshot: LoadedSnapshot) -> None:
    """Match pinned source versions/text before any rankings are observed."""
    if (dataset.source_snapshot_version != snapshot.manifest.snapshot_version
            or dataset.source_manifest_sha256 != snapshot.manifest_sha256):
        raise ValueError("EVALUATION_SNAPSHOT_MISMATCH")
    for item in dataset.items:
        for group in item.evidence_groups:
            for span in group.alternatives:
                doc = snapshot.documents.get(span.document_version_id)
                if doc is None or doc.normalized_text[span.start_char:span.end_char] != span.quote:
                    raise ValueError("GOLD_SOURCE_QUOTE_MISMATCH")


def span_covered(span: GoldSpan, chunks: Sequence[Chunk]) -> bool:
    """Cover an entire approved span with a gap-free union of source ranges."""
    cursor = span.start_char
    ranges = sorted((c.start_char, c.end_char) for c in chunks
                    if c.document_version_id == span.document_version_id)
    for start, end in ranges:
        if end <= cursor:
            continue
        if start > cursor:
            return False
        cursor = end
        if cursor >= span.end_char:
            return True
    return False


def evidence_coverage(item: EvalItem, chunks: Sequence[Chunk]) -> tuple[bool, ...]:
    return tuple(any(span_covered(a, chunks) for a in group.alternatives) for group in item.evidence_groups)


def score_retrieval(item: EvalItem, ranking: Sequence[str], context_ids: Sequence[str],
                    snapshot: LoadedSnapshot, *, k: int = 5) -> dict:
    if type(k) is not int or not 1 <= k <= 20:
        raise ValueError("INVALID_EVALUATION_K")
    for values in (ranking, context_ids):
        if len(set(values)) != len(values) or any(key not in snapshot.chunks for key in values):
            raise ValueError("INVALID_EVALUATION_CHUNK_IDS")
    if item.review_status == "REJECTED":
        raise ValueError("REJECTED_EVALUATION_ITEM")
    if not item.answerable:
        return {"required_groups": 0, "covered_groups_at_k": None, "recall_at_k": None,
                "complete_evidence_at_k": None, "mrr_overlap_at_k": None, "context_complete": None}
    top = [snapshot.chunks[key] for key in ranking[:k]]
    covered = evidence_coverage(item, top)
    context = evidence_coverage(item, [snapshot.chunks[key] for key in context_ids])
    # Explicit overlap diagnostic, not a claim that a partial span supports a fact.
    first = next((rank for rank, c in enumerate(top, 1) if any(
        c.document_version_id == span.document_version_id
        and c.start_char < span.end_char and span.start_char < c.end_char
        for g in item.evidence_groups for span in g.alternatives)), None)
    return {"required_groups": len(covered), "covered_groups_at_k": list(covered),
            "recall_at_k": sum(covered) / len(covered), "complete_evidence_at_k": int(all(covered)),
            "mrr_overlap_at_k": 1 / first if first else 0.0, "context_complete": int(all(context))}


def aggregate_retrieval(rows: Sequence[dict], items: Sequence[EvalItem], *, approved_only: bool) -> dict:
    """Per-question macro averages; unanswerable rows have no retrieval denominator."""
    lookup = {i.question_id: i for i in items}
    ids = [r["question_id"] for r in rows]
    if len(set(ids)) != len(ids) or set(ids) != set(lookup):
        raise ValueError("INCOMPLETE_OR_DUPLICATE_EVALUATION_ROWS")
    included = [r for r in rows if lookup[r["question_id"]].answerable
                and lookup[r["question_id"]].review_status != "REJECTED"
                and (not approved_only or lookup[r["question_id"]].review_status == "HUMAN_APPROVED")]
    metrics = ("recall_at_k", "complete_evidence_at_k", "mrr_overlap_at_k", "context_complete")
    return {"answerable_denominator": len(included),
            "intent_groups": len({lookup[r["question_id"]].intent_group_id for r in included}),
            "review_counts": dict(Counter(i.review_status for i in items)),
            "unanswerable_excluded": sum(not i.answerable for i in items),
            **{key: sum(r["metrics"][key] for r in included) / len(included) if included else None
               for key in metrics}}
