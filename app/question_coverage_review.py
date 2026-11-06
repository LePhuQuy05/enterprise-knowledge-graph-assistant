"""Offline review annotations: verify source presence, never infer entailment.

Each alternative witness is an AND of original spans; witnesses are ORed.
Reviewers assign semantic meaning. This evaluator only verifies bindings and
interval coverage. It must not be imported by ranking or used as an access grant.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.authorization_evidence import LiteralSpan
from app.contracts import (
    Chunk,
    DocumentVersion,
    Record,
    Sha256,
    canonical_json,
    sha256_bytes,
    stable_id,
)
from app.source_dependency_audit import missing_intervals

ORIGIN = "AI_AUTHORED_NONBLIND_DEVELOPMENT_REVIEW"
PRESENT = {"EXPLICIT_REVIEWED_WITNESS_PRESENT", "QUALIFIED_REVIEWED_WITNESS_PRESENT",
    "QUALIFIED_SYNTHESIS_PREMISES_PRESENT"}


class Requirement(Record):
    id: str
    intent: str
    claim: str
    actor: str
    action: str
    resource: str
    condition: str
    kind: Literal["EXPLICIT_SUPPORT", "QUALIFIED_INFERENCE"]
    requires: tuple[str, ...] = ()


class DevelopmentControl(Record):
    id: str
    group: str
    en: str
    vi: str
    required: tuple[str, ...]
    expected: str | None = None


class QuestionRubric(Record):
    version: Literal["question-coverage-rubric-v1"]
    origin: Literal["AI_AUTHORED_NONBLIND_DEVELOPMENT_REVIEW"]
    human_reviewers: Literal[0]
    exposure: str
    scope: str
    missing_semantics: str
    requirements: Annotated[tuple[Requirement, ...], Field(min_length=1, max_length=100)]
    controls_not_yet_retrieved: tuple[DevelopmentControl, ...]

    @model_validator(mode="after")
    def references(self):
        seen = {}
        for req in self.requirements:
            if (req.id in seen or len(set(req.requires)) != len(req.requires)
                    or bool(req.requires) != (req.kind == "QUALIFIED_INFERENCE")
                    or any(key not in seen or seen[key].intent != req.intent for key in req.requires)):
                raise ValueError("QUESTION_RUBRIC_DUPLICATE_OR_CYCLIC_REQUIREMENT")
            seen[req.id] = req
        controls = self.controls_not_yet_retrieved
        if (len({c.id for c in controls}) != len(controls) or len({c.group for c in controls}) != len(controls)
                or any(not set(c.required) <= set(seen) or not c.required and not c.expected for c in controls)):
            raise ValueError("QUESTION_RUBRIC_CONTROL_REFERENCES")
        return self


def load_rubric(raw, *, trusted_sha256):
    if sha256_bytes(raw) != trusted_sha256:
        raise ValueError("QUESTION_RUBRIC_TRUST_PIN")
    return QuestionRubric.model_validate_json(raw)


class ReviewedSpan(Record):
    document_version_id: str
    document_normalized_sha256: Sha256
    source_uri: str
    span: LiteralSpan


class ReviewedWitness(Record):
    version: Literal["reviewed-question-witness-v1"] = "reviewed-question-witness-v1"
    witness_id: str
    requirement_id: str
    evidence_kind: Literal["EXPLICIT_SUPPORT", "QUALIFIED_INFERENCE", "MIS_SCOPED_IF_GENERALIZED", "CONTRADICTORY"]
    spans: Annotated[tuple[ReviewedSpan, ...], Field(min_length=1, max_length=8)]
    rationale: Annotated[str, Field(min_length=1, max_length=1600)]
    origin: Literal["AI_AUTHORED_NONBLIND_DEVELOPMENT_REVIEW"] = ORIGIN

    @model_validator(mode="after")
    def identity(self):
        keys = [(s.document_version_id, s.span.start_char, s.span.end_char) for s in self.spans]
        expected = stable_id("question-witness-v1", **self.model_dump(mode="json", exclude={"witness_id"}))
        if len(set(keys)) != len(keys) or self.witness_id != expected:
            raise ValueError("QUESTION_WITNESS_IDENTITY_OR_DUPLICATE_SPAN")
        return self


def make_witness(requirement_id, evidence_kind, spans, rationale):
    data = {"version": "reviewed-question-witness-v1", "requirement_id": requirement_id,
        "evidence_kind": evidence_kind, "spans": [s.model_dump(mode="json") for s in spans],
        "rationale": rationale, "origin": ORIGIN}
    return ReviewedWitness.model_validate_json(canonical_json(data | {
        "witness_id": stable_id("question-witness-v1", **data)}))


def audit_question_coverage(rubric, witnesses, documents, chunks, selected_ids, *, intent, tenant_id):
    """Apply the same externally pinned annotations to any selected context.

    Absence of a reviewed witness is an unresolved review result, not an
    automatic judgement that no equivalent evidence exists. No ranking changes.
    """
    rubric = QuestionRubric.model_validate_json(rubric.model_dump_json())
    requirements = {r.id: r for r in rubric.requirements}
    if (not any(r.intent == intent for r in requirements.values()) or len(selected_ids) > 8
            or len(set(selected_ids)) != len(selected_ids) or len(witnesses) > 100
            or len({w.witness_id for w in witnesses}) != len(witnesses)):
        raise ValueError("QUESTION_REVIEW_BOUNDS_OR_INTENT")
    intervals = {}
    validated_docs = set()

    def valid_document(key):
        doc = documents.get(key)
        if (doc is None or doc.document_version_id != key or doc.tenant_id != tenant_id):
            raise ValueError("QUESTION_REVIEW_DOCUMENT_BINDING")
        if key not in validated_docs:
            DocumentVersion.model_validate_json(doc.model_dump_json())
            if sha256_bytes(doc.normalized_text.encode()) != doc.normalized_sha256:
                raise ValueError("QUESTION_REVIEW_DOCUMENT_HASH")
            validated_docs.add(key)
        return doc

    for key in selected_ids:
        chunk = chunks.get(key)
        if chunk is None or chunk.chunk_id != key or chunk.tenant_id != tenant_id:
            raise ValueError("QUESTION_REVIEW_CHUNK_BINDING")
        Chunk.model_validate_json(chunk.model_dump_json())
        doc = valid_document(chunk.document_version_id)
        if (doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text
                or sha256_bytes(chunk.text.encode()) != chunk.text_sha256):
            raise ValueError("QUESTION_REVIEW_CHUNK_TEXT")
        intervals.setdefault(doc.document_version_id, []).append((chunk.start_char, chunk.end_char))
    reviews = {}
    for original in witnesses:
        witness = ReviewedWitness.model_validate_json(original.model_dump_json())
        if witness.requirement_id not in requirements:
            raise ValueError("QUESTION_REVIEW_UNKNOWN_REQUIREMENT")
        spans = []
        for bound in witness.spans:
            doc = valid_document(bound.document_version_id)
            s = bound.span
            if (doc.normalized_sha256 != bound.document_normalized_sha256 or doc.source_uri != bound.source_uri
                    or doc.normalized_text[s.start_char:s.end_char] != s.quote):
                raise ValueError("QUESTION_REVIEW_WITNESS_SOURCE_BINDING")
            covered = intervals.get(doc.document_version_id, [])
            gaps = missing_intervals(s.start_char, s.end_char, covered)
            spans.append({"document_version_id": doc.document_version_id, "start_char": s.start_char,
                "end_char": s.end_char, "quote_sha256": sha256_bytes(s.quote.encode()), "missing_intervals": gaps,
                "any_overlap": any(a < s.end_char and b > s.start_char for a, b in covered)})
        reviews[witness.witness_id] = {"witness_id": witness.witness_id, "requirement_id": witness.requirement_id,
            "evidence_kind": witness.evidence_kind, "complete": all(not s["missing_intervals"] for s in spans),
            "any_overlap": any(s["any_overlap"] for s in spans), "spans": spans}
    rows, statuses = [], {}
    for req in rubric.requirements:
        if req.intent != intent:
            continue
        own = [r for r in reviews.values() if r["requirement_id"] == req.id]
        explicit = [r["witness_id"] for r in own if r["complete"] and r["evidence_kind"] == "EXPLICIT_SUPPORT"]
        qualified = [r["witness_id"] for r in own if r["complete"] and r["evidence_kind"] == "QUALIFIED_INFERENCE"]
        conflicts = [r["witness_id"] for r in own if r["complete"] and r["evidence_kind"] == "CONTRADICTORY"]
        if conflicts:
            status = "CONFLICTING_REVIEW_ANNOTATIONS_REQUIRE_ADJUDICATION"
        elif req.kind == "QUALIFIED_INFERENCE":
            status = ("QUALIFIED_SYNTHESIS_PREMISES_PRESENT" if all(statuses[k] in PRESENT for k in req.requires)
                else "NO_COMPLETE_REVIEWED_WITNESS")
        else:
            status = ("EXPLICIT_REVIEWED_WITNESS_PRESENT" if explicit else "QUALIFIED_REVIEWED_WITNESS_PRESENT"
                if qualified else "NO_COMPLETE_REVIEWED_WITNESS")
        statuses[req.id] = status
        rows.append({"requirement_id": req.id, "status": status, "explicit_witness_ids": explicit,
            "qualified_witness_ids": qualified, "premises": list(req.requires), "conflict_witness_ids": conflicts,
            "scope_cautions": [r["witness_id"] for r in own if r["any_overlap"]
                and r["evidence_kind"] == "MIS_SCOPED_IF_GENERALIZED"], "witness_checks": own})
    return {"version": "question-coverage-review-v1", "intent": intent, "requirements": rows,
        "reviewed_requirements_present": sum(r["status"] in PRESENT for r in rows),
        "requirements_total": len(rows), "all_reviewed_requirements_present": all(r["status"] in PRESENT for r in rows),
        "origin": ORIGIN, "human_reviewers": 0, "semantic_entailment_automatically_verified": False,
        "all_equivalent_sources_discovered": False, "answer_quality_measured": False,
        "permission_or_configuration_verified": False, "serving_gate_enabled": False,
        "selected_chunk_ids_unchanged": list(selected_ids)}
