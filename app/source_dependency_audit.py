"""Review-only audit of explicitly annotated same-document source dependencies.

Dependencies are source-review annotations, not facts, graph edges, benchmark
gold or automatic discovery of all qualifiers. The auditor never inserts a
source, changes a method, edits a claim, or promotes an answer's status. It
accepts any selected chunks whose interval union supplies the required text.
"""

from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.authorization_evidence import LiteralSpan
from app.contracts import Record, Sha256, canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError

POLICY = "REVIEWED_SAME_DOCUMENT_SOURCE_DEPENDENCY_AUDIT_V1"


class SourceDependency(Record):
    version: Literal["source-dependency-review-v1"] = "source-dependency-review-v1"
    dependency_id: str
    tenant_id: str
    document_version_id: str
    document_normalized_sha256: Sha256
    source_uri: str
    anchor: LiteralSpan
    required_spans: Annotated[tuple[LiteralSpan, ...], Field(min_length=1, max_length=4)]
    review_reason: Annotated[str, Field(min_length=1, max_length=500)]
    review_origin: Literal["AI_SOURCE_REVIEW_NOT_INDEPENDENT_GOLD", "SYNTHETIC_FIXTURE"]
    permission_or_configuration_verified: Literal[False] = False

    @model_validator(mode="after")
    def distinct_and_bound_identity(self):
        if len({(s.start_char, s.end_char) for s in self.required_spans}) != len(self.required_spans):
            raise ValueError("SOURCE_DEPENDENCY_DUPLICATE_REQUIREMENT")
        data = self.model_dump(mode="json", exclude={"dependency_id"})
        if self.dependency_id != stable_id("source-dependency-v1", **data):
            raise ValueError("SOURCE_DEPENDENCY_IDENTITY")
        return self


def make_source_dependency(document, anchor, required_spans, *, review_reason, fixture=False):
    data = {"version": "source-dependency-review-v1", "tenant_id": document.tenant_id,
        "document_version_id": document.document_version_id, "document_normalized_sha256": document.normalized_sha256,
        "source_uri": document.source_uri, "anchor": anchor.model_dump(mode="json"),
        "required_spans": [s.model_dump(mode="json") for s in required_spans], "review_reason": review_reason,
        "review_origin": "SYNTHETIC_FIXTURE" if fixture else "AI_SOURCE_REVIEW_NOT_INDEPENDENT_GOLD",
        "permission_or_configuration_verified": False}
    return SourceDependency.model_validate_json(canonical_json(data | {"dependency_id": stable_id("source-dependency-v1", **data)}))


def missing_intervals(start, end, covered):
    """Half-open Unicode document offsets, accepting overlaps and adjacency."""
    cursor, missing = start, []
    for left, right in sorted(covered):
        if right <= cursor or left >= end:
            continue
        if left > cursor:
            missing.append((cursor, min(left, end)))
        cursor = min(end, max(cursor, right))
        if cursor == end:
            break
    if cursor < end:
        missing.append((cursor, end))
    return tuple(missing)


def audit_source_dependencies(documents, chunks, selected_chunk_ids, dependencies, *, tenant_id):
    if (len(selected_chunk_ids) > 8 or len(set(selected_chunk_ids)) != len(selected_chunk_ids)
            or len(dependencies) > 100 or len({d.dependency_id for d in dependencies}) != len(dependencies)):
        raise GenerationError("SOURCE_DEPENDENCY_AUDIT_BOUNDS")
    selected, intervals = [], {}
    for key in selected_chunk_ids:
        chunk = chunks.get(key)
        doc = None if chunk is None else documents.get(chunk.document_version_id)
        if (doc is None or chunk.chunk_id != key or chunk.tenant_id != tenant_id or doc.tenant_id != tenant_id
                or doc.normalized_sha256 != sha256_bytes(doc.normalized_text.encode())
                or chunk.text_sha256 != sha256_bytes(chunk.text.encode())
                or doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
            raise GenerationError("SOURCE_DEPENDENCY_SELECTED_SOURCE_BINDING")
        selected.append(chunk)
        intervals.setdefault(doc.document_version_id, []).append((chunk.start_char, chunk.end_char))
    rows = []
    for dependency in dependencies:
        if type(dependency) is not SourceDependency:
            raise GenerationError("SOURCE_DEPENDENCY_REVIEW_TYPE")
        try:
            SourceDependency.model_validate_json(dependency.model_dump_json())
        except ValueError as exc:
            raise GenerationError("SOURCE_DEPENDENCY_REVIEW_IDENTITY") from exc
        doc = documents.get(dependency.document_version_id)
        if (doc is None or doc.tenant_id != tenant_id or dependency.tenant_id != tenant_id
                or doc.normalized_sha256 != dependency.document_normalized_sha256
                or sha256_bytes(doc.normalized_text.encode()) != doc.normalized_sha256 or doc.source_uri != dependency.source_uri):
            raise GenerationError("SOURCE_DEPENDENCY_REVIEW_DOCUMENT_BINDING")
        for span in (dependency.anchor, *dependency.required_spans):
            if doc.normalized_text[span.start_char:span.end_char] != span.quote:
                raise GenerationError("SOURCE_DEPENDENCY_REVIEW_SPAN_BINDING")
        covered = intervals.get(doc.document_version_id, [])
        anchor = dependency.anchor
        active = any(left < anchor.end_char and right > anchor.start_char for left, right in covered)
        missing_anchor = missing_intervals(anchor.start_char, anchor.end_char, covered) if active else ()
        requirements = [{"start_char": s.start_char, "end_char": s.end_char, "quote_sha256": sha256_bytes(s.quote.encode()),
            "missing_intervals": missing_intervals(s.start_char, s.end_char, covered) if active else (),
            "checked": active} for s in dependency.required_spans]
        incomplete = active and (bool(missing_anchor) or any(r["missing_intervals"] for r in requirements))
        rows.append({"dependency_id": dependency.dependency_id, "document_version_id": dependency.document_version_id,
            "status": "INCOMPLETE_REVIEWED_SOURCE_CONTEXT" if incomplete else "COMPLETE_REVIEWED_SOURCE_CONTEXT" if active else "NOT_TRIGGERED",
            "anchor_missing_intervals": missing_anchor, "required_spans": requirements,
            "selected_document_chunk_ids": [c.chunk_id for c in selected if c.document_version_id == doc.document_version_id],
            "annotation_origin": dependency.review_origin, "review_reason": dependency.review_reason})
    return {"policy": POLICY, "checks": rows, "incomplete_count": sum(r["status"] == "INCOMPLETE_REVIEWED_SOURCE_CONTEXT" for r in rows),
        "reviewed_dependencies": len(rows), "all_qualifiers_discovered": False, "semantic_correctness_measured": False,
        "selected_chunk_ids_unchanged": tuple(selected_chunk_ids), "serving_gate_enabled": False}
