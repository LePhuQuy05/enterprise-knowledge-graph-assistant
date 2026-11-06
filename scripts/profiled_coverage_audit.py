"""Evaluator-only adaptation of reviewed witnesses to a validated 8/12 context.

The old auditor still validates rubric/witness provenance on an empty context;
the new renderer validates all selected sources. Only interval presence is then
recomputed at the declared profile size. Nothing from this module enters search.
"""

from app.profiled_source_context import validate_profiled_context
from app.question_coverage_review import PRESENT, audit_question_coverage
from app.source_dependency_audit import missing_intervals


def audit_profiled_coverage(rubric, witnesses, documents, chunks, context, *, intent):
    validate_profiled_context(context, documents, chunks)
    result = audit_question_coverage(rubric, witnesses, documents, chunks, (), intent=intent, tenant_id=context.tenant_id)
    intervals = {}
    for key in context.chunk_ids:
        c = chunks[key]
        intervals.setdefault(c.document_version_id, []).append((c.start_char, c.end_char))
    requirements = {r.id: r for r in rubric.requirements}
    statuses = {}
    for row in result["requirements"]:
        req = requirements[row["requirement_id"]]
        for w in row["witness_checks"]:
            for s in w["spans"]:
                covered = intervals.get(s["document_version_id"], ())
                s["missing_intervals"] = missing_intervals(s["start_char"], s["end_char"], covered)
                s["any_overlap"] = any(a < s["end_char"] and b > s["start_char"] for a, b in covered)
            w["complete"] = all(not s["missing_intervals"] for s in w["spans"])
            w["any_overlap"] = any(s["any_overlap"] for s in w["spans"])
        own = row["witness_checks"]
        explicit = [w["witness_id"] for w in own if w["complete"] and w["evidence_kind"] == "EXPLICIT_SUPPORT"]
        qualified = [w["witness_id"] for w in own if w["complete"] and w["evidence_kind"] == "QUALIFIED_INFERENCE"]
        conflicts = [w["witness_id"] for w in own if w["complete"] and w["evidence_kind"] == "CONTRADICTORY"]
        if conflicts:
            status = "CONFLICTING_REVIEW_ANNOTATIONS_REQUIRE_ADJUDICATION"
        elif req.kind == "QUALIFIED_INFERENCE":
            status = "QUALIFIED_SYNTHESIS_PREMISES_PRESENT" if all(statuses[k] in PRESENT for k in req.requires) else "NO_COMPLETE_REVIEWED_WITNESS"
        else:
            status = "EXPLICIT_REVIEWED_WITNESS_PRESENT" if explicit else "QUALIFIED_REVIEWED_WITNESS_PRESENT" if qualified else "NO_COMPLETE_REVIEWED_WITNESS"
        statuses[req.id] = status
        row.update(status=status, explicit_witness_ids=explicit, qualified_witness_ids=qualified, conflict_witness_ids=conflicts,
            scope_cautions=[w["witness_id"] for w in own if w["any_overlap"] and w["evidence_kind"] == "MIS_SCOPED_IF_GENERALIZED"])
    result.update(version="profiled-question-coverage-review-v1", selected_chunk_ids_unchanged=list(context.chunk_ids),
        reviewed_requirements_present=sum(r["status"] in PRESENT for r in result["requirements"]),
        all_reviewed_requirements_present=all(r["status"] in PRESENT for r in result["requirements"]),
        context_profile_sha256=context.profile.fingerprint)
    return result
