"""Auditable deterministic review of native V7 receipts, not native V8 inference.

Retain the complete exact evidence as a scope qualifier. Split an overlapping
permission action only when the explicit resource is its exact trailing suffix.
No endpoints, predicates, types, polarity or source words may be invented.
"""

import re
from pathlib import Path

from app.contracts import Qualifier, canonical_json, sha256_bytes
from app.extraction import ExtractionError
from app.semantic_extraction_v7 import (
    CATALOG,
    ROOT,
    SemanticOutputV7,
    TypedFrameV7,
    TypedPermissionV7,
    validate_frame_v7,
    validate_permission_v7,
)


def review_recipe():
    return {"version": "semantic-v7-native-v8-source-scope-review-v1",
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "validator_sha256": sha256_bytes((ROOT / "app/semantic_extraction_v7.py").read_bytes()),
        "typed_catalog_sha256": sha256_bytes(CATALOG.read_bytes()),
        "rules": "EXACT_COMPLETE_EVIDENCE_SCOPE_AND_UNIQUE_TRAILING_RESOURCE_ACTION_SPLIT",
        "native_v8_inference": False, "serving_approval": False}


def preserve_full_scope(record):
    if any(record.evidence_quote == q.text for q in record.qualifiers):
        return record, False
    if len(record.qualifiers) >= 8:
        raise ExtractionError("V8_SCOPE_QUALIFIER_BOUND_NO_OVERWRITE")
    return record.model_copy(update={"qualifiers": record.qualifiers + (
        Qualifier(kind="LIMITATION", text=record.evidence_quote),)}), True


def split_permission_action(record):
    original, resource, quote = record.action_text, record.resource.name, record.evidence_quote
    action_start = quote.find(original)
    if action_start < 0 or quote.find(original, action_start + 1) >= 0:
        raise ExtractionError("V8_ORIGINAL_ACTION_MISSING_OR_AMBIGUOUS")
    resource_start = -1
    for _ in range(record.resource.occurrence):
        resource_start = quote.find(resource, resource_start + 1)
        if resource_start < 0:
            raise ExtractionError("V8_RESOURCE_OCCURRENCE_MISSING")
    if action_start + len(original) <= resource_start:
        return record, False
    if not original.endswith(resource) or action_start + len(original) != resource_start + len(resource):
        raise ExtractionError("V8_ACTION_RESOURCE_OVERLAP_NOT_A_TRAILING_SUFFIX")
    action = original[:-len(resource)].rstrip()
    for _ in range(3):
        shorter = re.sub(r"\s+(?:to|on|from|the|a|an)$", "", action, flags=re.I)
        if shorter == action:
            break
        action = shorter
    if not action or action not in original or not re.match(r"[A-Za-z]", action):
        raise ExtractionError("V8_ACTION_SPLIT_EMPTY_OR_NOT_SOURCE_EXACT")
    return record.model_copy(update={"action_text": action}), True


def review_output(output: SemanticOutputV7, text):
    decisions = []
    for kind, rows in (("RELATION", output.relationships), ("PERMISSION", output.permission_requirements)):
        for index, original in enumerate(rows):
            raw = original.model_dump(mode="json")
            item = {"kind": kind, "index": index, "raw_record_sha256": sha256_bytes(canonical_json(raw)),
                    "raw_record": raw, "status": "REJECTED", "reason": None,
                    "normalizations": [], "native_origin": "V7"}
            try:
                changed, scope_added = preserve_full_scope(original)
                if scope_added:
                    item["normalizations"].append("APPEND_FULL_EXACT_EVIDENCE_AS_SCOPE")
                if kind == "PERMISSION":
                    changed, action_split = split_permission_action(changed)
                    if action_split:
                        item["normalizations"].append("SPLIT_EXACT_TRAILING_RESOURCE_FROM_ACTION")
                    validated = validate_permission_v7(changed, text)
                    # JSON roundtrip validates model_copy updates under strict DTO.
                    TypedPermissionV7.model_validate_json(canonical_json(changed.model_dump(mode="json")))
                else:
                    validated = validate_frame_v7(changed, text)
                    TypedFrameV7.model_validate_json(canonical_json(changed.model_dump(mode="json")))
                item.update(status="CANDIDATE_SOURCE_GUARDS_PASS", normalized_record=changed.model_dump(mode="json"),
                            normalized_record_sha256=sha256_bytes(canonical_json(changed.model_dump(mode="json"))),
                            validation=validated, semantic_review="PENDING_NOT_SERVING")
            except ExtractionError as exc:
                item["reason"] = str(exc)
            decisions.append(item)
    return tuple(decisions)
