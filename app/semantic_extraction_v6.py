"""V6 deterministic review of V5 frames: minimal triggers and direct arguments.

This is explicitly post-processing of native V5 receipts, not native V6 calls.
It preserves every raw field and records each normalization before source checks.
Candidates still require semantic/type review; no automatic serving approval.
"""

import re
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.extraction import ExtractionError
from app.semantic_extraction import (
    SemanticExtractionOutputV5,
    endpoint_position,
    validate_frame,
    validate_permission,
)

VERBS = {
    "USES": r"(?:uses?|using|used|utilizes?|utilizing)",
    "SUPPORTS": r"(?:supports?|provides?|enables?|offers?)",
    "DEPENDS_ON": r"(?:requires?|needs?|depends? on)",
    "PART_OF": r"(?:part of|belongs? to|component of|contained in)",
    "INTEGRATES_WITH": r"(?:integrates? with|integrated with|integration with)",
    "REPLACES": r"(?:replaces?|replacing|replaced)",
}
AUXILIARIES = r"(?:(?:can|may|might|could|must|should|do|does|did|is|are|was|were|be|not|never|also)\s+)*"


def recipe_v6():
    return {"version": "semantic-frame-review-v6", "normalization": "MINIMAL_VERB_PREFIX_NO_SOURCE_REWORDING_V1",
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "serving_approval": False, "native_provider_inference": False}


def minimal_trigger(original, predicate):
    pattern = r"^" + AUXILIARIES + (VERBS[predicate] if predicate != "PERMISSION" else r"(?:needs? permission(?:s)?|requires? permission(?:s)?|must have permission(?:s)?)") + r"\b"
    match = re.match(pattern, original, re.I)
    if not match:
        raise ExtractionError("SEMANTIC_V6_MINIMAL_TRIGGER_REQUIRED")
    return match.group()


def direct_object_gap(gap):
    if re.fullmatch(r"\s*(?:(?:a|an|the)\s+)?", gap, re.I):
        return True
    # Explicit source list introducers support coordinated arguments. We do
    # not accept arbitrary noun phrases between a verb and the claimed object.
    match = re.match(r"\s*(?:the\s+)?(?:following\s+)?(?:protocols|services|features|resources)\s*:\s*", gap, re.I)
    if not match:
        return False
    prior = gap[match.end():]
    if re.search(r"\b(?:to|for|through|using|when|if|which|that)\b|[;.!?\n]", prior, re.I):
        return False
    return all(len(piece.strip().split()) <= 5 for piece in re.split(r",|\band\b|\bor\b", prior, flags=re.I))


def review_output_v6(output: SemanticExtractionOutputV5, text):
    results = []
    for kind, rows in (("RELATION", output.relationships), ("PERMISSION", output.permission_requirements)):
        for index, row in enumerate(rows):
            record = {"kind": kind, "index": index, "raw_record_sha256": sha256_bytes(canonical_json(row.model_dump(mode="json"))),
                "original_trigger": row.trigger_quote, "status": "REJECTED", "reason": None}
            try:
                normalized = minimal_trigger(row.trigger_quote, row.predicate if kind == "RELATION" else "PERMISSION")
                changed = row.model_copy(update={"trigger_quote": normalized})
                record.update(normalized_trigger=normalized, trigger_normalized=normalized != row.trigger_quote)
                if kind == "RELATION":
                    subject, obj = endpoint_position(row.subject, row.evidence_quote), endpoint_position(row.object, row.evidence_quote)
                    position = row.evidence_quote.find(normalized)
                    if position < 0 or row.evidence_quote.find(normalized, position + 1) >= 0:
                        raise ExtractionError("SEMANTIC_V6_TRIGGER_NOT_UNIQUE")
                    if row.voice == "ACTIVE":
                        if not subject[1] <= position < position + len(normalized) <= obj[0] or not direct_object_gap(row.evidence_quote[position + len(normalized):obj[0]]):
                            raise ExtractionError("SEMANTIC_V6_DIRECT_ARGUMENT_REQUIRED")
                        # The subsequent V5 guard remains conservative about
                        # later coordinated objects. Do not bypass its rejection.
                    validate_frame(changed, text)
                else:
                    validate_permission(changed, text)
                    principal = endpoint_position(row.principal, row.evidence_quote)
                    prefix = row.evidence_quote[:principal[0]]
                    if re.match(r"\s*(?:if|when|unless)\b", prefix, re.I):
                        condition = prefix.split(",", 1)[0].strip()
                        if not any(condition in q.text for q in row.qualifiers):
                            raise ExtractionError("SEMANTIC_V6_PERMISSION_CONDITION_REQUIRED")
                record.update(status="CANDIDATE_SOURCE_GUARDS_PASS", normalized_record=changed.model_dump(mode="json"))
            except ExtractionError as exc:
                record["reason"] = str(exc)
            results.append(record)
    return tuple(results)
