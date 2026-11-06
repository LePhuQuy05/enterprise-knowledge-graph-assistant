"""Promotion filter for dropped demonstratives and unresolved counted nouns.

Added after native V9 qualification; not retroactively a native V9 validator.
"""

import re
from pathlib import Path

from app.contracts import sha256_bytes
from app.semantic_extraction import endpoint_position


def policy_recipe():
    return {"version": "source-entity-reference-promotion-v1",
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()), "native_v9_inference": False}


def check_reference(endpoint, clause):
    start, _ = endpoint_position(endpoint, clause)
    if re.match(r"^(?:one|two|three|four|both|several|\d+)\s+", endpoint.name, re.I):
        return False, "ENTITY_REFERENCE_V1_UNRESOLVED_COUNTED_NOUN"
    if re.search(r"\b(?:this|that|these|those|such|both|other|same)(?:\s+(?:one|two|three|four|\d+))?\s+$", clause[:start], re.I):
        return False, "ENTITY_REFERENCE_V1_DROPPED_SOURCE_DETERMINER"
    return True, None
