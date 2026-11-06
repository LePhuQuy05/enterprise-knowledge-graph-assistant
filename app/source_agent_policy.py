"""Conservative promotion guard for active clause agents, separate from V7.

This blocks a documented false agent, not a general English entailment parser.
Raw V7 receipts and the frozen V8 review policy remain unchanged.
"""

import re
from pathlib import Path

from app.contracts import sha256_bytes
from app.semantic_extraction import endpoint_position
from app.semantic_extraction_v7 import TypedFrameV7


def policy_recipe():
    return {"version": "source-active-agent-promotion-v1",
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "scope": "CONSERVATIVE_FILTER_NOT_ENTAILMENT_OR_SERVING_APPROVAL"}


def check_agent(frame: TypedFrameV7):
    if frame.voice == "PASSIVE":
        # Explicit passive agent validation remains the responsibility of V7.
        return True, None
    _, subject_end = endpoint_position(frame.subject, frame.evidence_quote)
    triggers = list(re.finditer(re.escape(frame.trigger_quote), frame.evidence_quote, re.I))
    if len(triggers) != 1 or triggers[0].start() < subject_end:
        return False, "AGENT_V1_UNIQUE_ORDERED_TRIGGER_REQUIRED"
    gap = frame.evidence_quote[subject_end:triggers[0].start()]
    # A heading such as "Feature: Uses ..." is allowed; a second clause or a
    # different actor ("integrations, you can use") is never silently removed.
    allowed = r"\s*:?[ \t]*(?:(?:also|currently|only|typically|generally|can|may|might|could|must|should|does|do|did|not|never|is|are|was|were)\s+)*"
    if not re.fullmatch(allowed, gap, re.I):
        return False, "AGENT_V1_SUBJECT_NOT_DIRECT_CLAUSE_AGENT"
    return True, None
