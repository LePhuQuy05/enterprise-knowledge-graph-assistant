"""Conservative retention after V7 extraction and V8 exact-source normalization.

An admitted candidate still requires explicit source review before serving.
This policy rejects a reduced-relative phrase incorrectly treated as a claim.
"""

import re
from pathlib import Path

from app.contracts import sha256_bytes


def policy_recipe():
    return {"version": "source-candidate-retention-v1", "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "serving_approval": False, "scope": "No naked active using/being participles as universal class predicates"}


def retained(decision):
    if decision["status"] != "CANDIDATE_SOURCE_GUARDS_PASS":
        return False, decision.get("reason") or "SOURCE_GUARD_REJECTED"
    record = decision["normalized_record"]
    if decision["kind"] == "RELATION" and record["voice"] == "ACTIVE":
        trigger = record["trigger_quote"]
        # "REST APIs using VPC links" restricts a noun phrase; it does not
        # assert that REST APIs universally use VPC links. "are using" has
        # an explicit finite auxiliary and is not excluded by this rule.
        if re.fullmatch(r"(?:using|utilizing|replacing)", trigger, re.I):
            return False, "REDUCED_RELATIVE_PARTICIPLE_NOT_FINITE_ASSERTION"
    return True, None
