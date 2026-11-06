from copy import deepcopy
from types import SimpleNamespace

import pytest

from scripts.export_syntax_reviewed_graph import frame_signature


def fixture():
    clause = "CloudTrail delivers logs to the S3 bucket."
    candidate = {"source_sha256": "1" * 64, "kind": "ACTION", "record": {"clause_quote": clause,
        "action_family": "DELIVER", "polarity": "POSITIVE", "modality": "ASSERTED"},
        "evidence": {"quote": "Context. " + clause}, "verb_token": {"start": 11},
        "actor": {"root_token": 0}, "target": {"root_token": 1}, "payload_span": {"quote": "logs"}}
    parsed = SimpleNamespace(tokens=[{"start": 0}, {"start": clause.index("bucket")}])
    return candidate, parsed


def test_same_source_roots_deduplicate_despite_different_token_numbering():
    a, parsed_a = fixture()
    b = deepcopy(a)
    b["actor"]["root_token"], b["target"]["root_token"] = 2, 3
    parsed_b = SimpleNamespace(tokens=[{}, {}, *parsed_a.tokens])
    assert frame_signature(a, parsed_a) == frame_signature(b, parsed_b)


@pytest.mark.parametrize("change", ["source", "actor", "target", "verb", "polarity", "modality", "payload"])
def test_different_claims_or_source_roles_must_never_collapse(change):
    a, parsed_a = fixture()
    b, parsed_b = deepcopy(a), deepcopy(parsed_a)
    if change == "source":
        b["source_sha256"] = "2" * 64
    elif change in {"actor", "target"}:
        parsed_b.tokens[0 if change == "actor" else 1]["start"] += 1
    elif change == "verb":
        b["verb_token"]["start"] += 1
    elif change == "polarity":
        b["record"]["polarity"] = "NEGATIVE"
    elif change == "modality":
        b["record"]["modality"] = "CAPABILITY"
    else:
        b["payload_span"]["quote"] = "events"
    assert frame_signature(a, parsed_a) != frame_signature(b, parsed_b)
