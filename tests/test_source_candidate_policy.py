import pytest

from app.source_candidate_policy import retained


@pytest.mark.parametrize("trigger", ["using", "utilizing", "replacing"])
def test_source_scope_not_laundered_by_full_quote_qualifier(trigger):
    candidate = {"status": "CANDIDATE_SOURCE_GUARDS_PASS", "kind": "RELATION",
        "normalized_record": {"voice": "ACTIVE", "trigger_quote": trigger,
            "qualifiers": [{"kind": "LIMITATION", "text": "REST APIs using VPC links"}]}}
    assert retained(candidate) == (False, "REDUCED_RELATIVE_PARTICIPLE_NOT_FINITE_ASSERTION")


@pytest.mark.parametrize("trigger", ["uses", "are using", "can use", "supports"])
def test_finite_or_modal_predicate_is_not_rejected_as_noun_restriction(trigger):
    candidate = {"status": "CANDIDATE_SOURCE_GUARDS_PASS", "kind": "RELATION",
        "normalized_record": {"voice": "ACTIVE", "trigger_quote": trigger}}
    assert retained(candidate) == (True, None)


def test_policy_never_rescues_source_rejection():
    assert retained({"status": "REJECTED", "reason": "WRONG_AGENT"}) == (False, "WRONG_AGENT")
