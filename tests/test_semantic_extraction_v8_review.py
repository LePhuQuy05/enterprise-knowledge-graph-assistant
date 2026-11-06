from copy import deepcopy

import pytest

from app.contracts import canonical_json
from app.semantic_extraction_v7 import SemanticOutputV7
from app.semantic_extraction_v8_review import review_output
from tests.test_semantic_extraction_v7 import endpoint, frame


def review(text, *, frames=(), permissions=()):
    output = SemanticOutputV7.model_validate_json(canonical_json({"relationships": frames,
        "permission_requirements": permissions, "coverage_note": "HAS_CANDIDATES"}))
    return output, review_output(output, text)


def test_split_scope_qualifiers_and_source_suffix_do_not_erase_valid_relation():
    text = "Secrets Manager uses a symmetric encryption KMS key associated with a secret to generate a data key."
    row = frame(text, endpoint("Secrets Manager", "SERVICE_PRODUCT"), endpoint("symmetric encryption KMS key", "RESOURCE_TYPE"),
        predicate="USES", trigger="uses", qualifiers=[{"kind": "OTHER", "text": "associated with a secret"},
            {"kind": "OTHER", "text": "to generate a data key"}])
    original, decisions = review(text, frames=[row])
    candidate = decisions[0]
    assert candidate["status"] == "CANDIDATE_SOURCE_GUARDS_PASS"
    assert len(original.relationships[0].qualifiers) == 2
    assert len(candidate["normalized_record"]["qualifiers"]) == 3
    assert candidate["raw_record"] == row
    assert candidate["normalized_record"]["qualifiers"][-1]["text"] == text


@pytest.mark.parametrize("action,expected", [("access the target resource", "access"),
    ("write data to target resource", "write data")])
def test_permission_action_resource_overlap_has_exact_auditable_split(action, expected):
    text = "To deliver event data, EventBridge needs permission to " + action + "."
    row = dict(evidence_quote=text, principal=endpoint("EventBridge", "SERVICE_PRODUCT"),
        resource=endpoint("target resource", "RESOURCE_INSTANCE"), action_text=action,
        trigger_quote="needs permission", qualifiers=[])
    original, decisions = review(text, permissions=[row])
    assert original.permission_requirements[0].action_text == action
    assert decisions[0]["status"] == "CANDIDATE_SOURCE_GUARDS_PASS"
    assert decisions[0]["normalized_record"]["action_text"] == expected
    assert decisions[0]["normalizations"] == ["APPEND_FULL_EXACT_EVIDENCE_AS_SCOPE", "SPLIT_EXACT_TRAILING_RESOURCE_FROM_ACTION"]


def test_normalization_does_not_fix_wrong_type_or_relative_class_or_missing_sentence():
    texts = ["CloudWatch supports PromQL.", "EC2 instances that use internet gateways incur charges.",
             "If configured, CloudWatch supports TLS."]
    rows = [frame(texts[0], endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("PromQL", "PROTOCOL")),
        frame(texts[1], endpoint("EC2 instances", "RESOURCE_TYPE"), endpoint("internet gateways", "RESOURCE_TYPE"), predicate="USES", trigger="use"),
        frame("CloudWatch supports TLS.", endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("TLS", "PROTOCOL"))]
    expected = ["TYPE_MISMATCH", "RELATIVE_CLASS", "SENTENCE_SCOPE_TRUNCATED"]
    for text, row, reason in zip(texts, rows, expected, strict=True):
        _, decisions = review(text, frames=[row])
        assert decisions[0]["status"] == "REJECTED" and reason in decisions[0]["reason"]


def test_qualifier_bound_never_overwrites_original_conditions():
    text = "CloudWatch supports TLS."
    row = frame(text, endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("TLS", "PROTOCOL"),
        qualifiers=[{"kind": "LIMITATION", "text": "TLS"}] * 8)
    original, decisions = review(text, frames=[row])
    assert len(original.relationships[0].qualifiers) == 8
    assert decisions[0]["reason"] == "V8_SCOPE_QUALIFIER_BOUND_NO_OVERWRITE"


def test_permission_split_does_not_rewrite_nontrailing_or_invented_actions():
    text = "EventBridge needs permission to access target resource when configured."
    row = dict(evidence_quote=text, principal=endpoint("EventBridge", "SERVICE_PRODUCT"),
        resource=endpoint("target resource", "RESOURCE_INSTANCE"), action_text="access target resource when configured",
        trigger_quote="needs permission", qualifiers=[])
    _, result = review(text, permissions=[row])
    assert result[0]["reason"] == "V8_ACTION_RESOURCE_OVERLAP_NOT_A_TRAILING_SUFFIX"
    forged = deepcopy(row)
    forged["action_text"] = "s3:GetObject"
    _, result = review(text, permissions=[forged])
    assert result[0]["reason"] == "V8_ORIGINAL_ACTION_MISSING_OR_AMBIGUOUS"
