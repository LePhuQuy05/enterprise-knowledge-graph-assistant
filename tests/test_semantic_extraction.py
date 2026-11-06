import json

import pytest

from app.semantic_extraction import validate_semantic_output


def frame(text, subject, obj, predicate, trigger, **updates):
    return dict(evidence_quote=text, subject={"name": subject, "entity_type": "RESOURCE", "occurrence": 1},
        object={"name": obj, "entity_type": "CONCEPT", "occurrence": 1}, predicate=predicate,
        trigger_quote=trigger, voice="ACTIVE", polarity="POSITIVE", modality="ASSERTED", qualifiers=[], **updates)


def validate(text, frames=(), permissions=()):
    return validate_semantic_output(json.dumps({"relationships": frames, "permission_requirements": permissions}), text, stop_reason="end_turn")


@pytest.mark.parametrize("text,subject,obj,predicate,trigger", [
    ("If a service traces requests by using the X-Ray SDK, Amazon S3 can send the tracing headers.", "Amazon S3", "X-Ray SDK", "USES", "using"),
    ("Amazon RDS loses access to the KMS key for a DB instance when you disable the KMS key.", "Amazon RDS", "KMS key", "DEPENDS_ON", "loses access to"),
    ("You can't use an asymmetric KMS key.", "You", "asymmetric KMS key", "USES", "can't use"),
    ("CloudFormation uses the modified list to associate all security groups with the MyEC2Instance resource.", "CloudFormation", "MyEC2Instance", "USES", "uses"),
    ("By default, whenever a CloudFront event triggers a Lambda function, data is written to CloudWatch Logs.", "CloudWatch Logs", "CloudFront", "INTEGRATES_WITH", "triggers"),
    ("You can monitor rotation schedules, success rates, and performance metrics through Amazon CloudWatch.", "rotation schedules", "Amazon CloudWatch", "USES", "monitor"),
    ("You must enable trusted access for Amazon EC2.", "trusted access", "Amazon EC2", "DEPENDS_ON", "must enable"),
    ("If your Lambda function reads an object from an S3 bucket, the execution role needs permission to perform that action.", "execution role", "S3 bucket", "DEPENDS_ON", "needs"),
])
def test_eight_known_semantic_failures_cannot_be_accepted_as_graph_edges(text, subject, obj, predicate, trigger):
    result = validate(text, [frame(text, subject, obj, predicate, trigger)])
    assert not result.accepted_relationships and len(result.rejected_records) == 1


def test_explicit_positive_negative_optional_and_passive_frames_preserve_evidence():
    cases = [
        ("Service A supports TLS.", "Service A", "TLS", "SUPPORTS", "supports", {}),
        ("Service A does not support TLS.", "Service A", "TLS", "SUPPORTS", "does not support", {"polarity": "NEGATIVE"}),
        ("Service A can use Resource B.", "Service A", "Resource B", "USES", "can use", {"modality": "OPTIONAL", "qualifiers": [{"kind": "OTHER", "text": "can use"}]}),
        ("SDK is used by Service A.", "Service A", "SDK", "USES", "used by", {"voice": "PASSIVE"}),
    ]
    for text, subject, obj, predicate, trigger, updates in cases:
        row = frame(text, subject, obj, predicate, trigger)
        row.update(updates)
        result = validate(text, [row])
        assert len(result.accepted_relationships) == 1 and not result.rejected_records
        assert result.accepted_relationships[0][1].relationships[0].evidence.quote == text


def test_permission_is_triangular_source_record_never_a_dependency_edge():
    text = "The execution role needs permission to write data to CloudWatch Logs."
    record = dict(evidence_quote=text, principal={"name": "execution role", "entity_type": "RESOURCE", "occurrence": 1},
        action_text="write data", resource={"name": "CloudWatch Logs", "entity_type": "AWS_SERVICE", "occurrence": 1},
        trigger_quote="needs permission", qualifiers=[])
    result = validate(text, permissions=[record])
    assert len(result.accepted_permissions) == 1 and not result.accepted_relationships
    invalid = frame(text, "execution role", "CloudWatch Logs", "DEPENDS_ON", "needs")
    result = validate(text, [invalid])
    assert result.rejected_records[0]["reason"] == "SEMANTIC_AUTHORIZATION_REQUIRES_TYPED_RECORD"


@pytest.mark.parametrize("text,predicate,trigger,updates", [
    ("Service A does not need to use Resource B.", "USES", "use", {"polarity": "NEGATIVE"}),
    ("Service A can use Resource B.", "USES", "use", {}),
    ("Service A should use Resource B.", "USES", "should use", {"modality": "RECOMMENDED", "qualifiers": [{"kind": "OTHER", "text": "should"}]}),
    ("SDK is used by Service B while Service A monitors requests.", "USES", "used", {"voice": "PASSIVE"}),
])
def test_scope_modal_or_non_direct_passive_agent_is_quarantined(text, predicate, trigger, updates):
    subject, obj = ("Service A", "SDK") if "SDK" in text else ("Service A", "Resource B")
    row = frame(text, subject, obj, predicate, trigger)
    row.update(updates)
    assert not validate(text, [row]).accepted_relationships


def test_one_rejected_frame_does_not_erase_independent_valid_candidate_or_raw_record():
    text = "Service A supports TLS. You can use Resource B."
    result = validate(text, [frame(text, "Service A", "TLS", "SUPPORTS", "supports"),
        frame(text, "You", "Resource B", "USES", "can use")])
    assert len(result.output.relationships) == 2 and len(result.accepted_relationships) == 1
    assert len(result.rejected_records) == 1


def test_trailing_condition_and_modal_type_cannot_be_removed_or_inverted():
    text = "Service A supports TLS only when configured."
    row = frame(text, "Service A", "TLS", "SUPPORTS", "supports")
    assert validate(text, [row]).rejected_records[0]["reason"] == "SEMANTIC_TRAILING_CONDITION_MISSING"
    row["qualifiers"] = [{"kind": "CONFIGURATION", "text": "only when configured"}]
    assert validate(text, [row]).accepted_relationships
    text = "Service A can use Resource B."
    row = frame(text, "Service A", "Resource B", "USES", "can use")
    row.update(modality="REQUIRED", qualifiers=[{"kind": "OTHER", "text": "can use"}])
    assert validate(text, [row]).rejected_records[0]["reason"] == "SEMANTIC_MODALITY_CONTRADICTS_TRIGGER"
