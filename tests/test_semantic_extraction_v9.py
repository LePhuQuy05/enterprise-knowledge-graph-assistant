"""Synthetic mechanism fixtures, not independent AWS factual labels."""

import pytest

from app.contracts import canonical_json
from app.extraction import ExtractionError
from app.semantic_extraction_v9 import validate_semantic_output


def endpoint(name, kind, head=None, restrictions=(), occurrence=1):
    return dict(name=name, semantic_type=kind, occurrence=occurrence, head_quote=head or name, restrictions=list(restrictions))


def relation(quote, subject, obj, *, predicate="USES", trigger="uses", **changes):
    return dict(clause_quote=quote, subject=subject, object=obj, predicate=predicate, trigger_quote=trigger,
        voice="ACTIVE", polarity="POSITIVE", modality="ASSERTED") | changes


def validate(text, *, relations=(), permissions=(), deferred=(), note=None):
    raw = canonical_json({"relationships": list(relations), "permission_requirements": list(permissions),
        "deferred_statements": list(deferred), "coverage_note": note or ("HAS_CANDIDATES" if relations or permissions
        else "DEFERRED_ONLY" if deferred else "NO_ELIGIBLE_RELATION")}).decode()
    return validate_semantic_output(raw, text, stop_reason="end_turn")


def test_previous_sentences_are_retained_exactly_and_clause_has_its_own_span():
    clause = "CloudWatch uses Amazon SNS to notify users that an endpoint is unhealthy."
    text = "Optionally, configure health notifications. When configured, Route 53 sets an alarm. " + clause
    row = relation(clause, endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("Amazon SNS", "SERVICE_PRODUCT"))
    result = validate(text, relations=[row]).accepted_relationships[0]
    assert result["evidence"]["quote"] == text and result["qualifiers"][0]["text"] == text
    assert result["clause_span"]["start_char"] == text.index(clause)
    assert result["endpoint_identities"][0]["start_char"] == text.index("CloudWatch")


def test_resource_head_and_service_identity_are_distinct():
    text = "Secrets Manager uses an AWS Lambda function to rotate a secret."
    subject = endpoint("Secrets Manager", "SERVICE_PRODUCT")
    bad = relation(text, subject, endpoint("AWS Lambda", "SERVICE_PRODUCT"))
    result = validate(text, relations=[bad])
    assert result.rejected_records[0]["reason"] == "V9_SERVICE_SUBSTRING_OF_RESOURCE"
    good = relation(text, subject, endpoint("AWS Lambda function", "RESOURCE_TYPE", "function"))
    result = validate(text, relations=[good]).accepted_relationships[0]
    assert result["endpoint_identities"][1]["scope"] == "SOURCE_LOCAL"
    assert result["endpoint_identities"][1]["semantic_type"] == "RESOURCE_TYPE"


@pytest.mark.parametrize("wrong", ["type", "head", "restriction"])
def test_restricted_resource_class_must_preserve_type_head_and_restriction(wrong):
    text = "Amazon EKS supports clusters using Karpenter, although Karpenter works with any conformant Kubernetes cluster."
    obj = endpoint("clusters using Karpenter", "RESOURCE_TYPE", "clusters", ["using Karpenter"])
    good = relation(text, endpoint("Amazon EKS", "SERVICE_PRODUCT"), obj, predicate="SUPPORTS", trigger="supports")
    result = validate(text, relations=[good]).accepted_relationships[0]
    assert result["endpoint_identities"][1]["restrictions"] == ("using Karpenter",)
    if wrong == "type":
        obj["semantic_type"] = "CONCEPT"
    elif wrong == "head":
        obj["head_quote"] = "Karpenter"
    else:
        obj["restrictions"] = []
    assert validate(text, relations=[good]).rejected_records


@pytest.mark.parametrize("text", [
    "To build an API with Lambda integrations, you can use Lambda proxy integration.",
    "In either a Lambda proxy integration, you can use a Lambda function in another account.",
])
def test_introductory_service_is_not_the_user_action_agent(text):
    obj = endpoint("Lambda proxy integration", "FEATURE", "integration") if "To build" in text else endpoint("Lambda function", "RESOURCE_TYPE", "function")
    row = relation(text, endpoint("Lambda", "SERVICE_PRODUCT"), obj, trigger="can use", modality="CAPABILITY")
    assert validate(text, relations=[row]).rejected_records


def test_action_complement_is_deferred_instead_of_a_misleading_binary_edge():
    text = "A trail enables CloudTrail to deliver log files to an Amazon S3 bucket."
    row = relation(text, endpoint("trail", "RESOURCE_TYPE"), endpoint("CloudTrail", "SERVICE_PRODUCT"), predicate="SUPPORTS", trigger="enables")
    assert validate(text, relations=[row]).rejected_records[0]["reason"] == "V9_ENABLES_ACTION_DEFER_NON_BINARY"
    result = validate(text, deferred=[{"clause_quote": text, "reason": "ENABLES_ACTION"}])
    assert not result.accepted_relationships and not result.rejected_records
    assert result.deferred_statements[0]["binary_edge_projected"] is False


def test_enumeration_is_not_a_feature_identity_and_keeps_following_sdk_scope():
    clause = "Amazon SQS offers two options for deleting messages after processing:"
    text = clause + "\n\nManual deletion uses DeleteMessage. Automatic deletion is supported only in certain AWS SDKs."
    row = relation(clause, endpoint("Amazon SQS", "SERVICE_PRODUCT"),
        endpoint("two options for deleting messages after processing", "FEATURE", "options"), predicate="SUPPORTS", trigger="offers")
    assert validate(text, relations=[row]).rejected_records[0]["reason"] == "V9_UNRESOLVED_OR_ENUMERATED_ENDPOINT"
    result = validate(text, deferred=[{"clause_quote": clause, "reason": "ENUMERATED_ALTERNATIVES"}])
    assert result.deferred_statements[0]["evidence_quote"] == text


@pytest.mark.parametrize("mutation", ["negation", "modality", "resource_type", "truncated_quote", "word_boundary", "head"])
def test_invalid_semantics_never_pass_even_when_quotes_exist(mutation):
    text = "If configured, CloudWatch can use execution role."
    row = relation(text, endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("execution role", "RESOURCE_INSTANCE", "role"),
        trigger="can use", modality="CAPABILITY")
    if mutation == "negation":
        row["polarity"] = "NEGATIVE"
    elif mutation == "modality":
        row["modality"] = "ASSERTED"
    elif mutation == "resource_type":
        row["object"]["semantic_type"] = "CONCEPT"
    elif mutation == "truncated_quote":
        row["clause_quote"] = "CloudWatch can use execution role."
    elif mutation == "word_boundary":
        row["object"] = endpoint("role", "RESOURCE_INSTANCE")
        text = text.replace("execution role", "controlerole")
        row["clause_quote"] = text
    else:
        row["object"]["head_quote"] = "not in source"
    assert validate(text, relations=[row]).rejected_records


def test_permission_is_a_source_bound_triad_with_no_resource_overlap():
    text = "If you use these logs, the execution role needs permission to write data to CloudWatch Logs."
    row = dict(clause_quote=text, principal=endpoint("execution role", "RESOURCE_TYPE", "role"), action_text="write data",
        resource=endpoint("CloudWatch Logs", "SERVICE_PRODUCT"), trigger_quote="needs permission")
    result = validate(text, permissions=[row]).accepted_permissions[0]
    assert result["evidence"]["quote"] == text
    assert result["serving_projection"] == "NOT_IMPLEMENTED_PERMISSION_TRIAD"
    row["action_text"] = "write data to CloudWatch Logs"
    assert validate(text, permissions=[row]).rejected_records


def test_passive_direct_agent_and_protocol_types_still_work():
    quote = "The service-linked roles used by Service Quotas grant access to service principals."
    row = relation(quote, endpoint("Service Quotas", "SERVICE_PRODUCT"), endpoint("service-linked roles", "RESOURCE_TYPE", "roles"), trigger="used", voice="PASSIVE")
    assert len(validate(quote, relations=[row]).accepted_relationships) == 1
    quote = "A NAT gateway supports TCP."
    row = relation(quote, endpoint("NAT gateway", "RESOURCE_TYPE", "gateway"), endpoint("TCP", "PROTOCOL"), predicate="SUPPORTS", trigger="supports")
    assert len(validate(quote, relations=[row]).accepted_relationships) == 1


@pytest.mark.parametrize("case", ["coverage", "incomplete", "extra_field", "invented_source"])
def test_schema_and_source_failures_are_explicit(case):
    text = "CloudWatch supports TLS."
    if case == "invented_source":
        result = validate(text, deferred=[{"clause_quote": "Invented source.", "reason": "OUTSIDE_PREDICATES"}])
        assert result.rejected_records and not result.deferred_statements
        return
    raw = {"relationships": [], "permission_requirements": [], "deferred_statements": [], "coverage_note": "NO_ELIGIBLE_RELATION"}
    if case == "coverage":
        raw["coverage_note"] = "HAS_CANDIDATES"
    elif case == "extra_field":
        raw["hidden_tool"] = "execute"
    with pytest.raises(ExtractionError):
        validate_semantic_output(canonical_json(raw).decode(), text, stop_reason="length" if case == "incomplete" else "end_turn")
