"""Source-address regressions and counterexamples, not AWS accuracy labels."""

import copy

import pytest

from app.contracts import canonical_json
from app.extraction import ExtractionError
from app.semantic_extraction_v10 import semantic_request, validate_semantic_output
from app.source_normalization_v10 import normalize_record


def ep(name, kind="CONCEPT", head=None, *, scope=(), restrictions=(), occurrence=1):
    return dict(name=name, semantic_type=kind, head_quote=head or name, restrictions=list(restrictions),
        scope_qualifiers=list(scope), occurrence=occurrence)


def rel(text, subject, obj, **changes):
    return dict(clause_quote=text, subject=subject, object=obj, predicate="USES", trigger_quote="uses",
        voice="ACTIVE", polarity="POSITIVE", modality="ASSERTED") | changes


def action(text, actor, target, **changes):
    return dict(clause_quote=text, actor=actor, target=target, action_family="DELIVER", action_quote="delivers",
        target_role="DESTINATION", payload_quote="log files", construction="FINITE_ACTIVE", enabler=None,
        polarity="POSITIVE", modality="ASSERTED") | changes


def validate(text, *, relationships=(), permissions=(), actions=(), deferred=()):
    raw = dict(relationships=relationships, permission_requirements=permissions, actions=actions,
        deferred_statements=deferred, coverage_note="HAS_CANDIDATES" if relationships or permissions or actions
        else "DEFERRED_ONLY" if deferred else "NO_ELIGIBLE_RELATION")
    return validate_semantic_output(canonical_json(raw).decode(), text, stop_reason="end_turn")


def test_unique_occurrence_fix_is_source_bound_and_preserves_raw():
    text = "Amazon SQS integrates with Amazon SNS in the following ways:"
    row = rel(text, ep("Amazon SQS", "SERVICE_PRODUCT"), ep("Amazon SNS", "SERVICE_PRODUCT", occurrence=2),
        predicate="INTEGRATES_WITH", trigger_quote="integrates with")
    original = copy.deepcopy(row)
    accepted = validate(text, relationships=[row]).accepted_relationships[0]
    assert row == original and row["object"]["occurrence"] == 2
    assert accepted["record"]["object"]["occurrence"] == 1
    audit = accepted["normalization"]
    assert audit["raw_record"] == row and audit["raw_sha256"] != audit["normalized_sha256"]
    assert audit["edits"][0]["witness"] == "Amazon SNS"


def test_ambiguous_occurrence_and_duplicate_clause_are_not_guessed():
    text = "Amazon SQS uses Amazon SNS before Amazon SNS sends notifications."
    row = rel(text, ep("Amazon SQS", "SERVICE_PRODUCT"), ep("Amazon SNS", "SERVICE_PRODUCT", occurrence=3))
    normalized, audit = normalize_record(row, text, kind="RELATION")
    assert normalized == row and audit["edits"] == []
    assert validate(text, relationships=[row]).rejected_records
    with pytest.raises(ExtractionError):
        normalize_record(row, text + " " + text, kind="RELATION")


def test_permission_trailing_to_fix_does_not_invent_iam_action():
    text = "The caller must have permission to invoke the Lambda function."
    row = dict(clause_quote=text, principal=ep("caller"), action_text="invoke",
        resource=ep("Lambda function", "RESOURCE_TYPE", "function"), trigger_quote="must have permission to")
    accepted = validate(text, permissions=[row]).accepted_permissions[0]
    assert accepted["record"]["trigger_quote"] == "must have permission"
    assert accepted["record"]["action_text"] == "invoke"
    assert accepted["normalization"]["raw_record"] == row


def test_data_scope_stays_concept_and_preserves_full_source_context():
    text = "When configured, CloudFront uses cookies in viewer requests to identify cached objects."
    row = rel(text, ep("CloudFront", "SERVICE_PRODUCT"), ep("cookies in viewer requests", head="cookies", scope=["in viewer requests"]))
    accepted = validate(text, relationships=[row]).accepted_relationships[0]
    assert accepted["endpoint_identities"][1]["semantic_type"] == "CONCEPT"
    assert accepted["endpoint_identities"][1]["scope_qualifiers"] == ("in viewer requests",)
    assert accepted["evidence"]["quote"] == text
    row["object"]["scope_qualifiers"] = ["in origin requests"]
    assert validate(text, relationships=[row]).rejected_records[0]["reason"] == "V10_SCOPE_MUST_BE_EXACT_ENDPOINT_QUALIFIER"


@pytest.mark.parametrize("name", ["two calls", "calls"])
def test_unresolved_counted_reference_never_promotes(name):
    text = "These two calls use the taskToken provided by GetActivityTask to associate the result with that task."
    row = rel(text, ep(name, head="calls"), ep("taskToken"), trigger_quote="use")
    assert validate(text, relationships=[row]).rejected_records


def test_sns_positive_is_not_lost_to_a_trailing_list():
    text = "Amazon SNS uses the publisher-subscriber model, requiring three types of user account access:"
    row = rel(text, ep("Amazon SNS", "SERVICE_PRODUCT"), ep("publisher-subscriber model", head="model"))
    assert len(validate(text, relationships=[row]).accepted_relationships) == 1
    assert "publisher-subscriber model" in semantic_request(text)[0]


def test_enabled_delivery_preserves_facilitator_actor_payload_destination_and_modality():
    text = "A trail enables CloudTrail to deliver log files to an Amazon S3 bucket."
    row = action(text, ep("CloudTrail", "SERVICE_PRODUCT"), ep("Amazon S3 bucket", "RESOURCE_TYPE", "bucket"),
        action_quote="deliver", construction="ENABLED_ACTION", enabler=ep("trail", "RESOURCE_TYPE", "trail"), modality="CAPABILITY")
    accepted = validate(text, actions=[row]).accepted_actions[0]
    assert accepted["record"] == row
    assert len(accepted["endpoint_identities"]) == 3
    assert accepted["serving_projection"] == "TYPED_ACTION_CANDIDATE_NOT_LEGACY_EDGE"
    assert validate(text, actions=[row | {"construction": "FINITE_ACTIVE", "enabler": None}]).rejected_records
    assert validate(text, actions=[row | {"modality": "ASSERTED"}]).rejected_records


@pytest.mark.parametrize("family,verb,target,role,payload,clause", [
    ("INVOKE", "invokes", ep("Lambda function", "RESOURCE_TYPE", "function"), "CALLEE", None, "Amazon S3 invokes the Lambda function."),
    ("READ", "reads", ep("S3 bucket", "RESOURCE_TYPE", "bucket"), "DATA_RESOURCE", "objects", "CloudTrail reads objects from an S3 bucket."),
    ("WRITE", "writes", ep("S3 bucket", "RESOURCE_TYPE", "bucket"), "DATA_RESOURCE", "objects", "CloudTrail writes objects to an S3 bucket."),
    ("ENCRYPT", "encrypts", ep("KMS key", "RESOURCE_TYPE", "key"), "KEY", "logs", "CloudTrail encrypts logs with a KMS key."),
    ("DECRYPT", "decrypts", ep("data"), "DATA_RESOURCE", None, "CloudTrail decrypts the data."),
    ("ASSUME_ROLE", "assumes", ep("IAM role", "RESOURCE_TYPE", "role"), "ROLE", None, "CloudTrail assumes an IAM role."),
])
def test_action_families_have_explicit_argument_roles(family, verb, target, role, payload, clause):
    # Constructed syntax controls: no assertion these examples are AWS facts.
    actor = ep("Amazon S3" if family == "INVOKE" else "CloudTrail", "SERVICE_PRODUCT")
    row = action(clause, actor, target, action_family=family, action_quote=verb, target_role=role, payload_quote=payload)
    assert len(validate(clause, actions=[row]).accepted_actions) == 1


@pytest.mark.parametrize("change", [dict(payload_quote=None), dict(payload_quote="files"), dict(target_role="CALLEE"),
    dict(actor=ep("trail", "RESOURCE_TYPE", "trail")), dict(action_quote="sends"), dict(polarity="NEGATIVE")])
def test_delivery_cannot_drop_payload_invent_actor_or_reverse_meaning(change):
    text = "A trail exists. CloudTrail delivers log files to an Amazon S3 bucket."
    row = action("CloudTrail delivers log files to an Amazon S3 bucket.", ep("CloudTrail", "SERVICE_PRODUCT"),
        ep("Amazon S3 bucket", "RESOURCE_TYPE", "bucket")) | change
    assert validate(text, actions=[row]).rejected_records


def test_conditional_negative_and_key_role_remain_visible():
    clause = "CloudTrail does not deliver log files to an Amazon S3 bucket."
    text = "If delivery is disabled, the following limitation applies. " + clause
    row = action(clause, ep("CloudTrail", "SERVICE_PRODUCT"), ep("Amazon S3 bucket", "RESOURCE_TYPE", "bucket"),
        action_quote="does not deliver", polarity="NEGATIVE")
    accepted = validate(text, actions=[row]).accepted_actions[0]
    assert accepted["evidence"]["quote"] == text
    assert accepted["record"]["polarity"] == "NEGATIVE"
    text2 = "CloudTrail encrypts logs with a KMS key."
    bad = action(text2, ep("CloudTrail", "SERVICE_PRODUCT"), ep("KMS key", "RESOURCE_TYPE", "key"),
        action_family="ENCRYPT", action_quote="encrypts", target_role="DATA_RESOURCE", payload_quote=None)
    assert validate(text2, actions=[bad]).rejected_records


@pytest.mark.parametrize("text,quote", [
    ("CloudTrail can deliver log files to an Amazon S3 bucket.", "can deliver"),
    ("CloudTrail must deliver log files to an Amazon S3 bucket.", "must deliver"),
    ("CloudTrail should deliver log files to an Amazon S3 bucket.", "should deliver"),
    ("CloudTrail needs permission to deliver log files to an Amazon S3 bucket.", "deliver"),
    ("To use CloudTrail, you deliver log files to an Amazon S3 bucket.", "deliver"),
])
def test_modality_permission_and_user_action_are_not_asserted_delivery(text, quote):
    row = action(text, ep("CloudTrail", "SERVICE_PRODUCT"), ep("Amazon S3 bucket", "RESOURCE_TYPE", "bucket"), action_quote=quote)
    result = validate(text, actions=[row])
    # must is retained as REQUIRED only, not an asserted action.
    assert result.rejected_records


def test_four_fact_cap_cannot_be_evaded_with_different_arrays():
    text = "CloudTrail delivers log files to an Amazon S3 bucket."
    a = action(text, ep("CloudTrail", "SERVICE_PRODUCT"), ep("Amazon S3 bucket", "RESOURCE_TYPE", "bucket"))
    r = rel(text, ep("CloudTrail", "SERVICE_PRODUCT"), ep("log files"), trigger="delivers")
    with pytest.raises(ExtractionError, match="V10_OUTPUT_SCHEMA"):
        validate(text, actions=[a, a, a], relationships=[r, r])
