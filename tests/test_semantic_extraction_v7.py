import json

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction_v7 import (
    SemanticOutputV7,
    TypedEndpointV7,
    endpoint_identity,
    semantic_raw_recipe,
    semantic_recipe,
    validate_semantic_output,
)
from tests.test_extraction import aliases


def endpoint(name, semantic_type="CONCEPT", occurrence=1):
    return dict(name=name, semantic_type=semantic_type, occurrence=occurrence)


def frame(text, subject, obj, *, predicate="SUPPORTS", trigger="supports", qualifiers=(), **fields):
    return dict(evidence_quote=text, subject=subject, object=obj, predicate=predicate,
        trigger_quote=trigger, voice="ACTIVE", polarity="POSITIVE", modality="ASSERTED",
        qualifiers=list(qualifiers), **fields)


def validate(text, frames=(), permissions=(), coverage=None):
    return validate_semantic_output(json.dumps({"relationships": frames, "permission_requirements": permissions,
        "coverage_note": coverage or ("HAS_CANDIDATES" if frames or permissions else "NO_ELIGIBLE_RELATION")}),
        text, stop_reason="end_turn")


def test_query_language_and_resource_type_are_distinct_from_service_and_protocol():
    text = "CloudWatch supports PromQL for querying metrics."
    row = frame(text, endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("PromQL", "QUERY_LANGUAGE"),
                qualifiers=[{"kind": "LIMITATION", "text": "for querying metrics"}])
    good = validate(text, [row])
    assert good.accepted_relationships[0]["endpoint_identities"][1]["semantic_type"] == "QUERY_LANGUAGE"
    assert good.accepted_relationships[0]["core_projection"]["entities"][1]["entity_type"] == "CONCEPT"
    row["object"]["semantic_type"] = "PROTOCOL"
    assert validate(text, [row]).rejected_records[0]["reason"] == "V7_CATALOG_TYPE_MISMATCH"
    text = "A NAT gateway supports TCP."
    row = frame(text, endpoint("NAT gateway", "RESOURCE_TYPE"), endpoint("TCP", "PROTOCOL"))
    assert validate(text, [row]).accepted_relationships
    row["subject"]["semantic_type"] = "SERVICE_PRODUCT"
    assert validate(text, [row]).rejected_records


@pytest.mark.parametrize("object_name", ["TCP", "UDP", "ICMP"])
def test_real_colon_list_keeps_each_argument_and_other_list_members(object_name):
    text = "A NAT gateway supports the following protocols: TCP, UDP, and ICMP."
    row = frame(text, endpoint("NAT gateway", "RESOURCE_TYPE"), endpoint(object_name, "PROTOCOL"),
                qualifiers=[{"kind": "LIMITATION", "text": text}])
    assert validate(text, [row]).accepted_relationships


@pytest.mark.parametrize("name", ["This encryption", "some instance types", "those roles", "your resource"])
def test_unresolved_noun_phrases_are_never_global_or_source_local_entities(name):
    text = name + " uses TLS."
    row = frame(text, endpoint(name), endpoint("TLS", "PROTOCOL"), predicate="USES", trigger="uses")
    assert validate(text, [row]).rejected_records[0]["reason"] in {
        "V7_UNRESOLVED_OR_ACTION_ENDPOINT", "SEMANTIC_UNRESOLVED_OR_NON_ENTITY_ENDPOINT"}


def test_relative_clause_does_not_become_universal_class_edge_even_with_qualifier():
    text = "There are charges for EC2 instances that use internet gateways."
    row = frame(text, endpoint("EC2 instances", "RESOURCE_TYPE"), endpoint("internet gateways", "RESOURCE_TYPE"),
                predicate="USES", trigger="use", qualifiers=[{"kind": "LIMITATION", "text": text}])
    assert "RELATIVE_CLASS_RESTRICTION" in validate(text, [row]).rejected_records[0]["reason"]


@pytest.mark.parametrize("text,quote", [
    ("If configured, CloudWatch supports PromQL.", "CloudWatch supports PromQL."),
    ("CloudWatch supports PromQL only when configured.", "CloudWatch supports PromQL"),
])
def test_model_cannot_remove_a_condition_by_truncating_evidence(text, quote):
    row = frame(quote, endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("PromQL", "QUERY_LANGUAGE"))
    assert "SENTENCE_SCOPE_TRUNCATED" in validate(text, [row]).rejected_records[0]["reason"]


def test_association_and_purpose_must_be_preserved():
    text = "Secrets Manager uses a symmetric encryption KMS key associated with a secret to generate a data key."
    row = frame(text, endpoint("Secrets Manager", "SERVICE_PRODUCT"), endpoint("symmetric encryption KMS key", "RESOURCE_TYPE"),
                predicate="USES", trigger="uses")
    assert validate(text, [row]).rejected_records[0]["reason"] == "V7_TRAILING_SCOPE_QUALIFIER_REQUIRED"
    row["qualifiers"] = [{"kind": "LIMITATION", "text": "associated with a secret to generate a data key"}]
    accepted = validate(text, [row]).accepted_relationships
    assert accepted and accepted[0]["endpoint_identities"][1]["scope"] == "SOURCE_LOCAL"


def test_permission_keeps_purpose_and_exact_action_without_inventing_iam_api():
    text = "To deliver event data to a target, EventBridge needs permission to access the target resource."
    permission = dict(evidence_quote=text, principal=endpoint("EventBridge", "SERVICE_PRODUCT"),
        resource=endpoint("target resource", "RESOURCE_INSTANCE"), action_text="access",
        trigger_quote="needs permission", qualifiers=[])
    assert validate(text, permissions=[permission]).rejected_records[0]["reason"] == "V7_LEADING_SCOPE_QUALIFIER_REQUIRED"
    permission["qualifiers"] = [{"kind": "LIMITATION", "text": "To deliver event data to a target"}]
    good = validate(text, permissions=[permission]).accepted_permissions[0]
    assert good["record"]["action_text"] == "access"
    assert good["endpoint_identities"][1]["scope"] == "SOURCE_LOCAL"
    assert good["serving_projection"] == "NOT_IMPLEMENTED_PERMISSION_TRIAD"


def test_known_empty_miss_passive_role_use_is_representable_with_scope():
    text = "The service-linked roles used by Service Quotas grant access to the following service principals."
    row = frame(text, endpoint("Service Quotas", "SERVICE_PRODUCT"), endpoint("service-linked roles", "RESOURCE_INSTANCE"),
                predicate="USES", trigger="used", qualifiers=[{"kind": "LIMITATION", "text": text}])
    row["voice"] = "PASSIVE"
    assert validate(text, [row]).accepted_relationships


def test_indirect_object_hidden_in_trigger_is_rejected():
    text = "CloudFormation uses the modified list to associate groups with MyEC2Instance."
    row = frame(text, endpoint("CloudFormation", "SERVICE_PRODUCT"), endpoint("MyEC2Instance", "RESOURCE_INSTANCE"),
                predicate="USES", trigger="uses the modified list to associate groups with")
    assert validate(text, [row]).rejected_records[0]["reason"] == "V7_EXACT_MINIMAL_TRIGGER_REQUIRED"
    row["trigger_quote"] = "uses"
    assert validate(text, [row]).rejected_records[0]["reason"] == "V7_DIRECT_ARGUMENT_REQUIRED"


def test_unknown_names_do_not_merge_across_sources_or_evidence():
    e = TypedEndpointV7(**endpoint("execution role", "RESOURCE_INSTANCE"))
    quote = "The execution role needs permission."
    first = endpoint_identity(e, text_sha256="0" * 64, evidence_quote=quote)
    other = endpoint_identity(e, text_sha256="1" * 64, evidence_quote=quote)
    assert first["scope"] == other["scope"] == "SOURCE_LOCAL"
    assert first["canonical_key"] != other["canonical_key"]


@pytest.mark.parametrize("text,trigger,polarity,modality", [
    ("CloudWatch does not support TLS.", "does not support", "NEGATIVE", "ASSERTED"),
    ("CloudWatch can support TLS.", "can support", "POSITIVE", "CAPABILITY"),
])
def test_valid_negation_and_capability_keep_scope(text, trigger, polarity, modality):
    row = frame(text, endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("TLS", "PROTOCOL"), trigger=trigger,
                qualifiers=[{"kind": "LIMITATION", "text": text}])
    row.update(polarity=polarity, modality=modality)
    assert validate(text, [row]).accepted_relationships


def test_invalid_record_does_not_erase_independent_candidate():
    text = "CloudWatch supports TLS. This encryption uses TLS."
    valid = frame("CloudWatch supports TLS.", endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("TLS", "PROTOCOL"))
    invalid = frame("This encryption uses TLS.", endpoint("This encryption"), endpoint("TLS", "PROTOCOL"), predicate="USES", trigger="uses")
    result = validate(text, [valid, invalid])
    assert len(result.accepted_relationships) == 1 and len(result.rejected_records) == 1
    assert len(result.output.relationships) == 2


def test_coverage_note_is_diagnostic_and_not_permission_to_omit_schema():
    assert validate("No relation.", coverage="INSUFFICIENT_CONTEXT").output.coverage_note == "INSUFFICIENT_CONTEXT"
    with pytest.raises(ValueError, match="COVERAGE_NOTE_CONFLICT"):
        validate("No relation.", coverage="HAS_CANDIDATES")
    with pytest.raises(ValueError, match="OUTPUT_SCHEMA"):
        validate_semantic_output('{"relationships":[],"permission_requirements":[]}', "No relation.", stop_reason="end_turn")


def test_recipe_and_schema_are_distinct_and_parameters_are_pinned():
    recipe = semantic_recipe(aliases(), metadata_sha256="0" * 64)
    assert semantic_raw_recipe(recipe)["version"] == "deepseek-typed-scope-frames-v7"
    changed = recipe.model_copy(update={"parameters": recipe.parameters | {"typed_catalog_sha256": "0" * 64}})
    with pytest.raises(ValueError, match="UNSUPPORTED_RECIPE"):
        semantic_raw_recipe(changed)
    assert sha256_bytes(canonical_json(SemanticOutputV7.model_json_schema()))
