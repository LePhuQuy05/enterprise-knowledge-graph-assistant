import copy

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction_v7 import semantic_recipe
from app.semantic_extraction_v8_review import review_recipe
from app.source_candidate_policy import policy_recipe
from app.typed_graph_records import canonicalize_reviewed
from tests.test_extraction import aliases, source
from tests.test_semantic_extraction_v7 import endpoint, frame
from tests.test_semantic_extraction_v8_review import review


def fixture():
    text = "CloudWatch uses execution role."
    row = frame(text, endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("execution role", "RESOURCE_INSTANCE"),
                predicate="USES", trigger="uses")
    _, decisions = review(text, frames=[row])
    decision = decisions[0]
    judgment = {"status": "AI_SOURCE_SUPPORTED_SOURCE_SCOPED", "candidate_sha256": sha256_bytes(canonical_json(decision)),
                "reason_vi": "Synthetic fixture only; not an AWS factual assertion."}
    recipe = semantic_recipe(aliases(), metadata_sha256="0" * 64)
    composite = {"native_recipe_sha256": recipe.fingerprint, "review": review_recipe(), "retention": policy_recipe()}
    return text, decision, judgment, recipe, composite


def test_unicode_offsets_are_document_relative_and_local_resources_do_not_merge():
    text, decision, judgment, recipe, composite = fixture()
    doc_a, chunk_a = source(text, suffix="document-a")
    doc_b, chunk_b = source(text, suffix="document-b")
    a = canonicalize_reviewed(decision, judgment, chunk=chunk_a, document=doc_a, native_recipe=recipe, composite=composite)
    b = canonicalize_reviewed(decision, judgment, chunk=chunk_b, document=doc_b, native_recipe=recipe, composite=composite)
    assert a[0][0].entity_id == b[0][0].entity_id  # Catalog service identity.
    assert a[0][1].entity_id != b[0][1].entity_id  # Two local role references.
    assert doc_a.normalized_text[a[2].evidence.start_char:a[2].evidence.end_char] == text
    assert a[1][0].start_char >= chunk_a.start_char
    assert not a[3]["human_reviewed"] and not a[3]["serving_published"]


@pytest.mark.parametrize("mutation", ["review_status", "candidate", "validation", "document"])
def test_unreviewed_or_changed_sources_never_canonicalize(mutation):
    text, decision, judgment, recipe, composite = fixture()
    doc, chunk = source(text)
    decision, judgment = copy.deepcopy(decision), copy.deepcopy(judgment)
    if mutation == "review_status":
        judgment["status"] = "PENDING"
    elif mutation == "candidate":
        decision["normalized_record"]["predicate"] = "DEPENDS_ON"
    elif mutation == "validation":
        decision["validation"]["endpoint_identities"][0]["canonical_key"] = "forged"
        judgment["candidate_sha256"] = sha256_bytes(canonical_json(decision))
    else:
        doc, _ = source(text, suffix="other")
    with pytest.raises(ValueError):
        canonicalize_reviewed(decision, judgment, chunk=chunk, document=doc, native_recipe=recipe, composite=composite)


def test_source_review_label_cannot_override_wrong_clause_agent():
    text = "To build an API with Lambda integrations, you can use Lambda proxy integration."
    row = frame(text, endpoint("Lambda", "SERVICE_PRODUCT"), endpoint("Lambda proxy integration", "FEATURE"),
        predicate="USES", trigger="can use")
    row["modality"] = "CAPABILITY"
    _, decisions = review(text, frames=[row])
    decision = decisions[0]
    assert decision["status"] == "CANDIDATE_SOURCE_GUARDS_PASS"  # Frozen V7/V8 limitation.
    judgment = {"status": "AI_SOURCE_SUPPORTED_SOURCE_SCOPED", "candidate_sha256": sha256_bytes(canonical_json(decision)),
        "reason_vi": "Adversarial fixture: this label is deliberately wrong."}
    _, _, _, recipe, composite = fixture()
    doc, chunk = source(text)
    with pytest.raises(ValueError, match="SUBJECT_NOT_DIRECT_CLAUSE_AGENT"):
        canonicalize_reviewed(decision, judgment, chunk=chunk, document=doc, native_recipe=recipe, composite=composite)


def test_modality_is_explicit_in_sidecar_and_exact_scope_survives_projection():
    text = "CloudWatch can use execution role."
    row = frame(text, endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("execution role", "RESOURCE_INSTANCE"),
        predicate="USES", trigger="can use")
    row["modality"] = "CAPABILITY"
    _, decisions = review(text, frames=[row])
    decision = decisions[0]
    judgment = {"status": "AI_SOURCE_SUPPORTED_SOURCE_SCOPED", "candidate_sha256": sha256_bytes(canonical_json(decision)),
        "reason_vi": "Synthetic capability fixture, not a claim about AWS."}
    _, _, _, recipe, composite = fixture()
    doc, chunk = source(text)
    _, _, assertion, metadata = canonicalize_reviewed(decision, judgment, chunk=chunk, document=doc, native_recipe=recipe, composite=composite)
    assert metadata["modality"] == "CAPABILITY" and metadata["trigger_quote"] == "can use"
    assert any(q.text == text for q in assertion.qualifiers)
