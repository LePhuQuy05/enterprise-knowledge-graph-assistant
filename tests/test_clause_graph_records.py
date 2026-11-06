from copy import deepcopy

import pytest

from app.clause_graph_records import canonicalize_relation
from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction_v9 import semantic_recipe
from tests.test_extraction import aliases, source
from tests.test_semantic_extraction_v9 import endpoint, relation, validate


def fixture():
    clause = "CloudWatch can use execution role."
    text = "If configured, apply the documented notification settings. " + clause
    row = relation(clause, endpoint("CloudWatch", "SERVICE_PRODUCT"), endpoint("execution role", "RESOURCE_INSTANCE", "role"),
        trigger="can use", modality="CAPABILITY")
    candidate = validate(text, relations=[row]).accepted_relationships[0]
    judgment = {"status": "AI_SOURCE_SUPPORTED_SOURCE_SCOPED", "candidate_sha256": sha256_bytes(canonical_json(candidate)),
        "reason_vi": "Synthetic mechanism fixture only, not an AWS factual label."}
    return text, candidate, judgment, semantic_recipe(aliases(), metadata_sha256="0" * 64)


def test_full_context_clause_offset_and_local_identity_survive_canonicalization():
    text, candidate, judgment, recipe = fixture()
    doc_a, chunk_a = source(text, suffix="a")
    doc_b, chunk_b = source(text, suffix="b")
    a = canonicalize_relation(candidate, judgment, chunk=chunk_a, document=doc_a, native_recipe=recipe)
    b = canonicalize_relation(candidate, judgment, chunk=chunk_b, document=doc_b, native_recipe=recipe)
    assert a[0][0].entity_id == b[0][0].entity_id and a[0][1].entity_id != b[0][1].entity_id
    assert a[2].evidence.quote == text and a[2].qualifiers[0].text == text
    assert a[3]["modality"] == "CAPABILITY"
    span = a[3]["clause_span"]
    assert doc_a.normalized_text[span["start_char"]:span["end_char"]] == "CloudWatch can use execution role."


@pytest.mark.parametrize("change", ["source", "review", "candidate_hash", "endpoint_span", "context"])
def test_tampered_or_unreviewed_data_cannot_materialize(change):
    text, original, original_judgment, recipe = fixture()
    candidate, judgment = deepcopy(original), deepcopy(original_judgment)
    doc, chunk = source(text)
    if change == "source":
        doc, _ = source(text, suffix="wrong")
    elif change == "review":
        judgment["status"] = "PENDING"
    elif change == "candidate_hash":
        judgment["candidate_sha256"] = "0" * 64
    elif change == "endpoint_span":
        candidate["endpoint_identities"][1]["start_char"] += 1
        judgment["candidate_sha256"] = sha256_bytes(canonical_json(candidate))
    else:
        candidate["qualifiers"] = []
        judgment["candidate_sha256"] = sha256_bytes(canonical_json(candidate))
    with pytest.raises(ValueError):
        canonicalize_relation(candidate, judgment, chunk=chunk, document=doc, native_recipe=recipe)


def test_claimed_ai_support_cannot_promote_an_unresolved_count_reference():
    text = "These two calls use taskToken to associate the result with that task."
    row = relation(text, endpoint("two calls", "CONCEPT", "calls"), endpoint("taskToken", "CONCEPT"), trigger="use")
    candidate = validate(text, relations=[row]).accepted_relationships[0]
    judgment = {"status": "AI_SOURCE_SUPPORTED_SOURCE_SCOPED", "candidate_sha256": sha256_bytes(canonical_json(candidate))}
    _, _, _, recipe = fixture()
    doc, chunk = source(text)
    with pytest.raises(ValueError, match="UNRESOLVED_COUNTED_NOUN"):
        canonicalize_relation(candidate, judgment, chunk=chunk, document=doc, native_recipe=recipe)
