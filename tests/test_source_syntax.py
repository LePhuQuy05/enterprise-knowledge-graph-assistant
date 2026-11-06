import json
from copy import deepcopy
from pathlib import Path

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.extraction import ExtractionError
from app.semantic_extraction_v10 import RelationV10, SourceActionV1
from app.source_action_proposals import propose_actions
from app.source_syntax import SyntaxDocument, validate_syntax_record
from tests.test_semantic_extraction_v10 import action, ep, rel

FIXTURE = json.loads((Path(__file__).parent / "fixtures/source_syntax/annotations.json").read_bytes())
DOCUMENTS = {d["text"]: d for d in FIXTURE["documents"]}


def parse(text):
    return SyntaxDocument(DOCUMENTS[text])


def relation(text, actor="S3", obj="TLS"):
    return RelationV10.model_validate_json(canonical_json(rel(text, ep(actor, "SERVICE_PRODUCT"), ep(obj, "PROTOCOL"))))


@pytest.mark.parametrize("case", FIXTURE["real_cases"])
def test_actual_source_regressions_keep_exact_roles_scope_and_full_context(case):
    raw = deepcopy(case["record"])
    model = RelationV10 if case["kind"] == "RELATION" else SourceActionV1
    record = model.model_validate_json(canonical_json(raw))
    candidate = validate_syntax_record(record, case["source_text"], parse(record.clause_quote), kind=case["kind"])
    assert candidate["record"] == raw and candidate["evidence"]["quote"] == case["source_text"]
    assert candidate["semantic_review"] == "PENDING_NOT_SERVING"
    if "some or all" in record.clause_quote:
        q = candidate["target"]["quantifier"]
        assert q["quote"] == "some or all of the "
        assert record.clause_quote[q["start_char"]:q["end_char"]] == q["quote"]
        assert candidate["target"]["scope_attachment_note"].endswith("REQUIRES_SEMANTIC_REVIEW")
    if case["kind"] == "ACTION":
        assert candidate["subject_chain"][0]["kind"] == "SAME_SUBJECT_COORDINATION"
        assert candidate["target"]["source_annotations"][0]["quote"] == "that you specify"
        assert raw["target"]["restrictions"] == []  # raw receipt was not repaired in place.


@pytest.mark.parametrize("text", ["To configure S3, you can use TLS.", "S3 clients use TLS."])
def test_service_in_intro_or_compound_is_not_the_actual_actor(text):
    row = relation(text).model_copy(update={"trigger_quote": "use"})
    with pytest.raises(ExtractionError):
        validate_syntax_record(row, text, parse(text), kind="RELATION")


def test_adverb_needs_no_lexical_whitelist_and_wrong_polarity_is_rejected():
    text = "S3 internally uses TLS."
    candidate = validate_syntax_record(relation(text), text, parse(text), kind="RELATION")
    assert candidate["actor"]["span"]["quote"] == "S3"
    with pytest.raises(ExtractionError, match="POLARITY"):
        validate_syntax_record(relation(text).model_copy(update={"polarity": "NEGATIVE"}), text, parse(text), kind="RELATION")


@pytest.mark.parametrize("text", [
    "CloudTrail needs permission to deliver log files to an S3 bucket.",
    "CloudTrail does not need to deliver log files to an S3 bucket.",
    "CloudTrail captures API calls and Lambda delivers log files to an S3 bucket.",
])
def test_permission_negated_necessity_and_different_actor_never_become_cloudtrail_delivery(text):
    row = SourceActionV1.model_validate_json(canonical_json(action(text, ep("CloudTrail", "SERVICE_PRODUCT"),
        ep("S3 bucket", "RESOURCE_TYPE", "bucket"), action_quote="deliver" if "deliver " in text else "delivers")))
    with pytest.raises(ExtractionError):
        validate_syntax_record(row, text, parse(text), kind="ACTION")


def test_aspect_proposal_keeps_dynamodb_actor_modal_payload_and_propagation_condition():
    text = next(t for t in DOCUMENTS if t.startswith("After you disable the destination"))
    candidates, _ = propose_actions(parse(text), text)
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate["record"]["actor"]["name"] == "DynamoDB"
    assert candidate["record"]["modality"] == "CAPABILITY"
    assert candidate["record"]["payload_quote"] == "in-flight records"
    assert candidate["record"]["target"]["name"] == "the data stream"
    assert candidate["subject_chain"][0]["kind"] == "CONTINUE_ASPECT"
    assert "until the change fully propagates" in candidate["evidence"]["quote"]
    assert candidate["origin"] == "DEPENDENCY_SOURCE_PROPOSAL_NOT_NATIVE_LLM_EXTRACTION"


@pytest.mark.parametrize("text", ["CloudTrail needs permission to deliver log files to an S3 bucket.",
    "CloudTrail does not need to deliver log files to an S3 bucket.", "To configure S3, you can use TLS."])
def test_dependency_proposals_do_not_turn_user_procedures_or_obligations_into_service_actions(text):
    candidates, _ = propose_actions(parse(text), text)
    assert candidates == []


@pytest.mark.parametrize("case", FIXTURE["nonfinite_cases"])
def test_actual_allow_have_for_infinitives_never_become_asserted_service_actions(case):
    record = SourceActionV1.model_validate_json(canonical_json(case["record"]))
    with pytest.raises(ExtractionError, match="INFINITIVE_OR_NONFINITE"):
        validate_syntax_record(record, case["source_text"], parse(record.clause_quote), kind="ACTION")


@pytest.mark.parametrize("change", ["text", "span", "head", "cycle", "omission"])
def test_malformed_or_changed_parse_never_resolves_source(change):
    document = deepcopy(DOCUMENTS["S3 internally uses TLS."])
    if change == "text":
        document["text"] = document["text"].replace("TLS", "SSL")
    elif change == "span":
        document["tokens"][0]["end"] += 1
    elif change == "head":
        document["tokens"][0]["head"] = 99
    elif change == "cycle":
        document["tokens"][0]["head"], document["tokens"][1]["head"] = 1, 0
    else:
        document["tokens"].pop()
    with pytest.raises(ExtractionError):
        SyntaxDocument(document)


@pytest.mark.parametrize("a,b", [("A", "B"), ("60 seconds", "600 seconds")])
def test_account_ttl_and_exception_scopes_never_share_source_identity(a, b):
    text_a = next(t for t in DOCUMENTS if ("account " + a in t) if a in {"A", "B"}) if a == "A" else next(t for t in DOCUMENTS if a in t)
    text_b = text_a.replace(a, b)
    assert text_b in DOCUMENTS
    assert sha256_bytes(text_a.encode()) != sha256_bytes(text_b.encode())
    assert SyntaxDocument(DOCUMENTS[text_a]).text != SyntaxDocument(DOCUMENTS[text_b]).text
