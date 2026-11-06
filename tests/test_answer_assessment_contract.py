"""Assessment metadata preserves citation contracts and historical response shapes."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import AnswerAssessment, AnswerLimitation, QueryResponse


def response_data(status="ANSWERED"):
    data = {
        "request_id": "request",
        "answer": "Supported public fact.",
        "claims": [{"claim_id": "c1", "text": "Supported public fact.", "citation_ids": ["cite1"]}],
        "citations": [{
            "citation_id": "cite1", "chunk_id": "chunk1", "document_version_id": "docv1",
            "title": "Public source", "source_uri": "https://example.invalid/",
            "section_path": ["Section"], "quote": "Supported public fact.",
        }],
        "answer_status": status, "mode_requested": "hybrid", "mode_used": "hybrid",
        "used_graph": False, "snapshot_version": "snapshot1",
    }
    if status == "ABSTAINED":
        data.update(answer="No supported answer.", claims=[], citations=[])
    return data


def limitation_data(**changes):
    return {
        "code": "ACCOUNT_STATE_UNAVAILABLE", "reported_by": "MODEL_REPORTED",
        "question_part": "The actual route in my account",
        "message": "The supplied documentation does not show your account configuration.",
        "next_step": "Inspect the relevant configuration separately.",
    } | changes


def validate_response(data):
    return QueryResponse.model_validate_json(json.dumps(data))


def test_legacy_response_has_exact_historical_dump_and_json_shape():
    data = response_data()
    expected = data | {
        "retrieval_backend": "snapshot_numpy_adjacency", "warnings": [],
        "answer_language": "en", "evidence_bundle_sha256": None,
    }
    response = validate_response(data)
    assert response.assessment is None
    assert response.model_dump(mode="json") == expected
    assert json.loads(response.model_dump_json()) == expected
    # Only the newly introduced absent field is omitted: historical nulls remain.
    assert "evidence_bundle_sha256" in response.model_dump()
    assert "assessment" not in response.model_dump()
    assert validate_response(expected).model_dump_json() == response.model_dump_json()


def test_explicit_null_assessment_also_omits_only_that_field():
    legacy = validate_response(response_data())
    explicit = validate_response(response_data() | {"assessment": None})
    assert explicit.model_dump_json() == legacy.model_dump_json()


@pytest.mark.parametrize("status", ["ANSWERED", "PARTIAL", "ABSTAINED"])
def test_legacy_statuses_do_not_gain_new_assessment_requirements(status):
    response = validate_response(response_data(status))
    assert response.assessment is None
    assert "assessment" not in response.model_dump(mode="json")


@pytest.mark.parametrize("status", ["PARTIAL", "ABSTAINED"])
def test_new_partial_and_abstained_assessments_require_limitations(status):
    with pytest.raises(ValidationError, match="need a limitation"):
        validate_response(response_data(status) | {"assessment": {}})
    response = validate_response(response_data(status) | {
        "assessment": {"limitations": [limitation_data()]},
    })
    assert response.assessment.limitations[0].reported_by == "MODEL_REPORTED"
    assert response.model_dump(mode="json")["assessment"]["limitations"][0]["question_part"]
    assert QueryResponse.model_validate_json(response.model_dump_json()) == response


def test_answered_can_describe_policy_boundary_without_unanswered_limitations():
    response = validate_response(response_data() | {"assessment": {}})
    assessment = response.assessment
    assert assessment.source_scope == "SUPPLIED_DOCUMENTATION_ONLY"
    assert assessment.account_state_access == "NOT_CONNECTED"
    assert assessment.policy_origin == "SYSTEM_POLICY"
    assert assessment.limitations == ()
    with pytest.raises(ValidationError, match="cannot report unanswered limitations"):
        validate_response(response_data() | {"assessment": {"limitations": [limitation_data()]}})


def test_metadata_does_not_become_an_uncited_claim_or_modify_citation_text():
    data = response_data("PARTIAL")
    response = validate_response(data | {"assessment": {"limitations": [limitation_data()]}})
    assert response.answer == "\n".join(claim.text for claim in response.claims)
    assert response.citations[0].quote == data["citations"][0]["quote"]
    assert response.assessment.limitations[0].message not in response.answer
    with pytest.raises(ValidationError, match="rendered from cited claims"):
        validate_response(data | {
            "answer": data["answer"] + " Account configuration is unavailable.",
            "assessment": {"limitations": [limitation_data()]},
        })


@pytest.mark.parametrize("question_part", [None, "", "  ", "x" * 201, 23])
def test_model_reported_limitation_requires_bounded_question_part(question_part):
    with pytest.raises(ValidationError):
        AnswerLimitation.model_validate(limitation_data(question_part=question_part))


def test_system_policy_cannot_assert_a_question_part_assessment():
    with pytest.raises(ValidationError, match="cannot claim"):
        AnswerLimitation.model_validate(limitation_data(reported_by="SYSTEM_POLICY"))
    system = AnswerLimitation.model_validate(limitation_data(
        reported_by="SYSTEM_POLICY", question_part=None,
    ))
    assert system.question_part is None


@pytest.mark.parametrize("changes", [
    {"code": "VERIFIED_ACCOUNT_STATE"},
    {"reported_by": "HUMAN_VERIFIED"},
    {"source_id": "s1"},
    {"citation_ids": ["cite1"]},
])
def test_unrecognized_origins_codes_and_citation_fields_are_forbidden(changes):
    with pytest.raises(ValidationError):
        AnswerLimitation.model_validate(limitation_data(**changes))


@pytest.mark.parametrize("changes", [
    {"policy_origin": "MODEL_REPORTED"},
    {"source_scope": "ALL_AWS_KNOWLEDGE"},
    {"account_state_access": "VERIFIED"},
    {"origin": "SYSTEM_POLICY"},
    {"schema_version": 2},
])
def test_assessment_cannot_claim_a_different_policy_or_account_access(changes):
    with pytest.raises(ValidationError):
        AnswerAssessment.model_validate(changes)


@pytest.mark.parametrize("field", ["message", "next_step"])
@pytest.mark.parametrize("value", ["", "  ", "x" * 401, None, 42])
def test_required_metadata_strings_are_bounded(field, value):
    with pytest.raises(ValidationError):
        AnswerLimitation.model_validate(limitation_data(**{field: value}))


def test_limitation_text_bounds_are_unicode_characters_and_maximum_three_records():
    limitation = AnswerLimitation.model_validate(limitation_data(
        question_part="é" * 200, message="文" * 400, next_step="文" * 400,
    ))
    assert len(limitation.message) == 400
    assert len(limitation.message.encode()) > 400
    assert len(AnswerAssessment(limitations=(limitation,) * 3).limitations) == 3
    with pytest.raises(ValidationError):
        AnswerAssessment(limitations=(limitation,) * 4)


@pytest.mark.parametrize("code", [
    "MISSING_SUPPLIED_EVIDENCE", "ACCOUNT_STATE_UNAVAILABLE", "SOURCE_SCOPE_AMBIGUITY",
    "CONFLICTING_SUPPLIED_EVIDENCE", "OUT_OF_SCOPE",
])
def test_all_frozen_reason_codes_are_supported(code):
    assert AnswerLimitation.model_validate(limitation_data(code=code)).code == code


def test_checked_in_schema_is_generated_and_preserves_optional_legacy_contract():
    path = Path(__file__).resolve().parents[1] / "schemas/v2/QueryResponse.schema.json"
    schema = json.loads(path.read_text(encoding="utf-8"))
    assert schema == QueryResponse.model_json_schema()
    assert "assessment" not in schema["required"]
    assert schema["required"] == [
        "request_id", "answer", "claims", "citations", "answer_status", "mode_requested",
        "mode_used", "used_graph", "snapshot_version",
    ]
    assert schema["properties"]["assessment"]["default"] is None
    assert schema["$defs"]["AnswerAssessment"]["properties"]["limitations"]["maxItems"] == 3
    assert schema["properties"]["mode_requested"]["enum"] == [
        "bm25", "vector", "hybrid", "hybrid_reranked", "hybrid_multi_query",
        "hybrid_multi_query_balanced", "graph_1hop", "graph_2hop",
    ]
