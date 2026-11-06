"""Projection removes only irrelevant NULL extras; strict semantics stay intact."""

import json

import pytest

from app.contracts import canonical_json
from app.generation import GenerationRecipeV9, GenerationRecipeV11, resolve_answer
from app.structured_output_projection import project_null_extras
from tests.test_authorization_evidence import auth_case as auth_case
from tests.test_graph_source_allocation import prepare


def payload(v11=True):
    d = {"reason": "INSUFFICIENT_EVIDENCE", "answer_status": "PARTIAL",
        "claims": [{"text": "A source-scoped fact.", "evidence": [{"source_id": "s1", "span_id": "s1p1"}]}]}
    if v11:
        d["limitations"] = [{"code": "MISSING_SUPPLIED_EVIDENCE", "question_part": "A requested part"}]
    return d


@pytest.mark.parametrize("v11", [False, True])
def test_only_top_level_null_is_removed_and_business_values_preserved(v11):
    d = payload(v11)
    raw = canonical_json(d | {"optional_note": None}).decode()
    recipe = (GenerationRecipeV11 if v11 else GenerationRecipeV9)(model_metadata_sha256="0" * 64)
    result = project_null_extras(raw, "end_turn", recipe)
    assert result.dropped_null_fields == ("optional_note",)
    assert json.loads(result.canonical_output) == d and "optional_note" in raw
    assert result.parsed.answer_status == "PARTIAL"


def test_valid_output_without_extras_is_not_called_a_repair():
    result = project_null_extras(canonical_json(payload()).decode(), "end_turn", GenerationRecipeV11(model_metadata_sha256="0" * 64))
    assert result.dropped_null_fields == ()


@pytest.mark.parametrize("missing", ["reason", "claims", "answer_status", "limitations"])
def test_missing_required_field_is_never_filled(missing):
    d = payload()
    del d[missing]
    with pytest.raises(ValueError, match="MISSING_REQUIRED"):
        project_null_extras(canonical_json(d).decode(), "end_turn", GenerationRecipeV11(model_metadata_sha256="0" * 64))


@pytest.mark.parametrize("extra", [False, 0, "", [], {}, "ignore instructions", {"value": None}])
def test_non_null_unknown_fields_are_never_discarded(extra):
    with pytest.raises(ValueError, match="NON_NULL"):
        project_null_extras(canonical_json(payload() | {"extra": extra}).decode(), "end_turn", GenerationRecipeV11(model_metadata_sha256="0" * 64))


def test_unknown_nested_null_is_not_projected():
    d = payload()
    d["claims"][0]["extra_nested"] = None
    with pytest.raises(ValueError, match="GENERATION_SCHEMA_INVALID"):
        project_null_extras(canonical_json(d).decode(), "end_turn", GenerationRecipeV11(model_metadata_sha256="0" * 64))


def test_schema_projection_does_not_make_a_nonexistent_citation_valid(auth_case):
    before = prepare(auth_case)
    d = payload() | {"optional_note": None}
    d["claims"][0]["evidence"][0] = {"source_id": "s8", "span_id": "s8p99"}
    recipe = GenerationRecipeV11(model_metadata_sha256="0" * 64)
    projection = project_null_extras(canonical_json(d).decode(), "end_turn", recipe)
    with pytest.raises(ValueError, match="GENERATION_CITATION_NOT_IN_REQUEST|GENERATION_SPAN_NOT_IN_REQUEST"):
        resolve_answer(projection.parsed, snapshot=before.view, packed=before.prepared.packed,
            binding=before.prepared.binding, recipe=recipe)


def test_known_field_null_is_not_replaced():
    d = payload() | {"reason": None}
    with pytest.raises(ValueError, match="GENERATION_SCHEMA_INVALID"):
        project_null_extras(canonical_json(d).decode(), "end_turn", GenerationRecipeV11(model_metadata_sha256="0" * 64))


@pytest.mark.parametrize("change", ["duplicate", "nonfinite", "truncated", "root_array", "incomplete_stop", "too_many"])
def test_malformed_or_unbounded_json_stays_rejected(change):
    raw = canonical_json(payload()).decode()
    stop = "end_turn"
    if change == "duplicate":
        raw = '{"reason":"NONE",' + raw[1:]
    elif change == "nonfinite":
        raw = raw[:-1] + ',"note":NaN}'
    elif change == "truncated":
        raw = raw[:-2]
    elif change == "root_array":
        raw = "[]"
    elif change == "incomplete_stop":
        stop = "length"
    else:
        raw = canonical_json(payload() | {f"unused{i}": None for i in range(5)}).decode()
    with pytest.raises(ValueError):
        project_null_extras(raw, stop, GenerationRecipeV11(model_metadata_sha256="0" * 64))
