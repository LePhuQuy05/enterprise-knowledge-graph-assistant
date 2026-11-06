from copy import deepcopy

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction_v10 import RelationV10, SourceActionV1, semantic_recipe
from app.source_action_graph import SourceActionGraph
from app.source_syntax import validate_syntax_record
from app.syntax_graph_records import canonicalize
from tests.test_extraction import aliases, source
from tests.test_source_syntax import FIXTURE, parse


def fixture(kind="RELATION"):
    case = next(c for c in FIXTURE["real_cases"] if c["kind"] == kind)
    record = (RelationV10 if kind == "RELATION" else SourceActionV1).model_validate_json(canonical_json(case["record"]))
    parsed = parse(record.clause_quote)
    candidate = validate_syntax_record(record, case["source_text"], parsed, kind=kind)
    judgment = {"status": "AI_SOURCE_SUPPORTED_SOURCE_SCOPED", "candidate_sha256": sha256_bytes(canonical_json(candidate)),
        "reason_vi": "Mechanism fixture only, not independent human annotation."}
    doc, chunk = source(case["source_text"])
    recipe = semantic_recipe(aliases(), metadata_sha256="0" * 64)
    return candidate, judgment, doc, chunk, recipe, parsed


@pytest.mark.parametrize("kind", ["ACTION", "RELATION"])
def test_reviewed_composite_materialization_keeps_exact_mentions_source_and_kind(kind):
    c, j, doc, chunk, recipe, parsed = fixture(kind)
    entities, mentions, assertion, metadata = canonicalize(c, j, chunk=chunk, document=doc, parsed=parsed,
        native_recipe=recipe, installation=FIXTURE["installation"])
    assert len(entities) == len(mentions) == 2
    assert all(doc.normalized_text[m.start_char:m.end_char] == m.quote for m in mentions)
    assert metadata["syntax_witness"] == c and not metadata["native_v10_admission_claimed"]
    if kind == "ACTION":
        trace = SourceActionGraph([assertion]).evidence_paths(entities[0].entity_id)[0]
        assert trace["assertion"]["evidence"]["quote"] == chunk.text
        assert trace["assertion"]["nodes"][1]["source_witness"]["source_annotations"][0]["quote"] == "that you specify"
    else:
        assert assertion.evidence.quote == chunk.text
        assert metadata["record"]["object"]["name"].startswith("some or all")


@pytest.mark.parametrize("change", ["review", "hash", "source", "identity", "parse", "context", "parser"])
def test_forged_review_rehashed_candidate_and_changed_source_cannot_materialize(change):
    c, j, doc, chunk, recipe, parsed = fixture()
    c, j, installation = deepcopy(c), deepcopy(j), deepcopy(FIXTURE["installation"])
    if change == "review":
        j["status"] = "PENDING"
    elif change == "hash":
        j["candidate_sha256"] = "0" * 64
    elif change == "source":
        doc, _ = source(chunk.text, suffix="other")
    elif change == "identity":
        c["target"]["identity"]["canonical_key"] = "arbitrary"
    elif change == "parse":
        c["parse_sha256"] = "0" * 64
    elif change == "context":
        c["evidence"]["quote"] = c["record"]["clause_quote"]
    else:
        installation["model_wheel_sha256"] = "0" * 64
    if change not in {"hash", "review"}:
        j["candidate_sha256"] = sha256_bytes(canonical_json(c))
    with pytest.raises(ValueError):
        canonicalize(c, j, chunk=chunk, document=doc, parsed=parsed, native_recipe=recipe, installation=installation)
