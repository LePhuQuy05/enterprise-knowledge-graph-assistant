import pytest

from app.fixtures import fixture_content
from app.graph_store import GraphStoreError, graph_edges, graph_records, merge_canonical_rows


def test_stable_export_order_and_negative_assertions_are_preserved():
    content = fixture_content()
    rows = graph_records(content)
    assert [r["record_id"] for r in rows["assertions"]] == sorted(a.assertion_id for a in content.assertions)
    negative = next(a for a in content.assertions if a.polarity == "NEGATIVE")
    assert (negative.subject_id, negative.assertion_id) in graph_edges(content)["SUBJECT_OF"]


def test_alias_union_precedes_immutable_write_and_duplicate_assertions_deduplicate():
    content = fixture_content()
    original = content.entities[0]
    extra = original.model_copy(update={"aliases": (*original.aliases, "Synthetic alternate name")})
    row = {"entities": [e.model_dump(mode="json") for e in content.entities],
           "mentions": [m.model_dump(mode="json") for m in content.mentions],
           "assertions": [a.model_dump(mode="json") for a in content.assertions]}
    second = {"entities": [extra.model_dump(mode="json")], "mentions": [], "assertions": row["assertions"]}
    merged = merge_canonical_rows(content, [row, second])
    assert len(merged.entities) == len(content.entities) and len(merged.assertions) == len(content.assertions)
    assert "Synthetic alternate name" in next(e for e in merged.entities if e.entity_id == original.entity_id).aliases


def test_alias_merge_cannot_overwrite_same_id_with_different_identity_properties():
    content = fixture_content()
    original = content.entities[0]
    changed = original.model_copy(update={"canonical_name": "Different identity"})
    rows = [{"entities": [e.model_dump(mode="json")], "mentions": [], "assertions": []}
            for e in (original, changed)]
    with pytest.raises(GraphStoreError, match="IDENTITY_CONFLICT"):
        merge_canonical_rows(content, rows)
