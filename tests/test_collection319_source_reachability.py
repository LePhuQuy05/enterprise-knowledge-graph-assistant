"""Diagnostic stage distinctions; these fixtures do not define gold sources."""

from types import SimpleNamespace as NS

import pytest

from scripts.audit_collection319_methods import source_reachability


@pytest.fixture
def reachability_case():
    dependency = NS(dependency_id="synthetic-dependency", document_version_id="doc-a",
        anchor=NS(start_char=10, end_char=20), required_spans=(NS(start_char=30, end_char=40),))
    chunks = {"other": NS(document_version_id="other-doc", start_char=0, end_char=100),
        "unrelated": NS(document_version_id="doc-a", start_char=0, end_char=5),
        "anchor": NS(document_version_id="doc-a", start_char=10, end_char=20),
        "qualifier": NS(document_version_id="doc-a", start_char=30, end_char=40),
        "all": NS(document_version_id="doc-a", start_char=0, end_char=50)}
    for key, chunk in chunks.items():
        chunk.chunk_id = key
    return dependency, chunks


@pytest.mark.parametrize("pool,selected,expected", [
    (["other"], ["other"], "DOCUMENT_ABSENT_FROM_POOL"),
    (["other", "unrelated"], ["unrelated"], "ANCHOR_OUTSIDE_POOL"),
    (["anchor", "other"], ["anchor"], "REQUIRED_SPANS_OUTSIDE_POOL"),
    (["anchor", "qualifier"], ["anchor"], "ELIGIBLE_BUNDLE_NOT_SELECTED"),
    (["anchor", "qualifier"], ["anchor", "qualifier"], "COMPLETE_REVIEWED_BUNDLE_SELECTED"),
    (["all"], ["all"], "COMPLETE_REVIEWED_BUNDLE_SELECTED"),
])
def test_separate_retrieval_anchor_qualifier_and_selection_stages(reachability_case, pool, selected, expected):
    dependency, chunks = reachability_case
    result = source_reachability(dependency, pool, selected, chunks)
    assert result["stage"] == expected
    assert result["not_a_required_gold_source_or_semantic_quality_label"] is True
