import pytest

from app.collection_graph import expand_graph, reanchor_assertions


def test_source_reanchoring_requires_current_quote_and_keeps_direction_polarity_conditions(loaded):
    accepted, rejected, refs = reanchor_assertions(loaded.documents, loaded.content.assertions,
        loaded.content.documents, loaded.content.chunks)
    assert accepted and not rejected
    original = {a.assertion_id: a for a in loaded.content.assertions}
    for assertion, ref in zip(accepted, refs, strict=True):
        previous = original[ref["previous_assertion_id"]]
        assert (assertion.subject_id, assertion.object_id, assertion.predicate, assertion.polarity, assertion.qualifiers) == (
            previous.subject_id, previous.object_id, previous.predicate, previous.polarity, previous.qualifiers)
        d = loaded.documents[assertion.document_version_id]
        assert d.normalized_text[assertion.evidence.start_char:assertion.evidence.end_char] == assertion.evidence.quote


def test_missing_or_ambiguous_source_never_migrates_a_relation(loaded):
    accepted, rejected, _ = reanchor_assertions(loaded.documents, loaded.content.assertions, (), loaded.content.chunks)
    assert not accepted and len(rejected) == len(loaded.content.assertions)
    # A second current source version matching the same URI/quote is ambiguous.
    accepted, rejected, _ = reanchor_assertions(loaded.documents, loaded.content.assertions,
        loaded.content.documents * 2, loaded.content.chunks)
    assert not accepted and all(r["matches"] >= 2 for r in rejected)


def test_collection_expansion_retains_path_support_and_excludes_negative_assertions(loaded):
    scores = {key: 1.0 for key in loaded.chunk_ids}
    baseline = list(loaded.chunk_ids[:5])
    trace = expand_graph(loaded.content.entities, loaded.content.mentions, loaded.content.assertions,
        baseline, scores, hops=2)
    forbidden = {a.assertion_id for a in loaded.content.assertions if a.polarity == "NEGATIVE"}
    by_id = {a.assertion_id: a for a in loaded.content.assertions}
    assert trace["seed_entity_ids"] and trace["graph_candidates"]
    assert any(c["assertion_path"] for c in trace["graph_candidates"])
    for c in trace["graph_candidates"]:
        assert len(c["entity_path"]) == len(c["assertion_path"]) + 1
        assert not forbidden.intersection(c["assertion_path"])
        assert set(c["supporting_chunk_ids"]) == {by_id[a].chunk_id for a in c["assertion_path"]}
        assert len(c["assertion_path"]) <= 2
    assert len(trace["graph_candidates"]) <= 40 and trace["assertion_visits"] <= 200


def test_invalid_scope_or_hops_fails_before_traversal(loaded):
    with pytest.raises(ValueError, match="INPUT"):
        expand_graph(loaded.content.entities, (), (), [], {}, hops=3)
    bad = loaded.content.mentions[0].model_copy(update={"entity_id": "unknown"})
    with pytest.raises(ValueError, match="MENTION_BINDING"):
        expand_graph(loaded.content.entities, (bad,), (), [], {bad.chunk_id: 1.0})
