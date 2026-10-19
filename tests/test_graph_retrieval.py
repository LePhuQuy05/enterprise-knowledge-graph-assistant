import dataclasses

import pytest

from app.fixtures import fixture_vector
from app.graph_retrieval import GraphIndex, GraphLimits
from app.retrieval import retrieve_baseline


def test_zero_hop_is_exactly_hybrid_and_does_not_visit_graph(loaded):
    query = "Service A"
    result = GraphIndex(loaded).retrieve(query, fixture_vector(query), hops=0)
    assert result.ranked_chunk_ids == tuple(retrieve_baseline(loaded, query, fixture_vector(query)))
    assert not result.used_graph and not result.graph_candidates and result.assertion_visits == 0


def test_two_hop_paths_keep_all_support_and_exclude_negative_assertions(loaded):
    index = GraphIndex(loaded)
    start = next(e.entity_id for e in loaded.content.entities if e.canonical_name == "Service A")
    index.seeds = lambda *args: (start,)
    result = index.retrieve("Service A", fixture_vector("Service A"), hops=2)
    assert result.used_graph and result.mode_used == "graph_2hop"
    two_hop = [c for c in result.graph_candidates if len(c.assertion_path) == 2]
    assert two_hop
    negative = {a.assertion_id for a in loaded.content.assertions if a.polarity == "NEGATIVE"}
    for candidate in result.graph_candidates:
        assert not negative.intersection(candidate.assertion_path)
        assert len(candidate.entity_path) == len(candidate.assertion_path) + 1
        assert candidate.supporting_chunk_ids == tuple(dict.fromkeys(
            index.assertions[key].chunk_id for key in candidate.assertion_path))
        assert len(set(candidate.entity_path)) == len(candidate.entity_path)


def test_seeds_and_mention_only_entities_come_from_retrieved_chunks(loaded):
    index = GraphIndex(loaded)
    mention_only = next(e.entity_id for e in loaded.content.entities if e.canonical_name == "Service Z")
    assert index.chunks_by_entity[mention_only] and not index.adjacency[mention_only]
    query = "Service Z"
    baseline = retrieve_baseline(loaded, query, fixture_vector(query))
    limits = GraphLimits(seed_chunks=1)
    seeds = index.seeds(query, baseline, limits)
    assert set(seeds).issubset(index.mentions_by_chunk[baseline[0]])


def test_visit_and_candidate_caps_are_enforced_with_explicit_trace(loaded):
    index = GraphIndex(loaded)
    start = next(e.entity_id for e in loaded.content.entities if e.canonical_name == "Service B")
    index.seeds = lambda *args: (start,)
    result = index.retrieve("Service B", fixture_vector("Service B"), hops=2,
        limits=GraphLimits(max_assertion_visits=1, max_graph_candidate_chunks=1, max_visited_entities=2))
    assert result.assertion_visits <= 1 and result.visited_entities <= 2 and len(result.graph_candidates) <= 1
    assert "GRAPH_TRUNCATED" in result.warnings


def test_timeout_discards_partial_graph_and_keeps_identical_hybrid(loaded):
    times = iter((0, 0.301))
    result = GraphIndex(loaded).retrieve("Service A", fixture_vector("Service A"), clock=lambda: next(times))
    assert result.mode_used == "hybrid" and not result.used_graph
    assert result.ranked_chunk_ids == result.baseline_chunk_ids and result.warnings == ("GRAPH_TIMEOUT",)


def test_empty_graph_falls_back_without_hiding_integrity_errors(loaded):
    empty = dataclasses.replace(loaded, content=dataclasses.replace(loaded.content, entities=(), mentions=(), assertions=()))
    result = GraphIndex(empty).retrieve("Service A", fixture_vector("Service A"))
    assert not result.used_graph and result.warnings == ("GRAPH_EMPTY",)


@pytest.mark.parametrize("hops", [-1, 3, True])
def test_invalid_hop_count_rejected(loaded, hops):
    with pytest.raises(ValueError):
        GraphIndex(loaded).retrieve("Service A", fixture_vector("Service A"), hops=hops)


def test_resource_bounds_cannot_be_bypassed_with_bad_types_or_higher_defaults():
    for value in (0, 81, True):
        with pytest.raises(ValueError):
            GraphLimits(max_visited_entities=value)


def test_entity_cap_applies_to_seed_entities_before_bfs(loaded):
    result = GraphIndex(loaded).retrieve("Service A", fixture_vector("Service A"),
                                         limits=GraphLimits(max_visited_entities=1))
    assert len(result.seed_entity_ids) == 1 and result.visited_entities <= 1
    assert "GRAPH_TRUNCATED" in result.warnings
