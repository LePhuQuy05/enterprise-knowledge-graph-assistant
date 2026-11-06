"""Exact small set-cover controls, not tests of AWS semantic labels."""

import pytest

from scripts.diagnose_question_budget import minimum_contexts, nondominated


def test_sharing_a_chunk_across_requirements_counts_it_once():
    plans, steps = minimum_contexts(("R1", "R2"), {"R1": ({"a", "b"}, {"c"}), "R2": ({"c", "d"},)})
    assert min(map(len, plans)) == 2
    assert frozenset({"c", "d"}) in plans
    assert steps[-1]["minimum_chunks_so_far"] == 2


def test_and_requirements_cannot_be_satisfied_by_partial_alternative_witnesses():
    plans, _ = minimum_contexts(("R1",), {"R1": ({"a", "b"}, {"c", "d"})})
    assert frozenset({"a", "c"}) not in plans
    assert min(map(len, plans)) == 2


def test_alternative_shorter_review_changes_oracle_without_improving_retrieval():
    before, _ = minimum_contexts(("R1", "R2"), {"R1": ({"a", "b"},), "R2": ({"c"},)})
    after, _ = minimum_contexts(("R1", "R2"), {"R1": ({"a", "b"}, {"d"}), "R2": ({"c"},)})
    assert min(map(len, before)) == 3 and min(map(len, after)) == 2


def test_absent_reviewed_witness_is_unknown_feasibility_not_universal_impossibility():
    plans, steps = minimum_contexts(("R1",), {"R1": ()})
    assert plans == [] and steps[-1]["minimum_chunks_so_far"] is None


def test_supersets_are_removed_and_search_overflow_fails_without_a_minimum_claim():
    assert nondominated(({"a"}, {"a", "b"})) == [frozenset({"a"})]
    with pytest.raises(ValueError, match="NO_CONCLUSION"):
        nondominated(({"a"}, {"b"}), limit=1)
