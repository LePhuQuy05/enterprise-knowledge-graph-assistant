"""A same-document ancestor must not drop the original retrieved child."""

import json
from types import SimpleNamespace as NS

from app.bounded_ancestor_recall import (
    BoundedAncestorRecipe,
    bounded_anchor,
    build_bounded_ancestor_pool,
)
from app.generation import GenerationRecipeV9
from app.section_recall import pack_section_pool
from tests.test_authorization_evidence import auth_case as auth_case
from tests.test_section_recall import QUESTION, case_inputs


def chunks():
    return {str(i): NS(chunk_id=str(i), document_version_id="doc", start_char=i*10, end_char=i*10+8,
        section_path=("Title", "Parent") if i == 0 else ("Title", "Parent", "Child")) for i in range(12)}


def test_distant_parent_cannot_replace_child_anchor():
    own = chunks()
    assert bounded_anchor(own, "10", 2) == ("10", "PARENT_TOO_FAR_RETAIN_CHILD_CENTER")


def test_near_parent_is_retained_with_child_inside_window():
    own = chunks()
    assert bounded_anchor(own, "2", 2) == ("0", "PARENT_WITHIN_WINDOW")
    assert bounded_anchor(own, "2", 1) == ("2", "PARENT_TOO_FAR_RETAIN_CHILD_CENTER")


def test_invalid_radius_is_rejected():
    import pytest

    with pytest.raises(ValueError):
        bounded_anchor(chunks(), "2", 3)


def test_bounded_candidate_preserves_all_children_and_source_budget(auth_case):
    index, p, vectors, base = case_inputs(auth_case)
    pool = build_bounded_ancestor_pool(auth_case.collection, auth_case.lexical, index, question=QUESTION,
        search=QUESTION, language="en", plan=p, vectors=vectors, base_ids=base, recipe=BoundedAncestorRecipe(radius=2))
    trace = json.loads(pool.trace_json)
    assert trace["every_proposal_retains_retrieved_child"]
    assert all(p["retrieved_child_anchor_id"] in p["chunk_ids"] for p in trace["proposals"])
    for balanced in (False, True):
        answer, _ = pack_section_pool(auth_case.collection, auth_case.lexical, pool, balanced=balanced,
            generation_recipe=GenerationRecipeV9(model_metadata_sha256="0"*64), run_id="fixture", owner_id="fixture")
        assert len(answer.packed.chunk_ids) <= 8 and answer.packed.proxy_tokens <= 4096
        assert not answer.binding.used_graph
