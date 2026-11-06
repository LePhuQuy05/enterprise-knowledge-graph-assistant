"""Metadata parent leads preserve document/offset boundary and source budget."""

import json
from types import SimpleNamespace as NS

from app.ancestor_section_recall import (
    AncestorSectionRecipe,
    ancestor_lead,
    build_ancestor_section_pool,
)
from app.generation import GenerationRecipeV9
from app.section_recall import pack_section_pool
from tests.test_authorization_evidence import auth_case as auth_case
from tests.test_section_recall import QUESTION, case_inputs


def chunk(key, doc, path, start):
    return NS(chunk_id=key, document_version_id=doc, section_path=path, start_char=start)


def test_selects_nearest_existing_nonroot_ancestor_before_child():
    chunks = {c.chunk_id: c for c in [chunk("root", "d", ("Title",), 0),
        chunk("lead", "d", ("Title", "Role"), 10), chunk("later-parent", "d", ("Title", "Role"), 90),
        chunk("child", "d", ("Title", "Role", "Attach"), 30),
        chunk("foreign", "other", ("Title", "Role"), 5)]}
    assert ancestor_lead(chunks, "child") == "lead"


def test_does_not_jump_to_root_or_other_document_or_later_parent():
    chunks = {c.chunk_id: c for c in [chunk("root", "d", ("Title",), 0),
        chunk("leaf", "d", ("Title", "Leaf"), 20),
        chunk("child", "d", ("Title", "Role", "Attach"), 30),
        chunk("future", "d", ("Title", "Role"), 50), chunk("foreign", "x", ("Title", "Role"), 0)]}
    assert ancestor_lead(chunks, "leaf") == "leaf"
    assert ancestor_lead(chunks, "child") == "child"


def test_ancestor_candidate_keeps_trace_and_shared_atomic_context(auth_case):
    index, p, vectors, base = case_inputs(auth_case)
    pool = build_ancestor_section_pool(auth_case.collection, auth_case.lexical, index, question=QUESTION,
        search=QUESTION, language="en", plan=p, vectors=vectors, base_ids=base, recipe=AncestorSectionRecipe(radius=2))
    trace = json.loads(pool.trace_json)
    assert trace["metadata_ancestry_is_not_semantic_dependency"]
    assert all("retrieved_child_anchor_id" in b for b in trace["proposals"])
    for balanced in (False, True):
        answer, audit = pack_section_pool(auth_case.collection, auth_case.lexical, pool, balanced=balanced,
            generation_recipe=GenerationRecipeV9(model_metadata_sha256="0"*64), run_id="fixture", owner_id="fixture")
        assert len(answer.packed.chunk_ids) <= 8 and answer.packed.proxy_tokens <= 4096
        assert not answer.binding.used_graph and audit["pool_fingerprint"] == pool.fingerprint
