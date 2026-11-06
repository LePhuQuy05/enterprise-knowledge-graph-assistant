"""Parent collapse before topK, source-window identity and shared packer."""

import json
from types import SimpleNamespace as NS

import pytest

from app.generation import GenerationRecipeV9
from app.parent_section_recall import (
    ParentSectionRecipe,
    build_parent_section_pool,
    distinct_parents,
)
from app.section_recall import pack_section_pool
from tests.test_authorization_evidence import auth_case as auth_case
from tests.test_section_recall import QUESTION, case_inputs


def test_top20_counts_distinct_documents_before_depth_not_sections_of_one_long_document():
    rows = [{"document_version_id": "long-document"} for _ in range(50)]
    rows += [{"document_version_id": f"doc-{i}"} for i in range(30)]
    index = NS(rows=rows)
    selected = distinct_parents(index, range(len(rows)))
    assert selected == [0, *range(50, 69)]
    assert len({rows[i]["document_version_id"] for i in selected}) == 20


@pytest.mark.parametrize("mode", ["BM25", "DENSE", "RRF"])
@pytest.mark.parametrize("radius", [1, 2])
def test_parent_candidate_keeps_unique_document_proposals_and_shared_source_packer(auth_case, mode, radius):
    index, p, vectors, base = case_inputs(auth_case)
    pool = build_parent_section_pool(auth_case.collection, auth_case.lexical, index, question=QUESTION,
        search=QUESTION, language="en", plan=p, vectors=vectors, base_ids=base,
        recipe=ParentSectionRecipe(field_mode=mode, radius=radius))
    trace = json.loads(pool.trace_json)
    parents = [r["document_version_id"] for r in trace["proposals"]]
    assert len(parents) == len(set(parents))
    assert trace["policy"] == "PARENT_COLLAPSED_SECTION_FIELDS_V1"
    for rank in trace["rankings"]:
        for field in ("lexical_document_ranking", "dense_document_ranking", "ranking"):
            assert len(rank[field]) <= 20 and len(rank[field]) == len(set(rank[field]))
    fingerprints = []
    for balanced in (False, True):
        answer, audit = pack_section_pool(auth_case.collection, auth_case.lexical, pool, balanced=balanced,
            generation_recipe=GenerationRecipeV9(model_metadata_sha256="0"*64), run_id="fixture", owner_id="fixture")
        assert len(answer.packed.chunk_ids) <= 8 and answer.packed.proxy_tokens <= 4096
        assert not answer.binding.used_graph and not audit["hidden_fallbacks"]
        fingerprints.append(audit["pool_fingerprint"])
    assert fingerprints == [pool.fingerprint]*2


@pytest.mark.parametrize("change", [{"vectors": ()}, {"base_ids": ("unknown",)}])
def test_parent_candidate_rejects_invalid_vectors_and_sources(auth_case, change):
    index, p, vectors, base = case_inputs(auth_case)
    args = dict(question=QUESTION, search=QUESTION, language="en", plan=p, vectors=vectors,
        base_ids=base, recipe=ParentSectionRecipe())
    with pytest.raises(ValueError):
        build_parent_section_pool(auth_case.collection, auth_case.lexical, index, **(args | change))
