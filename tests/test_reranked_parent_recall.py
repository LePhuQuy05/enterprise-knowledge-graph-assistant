"""Reranker failure stays failure; source candidates are a bounded permutation."""

import json

import pytest

from app.generation import GenerationRecipeV9
from app.reranked_parent_recall import RerankedParentRecipe, build_reranked_parent_pool
from app.reranking import ChunkScore, RerankerRecipe, RerankingError, RerankResult, WindowScore
from app.section_recall import pack_section_pool
from tests.test_authorization_evidence import auth_case as auth_case
from tests.test_section_recall import QUESTION, case_inputs


class FixtureReranker:
    fixture_only = True

    def __init__(self, bad=False, error=False):
        self.recipe = RerankerRecipe(artifact_manifest_sha256="0"*64, onnxruntime_version="fixture", tokenizers_version="fixture")
        self.calls, self.bad, self.error = [], bad, error

    def rerank(self, query, chunks, tenant):
        self.calls.append((query, chunks, tenant))
        if self.error:
            raise RerankingError("fixture failure")
        order = tuple(c.chunk_id for c in reversed(chunks))
        if self.bad:
            order = order[:-1]
        return RerankResult(order, tuple(ChunkScore(c.chunk_id, i+1, float(i), 1) for i, c in enumerate(chunks)),
            tuple(WindowScore(c.chunk_id, i+1, 1, 0, len(c.text), 40, float(i)) for i, c in enumerate(chunks)), self.recipe.fingerprint)


def build(case, reranker):
    index, p, vectors, base = case_inputs(case)
    return build_reranked_parent_pool(case.collection, case.lexical, index, question=QUESTION,
        search=QUESTION, language="en", plan=p, vectors=vectors, base_ids=base,
        recipe=RerankedParentRecipe(), reranker=reranker)


def test_reranker_gets_full_original_search_and_max40_original_anchors(auth_case):
    reranker = FixtureReranker()
    pool = build(auth_case, reranker)
    assert len(reranker.calls) == 1
    query, chunks, tenant = reranker.calls[0]
    assert query == QUESTION and tenant == "demo" and len(chunks) <= 40
    trace = json.loads(pool.trace_json)
    assert not trace["reranking"]["fallback_executed"]
    assert set(trace["reranking"]["candidate_anchor_ids"]) == set(trace["reranking"]["ranked_anchor_ids"])
    assert len(pool.pool_ids) <= 40


def test_missing_ranked_candidate_is_error_not_baseline_fallback(auth_case):
    with pytest.raises(ValueError, match="PERMUTATION"):
        build(auth_case, FixtureReranker(bad=True))


def test_reranker_failure_propagates_without_substitution(auth_case):
    reranker = FixtureReranker(error=True)
    with pytest.raises(RerankingError, match="fixture failure"):
        build(auth_case, reranker)
    assert len(reranker.calls) == 1


def test_fixture_namespace_checked_before_reranking(auth_case):
    reranker = FixtureReranker()
    reranker.fixture_only = False
    with pytest.raises(ValueError, match="NAMESPACE"):
        build(auth_case, reranker)
    assert not reranker.calls


def test_reranked_parent_recipe_can_use_shared_atomic_packer(auth_case):
    pool = build(auth_case, FixtureReranker())
    for balanced in (False, True):
        answer, audit = pack_section_pool(auth_case.collection, auth_case.lexical, pool, balanced=balanced,
            generation_recipe=GenerationRecipeV9(model_metadata_sha256="0"*64), run_id="fixture", owner_id="fixture")
        assert len(answer.packed.chunk_ids) <= 8 and answer.packed.proxy_tokens <= 4096
        assert not answer.binding.used_graph and audit["pool_fingerprint"] == pool.fingerprint
