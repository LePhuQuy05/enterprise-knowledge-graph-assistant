import json
from dataclasses import FrozenInstanceError
from datetime import UTC, datetime

import numpy as np
import pytest

from app.answer_cache import AnswerCache
from app.collection_answering import prepare_collection_answer
from app.collection_lexical import build_global_lexical_index, load_global_lexical_index
from app.collections import build_collection, load_collection
from app.contracts import Chunk, DocumentVersion, SnapshotManifest, stable_id, text_hash
from app.fixtures import FIXTURE_EMBEDDING_MODEL, FIXTURE_TENANT, fixture_recipes, fixture_vector
from app.generation import (
    GenerationError,
    GenerationRecipeV9,
    authorized_spans,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.reranking import ChunkScore, RerankerRecipe, RerankResult, WindowScore
from app.snapshot import SnapshotContent, build_snapshot


class FixtureReranker:
    fixture_only = True

    def __init__(self):
        self.recipe = RerankerRecipe(artifact_manifest_sha256="a" * 64,
                                    onnxruntime_version="fixture", tokenizers_version="fixture")
        self.inference_calls = 0
        self.seen = []
        self.fail = False

    def rerank(self, query, chunks, tenant_id):
        assert all(chunk.tenant_id == tenant_id for chunk in chunks)
        self.seen.append((query, tuple(chunk.chunk_id for chunk in chunks)))
        if self.fail:
            raise RuntimeError("SYNTHETIC_RERANKER_FAILURE")
        if chunks:
            self.inference_calls += 1
        scores = tuple(ChunkScore(chunk.chunk_id, rank, float(rank), 1) for rank, chunk in reversed(list(enumerate(chunks, 1))))
        windows = tuple(WindowScore(chunk.chunk_id, rank, 1, 0, len(chunk.text), 4, float(rank))
                        for rank, chunk in enumerate(chunks, 1))
        return RerankResult(tuple(score.chunk_id for score in scores), scores, windows, self.recipe.fingerprint)


def sources(root, *, documents=3, label="fixture"):
    extraction, embedding = fixture_recipes()
    shards = []
    for start in range(0, documents, 9):
        docs, chunks = [], []
        for index in range(start, min(start + 9, documents)):
            prefix = "Tiếng Việt 🚀\n"
            body = f"Service {index} uses another service only when option {index} is enabled."
            text = prefix + body
            doc = DocumentVersion.create(
                tenant_id=FIXTURE_TENANT, source_uri=f"https://example.invalid/{label}/{index}",
                title=f"Synthetic service {index}", fetched_at=datetime(2026, 10, 24, tzinfo=UTC),
                raw_sha256=text_hash(text), normalizer_version="normalize-v1",
                normalized_text_path=f"sources/{index}.txt", normalized_text=text, source_kind="TEXT",
            )
            docs.append(doc)
            for position, (begin, own_text) in enumerate(((0, prefix.rstrip()), (len(prefix), body))):
                chunks.append(Chunk.create(
                    tenant_id=FIXTURE_TENANT, document_version_id=doc.document_version_id,
                    chunking_version="chunk-v1", section_path=("Synthetic conditions",), position=position,
                    start_char=begin, end_char=begin + len(own_text), text=own_text,
                    token_count=len(own_text.split()), tokenizer_version="fixture-words-v1",
                ))
        shards.append(build_snapshot(
            root, SnapshotContent(tuple(docs), tuple(chunks)),
            {chunk.chunk_id: fixture_vector(chunk.text) for chunk in chunks},
            tenant_id=FIXTURE_TENANT, purpose="FIXTURE", extraction_recipe=extraction,
            embedding_recipe=embedding, corpus_version=f"{label}-{start}",
            normalizer_version="normalize-v1", chunking_version="chunk-v1",
            tokenizer_version="fixture-words-v1",
        ))
    path = build_collection(root, shards, tenant_id=FIXTURE_TENANT,
                            embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    collection = load_collection(path, tenant_id=FIXTURE_TENANT,
                                 embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True,
                                 cache_shards=1)
    lexical = build_global_lexical_index(collection, root / "lexical")
    return collection, load_global_lexical_index(lexical, collection)


@pytest.fixture
def collection(tmp_path):
    # Ten documents force at least two source shards.
    collection, index = sources(tmp_path / "sources", documents=10)
    try:
        yield collection, index
    finally:
        index.close()


def prepare(collection, index, **overrides):
    kwargs = dict(tenant_id=FIXTURE_TENANT, owner_id="reviewer-a", run_id="collection-answer",
                  recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64), mode="hybrid") | overrides
    return prepare_collection_answer(collection, index, "Which synthetic service uses another service?",
                                      fixture_vector("service uses option"), **kwargs)


def test_cross_shard_view_preserves_original_offsets_and_trusted_citation_identity(collection):
    current, index = collection
    view, prepared = prepare(current, index, mode="bm25", search_query="uses option 9")
    assert len({view.source_refs[key].shard_id for key in view.chunks}) == 2
    assert not isinstance(view.manifest, SnapshotManifest)
    assert view.manifest.snapshot_version == current.manifest.collection_version
    assert view.manifest_sha256 == current.manifest_sha256
    assert not hasattr(view, "vectors") and view.content == SnapshotContent((), ())
    assert not view.entities
    with pytest.raises(TypeError):
        view.chunks["forged"] = next(iter(view.chunks.values()))
    with pytest.raises(FrozenInstanceError):
        view.manifest_sha256 = "0" * 64
    _, lookup = authorized_spans(view, prepared.packed, FIXTURE_TENANT, include_titles=True)
    claims, expected = [], []
    for source_id, chunk_id in zip(prepared.packed.source_ids, prepared.packed.chunk_ids, strict=True):
        chunk = view.chunks[chunk_id]
        if chunk.start_char == 0:
            continue
        span_id = next(span for source, span in lookup if source == source_id)
        start, end, quote = lookup[(source_id, span_id)]
        claims.append({"text": chunk.text,
                       "evidence": [{"source_id": source_id, "span_id": span_id}]})
        expected.append((chunk, start, end, quote))
        if len(claims) == 2:
            break
    assert len(claims) == 2
    assert len({view.source_refs[chunk.chunk_id].shard_id for chunk, *_ in expected}) == 2
    parsed = parse_generated(json.dumps({"answer_status": "ANSWERED", "reason": "NONE", "claims": claims}),
                             "end_turn", prepared.recipe)
    response = resolve_answer(parsed, snapshot=view, packed=prepared.packed,
                               binding=prepared.binding, recipe=prepared.recipe)
    for citation, (chunk, start, end, quote) in zip(response.citations, expected, strict=True):
        document = view.documents[chunk.document_version_id]
        assert citation.quote == document.normalized_text[chunk.start_char + start:chunk.start_char + end]
        assert citation.quote == quote and citation.source_uri == document.source_uri
        assert citation.citation_id == stable_id(
            "cite", tenant=FIXTURE_TENANT, owner="reviewer-a", request=prepared.binding.request_id,
            snapshot=current.manifest.collection_version, chunk=chunk.chunk_id,
            start=chunk.start_char + start, end=chunk.start_char + end,
        )


def test_original_question_is_used_for_generation_and_search_query_for_retrieval(collection):
    current, index = collection
    original, search = "Dịch vụ giả lập nào cần bật tùy chọn?", "uses option"
    view, prepared = prepare_collection_answer(
        current, index, original, None, tenant_id=FIXTURE_TENANT, owner_id="reviewer-a",
        run_id="collection-answer", recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64),
        mode="bm25", search_query=search,
    )
    assert prepared.question == original and prepared.retrieved_chunk_ids
    _, prompt = generation_request(original, view, prepared.packed, FIXTURE_TENANT,
                                    prepared.recipe, answer_language="vi")
    assert json.loads(prompt)["untrusted_question"] == original


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_all_modes_preserve_scope_and_caps(collection, mode):
    current, index = collection
    view, prepared = prepare(current, index, mode=mode)
    assert len(view.chunks) <= 40 and len(view.documents) <= 40
    assert len(prepared.packed.chunk_ids) <= 8 and prepared.packed.proxy_tokens <= 4096
    assert prepared.binding.mode_used == prepared.binding.mode_requested == mode
    assert prepared.binding.used_graph is False and prepared.packed.assertion_ids == ()
    assert set(prepared.packed.chunk_ids).issubset(view.chunks)


def test_request_identity_binds_owner_method_query_recipe_and_vector(collection):
    current, index = collection
    _, first = prepare(current, index)
    _, again = prepare(current, index)
    variants = [
        prepare(current, index, owner_id="reviewer-b")[1],
        prepare(current, index, mode="vector")[1],
        prepare(current, index, search_query="different exact task")[1],
        prepare(current, index, recipe=GenerationRecipeV9(model_metadata_sha256="1" * 64))[1],
    ]
    assert first.context.request_id == again.context.request_id
    assert len({item.context.request_id for item in [first, *variants]}) == 5
    _, explicit = prepare(current, index, request_id="owned-explicit-id")
    assert explicit.context.request_id == explicit.binding.request_id == "owned-explicit-id"


@pytest.mark.parametrize("mode", ["bm25", "vector", "hybrid"])
def test_foreign_index_rejected_for_every_mode_before_vector_search(collection, tmp_path, monkeypatch, mode):
    current, _ = collection
    _, other = sources(tmp_path / "other", label="other")

    def forbidden(*args, **kwargs):
        pytest.fail("Foreign index must be rejected before retrieval")

    monkeypatch.setattr(current, "search_vector", forbidden)
    try:
        with pytest.raises(GenerationError, match="LEXICAL_BINDING_MISMATCH"):
            prepare(current, other, mode=mode)
    finally:
        other.close()


@pytest.mark.parametrize("mode", ["graph_1hop", "graph_2hop"])
def test_graph_answers_remain_disabled(collection, mode):
    with pytest.raises(GenerationError, match="GRAPH_NOT_QUALITY_APPROVED"):
        prepare(*collection, mode=mode)


def test_foreign_tenant_rejected(collection):
    with pytest.raises(GenerationError, match="TENANT_MISMATCH"):
        prepare(*collection, tenant_id="other")


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_vector_dimensions_are_checked_and_bm25_needs_no_vector(collection, mode):
    current, index = collection
    kwargs = dict(tenant_id=FIXTURE_TENANT, owner_id="reviewer-a", run_id="collection-answer",
                  recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64))
    with pytest.raises(ValueError, match="Query vector"):
        prepare_collection_answer(current, index, "uses option", np.ones(384, dtype=np.float32),
                                   mode=mode, **kwargs)
    _, prepared = prepare_collection_answer(current, index, "uses option", None, mode="bm25", **kwargs)
    assert prepared.packed.chunk_ids


def test_no_lexical_evidence_abstains_without_generator_or_receipt(collection, tmp_path):
    current, index = collection
    view, prepared = prepare(current, index, mode="bm25", search_query="neverindexedtoken")
    result = AnswerCache(tmp_path, namespace="FIXTURE").resolve(
        prepared.context, question=prepared.question, snapshot=view, packed=prepared.packed,
        binding=prepared.binding, recipe=prepared.recipe, answer_language="vi",
    )
    assert result.deterministic_no_evidence and result.receipt is None
    assert result.response.answer_status == "ABSTAINED" and not result.response.citations
    assert result.new_cost_micro_usd == 0 and result.response.answer_language == "vi"


def test_oversized_candidate_union_is_rejected_before_materializing_sources(tmp_path, monkeypatch):
    current, index = sources(tmp_path / "large", documents=25)
    try:
        all_ids = [key for ref in current.manifest.shards for key in current.get_shard(ref.shard_id).chunk_ids]
        monkeypatch.setattr("app.collection_answering.retrieve_collection_hybrid", lambda *a, **k: all_ids[:41])
        with pytest.raises(GenerationError, match="CANDIDATE_LIMIT"):
            prepare(current, index)
    finally:
        index.close()


def test_reranked_adapter_uses_exact_hybrid_pool_original_sources_and_common_caps(collection):
    current, index = collection
    before_view, before = prepare(current, index)
    reranker, traces = FixtureReranker(), []
    view, after = prepare(current, index, mode="hybrid_reranked", reranker=reranker, rerank_trace=traces)
    assert reranker.seen[0][1] == traces[0].baseline_candidate_ids == before.retrieved_chunk_ids
    assert after.retrieved_chunk_ids == tuple(reversed(before.retrieved_chunk_ids))
    assert set(view.chunks) == set(before_view.chunks)
    assert all(view.chunks[key] == before_view.chunks[key] for key in view.chunks)
    assert view.source_refs == before_view.source_refs and view.documents == before_view.documents
    assert after.binding.mode_requested == after.binding.mode_used == "hybrid_reranked"
    assert not after.binding.used_graph and len(after.packed.chunk_ids) <= 8 and after.packed.proxy_tokens <= 4096
    assert traces[0].recipe_sha256 == reranker.recipe.fingerprint
    restored = type(traces[0]).model_validate_json(traces[0].model_dump_json())
    assert restored == traces[0]


def test_reranker_recipe_binds_only_r_identity_and_missing_r_is_rejected(collection):
    current, index = collection
    scorer = FixtureReranker()
    _, baseline = prepare(current, index)
    _, unchanged = prepare(current, index, reranker=scorer)
    assert baseline.context.request_id == unchanged.context.request_id and not scorer.seen
    with pytest.raises(GenerationError, match="VERIFIED_RERANKER_REQUIRED"):
        prepare(current, index, mode="hybrid_reranked")
    _, first = prepare(current, index, mode="hybrid_reranked", reranker=scorer)
    scorer.recipe = scorer.recipe.model_copy(update={"artifact_manifest_sha256": "b" * 64})
    _, second = prepare(current, index, mode="hybrid_reranked", reranker=scorer)
    assert first.context.request_id != second.context.request_id != baseline.context.request_id


@pytest.mark.parametrize("kind", ["unknown_source", "window_gap"])
def test_reranker_cannot_invent_a_source_or_publish_incomplete_window_coverage(collection, kind):
    current, index = collection
    scorer = FixtureReranker()
    original = scorer.rerank

    def forged(query, chunks, tenant_id):
        result = original(query, chunks, tenant_id)
        bad = result.windows[0]
        from dataclasses import replace

        if kind == "unknown_source":
            return replace(result, ranked_chunk_ids=("chk_invented", *result.ranked_chunk_ids[1:]))
        return replace(result, windows=(replace(bad, end_char=bad.end_char - 1), *result.windows[1:]))

    scorer.rerank = forged
    with pytest.raises(ValueError, match="CANDIDATE_MISMATCH" if kind == "unknown_source" else "WINDOW_SOURCE_MISMATCH"):
        prepare(current, index, mode="hybrid_reranked", reranker=scorer)


def test_reranked_foreign_tenant_rejected_before_scorer(collection):
    scorer = FixtureReranker()
    with pytest.raises(GenerationError, match="TENANT_MISMATCH"):
        prepare(*collection, mode="hybrid_reranked", reranker=scorer, tenant_id="other")
    assert not scorer.seen and scorer.inference_calls == 0


def test_oversized_reranked_union_is_rejected_before_scorer(tmp_path, monkeypatch):
    current, index = sources(tmp_path / "large-reranked", documents=25)
    scorer = FixtureReranker()
    try:
        all_ids = [key for ref in current.manifest.shards for key in current.get_shard(ref.shard_id).chunk_ids]
        monkeypatch.setattr("app.collection_answering.retrieve_collection_hybrid", lambda *a, **k: all_ids[:41])
        with pytest.raises(GenerationError, match="CANDIDATE_LIMIT"):
            prepare(current, index, mode="hybrid_reranked", reranker=scorer)
        assert scorer.inference_calls == 0 and not scorer.seen
    finally:
        index.close()
