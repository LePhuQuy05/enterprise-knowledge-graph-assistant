"""Bounded collection evidence adapter for the existing source-only answer path.

The view is not a v2 snapshot or a new release. It retains only authorized
candidate records, keeping their original identities, complete source documents
and Unicode offsets. No vector arrays or LoadedSnapshot objects are retained.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from types import MappingProxyType

import numpy as np

from app.answer_pipeline import PreparedAnswer
from app.collection_lexical import GlobalLexicalIndex, retrieve_collection_hybrid
from app.collections import CollectionError, SnapshotCollection
from app.context import pack_context
from app.context_selection import ContextSelectionRecipe, select_context_order
from app.contracts import (
    Chunk,
    DocumentVersion,
    Entity,
    QueryRequest,
    canonical_json,
    sha256_bytes,
    stable_id,
)
from app.generation import AnswerBinding, GenerationError, GenerationRecipe
from app.graph_retrieval import GraphRetrievalResult
from app.model_gateway import CallContext
from app.multi_query import retrieve_multi_query
from app.rag_methods import RerankTraceV1
from app.reranking import OnnxCrossEncoder, RerankerRecipe
from app.snapshot import SnapshotContent

MAX_CANDIDATE_CHUNKS = 40
Q_FAMILY = {"hybrid_multi_query", "hybrid_multi_query_balanced"}


@dataclass(frozen=True)
class CollectionEvidencePolicy:
    tenant_id: str
    snapshot_version: str


@dataclass(frozen=True)
class CollectionSourceReference:
    shard_id: str
    manifest_sha256: str


@dataclass(frozen=True)
class CollectionEvidenceView:
    manifest: CollectionEvidencePolicy
    manifest_sha256: str
    documents: Mapping[str, DocumentVersion]
    chunks: Mapping[str, Chunk]
    entities: Mapping[str, Entity]
    # The common packer needs the graph records only. This is intentionally not
    # fabricated full-snapshot content or a claim of full document chunk coverage.
    content: SnapshotContent
    source_refs: Mapping[str, CollectionSourceReference]


def validate_collection_lexical_binding(collection: SnapshotCollection, index: GlobalLexicalIndex) -> None:
    expected = {
        "collection_version": collection.manifest.collection_version,
        "collection_manifest_sha256": collection.manifest_sha256,
        "corpus_version": collection.manifest.corpus_version,
        "embedding_recipe_sha256": collection.manifest.embedding_recipe.fingerprint,
        "chunks": collection.manifest.counts.chunks,
    }
    if any(getattr(index.manifest, key) != value for key, value in expected.items()):
        raise GenerationError("COLLECTION_LEXICAL_BINDING_MISMATCH")


def validate_collection_reranker(collection, reranker):
    """A synthetic scorer is accepted only for explicitly synthetic collections."""
    if (reranker is None or not isinstance(getattr(reranker, "recipe", None), RerankerRecipe)
            or not callable(getattr(reranker, "rerank", None))):
        raise GenerationError("LOCAL_RUN_VERIFIED_RERANKER_REQUIRED")
    if type(reranker) is not OnnxCrossEncoder and not (
            collection.manifest.purpose == "FIXTURE" and getattr(reranker, "fixture_only", False) is True):
        raise GenerationError("LOCAL_RUN_RERANKER_NAMESPACE_MISMATCH")
    if collection.manifest.purpose != "FIXTURE" and getattr(reranker, "fixture_only", False):
        raise GenerationError("LOCAL_RUN_RERANKER_NAMESPACE_MISMATCH")
    try:
        RerankerRecipe.model_validate_json(reranker.recipe.model_dump_json())
    except ValueError:
        raise GenerationError("LOCAL_RUN_RERANKER_RECIPE_INVALID") from None


def validate_context_selection_recipe(recipe=None):
    """Only the preregistered identity-preserving selector may answer at runtime."""
    if recipe is None:
        recipe = ContextSelectionRecipe()
    if not isinstance(recipe, ContextSelectionRecipe):
        raise GenerationError("LOCAL_RUN_CONTEXT_SELECTION_RECIPE_INVALID")
    try:
        validated = ContextSelectionRecipe.model_validate_json(recipe.model_dump_json())
    except ValueError:
        raise GenerationError("LOCAL_RUN_CONTEXT_SELECTION_RECIPE_INVALID") from None
    if validated.policy != "RR_ID_ONLY":
        raise GenerationError("LOCAL_RUN_CONTEXT_SELECTION_RUNTIME_POLICY_UNAVAILABLE")
    return validated


def _evidence_view(collection, index, ranked):
    if len(ranked) > MAX_CANDIDATE_CHUNKS or len(set(ranked)) != len(ranked):
        raise GenerationError("COLLECTION_CANDIDATE_LIMIT_OR_DUPLICATES")
    groups = defaultdict(list)
    for key in ranked:
        groups[index.shard_for_chunk(key)].append(key)
    refs = {ref.shard_id: ref for ref in collection.manifest.shards}
    chunks, documents, sources = {}, {}, {}
    for shard_id, own_ids in groups.items():
        ref = refs.get(shard_id)
        if ref is None:
            raise GenerationError("COLLECTION_SOURCE_SHARD_UNKNOWN")
        snapshot = collection.get_shard(shard_id)
        if snapshot.manifest_sha256 != ref.manifest_sha256:
            raise GenerationError("COLLECTION_SOURCE_MANIFEST_MISMATCH")
        for key in own_ids:
            chunk = snapshot.chunks.get(key)
            if chunk is None or chunk.tenant_id != collection.manifest.tenant_id:
                raise GenerationError("COLLECTION_SOURCE_CHUNK_UNKNOWN_OR_FOREIGN")
            document = snapshot.documents.get(chunk.document_version_id)
            if (document is None or document.tenant_id != collection.manifest.tenant_id
                    or document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
                raise GenerationError("COLLECTION_SOURCE_SLICE_MISMATCH")
            chunks[key], documents[document.document_version_id] = chunk, document
            sources[key] = CollectionSourceReference(shard_id, ref.manifest_sha256)
        del snapshot
    return CollectionEvidenceView(
        CollectionEvidencePolicy(collection.manifest.tenant_id, collection.manifest.collection_version),
        collection.manifest_sha256, MappingProxyType(documents), MappingProxyType(chunks),
        MappingProxyType({}), SnapshotContent((), ()), MappingProxyType(sources),
    )


def prepare_collection_answer(
    collection: SnapshotCollection, index: GlobalLexicalIndex, question: str,
    query_vector: np.ndarray | None, *, tenant_id: str, owner_id: str, run_id: str,
    recipe: GenerationRecipe, mode="hybrid", request_id=None, search_query=None,
    reranker=None, rerank_trace=None, subqueries=(), subquery_vectors=(), multi_query_trace=None,
    context_selection_recipe=None, context_selection_trace=None, context_selection_timing=None,
) -> tuple[CollectionEvidenceView, PreparedAnswer]:
    """Retrieve and pack original sources; generation keeps the original question.

    BM25 accepts a None vector. Vector/hybrid validate its dtype, dimension and
    normalization. An unlabelled ndarray cannot prove encoder identity: callers
    must also match their encoder's document recipe to the collection recipe.
    """
    request = QueryRequest(question=question, mode=mode)
    if tenant_id != collection.manifest.tenant_id:
        raise GenerationError("ANSWER_TENANT_MISMATCH")
    if request.mode not in {"bm25", "vector", "hybrid", "hybrid_reranked", *Q_FAMILY}:
        raise GenerationError("GRAPH_NOT_QUALITY_APPROVED_FOR_ANSWERS")
    if request.mode not in Q_FAMILY and (subqueries or subquery_vectors):
        raise GenerationError("UNEXPECTED_MULTI_QUERY_INPUT")
    selector = validate_context_selection_recipe(context_selection_recipe) if request.mode == "hybrid_multi_query_balanced" else None
    if request.mode == "hybrid_reranked":
        validate_collection_reranker(collection, reranker)
    validate_collection_lexical_binding(collection, index)
    search = request.question if search_query is None else QueryRequest(
        question=search_query, mode=request.mode,
    ).question
    if request.mode in Q_FAMILY:
        query_trace = retrieve_multi_query(collection, index, search, query_vector,
            subqueries=subqueries, subquery_vectors=subquery_vectors)
        ranked = query_trace.ranked_chunk_ids
        if multi_query_trace is not None:
            multi_query_trace.append(query_trace)
    elif request.mode == "bm25":
        ranked = tuple(key for key, _ in index.search(search, k=20))
    elif request.mode == "vector":
        ranked = tuple(hit.chunk_id for hit in collection.search_vector(query_vector, k=20))
    else:
        ranked = tuple(retrieve_collection_hybrid(collection, index, search, query_vector,
                                                  vector_k=20, lexical_k=20))
    try:
        view = _evidence_view(collection, index, ranked)
    except CollectionError as exc:
        raise GenerationError("COLLECTION_SOURCE_UNAVAILABLE") from exc
    if selector is not None:
        started = perf_counter()
        selection = select_context_order(query_trace, tuple(view.chunks[key] for key in ranked),
                                         tenant_id=tenant_id, recipe=selector)
        duration_ms = round((perf_counter() - started) * 1000, 3)
        if context_selection_trace is not None:
            context_selection_trace.append(selection)
        if context_selection_timing is not None:
            context_selection_timing.append(duration_ms)
        ranked = selection.ordered_chunk_ids
    if request.mode == "hybrid_reranked":
        started = perf_counter()
        scored = reranker.rerank(search, tuple(view.chunks[key] for key in ranked), tenant_id=tenant_id)
        trace = RerankTraceV1(recipe_sha256=scored.recipe_sha256, baseline_candidate_ids=ranked,
            ranked_chunk_ids=scored.ranked_chunk_ids, scores=scored.scores, windows=scored.windows,
            duration_ms=round((perf_counter() - started) * 1000, 3))
        if trace.recipe_sha256 != reranker.recipe.fingerprint:
            raise GenerationError("COLLECTION_RERANKER_RECIPE_MISMATCH")
        for key in ranked:
            own = [window for window in trace.windows if window.chunk_id == key]
            if (own[0].start_char != 0 or own[-1].end_char != len(view.chunks[key].text)
                    or tuple(window.window_index for window in own) != tuple(range(1, len(own) + 1))
                    or any(not 0 <= window.start_char < window.end_char <= len(view.chunks[key].text) for window in own)
                    or any(left.end_char != window.start_char
                           for left, window in zip(own, own[1:]))):
                raise GenerationError("COLLECTION_RERANKER_WINDOW_SOURCE_MISMATCH")
        if rerank_trace is not None:
            rerank_trace.append(trace)
        ranked = trace.ranked_chunk_ids
    retrieval = GraphRetrievalResult(ranked, ranked, (), (), request.mode, False, (), 0, 0)
    packed = pack_context(view, retrieval)
    if request_id is None:
        extra = {"reranker_recipe": reranker.recipe.fingerprint} if request.mode == "hybrid_reranked" else {}
        if request.mode in Q_FAMILY:
            extra["query_bundle_sha256"] = sha256_bytes(canonical_json([
                [query, sha256_bytes(vector.tobytes(order="C"))]
                for query, vector in zip(subqueries, subquery_vectors, strict=True)]))
        if selector is not None:
            extra["context_selection_recipe"] = selector.fingerprint
        request_id = stable_id(
            "q", tenant=tenant_id, owner=owner_id, question=request.question, search_query=search,
            mode=request.mode, collection_sha=collection.manifest_sha256,
            lexical_index=index.manifest.index_version, recipe=recipe.fingerprint,
            query_vector_sha256=None if request.mode == "bm25" else sha256_bytes(query_vector.tobytes(order="C")),
            **extra,
        )
    binding = AnswerBinding(
        request_id=request_id, tenant_id=tenant_id, owner_id=owner_id,
        mode_requested=request.mode, mode_used=request.mode, used_graph=False,
    )
    context = CallContext(tenant_id=tenant_id, run_id=run_id, request_id=request_id, stage="GENERATE")
    return view, PreparedAnswer(context, binding, packed, request.question, recipe, ranked)
