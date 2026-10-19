"""Shared local/future-cloud retrieval-to-answer path; authentication belongs to the caller."""

from dataclasses import dataclass

import numpy as np

from app.answer_cache import AnswerCache, AnswerLookup
from app.context import PackedContext, pack_context
from app.contracts import QueryRequest, stable_id
from app.generation import AnswerBinding, GenerationError, GenerationRecipe
from app.graph_retrieval import GraphRetrievalResult
from app.model_gateway import CallContext
from app.retrieval import retrieve_baseline
from app.snapshot import LoadedSnapshot


@dataclass(frozen=True)
class PreparedAnswer:
    context: CallContext
    binding: AnswerBinding
    packed: PackedContext
    question: str
    recipe: GenerationRecipe
    retrieved_chunk_ids: tuple[str, ...]


def prepare_answer(snapshot: LoadedSnapshot, question: str, query_vector: np.ndarray, *, tenant_id: str,
                   owner_id: str, run_id: str, recipe: GenerationRecipe, mode="hybrid", request_id=None) -> PreparedAnswer:
    request = QueryRequest(question=question, mode=mode)
    if tenant_id != snapshot.manifest.tenant_id:
        raise GenerationError("ANSWER_TENANT_MISMATCH")
    if request.mode not in {"vector", "hybrid"}:
        raise GenerationError("GRAPH_NOT_QUALITY_APPROVED_FOR_ANSWERS")
    ranked = tuple(retrieve_baseline(snapshot, request.question, query_vector, mode=request.mode))
    result = GraphRetrievalResult(ranked, ranked, (), (), request.mode, False, (), 0, 0)
    packed = pack_context(snapshot, result)
    request_id = request_id or stable_id("q", tenant=tenant_id, owner=owner_id, question=request.question,
        mode=request.mode, snapshot_sha=snapshot.manifest_sha256, recipe=recipe.fingerprint)
    binding = AnswerBinding(request_id=request_id, tenant_id=tenant_id, owner_id=owner_id,
        mode_requested=request.mode, mode_used=request.mode, used_graph=False)
    context = CallContext(tenant_id=tenant_id, run_id=run_id, request_id=request_id, stage="GENERATE")
    return PreparedAnswer(context, binding, packed, request.question, recipe, ranked)


def answer_prepared(prepared: PreparedAnswer, snapshot: LoadedSnapshot, cache: AnswerCache, *, gateway=None) -> AnswerLookup:
    return cache.resolve(prepared.context, question=prepared.question, snapshot=snapshot, packed=prepared.packed,
                         binding=prepared.binding, recipe=prepared.recipe, gateway=gateway)
