"""Local L/D/H and versioned R contracts; execution is separate from quality approval."""

import math
from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, field_validator, model_serializer, model_validator

from app.context_selection import ContextSelectionRecipe, ContextSelectionResult
from app.contracts import QueryResponse, Record, Sha256, canonical_json, sha256_bytes
from app.multi_query import MultiQueryTraceV1, PlanningTraceV1
from app.query_planning import PlannerRecipe
from app.reranking import ChunkScore, RerankerRecipe, WindowScore

BaselineMethodId = Literal["bm25", "vector", "hybrid"]
MethodId = Literal["bm25", "vector", "hybrid", "hybrid_reranked", "hybrid_multi_query", "hybrid_multi_query_balanced"]


class MethodConfigV1(Record):
    schema_version: Literal[1] = 1
    method_id: BaselineMethodId
    revision: Literal["collection-ldh-v1"] = "collection-ldh-v1"
    supported_scopes: tuple[Literal["local_collection"], ...] = ("local_collection",)
    feature_state: Literal["experimental"] = "experimental"
    candidates_per_branch: Literal[20] = 20
    fusion: Literal["NONE", "RRF60"]
    max_context_chunks: Literal[8] = 8
    context_proxy_token_budget: Literal[4096] = 4096
    max_output_tokens: Literal[800] = 800
    collection_manifest_sha256: Sha256
    lexical_index_version: str
    document_recipe_sha256: Sha256
    query_recipe_sha256: Sha256 | None
    generation_recipe_sha256: Sha256

    @model_validator(mode="after")
    def actual_method(self) -> Self:
        if self.fusion != ("RRF60" if self.method_id == "hybrid" else "NONE"):
            raise ValueError("METHOD_FUSION_MISMATCH")
        if (self.query_recipe_sha256 is None) != (self.method_id == "bm25"):
            raise ValueError("METHOD_ENCODER_MISMATCH")
        return self

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


class RerankedMethodConfigV1(MethodConfigV1):
    schema_version: Literal[2] = 2
    method_id: Literal["hybrid_reranked"] = "hybrid_reranked"
    revision: Literal["collection-hybrid-reranked-v1"] = "collection-hybrid-reranked-v1"
    max_candidates: Literal[40] = 40
    reranker_recipe: RerankerRecipe
    reranker_recipe_sha256: Sha256

    @model_validator(mode="after")
    def actual_method(self) -> Self:
        if self.fusion != "RRF60" or self.query_recipe_sha256 is None:
            raise ValueError("METHOD_RERANKED_RETRIEVAL_MISMATCH")
        if self.reranker_recipe_sha256 != self.reranker_recipe.fingerprint:
            raise ValueError("METHOD_RERANKER_RECIPE_MISMATCH")
        return self


class RerankTraceV1(Record):
    schema_version: Literal[1] = 1
    recipe_sha256: Sha256
    baseline_candidate_ids: Annotated[tuple[str, ...], Field(max_length=40)]
    ranked_chunk_ids: Annotated[tuple[str, ...], Field(max_length=40)]
    scores: Annotated[tuple[ChunkScore, ...], Field(max_length=40)]
    windows: Annotated[tuple[WindowScore, ...], Field(max_length=160)]
    duration_ms: Annotated[float, Field(ge=0)]

    @model_validator(mode="after")
    def source_pool_and_scores(self) -> Self:
        before, after = self.baseline_candidate_ids, self.ranked_chunk_ids
        if (len(before) != len(set(before)) or len(after) != len(set(after)) or set(before) != set(after)
                or tuple(score.chunk_id for score in self.scores) != after):
            raise ValueError("METHOD_RERANKER_CANDIDATE_MISMATCH")
        for score in self.scores:
            own = [window for window in self.windows if window.chunk_id == score.chunk_id]
            if (not math.isfinite(score.raw_logit) or score.original_rank != before.index(score.chunk_id) + 1
                    or not 1 <= score.window_count == len(own) <= 8
                    or score.raw_logit != max(window.raw_logit for window in own)):
                raise ValueError("METHOD_RERANKER_SCORE_MISMATCH")
        if any(window.chunk_id not in before or not math.isfinite(window.raw_logit)
               or not 1 <= window.input_tokens <= 512 or window.offset_basis != "CHUNK_LOCAL_UNICODE"
               or window.original_rank != before.index(window.chunk_id) + 1 for window in self.windows):
            raise ValueError("METHOD_RERANKER_WINDOW_MISMATCH")
        if tuple(sorted(self.scores, key=lambda score: (-score.raw_logit, score.original_rank, score.chunk_id))) != self.scores:
            raise ValueError("METHOD_RERANKER_ORDER_MISMATCH")
        return self


class MultiQueryMethodConfigV1(MethodConfigV1):
    schema_version: Literal[3] = 3
    method_id: Literal["hybrid_multi_query"] = "hybrid_multi_query"
    revision: Literal["collection-bounded-multi-query-v1"] = "collection-bounded-multi-query-v1"
    max_queries: Literal[3] = 3
    max_candidates: Literal[40] = 40
    max_transient_candidates: Literal[120] = 120
    query_fusion: Literal["QUERY_HYBRID_THEN_EQUAL_RRF60_CAP40"] = "QUERY_HYBRID_THEN_EQUAL_RRF60_CAP40"
    planner_recipe: PlannerRecipe
    planner_recipe_sha256: Sha256

    @model_validator(mode="after")
    def actual_method(self) -> Self:
        if self.fusion != "RRF60" or self.query_recipe_sha256 is None:
            raise ValueError("METHOD_MULTI_QUERY_RETRIEVAL_MISMATCH")
        if self.planner_recipe_sha256 != self.planner_recipe.fingerprint:
            raise ValueError("METHOD_PLANNER_RECIPE_MISMATCH")
        return self


class BalancedMultiQueryMethodConfigV1(MultiQueryMethodConfigV1):
    schema_version: Literal[4] = 4
    method_id: Literal["hybrid_multi_query_balanced"] = "hybrid_multi_query_balanced"
    revision: Literal["collection-bounded-multi-query-balanced-v1"] = "collection-bounded-multi-query-balanced-v1"
    context_selection_recipe: ContextSelectionRecipe
    context_selection_recipe_sha256: Sha256

    @model_validator(mode="after")
    def selector_integrity(self) -> Self:
        if self.context_selection_recipe.policy != "RR_ID_ONLY":
            raise ValueError("METHOD_CONTEXT_SELECTION_RUNTIME_POLICY_UNAVAILABLE")
        if self.context_selection_recipe_sha256 != self.context_selection_recipe.fingerprint:
            raise ValueError("METHOD_CONTEXT_SELECTION_RECIPE_MISMATCH")
        return self


class RunInputV1(Record):
    schema_version: Literal[1] = 1
    original_question: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    language: Literal["auto", "en", "vi"] = "auto"
    source_scope: Literal["local_collection"] = "local_collection"
    method_ids: Annotated[tuple[MethodId, ...], Field(min_length=1, max_length=3)] = ("bm25", "vector", "hybrid")
    run_kind: Literal["retrieval", "answer"] = "retrieval"
    idempotency_key: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,80}$")]

    @field_validator("original_question")
    @classmethod
    def trim_question(cls, value):
        value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not value:
            raise ValueError("EMPTY_QUESTION")
        return value

    @field_validator("method_ids")
    @classmethod
    def distinct_methods(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("DUPLICATE_METHODS")
        return value


class StageEventV1(Record):
    stage: Literal["query_rewrite", "query_embedding", "query_planning", "subquery_embeddings", "retrieval_and_pack", "reranking", "context_selection", "generation_and_validation"]
    duration_ms: Annotated[float, Field(ge=0)]
    status: Literal["COMPLETED", "CACHE_MISS", "FAILED"]
    input_sha256: Sha256
    output_count: Annotated[int, Field(ge=0)] = 0
    error_category: str | None = None


class MethodResultV1(Record):
    schema_version: Literal[1] = 1
    requested_method: MethodId
    effective_method: MethodId
    config: MethodConfigV1 | RerankedMethodConfigV1 | MultiQueryMethodConfigV1 | BalancedMultiQueryMethodConfigV1
    config_sha256: Sha256
    collection_version: str
    status: Literal["RETRIEVED", "COMPLETED", "PLANNED_NO_MODEL_CALL", "FAILED"]
    ranked_chunk_ids: tuple[str, ...] = ()
    context_chunk_ids: tuple[str, ...] = ()
    context_proxy_tokens: Annotated[int, Field(ge=0)] = 0
    context_sources: Annotated[tuple[dict, ...], Field(max_length=8)] = ()
    response: QueryResponse | None = None
    cache_hit: bool = False
    new_cost_micro_usd: Annotated[int, Field(ge=0)] | None = 0
    original_receipt_cost_micro_usd: Annotated[int, Field(ge=0)] | None = None
    cost_basis: Literal["PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND"] = "PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND"
    usage: dict | None = None
    trace: tuple[StageEventV1, ...] = ()
    error_category: str | None = None
    semantic_review: Literal["NOT_REVIEWED"] = "NOT_REVIEWED"
    rerank_trace: RerankTraceV1 | None = None
    planning_trace: PlanningTraceV1 | None = None
    multi_query_trace: MultiQueryTraceV1 | None = None
    context_selection_trace: ContextSelectionResult | None = None

    @model_serializer(mode="wrap")
    def preserve_baseline_shape(self, handler):
        result = handler(self)
        for name in ("rerank_trace", "planning_trace", "multi_query_trace", "context_selection_trace"):
            if getattr(self, name) is None:
                result.pop(name, None)
        return result

    @model_validator(mode="after")
    def method_integrity(self) -> Self:
        if self.requested_method != self.effective_method or self.config.method_id != self.effective_method:
            raise ValueError("METHOD_ID_MISMATCH")
        if self.config_sha256 != self.config.fingerprint:
            raise ValueError("METHOD_CONFIG_HASH_MISMATCH")
        if self.rerank_trace is not None:
            if (not isinstance(self.config, RerankedMethodConfigV1)
                    or self.rerank_trace.recipe_sha256 != self.config.reranker_recipe_sha256
                    or self.ranked_chunk_ids != self.rerank_trace.ranked_chunk_ids):
                raise ValueError("METHOD_RERANKER_TRACE_MISMATCH")
        elif self.effective_method == "hybrid_reranked" and self.status != "FAILED":
            raise ValueError("METHOD_RERANKER_TRACE_REQUIRED")
        if self.planning_trace is not None:
            if (not isinstance(self.config, MultiQueryMethodConfigV1)
                    or self.planning_trace.recipe_sha256 != self.config.planner_recipe_sha256):
                raise ValueError("METHOD_PLANNER_TRACE_MISMATCH")
        if self.multi_query_trace is not None:
            if (not isinstance(self.config, MultiQueryMethodConfigV1)
                    or self.planning_trace is None
                    or tuple(row.query for row in self.multi_query_trace.queries[1:]) != self.planning_trace.subqueries
                    or (self.effective_method != "hybrid_multi_query_balanced"
                        and self.multi_query_trace.ranked_chunk_ids != self.ranked_chunk_ids)):
                raise ValueError("METHOD_MULTI_QUERY_TRACE_MISMATCH")
        elif self.effective_method in {"hybrid_multi_query", "hybrid_multi_query_balanced"} and self.status in {"RETRIEVED", "COMPLETED"}:
            raise ValueError("METHOD_MULTI_QUERY_TRACE_REQUIRED")
        if self.context_selection_trace is not None:
            selection = self.context_selection_trace
            if (not isinstance(self.config, BalancedMultiQueryMethodConfigV1)
                    or self.multi_query_trace is None
                    or selection.recipe != self.config.context_selection_recipe
                    or selection.recipe_sha256 != self.config.context_selection_recipe_sha256
                    or selection.original_pool_ids != self.multi_query_trace.ranked_chunk_ids
                    or selection.source_trace_sha256 != sha256_bytes(canonical_json(self.multi_query_trace.model_dump(mode="json")))
                    or selection.ordered_chunk_ids != self.ranked_chunk_ids
                    or not set(self.context_chunk_ids).issubset(selection.ordered_chunk_ids)):
                raise ValueError("METHOD_CONTEXT_SELECTION_TRACE_MISMATCH")
            candidates = {row.chunk_id: row for row in selection.candidates}
            for key, candidate in candidates.items():
                expected_occurrences = tuple((index, query.ranked_chunk_ids.index(key) + 1, query.kind)
                    for index, query in enumerate(self.multi_query_trace.queries) if key in query.ranked_chunk_ids)
                actual_occurrences = tuple((row.query_index, row.query_rank, row.kind)
                                           for row in candidate.query_occurrences)
                if actual_occurrences != expected_occurrences:
                    raise ValueError("METHOD_CONTEXT_SELECTION_QUERY_PROVENANCE_MISMATCH")
            for source in self.context_sources:
                candidate = candidates.get(source.get("chunk_id"))
                if (candidate is None or candidate.document_version_id != source.get("document_version_id")
                        or candidate.text_sha256 != source.get("text_sha256")
                        or list(candidate.section_path) != source.get("section_path")
                        or candidate.start_char != source.get("start_char")
                        or candidate.end_char != source.get("end_char")):
                    raise ValueError("METHOD_CONTEXT_SELECTION_SOURCE_PROVENANCE_MISMATCH")
        elif self.effective_method == "hybrid_multi_query_balanced" and self.status in {"RETRIEVED", "COMPLETED"}:
            raise ValueError("METHOD_CONTEXT_SELECTION_TRACE_REQUIRED")
        if len(self.context_chunk_ids) > 8 or self.context_proxy_tokens > 4096:
            raise ValueError("METHOD_CONTEXT_LIMIT")
        if self.status == "COMPLETED" and self.response is None:
            raise ValueError("METHOD_RESPONSE_REQUIRED")
        if self.status != "COMPLETED" and self.response is not None:
            raise ValueError("METHOD_UNEXPECTED_RESPONSE")
        if self.response is not None and (self.response.mode_used != self.effective_method
                or self.response.used_graph or self.response.snapshot_version != self.collection_version
                or self.response.retrieval_backend != "collection_numpy_global_bm25"):
            raise ValueError("METHOD_RESPONSE_BINDING_MISMATCH")
        return self
