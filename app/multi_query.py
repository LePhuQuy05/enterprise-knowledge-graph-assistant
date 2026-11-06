"""Bounded multi-query retrieval: per-query hybrid, then equal-weight rank fusion.

Planning lives separately. This module consumes validated query strings/vectors,
never an expected answer, target passage or relevance label.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from app.collection_lexical import retrieve_collection_hybrid
from app.contracts import Record, Sha256
from app.retrieval import reciprocal_rank_fusion


class QueryRankingV1(Record):
    query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    kind: Literal["BASE", "SUBQUERY"]
    ranked_chunk_ids: Annotated[tuple[str, ...], Field(max_length=40)]

    @model_validator(mode="after")
    def unique_ids(self) -> Self:
        if len(set(self.ranked_chunk_ids)) != len(self.ranked_chunk_ids):
            raise ValueError("Q_DUPLICATE_PER_QUERY_CHUNK")
        return self


class MultiQueryTraceV1(Record):
    schema_version: Literal[1] = 1
    fusion: Literal["QUERY_HYBRID_THEN_EQUAL_RRF60_CAP40"] = "QUERY_HYBRID_THEN_EQUAL_RRF60_CAP40"
    base_search_query: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    queries: Annotated[tuple[QueryRankingV1, ...], Field(min_length=1, max_length=3)]
    ranked_chunk_ids: Annotated[tuple[str, ...], Field(max_length=40)]
    unique_union_count: Annotated[int, Field(ge=0, le=120)]
    truncated_count: Annotated[int, Field(ge=0, le=80)]

    @model_validator(mode="after")
    def actual_fusion(self) -> Self:
        if self.queries[0].kind != "BASE" or self.queries[0].query != self.base_search_query:
            raise ValueError("Q_BASE_ANCHOR_MISMATCH")
        normalized = [" ".join(row.query.split()).casefold() for row in self.queries]
        if (len(set(normalized)) != len(normalized)
                or any(row.kind != "SUBQUERY" or len(row.query) > 240 for row in self.queries[1:])):
            raise ValueError("Q_QUERY_DUPLICATE_OR_SUBQUERY_BOUND")
        fused = tuple(reciprocal_rank_fusion([row.ranked_chunk_ids for row in self.queries]))
        if (self.ranked_chunk_ids != fused[:40] or self.unique_union_count != len(fused)
                or self.truncated_count != max(0, len(fused) - 40)):
            raise ValueError("Q_FUSION_TRACE_MISMATCH")
        return self


class PlanningTraceV1(Record):
    schema_version: Literal[1] = 1
    recipe_sha256: Sha256
    subqueries: Annotated[tuple[str, ...], Field(max_length=2)]
    cache_hit: bool
    new_cost_micro_usd: Annotated[int, Field(ge=0)]
    receipt: dict

    @model_validator(mode="after")
    def receipt_cost_binding(self) -> Self:
        if (self.receipt.get("recipe_sha256") != self.recipe_sha256
                or type(self.receipt.get("cost_micro_usd")) is not int
                or self.receipt["cost_micro_usd"] < 0
                or self.new_cost_micro_usd != (0 if self.cache_hit else self.receipt["cost_micro_usd"])
                or any(not query or len(query) > 240 for query in self.subqueries)):
            raise ValueError("Q_PLANNER_TRACE_RECEIPT_MISMATCH")
        return self


def retrieve_multi_query(collection, lexical, base_query, base_vector, *, subqueries, subquery_vectors):
    """At most 120 transient IDs, capped at 40 before source materialization."""
    if not isinstance(subqueries, tuple) or not isinstance(subquery_vectors, tuple) or len(subqueries) != len(subquery_vectors):
        raise ValueError("Q_QUERY_VECTOR_PAIR_MISMATCH")
    if len(subqueries) > 2:
        raise ValueError("Q_SUBQUERY_CAP")
    queries = (base_query, *subqueries)
    normalized = [" ".join(query.split()).casefold() if isinstance(query, str) else "" for query in queries]
    if (any(not value for value in normalized) or len(normalized) != len(set(normalized))
            or not isinstance(base_query, str) or len(base_query) > 2000
            or any(not isinstance(query, str) or len(query) > 240 for query in subqueries)):
        raise ValueError("Q_QUERY_INPUT_INVALID")
    rankings = tuple(QueryRankingV1(query=query, kind="BASE" if index == 0 else "SUBQUERY",
        ranked_chunk_ids=tuple(retrieve_collection_hybrid(collection, lexical, query, vector,
                                                        vector_k=20, lexical_k=20)))
        for index, (query, vector) in enumerate(zip(queries, (base_vector, *subquery_vectors), strict=True)))
    fused = tuple(reciprocal_rank_fusion([row.ranked_chunk_ids for row in rankings]))
    return MultiQueryTraceV1(base_search_query=base_query, queries=rankings, ranked_chunk_ids=fused[:40],
                            unique_union_count=len(fused), truncated_count=max(0, len(fused) - 40))
