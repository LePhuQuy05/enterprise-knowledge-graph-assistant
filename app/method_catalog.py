"""Public-safe deployment capabilities; local experiments do not enable cloud methods."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from app.contracts import Record


class PublicMethod(Record):
    method_id: Literal["bm25", "vector", "hybrid", "hybrid_reranked", "hybrid_multi_query",
                       "hybrid_multi_query_balanced", "graph_1hop", "graph_2hop"]
    label: Annotated[str, StringConstraints(min_length=1, max_length=100)]
    description: Annotated[str, StringConstraints(min_length=1, max_length=300)]
    available_scopes: tuple[Literal["snapshot", "aws_official"], ...]
    planned_scopes: tuple[Literal["snapshot", "local_collection"], ...]
    state: Literal["AVAILABLE", "NOT_SERVED"]
    unavailable_reason: Literal["COLLECTION_SERVING_NOT_ENABLED", "GRAPH_ANSWER_GATE_CLOSED"] | None
    execution_endpoint: Literal["/v1/queries"] | None
    run_kinds: tuple[Literal["answer"], ...]
    quality_metrics: None = None

    @model_validator(mode="after")
    def consistent_capabilities(self):
        for values in (self.available_scopes, self.planned_scopes, self.run_kinds):
            if len(set(values)) != len(values):
                raise ValueError("METHOD_CAPABILITIES_MUST_BE_UNIQUE")
        if self.state == "AVAILABLE":
            if not self.available_scopes or self.unavailable_reason is not None or (
                self.execution_endpoint is None or self.run_kinds != ("answer",)
            ):
                raise ValueError("AVAILABLE_METHOD_REQUIRES_EXECUTION_CAPABILITY")
        elif self.available_scopes or self.execution_endpoint is not None or self.run_kinds or (
            self.unavailable_reason is None
        ):
            raise ValueError("UNSERVED_METHOD_CANNOT_ADVERTISE_EXECUTION")
        return self


class PublicMethodCatalog(Record):
    schema_version: Literal[1] = 1
    catalog_version: Literal["deployment-method-catalog-v1"] = "deployment-method-catalog-v1"
    active_snapshot_version: Annotated[str, StringConstraints(min_length=1, max_length=160)]
    collection_serving_enabled: Literal[False] = False
    async_runs_enabled: Literal[False] = False
    methods: Annotated[tuple[PublicMethod, ...], Field(min_length=8, max_length=8)]

    @model_validator(mode="after")
    def unique_methods(self):
        if len({row.method_id for row in self.methods}) != len(self.methods):
            raise ValueError("METHOD_IDS_MUST_BE_UNIQUE")
        return self


METHODS = (
    ("bm25", "Lexical RAG", "BM25 keyword retrieval with source-grounded generation.", "local_collection"),
    ("vector", "Dense RAG", "Vector retrieval with source-grounded generation.", "local_collection"),
    ("hybrid", "Hybrid RAG", "Lexical and dense retrieval combined by rank fusion.", "local_collection"),
    ("hybrid_reranked", "Hybrid + reranker", "Reorder a bounded hybrid candidate pool with a cross-encoder.", "local_collection"),
    ("hybrid_multi_query", "Multi-query RAG", "Bounded query planning and rank fusion across query variants.", "local_collection"),
    ("hybrid_multi_query_balanced", "Query-balanced RAG", "Select context in turns within the frozen multi-query pool.", "local_collection"),
    ("graph_1hop", "Graph-assisted RAG · 1 hop", "Expand evidence through source-backed relation assertions.", "snapshot"),
    ("graph_2hop", "Graph-assisted RAG · 2 hops", "Expand evidence through two bounded logical relation hops.", "snapshot"),
)


def deployment_method_catalog(snapshot_version, *, official_search_available):
    """Caller supplies trusted server capabilities, never a browser-selected feature flag."""
    if type(official_search_available) is not bool:
        raise ValueError("METHOD_CATALOG_CAPABILITY_TYPE")
    rows = []
    for method, label, description, planned_scope in METHODS:
        live = method in {"vector", "hybrid"}
        scopes = (("snapshot", "aws_official") if method == "hybrid" and official_search_available
                  else ("snapshot",)) if live else ()
        rows.append(PublicMethod(method_id=method, label=label, description=description,
            available_scopes=scopes, planned_scopes=(planned_scope,), state="AVAILABLE" if live else "NOT_SERVED",
            unavailable_reason=None if live else (
                "GRAPH_ANSWER_GATE_CLOSED" if method.startswith("graph_") else "COLLECTION_SERVING_NOT_ENABLED"),
            execution_endpoint="/v1/queries" if live else None, run_kinds=("answer",) if live else ()))
    return PublicMethodCatalog(active_snapshot_version=snapshot_version, methods=tuple(rows)).model_dump(mode="json")
