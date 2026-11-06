"""Read-only benchmark readiness and explicit, non-binding cost scenarios."""

from decimal import ROUND_CEILING, Decimal

METHODS = (
    ("bm25", "BM25", "Keywords and exact AWS identifiers."),
    ("vector", "Vector", "Semantic similarity using local BGE embeddings."),
    ("hybrid", "Hybrid", "BM25 and vector ranks fused with RRF."),
    ("hybrid_reranked", "Hybrid + Reranker", "A cross-encoder reorders the bounded hybrid pool."),
    ("hybrid_multi_query", "Multi-query RAG", "A bounded planner proposes additional search queries."),
    ("hybrid_multi_query_balanced", "Query-balanced RAG", "Allocate context across the multi-query rankings."),
    ("graph_1hop", "Graph-assisted RAG · 1 hop", "Follow source-backed assertions to additional evidence."),
    ("graph_2hop", "Graph-assisted RAG · 2 hops", "Explore bounded two-relation paths, preserving their sources."),
)


def readiness_methods(*, assertion_count, recorded_cloud_release):
    """Availability is not quality; nonempty graph alone never opens a serving gate."""
    if type(assertion_count) is not int or assertion_count < 0 or type(recorded_cloud_release) is not bool:
        raise ValueError("BENCHMARK_READINESS_INPUT_INVALID")
    rows = []
    for method, label, mechanism in METHODS:
        graph = method.startswith("graph_")
        cloud = recorded_cloud_release and method in {"bm25", "vector", "hybrid", "hybrid_reranked"}
        rows.append({
            "id": method, "label": label, "mechanism": mechanism,
            "local": "PILOT_GRAPH_ONLY" if graph else "COLLECTION_IMPLEMENTED",
            "cloud_retrieval": "RECORDED_DEPLOYED" if cloud else "NOT_RELEASED",
            "cloud_answer": "NOT_RELEASED", "collection_quality": None,
            "next_gate": ("BUILD_SOURCE_BACKED_COLLECTION_GRAPH" if not assertion_count else
                          "COLLECTION_GRAPH_REVIEW_AND_RUNNER") if graph else (
                              "FROZEN_SOURCE_LABELS_AND_EVALUATION" if cloud else
                              "BOUNDED_PAID_STAGES_AND_CLOUD_INTEGRATION"),
        })
    return rows


def cost_scenario(calls, *, input_tokens, output_tokens, input_micro_per_million,
                  output_micro_per_million, extra_attempt_fraction=Decimal("0")):
    """Forecast only. Token assumptions do not replace the gateway's reservation bound."""
    values = (calls, input_tokens, output_tokens, input_micro_per_million, output_micro_per_million)
    if any(type(value) is not int or value < 0 for value in values) or not (
        Decimal(0) <= extra_attempt_fraction <= Decimal(1)
    ):
        raise ValueError("BENCHMARK_COST_INPUT_INVALID")
    attempts = int((Decimal(calls) * (1 + extra_attempt_fraction)).to_integral_value(rounding=ROUND_CEILING))
    per_call = (Decimal(input_tokens) * input_micro_per_million +
                Decimal(output_tokens) * output_micro_per_million) / Decimal(1_000_000)
    micro = int((per_call * attempts).to_integral_value(rounding=ROUND_CEILING))
    return {"base_calls": calls, "assumed_attempts": attempts, "input_tokens_per_attempt": input_tokens,
            "output_tokens_per_attempt": output_tokens, "estimated_micro_usd": micro,
            "estimated_usd": micro / 1_000_000, "kind": "ASSUMPTION_NOT_INVOICE_OR_AUTHORIZATION"}
