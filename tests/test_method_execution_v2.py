import pytest
from pydantic import ValidationError

from app.method_execution import MethodExecutionV2


def test_strict_contract_rejects_hybrid_fallback_and_false_graph_success():
    base = dict(requested_method="graph_1hop", mechanism_status="USED_GRAPH", executed_stages=("TRAVERSE_GRAPH",),
        effective_method="graph_1hop", graph_path_used=True, context_chunk_ids=("chunk",),
        context_assertion_ids=("assertion",), answer_status="READY")
    assert MethodExecutionV2(**base).graph_path_used
    for mutation in ({"effective_method": "hybrid"}, {"context_assertion_ids": ()},
                     {"graph_path_used": False}, {"answer_status": "NOT_REQUESTED"}):
        with pytest.raises(ValidationError):
            MethodExecutionV2(**(base | mutation))


def test_unsupported_graph_cannot_prepare_generation_or_claim_a_used_method():
    base = dict(requested_method="graph_2hop", mechanism_status="NO_SUPPORTED_RELATION",
        executed_stages=("LINK_ENTITIES",), answer_status="NOT_REQUESTED")
    assert MethodExecutionV2(**base).effective_method is None
    for mutation in ({"effective_method": "graph_2hop"}, {"answer_status": "READY"},
                     {"graph_path_used": True}, {"context_assertion_ids": ("fake",)}):
        with pytest.raises(ValidationError):
            MethodExecutionV2(**(base | mutation))


def test_qb_requires_actual_planning_and_selection_while_zero_decomposition_is_explicit():
    base = dict(requested_method="hybrid_multi_query_balanced", mechanism_status="USED_MULTI_QUERY",
        effective_method="hybrid_multi_query_balanced", planning_status="VALID", subquery_count=2,
        context_selection_used=True, executed_stages=("ALLOCATE_CONTEXT",), context_chunk_ids=("chunk",),
        answer_status="READY")
    assert MethodExecutionV2(**base).subquery_count == 2
    for mutation in ({"planning_status": "FAILED"}, {"subquery_count": 0}, {"context_selection_used": False}):
        with pytest.raises(ValidationError):
            MethodExecutionV2(**(base | mutation))
    zero = base | {"mechanism_status": "NO_DECOMPOSITION_NEEDED", "subquery_count": 0}
    assert MethodExecutionV2(**zero).mechanism_status == "NO_DECOMPOSITION_NEEDED"
