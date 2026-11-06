from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.benchmark_queries import AugmentedRetrievalReceipt
from app.collection_benchmark import RetrievalTask
from app.context_selection import ContextSelectionRecipe
from app.contracts import sha256_bytes
from app.embeddings import EmbeddingCache
from app.generation import GenerationRecipeV9
from scripts.run_augmented_collection_benchmark import AugmentedRuntime, execute_remaining
from tests.test_collection_evaluation import dataset, group, scenario
from tests.test_collection_evaluation import index as index
from tests.test_collection_review import review_context as review_context
from tests.test_rag_runner import Encoder


def test_query_and_balanced_use_identical_shared_pool_and_embedding_vectors(review_context, tmp_path):
    collection, lexical, _, _, params = review_context
    runtime = AugmentedRuntime.__new__(AugmentedRuntime)
    runtime.collection, runtime.lexical = collection, lexical
    runtime.encoder = Encoder(params["document_recipe"], params["query_recipe"])
    runtime.query_recipe = params["query_recipe"]
    runtime.cache = EmbeddingCache(tmp_path / "queries", namespace="FIXTURE")
    runtime.vectors, runtime.embedding_events = {}, []
    runtime.reranker, runtime.recipe = None, GenerationRecipeV9(model_metadata_sha256="0" * 64)
    runtime.selector = ContextSelectionRecipe()
    runtime.rows = {("case1", "vi"): SimpleNamespace(search_query="Service B", subqueries=("Service A",))}
    task = RetrievalTask("case1", "vi", "Dịch vụ B hoạt động ra sao?")
    q = runtime(task, "hybrid_multi_query")
    qb = runtime(task, "hybrid_multi_query_balanced")
    assert set(q["ranked_chunk_ids"]) == set(qb["ranked_chunk_ids"])
    assert runtime.encoder.calls == 2  # Base and one subquery, shared across Q/QB.
    assert not q["context_selection_used"] and qb["context_selection_used"]
    # Changing method never substitutes the displayed Vietnamese question with gold EN.
    assert len(runtime.rows) == 1


def test_remaining_matrix_is_two_en_methods_and_all_eight_vi_with_failure_retained(index):
    chunk = next(iter(index.chunks.values()))
    case = scenario((group(chunk),))
    source = dataset(index, (case,))
    row = SimpleNamespace(search_query="English search", subqueries=("One supporting search",), preparation_status="READY",
        planner_receipt=SimpleNamespace(operation_id="plan-shared"),
        rewrite_receipt=SimpleNamespace(operation_id="rewrite-shared"))

    class Runtime:
        rows = {(case.scenario_id, lang): row for lang in ("en", "vi")}
        bundle_sha = "c" * 64

        def __call__(self, task, method):
            if task.language == "vi" and method == "vector":
                raise ValueError("SYNTHETIC_EMBEDDING_FAILURE")
            return dict(effective_method="hybrid" if method.startswith("graph_") else method,
                ranked_chunk_ids=(chunk.chunk_id,), context_chunk_ids=(chunk.chunk_id,), duration_ms=1.0,
                context_selection_used=method == "hybrid_multi_query_balanced")

    receipts = []
    execute_remaining(source, Runtime(), emit=receipts.append)
    assert len(receipts) == 10
    assert sum(r.status == "FAILED" for r in receipts) == 1
    assert sum(r.language == "en" for r in receipts) == 2
    assert sum(r.query_track == "VI_SHARED_REWRITE" for r in receipts) == 8
    q = [r for r in receipts if r.language == "vi" and "multi_query" in r.requested_method]
    assert len({r.planner_operation_id for r in q}) == 1
    assert len({r.rewrite_operation_id for r in receipts if r.language == "vi"}) == 1
    row.preparation_status, row.failure_code, row.subqueries = "PLAN_FAILED", "QUERY_PLAN_OUTPUT_INVALID", ()
    failed_preparation = []
    execute_remaining(source, Runtime(), emit=failed_preparation.append)
    dependent = [r for r in failed_preparation if "multi_query" in r.requested_method]
    assert len(dependent) == 4 and all(r.status == "FAILED" for r in dependent)
    assert all(r.preparation_error == "QUERY_PLAN_OUTPUT_INVALID" and not r.ranked_chunk_ids for r in dependent)


def test_augmented_receipt_cannot_mislabel_raw_vi_or_hide_missing_planner():
    values = dict(schema_version=1, scenario_id="s1", language="vi", query_track="VI_SHARED_REWRITE",
        question_sha256=sha256_bytes(b"fixture"), requested_method="hybrid_multi_query",
        effective_method="hybrid_multi_query", status="COMPLETED", duration_ms=1.0,
        search_query_sha256="a" * 64, shared_bundle_sha256="b" * 64,
        planner_operation_id="planner", rewrite_operation_id="rewrite")
    assert AugmentedRetrievalReceipt(**values).model_calls == 0
    with pytest.raises(ValidationError):
        AugmentedRetrievalReceipt(**(values | {"query_track": "VI_RAW_DIAGNOSTIC"}))
    with pytest.raises(ValidationError, match="STAGE_IDENTITY"):
        AugmentedRetrievalReceipt(**(values | {"planner_operation_id": None}))
