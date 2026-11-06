from dataclasses import fields

import pytest
from pydantic import ValidationError

from app.collection_benchmark import (
    BenchmarkProtocol,
    ExposureCase,
    RetrievalReceipt,
    RetrievalTask,
    evidence_at_k_feasibility,
    score_receipts,
    summarize_receipts,
    validate_protocol,
)
from app.contracts import canonical_json, sha256_bytes
from scripts.run_collection_benchmark import CONFIG, execute_tasks
from tests.test_collection_evaluation import dataset, group, scenario
from tests.test_collection_evaluation import index as index


def receipt(case, *, method="hybrid", **changes):
    values = dict(schema_version=1, scenario_id=case.scenario_id, language="en", query_track="EN_ORIGINAL",
        question_sha256=sha256_bytes(case.question_en.encode("utf-8")), requested_method=method,
        effective_method=method, status="COMPLETED", duration_ms=1.0)
    values.update(changes)
    return RetrievalReceipt(**values)


def test_frozen_protocol_and_semantic_groups_bind_exact_original_content():
    protocol = BenchmarkProtocol.model_validate_json((CONFIG / "collection316-benchmark-protocol-v1.json").read_bytes())
    kwargs = dict(frozen_bytes=(CONFIG / "collection316-benchmark-frozen-v1.json").read_bytes(),
        inventory_bytes=(CONFIG / "collection316-exposure-inventory-v1.json").read_bytes(),
        authoring_bytes=(CONFIG / "collection316-annotations-v2.json").read_bytes())
    frozen = validate_protocol(protocol, **kwargs)
    assert len(frozen.scenarios) == 100
    assert frozen.split_status == "GROUPED_FROZEN"
    assert {s.split for s in frozen.scenarios} == {"DEVELOPMENT", "EVALUATION"}
    kwargs["inventory_bytes"] += b" "
    with pytest.raises(ValueError, match="HASH_MISMATCH"):
        validate_protocol(protocol, **kwargs)


def test_related_prior_inputs_cannot_be_claimed_reserved_evaluation():
    with pytest.raises(ValidationError, match="PREVIOUSLY_RELATED"):
        ExposureCase(scenario_id="s1", intent_group_id="g1", grouping_reason="Related facts",
            prior_related_inputs=("prior#1",), exposure="PRIOR_RELATED", split="EVALUATION")


def test_failure_denominator_and_missing_are_distinct(index):
    chunk = next(iter(index.chunks.values()))
    case = scenario((group(chunk),), split="DEVELOPMENT")
    source = dataset(index, (case,), split_status="GROUPED_FROZEN")
    rows = score_receipts(source, [receipt(case, ranked_chunk_ids=(chunk.chunk_id,),
        context_chunk_ids=(chunk.chunk_id,)), receipt(case, method="vector", status="FAILED",
        effective_method=None, error_category="FIXTURE_FAILURE", duration_ms=None)], index)
    summary = summarize_receipts(source, rows, languages=("en",), methods=("hybrid", "vector", "bm25"))
    by_method = {r["method"]: r for r in summary if r["split"] == "ALL"}
    assert by_method["hybrid"]["complete_evidence_at_k"] == 1
    assert by_method["vector"]["complete_evidence_at_k"] == 0
    assert by_method["vector"]["failed"] == 1
    assert by_method["bm25"]["complete_evidence_at_k"] is None
    assert by_method["bm25"]["not_run"] == 1


def test_controls_never_become_positive_retrieval_labels(index):
    case = scenario(answerable=False, kind="account_unavailable", split="DEVELOPMENT",
        control_reason="No account state", expected_behavior="ACKNOWLEDGE_ACCOUNT_LIMITATION")
    source = dataset(index, (case,), split_status="GROUPED_FROZEN")
    rows = score_receipts(source, [receipt(case)], index)
    total = summarize_receipts(source, rows, languages=("en",), methods=("hybrid",))[-1]
    assert total["complete_matrix"] and total["controls"] == 1
    assert total["answerable_denominator"] == 0 and total["complete_evidence_at_k"] is None


def test_runtime_only_receives_question_task_and_failures_do_not_drop_peers(index):
    chunk = next(iter(index.chunks.values()))
    case = scenario((group(chunk),))
    source = dataset(index, (case,))
    seen, receipts = [], []

    def runtime(task, method):
        assert {f.name for f in fields(task)} == {"scenario_id", "language", "question"}
        assert isinstance(task, RetrievalTask)
        seen.append(method)
        if method == "vector":
            raise ValueError("FIXTURE_RETRIEVAL_FAILURE")
        return dict(effective_method=method, ranked_chunk_ids=(chunk.chunk_id,),
                    context_chunk_ids=(chunk.chunk_id,), duration_ms=1.0)

    execute_tasks(source, runtime, languages=("en",), methods=("vector", "hybrid"), emit=receipts.append)
    assert seen == ["vector", "hybrid"]
    assert [r.status for r in receipts] == ["FAILED", "COMPLETED"]


def test_replay_scoring_rejects_question_and_duplicate_row_tampering(index):
    chunk = next(iter(index.chunks.values()))
    case = scenario((group(chunk),))
    source = dataset(index, (case,))
    row = receipt(case)
    with pytest.raises(ValueError, match="DUPLICATE_METHOD_CASE"):
        score_receipts(source, [row, row], index)
    with pytest.raises(ValueError, match="QUESTION_MISMATCH"):
        score_receipts(source, [row.model_copy(update={"question_sha256": "f" * 64})], index)


def test_graph_wins_cannot_be_fabricated_without_packed_assertion_evidence():
    with pytest.raises(ValidationError, match="GRAPH_EVIDENCE_REQUIRED"):
        receipt(scenario(annotation_status="NEEDS_EVIDENCE"), method="graph_1hop", used_graph=True)


def test_gold_oracle_respects_k_and_search_bound(index):
    chunks = list(index.chunks.values())[:2]
    case = scenario(tuple(group(c, n) for n, c in enumerate(chunks, 1)))
    result = evidence_at_k_feasibility(case, index, k=2)
    assert result["status"] == "REACHABLE_AT_K" and result["min_chunks"] == 2
    assert evidence_at_k_feasibility(case, index, k=1)["status"] == "NOT_REACHABLE_AT_K_FOR_CURRENT_SPANS"
    assert evidence_at_k_feasibility(case, index, max_combinations=0)["status"] == "UNKNOWN_SEARCH_LIMIT"


def test_source_content_cannot_be_edited_during_freeze_even_with_new_hash():
    protocol = BenchmarkProtocol.model_validate_json((CONFIG / "collection316-benchmark-protocol-v1.json").read_bytes())
    import json
    value = json.loads((CONFIG / "collection316-benchmark-frozen-v1.json").read_bytes())
    value["scenarios"][0]["reference_answer_en"] = "A convenient new answer after seeing a ranking"
    frozen = canonical_json(value)
    protocol = protocol.model_copy(update={"frozen_dataset_sha256": sha256_bytes(frozen)})
    with pytest.raises(ValueError, match="FROZEN_CONTENT_CHANGED"):
        validate_protocol(protocol, frozen, (CONFIG / "collection316-exposure-inventory-v1.json").read_bytes(),
                          authoring_bytes=(CONFIG / "collection316-annotations-v2.json").read_bytes())
