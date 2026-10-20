from types import SimpleNamespace

import pytest

from app.release_experiment import ExperimentStopped
from scripts.evaluate_expanded_answers import ExpandedAnswerExperiment, summary


def test_generation_totals_keep_format_failures_and_missing_cases():
    inputs = [{"language": lang} for _ in range(3) for lang in ("en", "vi")]
    rows = [{"question_id": "q1", "scenario_id": "scenario1", "language": "en", "answerable_in_corpus": False,
             "structural_status": "FAIL", "answer_status": None, "cost_upper_bound_micro_usd": 1200}]
    report = summary(rows, inputs)
    assert report["attempted"] == 1 and report["planned"] == 6
    assert report["languages"]["en"]["structural_fail"] == 1
    assert report["languages"]["en"]["unanswerable_returned_abstained"] == 0
    assert report["languages"]["vi"]["attempted"] == 0
    assert report["languages"]["en"]["semantic_correctness"] is None


def test_experiment_never_removes_another_writer_lock(tmp_path):
    journal = ExpandedAnswerExperiment(tmp_path, SimpleNamespace())
    journal.lock.write_text("another writer", encoding="utf-8")
    with pytest.raises(ExperimentStopped):
        journal.__enter__()
    assert journal.lock.read_text(encoding="utf-8") == "another writer"


def test_expanded_policy_does_not_change_legacy_money_or_operation_caps():
    from app.release_experiment import ReleaseExperiment

    assert ReleaseExperiment.MAX_OPERATIONS == 80 and ReleaseExperiment.BATCH_STOP == 80_000
    assert ExpandedAnswerExperiment.MAX_OPERATIONS == 100
    assert ExpandedAnswerExperiment.BATCH_STOP == 250_000
