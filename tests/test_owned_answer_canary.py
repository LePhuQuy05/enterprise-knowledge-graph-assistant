import pytest

from scripts import run_owned_answer_canary as workflow


def test_execution_marker_blocks_rerunning_before_provider_or_aws(tmp_path, monkeypatch):
    (tmp_path / "execution-started.json").write_text("{}", encoding="utf-8")
    def forbidden():
        pytest.fail("must not call paid/provider/AWS preflight after an execution marker exists")
    monkeypatch.setattr(workflow, "preflight", forbidden)
    with pytest.raises(ValueError, match="NO_AUTOMATIC_RETRY"):
        workflow.main(["--execute", "--directory", str(tmp_path)])


def test_group_budget_failure_stops_before_pipeline_queue_or_admission(tmp_path, monkeypatch):
    monkeypatch.setattr(workflow, "preflight", lambda: (None, {"status": "INSUFFICIENT_BUDGET", "quota_remaining": 286}))
    with pytest.raises(ValueError, match="ADMISSION_NOT_READY"):
        workflow.main(["--execute", "--directory", str(tmp_path)])
    assert not list(tmp_path.iterdir())


def test_canary_is_fixed_to_one_english_hybrid_question():
    assert workflow.AUTH.user_id.endswith("000000000002")
    assert "KMS" in workflow.QUESTION
    with pytest.raises(SystemExit):
        workflow.main(["--execute", "--methods", "hybrid", "vector"])


def test_replay_compares_canonical_json_without_mutating_saved_types():
    detail = {"request": {"method_ids": ["hybrid"]}}
    results = [{"context_chunk_ids": ("chunk-a", "chunk-b")}]
    old = {"run": detail, "results": [{"context_chunk_ids": ["chunk-a", "chunk-b"]}]}
    assert workflow.same_recorded_result(detail, results, old)
    assert type(results[0]["context_chunk_ids"]) is tuple
    old["results"][0]["context_chunk_ids"].reverse()
    assert not workflow.same_recorded_result(detail, results, old)


def test_replay_only_runner_patch_preserves_original_code_and_core_pins(tmp_path):
    from app.contracts import sha256_bytes

    own = "scripts/run_owned_answer_canary.py"
    stored = {"request": "fixed", "implementation_sha256": {own: sha256_bytes(b"executed"), "app/run_budget.py": "a" * 64}}
    current = stored | {"implementation_sha256": stored["implementation_sha256"] | {own: sha256_bytes(b"fixed replay")}}
    with pytest.raises(ValueError, match="ARCHIVE_MISSING"):
        workflow.validate_saved_plan(stored, current, tmp_path, replay=True)
    (tmp_path / "execution-runner.py").write_bytes(b"executed")
    workflow.validate_saved_plan(stored, current, tmp_path, replay=True)
    with pytest.raises(ValueError, match="PLAN_CHANGED"):
        workflow.validate_saved_plan(stored, current, tmp_path, replay=False)
    changed_core = current | {"implementation_sha256": current["implementation_sha256"] | {"app/run_budget.py": "b" * 64}}
    with pytest.raises(ValueError, match="PLAN_CHANGED"):
        workflow.validate_saved_plan(stored, changed_core, tmp_path, replay=True)
