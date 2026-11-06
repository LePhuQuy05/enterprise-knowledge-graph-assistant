"""Real-corpus source/identity parity plus no-evaluator-input boundaries."""

import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.context import PackedContext
from app.contracts import canonical_json
from app.generation import authorized_sources
from app.graph_collection_migration import collection_records
from app.profiled_retrieval import ProfileRetrievalTask, retrieve_profile_pool
from app.profiled_source_context import SourceContextProfile, render_profiled_context
from app.question_coverage_review import audit_question_coverage
from scripts.migrate_collection319_graph import load_migrated
from scripts.profiled_coverage_audit import audit_profiled_coverage
from scripts.run_context_profiles import control_review, load_review, recorded_pools

ROOT = Path(__file__).resolve().parents[1]
LOCAL_REPLAY_INPUTS = (
    "results/collection319-local/runtime.json",
    "results/collection319-graph-migration-v1/selection.json",
    "results/question-coverage-v1/semantic-amendment-v2/witnesses.json",
    "results/question-coverage-v1/budget-feasibility-v1/report.json",
    "results/facet-recall-v5-canary-v1/receipts.json",
    "results/collection319-methods-v1/receipts.json",
    "results/parent-section-recall-v1/receipts.json",
    "results/bounded-ancestor-recall-v1/receipts.json",
)
requires_recorded_profiles = pytest.mark.skipif(
    not all((ROOT / name).is_file() for name in LOCAL_REPLAY_INPUTS),
    reason="Requires local collection319 corpus, reviewed witnesses and recorded profile receipts",
)


@pytest.fixture(scope="module")
def sources():
    collection, _, _ = load_migrated()
    docs, chunks, _ = collection_records(collection)
    return collection, docs, chunks


@requires_recorded_profiles
def test_profiled_audit_matches_legacy_annotation_semantics_on_all_four_existing_inputs(sources):
    collection, docs, chunks = sources
    rubric, witnesses, _ = load_review()
    payload = json.loads((ROOT / "results/facet-recall-v5-canary-v1/receipts.json").read_bytes())
    for row in (r for r in payload["recall_cells"] if r["arm"] == "V5_WINDOW_R2_Q"):
        keys = tuple(row["context_chunk_ids"])
        context = render_profiled_context(docs, chunks, keys, question=row["question"], language=row["language"],
            tenant_id="demo", collection_sha256=collection.manifest_sha256, profile=SourceContextProfile(max_chunks=8))
        old = audit_question_coverage(rubric, witnesses, docs, chunks, keys, intent=row["intent"], tenant_id="demo")
        new = audit_profiled_coverage(rubric, witnesses, docs, chunks, context, intent=row["intent"])
        new.pop("context_profile_sha256")
        new["version"] = old["version"]
        assert canonical_json(new) == canonical_json(old)


@requires_recorded_profiles
def test_oracle_source_set_fits_profile12_and_meets_review_but_legacy_cap8_stays_unchanged(sources):
    collection, docs, chunks = sources
    rubric, witnesses, _ = load_review()
    oracle = json.loads((ROOT / "results/question-coverage-v1/budget-feasibility-v1/report.json").read_bytes())
    row = next(r for r in oracle["intents"] if r["intent"] == "s3-permission-layers")
    keys = tuple(s["chunk_id"] for s in row["example_annotation_oracle_sources"])
    assert len(keys) == 9
    context = render_profiled_context(docs, chunks, keys, question="Synthetic budget feasibility probe?", language="en",
        tenant_id="demo", collection_sha256=collection.manifest_sha256, profile=SourceContextProfile(max_chunks=12))
    review = audit_profiled_coverage(rubric, witnesses, docs, chunks, context, intent=row["intent"])
    assert review["all_reviewed_requirements_present"] and context.proxy_tokens < 4096
    packed = PackedContext(keys, tuple(f"s{i+1}" for i in range(9)), (), "", 0, ())
    with pytest.raises(ValueError, match="CONTEXT_LIMIT_OR_IDS"):
        authorized_sources(SimpleNamespace(manifest=SimpleNamespace(tenant_id="demo")), packed, "demo")


@requires_recorded_profiles
def test_recorded_q_qb_keep_identical_native_query_pool_and_bundles():
    rows = recorded_pools()
    assert len(rows) == 64
    for case in ("s3-permission-layers", "kms-cross-account"):
        for lang in ("en", "vi"):
            q, qb = [next(r for r in rows if r["family"] == "recorded-eight-methods" and r["task"]["task_id"] == case
                and r["task"]["language"] == lang and r["method"] == m)
                for m in ("hybrid_multi_query", "hybrid_multi_query_balanced")]
            assert q["pool_ids"] == qb["pool_ids"] and q["pool_fingerprint"] == qb["pool_fingerprint"]
            assert not q["recorded_qb_allocator_used"] and qb["recorded_qb_allocator_used"]


@requires_recorded_profiles
def test_control_requirements_are_scoped_before_retrieval_not_mechanically_copied():
    rubric, witnesses, scope = load_review()
    one, _ = control_review(rubric, witnesses, scope, "control-01")
    assert one.requirements[-1].requires == ("S1", "S2", "S3")
    assert all(r.id not in {"S4", "S7"} for r in one.requirements)
    two, _ = control_review(rubric, witnesses, scope, "control-02")
    assert [r.id for r in two.requirements] == ["S4"]
    six, evidence = control_review(rubric, witnesses, scope, "control-06")
    assert [r.id for r in six.requirements] == ["C6_KEY_TYPE"] and len(evidence) == len(evidence[0].spans) == 1
    assert control_review(rubric, witnesses, scope, "control-07") is None
    assert control_review(rubric, witnesses, scope, "control-08") is None


def test_runtime_task_rejects_evaluator_fields_and_native_q_is_not_substituted():
    data = {"task_id": "control", "question": "Synthetic AWS question?", "search": "Synthetic AWS question?",
        "language": "en", "query_track": "ORIGINAL_EN"}
    with pytest.raises(ValueError):
        ProfileRetrievalTask.model_validate_json(canonical_json(data | {"required_evidence": ["gold"]}))
    task = ProfileRetrievalTask.model_validate_json(canonical_json(data))
    with pytest.raises(ValueError, match="CORE_TASK_REQUIRED"):
        retrieve_profile_pool(None, None, None, None, task, "hybrid_multi_query")


def test_runtime_modules_do_not_import_review_or_oracle():
    for filename in ("app/profiled_source_context.py", "app/profiled_generation.py", "app/profiled_retrieval.py"):
        tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"))
        modules = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert not any(m and any(x in m for x in ("question_coverage", "profiled_coverage", "diagnose_question", "run_context_profiles")) for m in modules)


def test_reranking_error_propagates_instead_of_method_substitution(monkeypatch):
    import app.profiled_retrieval as module

    def fail(*args, **kwargs):
        raise RuntimeError("SYNTHETIC_SCORER_FAILURE")

    monkeypatch.setattr(module, "prepare_collection_answer", fail)
    task = ProfileRetrievalTask(task_id="test", question="Test?", search="Test?", language="en", query_track="ORIGINAL_EN")
    runtime = SimpleNamespace(recipe=None, reranker=None)
    with pytest.raises(RuntimeError, match="SCORER_FAILURE"):
        retrieve_profile_pool(SimpleNamespace(manifest=SimpleNamespace(tenant_id="demo")), None, None, runtime, task, "bm25")


@requires_recorded_profiles
def test_scope_freeze_binds_lexical_manifest_and_all_sixteen_control_inputs(tmp_path, monkeypatch):
    import scripts.run_context_profiles as workflow

    monkeypatch.setattr(workflow, "OUTPUT", tmp_path)
    workflow.freeze()
    scope = json.loads((tmp_path / "scope.json").read_bytes())
    assert len(scope["tasks"]) == 16 and scope["context_cells"] == 320
    assert len(scope["lexical_manifest_sha256"]) == 64
    assert workflow.check_pins() == scope
