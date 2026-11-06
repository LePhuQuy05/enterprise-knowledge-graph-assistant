import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.collection_answering import prepare_collection_answer
from app.contracts import canonical_json, sha256_bytes
from app.generation import GenerationRecipeV9
from app.reranking import ChunkScore, RerankerRecipe, RerankResult, WindowScore
from scripts.compare_citation_recipes import pin_comparison_plan
from scripts.compare_reranker_contexts import compare_prepared, main, verify_saved
from tests.test_collection_review import review_context as review_context


class ReversedFixtureReranker:
    """Synthetic ordering only; no claim of model or AWS quality."""

    def __init__(self, behavior="normal"):
        self.inference_calls = 0
        self.behavior = behavior
        self.queries = []
        self.recipe = RerankerRecipe(artifact_manifest_sha256="a" * 64,
                                     onnxruntime_version="fixture", tokenizers_version="fixture")

    def rerank(self, query, chunks, tenant_id):
        self.queries.append(query)
        if self.behavior == "failure":
            raise RuntimeError("SYNTHETIC_RERANK_FAILURE")
        if chunks:
            self.inference_calls += 1
        ids = tuple(chunk.chunk_id for chunk in reversed(chunks))
        scores = tuple(ChunkScore(key, next(i for i, c in enumerate(chunks, 1) if c.chunk_id == key),
                                  float(len(ids) - n), 1) for n, key in enumerate(ids))
        windows = tuple(WindowScore(c.chunk_id, i, 1, 0, len(c.text), 20, float(i)) for i, c in enumerate(chunks, 1))
        if self.behavior == "changed_pool" and ids:
            ids = ("invented-source", *ids[1:])
        if self.behavior == "unstable" and len(self.queries) > 1:
            ids = ids[::-1]
        return RerankResult(ids, scores, windows, self.recipe.fingerprint)


def prepare(review_context, *, empty=False):
    collection, lexical, _, vectors, _ = review_context
    view, ready = prepare_collection_answer(collection, lexical, "Fixture question", None if empty else vectors["dev-01:en"],
        tenant_id=collection.manifest.tenant_id, owner_id="fixture-owner", run_id="fixture-run",
        recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64),
        mode="bm25" if empty else "hybrid", search_query="qzxvplmunknown" if empty else "Service B")
    return view, ready, collection.manifest.tenant_id


def test_same_pool_comparison_changes_order_but_keeps_original_sources_and_caps(review_context):
    view, prepared, tenant = prepare(review_context)
    fake = ReversedFixtureReranker()
    original = prepared.packed
    report = compare_prepared(view, prepared, fake, search_query="Service B", tenant_id=tenant)
    assert prepared.packed is original
    assert tuple(report["reranked_chunk_ids"]) == prepared.retrieved_chunk_ids[::-1]
    assert report["same_candidate_pool"] and report["generation_calls"] == report["paid_api_calls"] == 0
    assert report["reranker_onnx_batch_calls_this_case"] == 3
    assert fake.queries == ["Service B"] * 3
    assert report["quality_metrics"] is None and not report["human_reviewed"]
    for method in ("H", "R"):
        assert len(report[method]["chunk_ids"]) <= 8 and report[method]["proxy_tokens"] <= 4096
        for row in report[method]["sources"]:
            chunk = view.chunks[row["chunk_id"]]
            doc = view.documents[chunk.document_version_id]
            assert row["text"] == doc.normalized_text[row["start_char"]:row["end_char"]]
            assert row["shard_manifest_sha256"] == view.source_refs[chunk.chunk_id].manifest_sha256


def test_empty_control_is_zero_inference_and_empty_context(review_context):
    view, prepared, tenant = prepare(review_context, empty=True)
    fake = ReversedFixtureReranker()
    report = compare_prepared(view, prepared, fake, search_query="qzxvplmunknown", tenant_id=tenant)
    assert fake.inference_calls == 0 and report["candidate_count"] == 0
    assert report["H"]["chunk_ids"] == report["R"]["chunk_ids"] == []


@pytest.mark.parametrize("behavior,error", [("changed_pool", "CHANGED_POOL"), ("unstable", "NONDETERMINISTIC")])
def test_invalid_reranker_result_cannot_be_published_as_same_pool(review_context, behavior, error):
    view, prepared, tenant = prepare(review_context)
    with pytest.raises(ValueError, match=error):
        compare_prepared(view, prepared, ReversedFixtureReranker(behavior), search_query="Service B", tenant_id=tenant)


def test_reranker_failure_is_not_hidden_by_returning_hybrid_context(review_context):
    view, prepared, tenant = prepare(review_context)
    with pytest.raises(RuntimeError, match="SYNTHETIC_RERANK_FAILURE"):
        compare_prepared(view, prepared, ReversedFixtureReranker("failure"), search_query="Service B", tenant_id=tenant)


def test_completed_experiment_cannot_be_overwritten_before_loading_models(tmp_path, monkeypatch):
    path = tmp_path / "experiment.json"
    original = b'{"status":"original measured experiment"}\n'
    path.write_bytes(original)
    monkeypatch.setattr("scripts.compare_reranker_contexts.OnnxCrossEncoder",
                        lambda *args, **kwargs: pytest.fail("No model load after existing receipt"))
    with pytest.raises(ValueError, match="IMMUTABLE_USE_VERIFY"):
        main(["--cases", str(tmp_path / "missing.json"), "--output-dir", str(tmp_path)])
    assert path.read_bytes() == original


def test_candidate_cap_is_checked_before_reranker_or_packer(review_context):
    view, prepared, tenant = prepare(review_context)
    oversized = SimpleNamespace(retrieved_chunk_ids=("bad",) * 41)
    fake = ReversedFixtureReranker()
    with pytest.raises(ValueError, match="CANDIDATE_BOUND"):
        compare_prepared(view, oversized, fake, search_query="Service B", tenant_id=tenant)
    assert fake.queries == []


@pytest.fixture
def saved_comparison(review_context, tmp_path, monkeypatch):
    view, prepared, tenant = prepare(review_context)
    fake = ReversedFixtureReranker()
    report = {"case_id": "synthetic", **compare_prepared(view, prepared, fake, search_query="Service B", tenant_id=tenant)}
    ids = list(prepared.retrieved_chunk_ids)
    collection, lexical, _, _, _ = review_context
    monkeypatch.setattr("scripts.compare_reranker_contexts.load_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr("scripts.compare_reranker_contexts.load_global_lexical_index", lambda *args, **kwargs: nullcontext(lexical))
    plan = pin_comparison_plan(tmp_path / "plan.json", {
        "created_at": "fixture", "collection_path": "fixture", "lexical_manifest_path": "fixture",
        "collection_manifest_sha256": collection.manifest_sha256,
        "reranker_recipe": fake.recipe.model_dump(mode="json"), "reranker_recipe_sha256": fake.recipe.fingerprint,
        "cases": [{"input": {"id": "synthetic"}, "candidate_chunk_ids": ids,
            "candidate_pool_sha256": sha256_bytes(canonical_json(ids)),
            "baseline_context_sha256": report["H"]["serialized_sha256"]}]})
    result = {"plan_sha256": plan["plan_sha256"], "cases": [report]}
    (tmp_path / "experiment.json").write_bytes(canonical_json(result) + b"\n")
    return tmp_path, result, fake


def test_saved_verification_reconstructs_source_and_context_without_running_models(saved_comparison):
    directory, _, fake = saved_comparison
    previous = fake.inference_calls
    receipt = verify_saved(directory)
    assert receipt["status"] == "PASS_SOURCE_RANK_CONTEXT_REBUILD" and receipt["cases"] == 1
    assert receipt["model_encoder_calls"] == receipt["paid_api_calls"] == receipt["aws_calls"] == 0
    assert fake.inference_calls == previous


@pytest.mark.parametrize("tamper,error", [
    ("foreign_window", "WINDOW_POOL"), ("too_many_windows", "WINDOW_POOL"),
    ("missing_windows", "WINDOW_COUNT"), ("uncovered_tail", "UNCOVERED_SOURCE"),
    ("score", "SCORE_AGGREGATION"), ("context", "CONTEXT_OR_PROVENANCE"),
    ("failed_case", "INCOMPLETE_CASE"), ("wrong_plan_hash", "PLAN_HASH"),
])
def test_saved_verifier_rejects_invalid_receipts(saved_comparison, tamper, error):
    directory, original, _ = saved_comparison
    result = copy.deepcopy(original)
    case = result["cases"][0]
    if tamper == "foreign_window":
        case["windows"].append({**case["windows"][0], "chunk_id": "foreign"})
    elif tamper == "too_many_windows":
        case["windows"] = [case["windows"][0]] * 161
    elif tamper == "missing_windows":
        case["windows"] = []
    elif tamper == "uncovered_tail":
        case["windows"][0]["end_char"] -= 1
    elif tamper == "score":
        case["scores"][0]["raw_logit"] += 1
    elif tamper == "context":
        case["R"]["sources"][0]["text"] = "Changed claim is not an original source slice"
    elif tamper == "failed_case":
        case["status"] = "FAILED_RERANKER_NO_FALLBACK"
    else:
        result["plan_sha256"] = "0" * 64
    (directory / "experiment.json").write_bytes(canonical_json(result) + b"\n")
    with pytest.raises(ValueError, match=error):
        verify_saved(directory)
    assert json.loads((directory / "experiment.json").read_bytes()) == result
