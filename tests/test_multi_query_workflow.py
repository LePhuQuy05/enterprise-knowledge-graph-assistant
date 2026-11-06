import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.contracts import canonical_json
from app.external_gateway import ExternalRoutes
from scripts import run_collection_methods as cli
from scripts import run_multi_query_pilot as pilot
from scripts.replay_collection_run import assert_replay


@pytest.mark.parametrize("method,expects_planner", [("hybrid_multi_query", True), ("hybrid", False), ("bm25", False)])
def test_cli_q_answer_creates_planner_only_when_requested(tmp_path, monkeypatch, method, expects_planner):
    monkeypatch.setattr(cli, "ROOT", tmp_path)
    routes_path = tmp_path / "configs/models/external-direct-v1.json"
    routes_path.parent.mkdir(parents=True)
    routes_path.write_text(ExternalRoutes(account_model_metadata_sha256="0" * 64).model_dump_json(), encoding="utf-8")
    pipeline_path = tmp_path / "pipeline.json"
    pipeline_path.write_bytes(canonical_json({"collection_path": str(tmp_path / "collection.json"),
                                              "lexical_manifest_path": str(tmp_path / "lexical.json")}))
    encoder = SimpleNamespace(inference_calls=0, recipe=lambda role: SimpleNamespace(model_id="synthetic"))
    model_loads, planned, observed = [], [], []

    def load_encoder(*args, **kwargs):
        model_loads.append(True)
        return encoder

    def forbidden(*args, **kwargs):
        pytest.fail("CLI cache-only planning cannot invoke managed gateway, reranker or provider.")

    def planner(path, *, namespace):
        value = SimpleNamespace(path=path, namespace=namespace)
        planned.append(value)
        return value

    def run(collection, lexical, request, **kwargs):
        observed.append((request, kwargs))
        return {"status": "COMPLETED", "methods": [{"effective_method": method,
                 "status": "COMPLETED", "cache_hit": True}], "new_known_cost_micro_usd": 0,
                "cloud_deployed": False}

    monkeypatch.setattr(cli, "BgeEncoder", load_encoder)
    monkeypatch.setattr(cli, "OnnxCrossEncoder", forbidden)
    monkeypatch.setattr(cli, "managed_gateway", forbidden)
    monkeypatch.setattr(cli, "QueryPlanCache", planner)
    monkeypatch.setattr(cli, "load_collection", lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "load_global_lexical_index", lambda *args, **kwargs: nullcontext(object()))
    for name in ("AnswerCache", "EmbeddingCache", "RewriteCache"):
        monkeypatch.setattr(cli, name, lambda *args, **kwargs: object())
    monkeypatch.setattr(cli, "run_methods", run)
    output = tmp_path / "report.json"
    assert cli.main(["Synthetic AWS question", "--pipeline", str(pipeline_path), "--methods", method,
        "--run-kind", "answer", "--idempotency-key", "workflow-fixture", "--output", str(output)]) == 0
    request, kwargs = observed[0]
    assert request.run_kind == "answer" and request.method_ids == (method,)
    assert kwargs["gateway"] is None and kwargs["reranker"] is None
    assert len(planned) == int(expects_planner)
    assert kwargs["query_plan_cache"] is (planned[0] if expects_planner else None)
    if planned:
        assert planned[0].namespace == "LIVE" and planned[0].path == tmp_path / "data/cache/collection-query-plans"
    assert bool(model_loads) == (method != "bm25")
    saved = json.loads(output.read_bytes())
    assert saved["model_dispatch_attempts_this_invocation"] == saved["local_query_onnx_calls_this_invocation"] == 0


def comparison_reports():
    common = {"collection_manifest_sha256": "c" * 64, "lexical_index_version": "lexical_fixture",
        "document_recipe_sha256": "d" * 64, "query_recipe_sha256": "q" * 64,
        "generation_recipe_sha256": "g" * 64, "max_context_chunks": 8,
        "context_proxy_token_budget": 4096, "max_output_tokens": 800, "candidates_per_branch": 20, "fusion": "RRF60"}
    hybrid = {"requested_method": "hybrid", "effective_method": "hybrid", "status": "COMPLETED",
        "config": common | {"method_id": "hybrid", "schema_version": 1}, "config_sha256": "h" * 64,
        "ranked_chunk_ids": ["chunk-a", "chunk-b"], "context_chunk_ids": ["chunk-a"],
        "context_sources": [{"chunk_id": "chunk-a", "text": "Synthetic source text."}],
        "response": {"mode_used": "hybrid", "answer": "Synthetic source."}, "usage": {"input_tokens": 20},
        "original_receipt_cost_micro_usd": 7, "cache_hit": False, "new_cost_micro_usd": 7}
    multi = copy.deepcopy(hybrid)
    multi.update(requested_method="hybrid_multi_query", effective_method="hybrid_multi_query",
        config=common | {"method_id": "hybrid_multi_query", "schema_version": 3,
                         "planner_recipe": {"version": "bounded-query-plan-v1", "max_subqueries": 2}},
        config_sha256="m" * 64, response={"mode_used": "hybrid_multi_query", "answer": "Synthetic source."},
        new_cost_micro_usd=10,
        multi_query_trace={"base_search_query": "Synthetic search", "queries": [{"kind": "BASE",
            "query": "Synthetic search", "ranked_chunk_ids": ["chunk-a", "chunk-b"]},
            {"kind": "SUBQUERY", "query": "AWS permissions", "ranked_chunk_ids": ["chunk-b"]}],
            "ranked_chunk_ids": ["chunk-a", "chunk-b"]},
        planning_trace={"recipe_sha256": "p" * 64, "subqueries": ["AWS permissions"],
            "cache_hit": False, "new_cost_micro_usd": 3, "receipt": {"operation_id": "op-plan-original",
            "recipe_sha256": "p" * 64, "raw_output": '{"subqueries":["AWS permissions"]}', "cost_micro_usd": 3}})
    original = {"status": "COMPLETED", "cost_complete": True, "run_identity": "fixture-run",
        "input": {"original_question": "Synthetic question", "method_ids": ["hybrid", "hybrid_multi_query"]},
        "collection_version": "fixture-collection", "collection_manifest_sha256": "c" * 64,
        "search_query": "Synthetic search", "query_transform": "NONE", "new_known_cost_micro_usd": 17,
        "methods": [hybrid, multi]}
    replay = copy.deepcopy(original)
    replay["new_known_cost_micro_usd"] = 0
    for row in replay["methods"]:
        row.update(cache_hit=True, new_cost_micro_usd=0)
    replay["methods"][1]["planning_trace"].update(cache_hit=True, new_cost_micro_usd=0)
    return original, replay


def test_q_replay_ignores_only_new_planner_cost_and_cache_hit_inside_planning_trace():
    original, replay = comparison_reports()
    assert_replay(original, replay)
    before = original["methods"][1]["planning_trace"]
    after = replay["methods"][1]["planning_trace"]
    assert {key for key in before if before[key] != after[key]} == {"cache_hit", "new_cost_micro_usd"}


@pytest.mark.parametrize("tamper", ["subqueries", "operation", "recipe", "body", "extra", "missing", "not_hit", "new_cost"])
def test_q_replay_rejects_changed_plan_identity_receipt_or_subqueries(tamper):
    original, replay = comparison_reports()
    planning = replay["methods"][1]["planning_trace"]
    if tamper == "subqueries":
        planning["subqueries"] = ["Changed task"]
    elif tamper == "operation":
        planning["receipt"]["operation_id"] = "op-other-request"
    elif tamper == "recipe":
        planning["recipe_sha256"] = "other-recipe"
    elif tamper == "body":
        planning["receipt"]["raw_output"] = '{"subqueries":[]}'
    elif tamper == "extra":
        planning["new_unreviewed_field"] = "tampered"
    elif tamper == "missing":
        replay["methods"][1].pop("planning_trace")
    elif tamper == "not_hit":
        planning["cache_hit"] = False
    else:
        planning["new_cost_micro_usd"] = 1
    with pytest.raises(ValueError, match="PLANNING_TRACE_MISMATCH"):
        assert_replay(original, replay)


def test_q_replay_compares_full_config_body_even_with_original_hash_field():
    original, replay = comparison_reports()
    replay["methods"][1]["config"]["planner_recipe"]["max_subqueries"] = 3
    assert replay["methods"][1]["config_sha256"] == original["methods"][1]["config_sha256"]
    with pytest.raises(ValueError, match="METHOD_MISMATCH"):
        assert_replay(original, replay)


@pytest.mark.parametrize("guard", ["execution-started.json", "summary.json"])
def test_existing_q_pilot_execution_guard_runs_before_models_keys_or_network(tmp_path, monkeypatch, guard):
    (tmp_path / guard).write_text("{}", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("An existing pilot must stop before model/cache setup, key reads or network.")

    for name in ("BgeEncoder", "QueryPlanCache", "load_collection", "load_private_key", "deepseek_preflight", "managed_gateway"):
        monkeypatch.setattr(pilot, name, forbidden)
    monkeypatch.setattr(pilot, "ROOT", tmp_path / "absent-config-root")
    with pytest.raises(ValueError, match="EXECUTION_ALREADY_STARTED_USE_REPLAY"):
        pilot.main(["--execute", "--output-directory", str(tmp_path)])


def test_q_pilot_accepts_matching_h_anchor_and_shared_configuration():
    original, _ = comparison_reports()
    pilot.check_pair(original)


@pytest.mark.parametrize("tamper", ["base_query", "base_ranking", "effective_method"])
def test_q_pilot_rejects_changed_base_anchor_or_fake_fallback(tamper):
    original, _ = comparison_reports()
    multi = original["methods"][1]
    if tamper == "base_query":
        multi["multi_query_trace"]["base_search_query"] = "Different question"
    elif tamper == "base_ranking":
        multi["multi_query_trace"]["queries"][0]["ranked_chunk_ids"] = ["other-source"]
    else:
        multi["effective_method"] = "hybrid"
    with pytest.raises(ValueError, match="BASE_ANCHOR_MISMATCH"):
        pilot.check_pair(original)


@pytest.mark.parametrize("field", ["collection_manifest_sha256", "lexical_index_version", "document_recipe_sha256",
    "query_recipe_sha256", "generation_recipe_sha256", "max_context_chunks", "context_proxy_token_budget",
    "max_output_tokens", "candidates_per_branch", "fusion"])
def test_q_pilot_rejects_different_common_configuration(field):
    original, _ = comparison_reports()
    original["methods"][1]["config"][field] = "DIFFERENT_CONFIG"
    with pytest.raises(ValueError, match="COMMON_CONFIGURATION_MISMATCH"):
        pilot.check_pair(original)


@pytest.mark.parametrize("tamper", ["partial", "incomplete_cost", "missing_peer"])
def test_q_pilot_stops_on_failed_or_incomplete_pair(tamper):
    original, _ = comparison_reports()
    if tamper == "partial":
        original["status"] = "PARTIAL"
    elif tamper == "incomplete_cost":
        original["cost_complete"] = False
    else:
        original["methods"].pop()
    with pytest.raises(ValueError, match="PAIR_NOT_COMPLETED_STOP_NO_RETRY"):
        pilot.check_pair(original)
