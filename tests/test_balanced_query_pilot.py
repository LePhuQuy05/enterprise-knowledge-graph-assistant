"""Synthetic pilot safety checks; no real retrieval, encoder, provider or AWS."""

import copy
import json
from contextlib import nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalRoutes
from app.model_gateway import CallContext
from app.rag_runner import run_methods
from scripts import run_balanced_query_pilot as pilot
from tests.test_collection_review import review_context as review_context
from tests.test_multi_query_runner import provider
from tests.test_multi_query_runner import q_setup as q_setup
from tests.test_rag_runner import fixture_gateway, request
from tests.test_rag_runner import setup_runner as setup_runner

REPO = Path(__file__).resolve().parents[1]


def test_exact_prior_inputs_have_new_q_qb_methods_and_durable_keys_without_targets():
    path = REPO / "configs/evaluation/multi-query-answer-pilot-v1.json"
    inputs, requests = pilot.load_cases(path)
    assert tuple(case["id"] for case in inputs["cases"]) == pilot.CASE_IDS
    assert [item.language for item in requests] == ["en", "vi", "en"]
    assert all(item.method_ids == pilot.METHODS and item.run_kind == "answer" for item in requests)
    assert [item.original_question for item in requests] == [case["question"] for case in inputs["cases"]]
    assert len({item.idempotency_key for item in requests}) == 3
    for item in requests:
        assert item.idempotency_key.startswith("qb-answer-v1-")
        assert pilot.SOURCE_CASES_SHA256[:16] in item.idempotency_key
        assert not set(item.model_dump()).intersection({"gold", "required_facts", "target_ids", "expected_answer"})


def test_modified_prior_question_rejected_even_before_a_new_plan_exists(tmp_path):
    data = json.loads((REPO / "configs/evaluation/multi-query-answer-pilot-v1.json").read_bytes())
    data["cases"][0]["question"] = "Outcome-driven easier replacement?"
    path = tmp_path / "cases.json"
    path.write_bytes(canonical_json(data))
    with pytest.raises(ValueError, match="SOURCE_CASES_PIN_MISMATCH"):
        pilot.load_cases(path)


@pytest.mark.parametrize("tamper", ["target", "methods", "languages", "ids", "gold", "count"])
def test_runtime_case_contract_rejects_targets_or_changed_design(tmp_path, monkeypatch, tamper):
    data = json.loads((REPO / "configs/evaluation/multi-query-answer-pilot-v1.json").read_bytes())
    if tamper == "target":
        data["cases"][0]["target_ids"] = ["Do not inject source labels"]
    elif tamper == "methods":
        data["cases"][0]["methods"].reverse()
    elif tamper == "languages":
        data["cases"][1]["language"] = "en"
    elif tamper == "ids":
        data["cases"][0]["id"] = "different-case"
    elif tamper == "gold":
        data["gold"] = True
    else:
        data["cases"].pop()
    path = tmp_path / "cases.json"
    path.write_bytes(canonical_json(data))
    monkeypatch.setattr(pilot, "SOURCE_CASES_SHA256", sha256_bytes(path.read_bytes()))
    with pytest.raises(ValueError):
        pilot.load_cases(path)


@pytest.fixture
def paired_report(q_setup):
    collection, lexical, kwargs = q_setup
    gateway = fixture_gateway(kwargs)
    provider(gateway)
    return run_methods(collection, lexical, request(run_kind="answer", method_ids=pilot.METHODS),
                       gateway=gateway, **kwargs)


def test_pair_check_accepts_actual_synthetic_runner_shared_planner_and_selector(paired_report):
    pilot.check_pair(paired_report)
    assert paired_report["methods"][0]["planning_trace"]["receipt"] == paired_report["methods"][1]["planning_trace"]["receipt"]
    assert paired_report["methods"][1]["planning_trace"]["cache_hit"]


@pytest.mark.parametrize("tamper,category", [
    ("schema", "METHOD_OR_SCHEMA"), ("method", "METHOD_OR_SCHEMA"),
    ("policy", "RUNTIME_POLICY"), ("pool", "SAME_FROZEN_POOL"),
    ("base", "SAME_FROZEN_POOL"), ("receipt", "SHARED_PLANNER"),
    ("subqueries", "SHARED_PLANNER"), ("cache", "SHARED_PLANNER"),
    ("planner_cost", "SHARED_PLANNER"), ("sharing", "SHARED_PLANNER"),
    ("common_config", "COMMON_CONFIGURATION"), ("partial", "PAIR_NOT_COMPLETED"),
    ("unknown_cost", "PAIR_NOT_COMPLETED"), ("failed_peer", "PAIR_NOT_COMPLETED"),
])
def test_pair_check_rejects_design_drift_or_failed_unknown_pair(paired_report, tamper, category):
    qb = paired_report["methods"][1]
    if tamper == "schema":
        qb["config"]["schema_version"] = 3
    elif tamper == "method":
        qb["effective_method"] = "hybrid"
    elif tamper == "policy":
        qb["config"]["context_selection_recipe"]["policy"] = "RR_SCOPED_EXACT_DEDUP"
    elif tamper == "pool":
        qb["multi_query_trace"]["ranked_chunk_ids"].reverse()
    elif tamper == "base":
        paired_report["search_query"] = "Changed base anchor"
    elif tamper == "receipt":
        qb["planning_trace"]["receipt"]["operation_id"] = "op-foreign"
    elif tamper == "subqueries":
        qb["planning_trace"]["subqueries"] = ["Different search"]
    elif tamper == "cache":
        qb["planning_trace"]["cache_hit"] = False
    elif tamper == "planner_cost":
        qb["planning_trace"]["new_cost_micro_usd"] = 1
    elif tamper == "sharing":
        paired_report["planner_sharing"] = "SEPARATE_MODEL_PLANS"
    elif tamper == "common_config":
        qb["config"]["max_output_tokens"] = 999
    elif tamper == "partial":
        paired_report["status"] = "PARTIAL"
    elif tamper == "failed_peer":
        qb["status"] = "FAILED"
    else:
        paired_report["cost_complete"] = False
    with pytest.raises(ValueError, match=category):
        pilot.check_pair(paired_report)


def test_typed_validation_rejects_changed_selection_trace_with_unchanged_hash_fields(paired_report):
    paired_report["methods"][1]["context_selection_trace"]["source_trace_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="CONTEXT_SELECTION_TRACE_MISMATCH"):
        pilot.check_pair(paired_report)


def test_pair_check_rejects_generator_outside_pinned_v9(paired_report):
    with pytest.raises(ValueError, match="GENERATOR_PIN_MISMATCH"):
        pilot.check_pair(paired_report, generation_recipe_sha256="0" * 64)


def inner_gateway(*, fail=False):
    calls = []

    def generate(context, **kwargs):
        calls.append((context, kwargs))
        if fail:
            raise TimeoutError("synthetic transport failure")
        return "synthetic-result"

    raw = SimpleNamespace(provider=SimpleNamespace(dispatch_attempts=0), routes=object(),
                          ledger=object(), generate=generate)
    return raw, calls


def context(stage, **changes):
    return CallContext(tenant_id="demo", run_id=pilot.RUN, request_id="synthetic-pilot", stage=stage,
                       **changes)


@pytest.mark.parametrize("stage", ["GENERATE", "PLAN_QUERY", "REWRITE"])
def test_stage_ceiling_is_enforced_before_inner_gateway_or_provider(stage):
    raw, calls = inner_gateway()
    gateway = pilot.BoundedPilotGateway(raw)
    for _ in range(pilot.STAGE_LIMITS[stage]):
        assert gateway.generate(context(stage), system="s", prompt="p",
                                max_output_tokens=pilot.OUTPUT_LIMITS[stage]) == "synthetic-result"
    with pytest.raises(ValueError, match="PRE_DISPATCH_CEILING_OR_SCOPE"):
        gateway.generate(context(stage), system="s", prompt="p", max_output_tokens=pilot.OUTPUT_LIMITS[stage])
    assert len(calls) == pilot.STAGE_LIMITS[stage] and gateway.halted


def test_total_ten_attempts_and_failure_halt_never_allow_automatic_retry():
    raw, calls = inner_gateway()
    gateway = pilot.BoundedPilotGateway(raw)
    for stage, limit in pilot.STAGE_LIMITS.items():
        for _ in range(limit):
            gateway.generate(context(stage), system="s", prompt="p", max_output_tokens=pilot.OUTPUT_LIMITS[stage])
    assert len(calls) == sum(gateway.stage_attempts.values()) == 10
    with pytest.raises(ValueError):
        gateway.generate(context("GENERATE"), system="s", prompt="p")
    assert len(calls) == 10
    raw, calls = inner_gateway(fail=True)
    gateway = pilot.BoundedPilotGateway(raw)
    with pytest.raises(TimeoutError):
        gateway.generate(context("PLAN_QUERY"), system="s", prompt="p", max_output_tokens=256)
    with pytest.raises(ValueError):
        gateway.generate(context("GENERATE"), system="s", prompt="p")
    assert len(calls) == 1 and gateway.halted


@pytest.mark.parametrize("bad_context,cap", [
    (context("EXTRACT"), 800), (context("GENERATE", attempt=2), 800),
    (CallContext(tenant_id="other", run_id=pilot.RUN, request_id="x", stage="GENERATE"), 800),
    (CallContext(tenant_id="demo", run_id="unapproved-run", request_id="x", stage="GENERATE"), 800),
    (context("GENERATE"), 801),
])
def test_unapproved_stage_scope_attempt_or_output_cap_never_reaches_gateway(bad_context, cap):
    raw, calls = inner_gateway()
    with pytest.raises(ValueError):
        pilot.BoundedPilotGateway(raw).generate(bad_context, system="s", prompt="p", max_output_tokens=cap)
    assert not calls


@pytest.mark.parametrize("guard", ["execution-started.json", "summary.json", "case-1.json"])
def test_existing_execution_stops_before_models_keys_provider_or_history(tmp_path, monkeypatch, guard):
    (tmp_path / guard).write_bytes(b"{}")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing execution must not reach setup or external systems.")

    for name in ("historical_artifact_hashes", "load_cases", "BgeEncoder", "load_collection", "_preflight"):
        monkeypatch.setattr(pilot, name, forbidden)
    with pytest.raises(ValueError, match="EXECUTION_ALREADY_STARTED_USE_REPLAY"):
        pilot.main(["--execute", "--output-directory", str(tmp_path)])


@pytest.fixture
def prepared_driver(tmp_path, monkeypatch):
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    cases = tmp_path / "configs/evaluation/multi-query-answer-pilot-v1.json"
    cases.parent.mkdir(parents=True)
    cases.write_bytes((REPO / "configs/evaluation/multi-query-answer-pilot-v1.json").read_bytes())
    routes = ExternalRoutes(account_model_metadata_sha256="0" * 64)
    route_path = tmp_path / "configs/models/external-direct-v1.json"
    route_path.parent.mkdir(parents=True)
    route_path.write_bytes(canonical_json(routes.model_dump(mode="json")))
    prices = json.loads((REPO / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    prices.update(checked_at=(datetime.now(UTC) - timedelta(days=1)).isoformat(),
                  expires_at=(datetime.now(UTC) + timedelta(days=1)).isoformat())
    prices_path = tmp_path / "configs/prices/external-direct-2026-10-24.json"
    prices_path.parent.mkdir(parents=True)
    prices_path.write_bytes(canonical_json(prices))
    lexical = tmp_path / "lexical.json"
    lexical.write_bytes(b"SYNTHETIC_LEXICAL_METADATA")
    pipeline = tmp_path / "results/corpus-expansion-v2/pipeline.json"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_bytes(canonical_json({"collection_path": str(tmp_path / "collection.json"),
                                         "lexical_manifest_path": str(lexical)}))
    for relative in pilot.HISTORICAL_DIRECTORIES:
        old = tmp_path / relative
        old.mkdir(parents=True)
        for name in pilot.HISTORICAL_NAMES:
            (old / name).write_bytes(b"HISTORICAL_BYTES_HASH_ONLY_NO_OUTCOME_PARSE")
    for relative in (*pilot.IMPLEMENTATIONS, "scripts/compare_context_allocations.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SYNTHETIC_IMPLEMENTATION_HASH_ONLY")
    fake_encoder = SimpleNamespace(inference_calls=0, recipe=lambda role: SimpleNamespace(
        model_id="synthetic", fingerprint=("d" if role == "document" else "q") * 64))
    monkeypatch.setattr(pilot, "BgeEncoder", lambda *args, **kwargs: fake_encoder)
    monkeypatch.setattr(pilot, "load_collection", lambda *args, **kwargs: SimpleNamespace(manifest_sha256="c" * 64))
    monkeypatch.setattr(pilot, "load_global_lexical_index", lambda *args, **kwargs: nullcontext(object()))
    for name in ("QueryPlanCache", "AnswerCache", "EmbeddingCache", "RewriteCache"):
        monkeypatch.setattr(pilot, name, lambda *args, **kwargs: object())
    raw, calls = inner_gateway()
    raw.routes = routes
    preflights = []

    def preflight(observed_routes, path):
        assert observed_routes == routes and path == prices_path
        preflights.append(True)
        return raw, {"provider": {"available": True}, "counters": {"synthetic": {"cap_micro_usd": 500000}}}

    monkeypatch.setattr(pilot, "_preflight", preflight)
    output = tmp_path / "results/context-allocation/answer-pilot-v1"
    return SimpleNamespace(root=tmp_path, output=output, raw=raw, calls=calls, preflights=preflights)


def test_plan_freezes_history_and_recipes_without_key_provider_or_model_dispatch(prepared_driver):
    fixture = prepared_driver
    assert pilot.main(["--output-directory", str(fixture.output)]) == 0
    assert not fixture.preflights and not fixture.calls
    plan_path = fixture.output / "plan.json"
    original = plan_path.read_bytes()
    plan = json.loads(original)
    assert len(plan["historical_artifact_sha256"]) == 18
    assert plan["context_selection_recipe"]["policy"] == "RR_ID_ONLY"
    assert plan["max_new_generation_calls"] == 6 and plan["max_new_planner_calls"] == 3
    assert plan["max_shared_vi_rewrites"] == 1 and plan["max_new_model_dispatches"] == 10
    assert plan["generation_recipe"]["version"] == "deepseek-grounded-answer-v9"
    assert plan["human_reviewed"] is plan["holdout"] is plan["gold"] is False
    assert pilot.main(["--output-directory", str(fixture.output)]) == 0
    assert plan_path.read_bytes() == original


def test_read_only_preflight_does_not_start_execution_or_call_runner(prepared_driver, monkeypatch):
    fixture = prepared_driver
    monkeypatch.setattr(pilot, "run_methods", lambda *args, **kwargs: pytest.fail("No inference during preflight."))
    assert pilot.main(["--preflight", "--output-directory", str(fixture.output)]) == 0
    assert len(fixture.preflights) == 1 and not fixture.calls
    assert not (fixture.output / "execution-started.json").exists()
    assert not (fixture.output / "summary.json").exists()
    receipt = json.loads(next(fixture.output.glob("preflight-*.json")).read_bytes())
    assert receipt["status"] == "READY_READ_ONLY_NO_INFERENCE"
    assert receipt["model_dispatch_attempts"] == receipt["ledger_writes"] == 0


@pytest.mark.parametrize("changed", ["history", "selector", "plan"])
def test_immutable_plan_rejects_changed_inputs_before_external_preflight(prepared_driver, changed):
    fixture = prepared_driver
    pilot.main(["--output-directory", str(fixture.output)])
    if changed == "history":
        path = fixture.root / pilot.HISTORICAL_DIRECTORIES[0] / "case-1.json"
    elif changed == "selector":
        path = fixture.root / "app/context_selection.py"
    else:
        path = fixture.output / "plan.json"
    path.write_bytes(b"{}" if changed == "plan" else b"CHANGED_PINNED_BYTES")
    with pytest.raises(ValueError, match="PLAN_PIN_MISMATCH"):
        pilot.main(["--execute", "--output-directory", str(fixture.output)])
    assert not fixture.preflights and not fixture.calls


def test_output_cannot_alias_historical_pilot_directory(prepared_driver):
    fixture = prepared_driver
    with pytest.raises(ValueError, match="OUTPUT_OVERLAPS_HISTORICAL"):
        pilot.historical_artifact_hashes(fixture.root / pilot.HISTORICAL_DIRECTORIES[0] / "new-child")


@pytest.mark.parametrize("phase", ["preflight", "failed_pair", "unknown_exception"])
def test_failures_stop_after_first_case_persist_receipts_and_never_resume(prepared_driver, monkeypatch, phase):
    fixture = prepared_driver
    observed = []
    if phase == "preflight":
        monkeypatch.setattr(pilot, "_preflight", lambda *args: (_ for _ in ()).throw(TimeoutError("synthetic")))

    def run(collection, lexical, item, **kwargs):
        assert (fixture.output / "plan.json").exists() and (fixture.output / "execution-started.json").exists()
        assert item.method_ids == pilot.METHODS and item.idempotency_key.startswith("qb-answer-v1-")
        assert kwargs["context_selection_recipe"].policy == "RR_ID_ONLY"
        observed.append(item)
        fixture.raw.provider.dispatch_attempts += 1
        if phase == "unknown_exception":
            raise TimeoutError("uncertain synthetic result")
        return {"status": "PARTIAL", "cost_complete": False, "methods": [], "new_known_cost_micro_usd": 137}

    monkeypatch.setattr(pilot, "run_methods", run)
    args = ["--execute", "--output-directory", str(fixture.output)]
    assert pilot.main(args) == 2
    if phase == "preflight":
        assert not observed and not (fixture.output / "execution-started.json").exists()
        receipt = json.loads(next(fixture.output.glob("preflight-*.json")).read_bytes())
        assert receipt["status"] == "BLOCKED_BEFORE_EXECUTION"
        return
    assert len(observed) == 1 and not (fixture.output / "case-2.json").exists()
    summary = json.loads((fixture.output / "summary.json").read_bytes())
    assert summary["status"] == "PARTIAL_STOPPED_NO_RETRY" and summary["cost_complete"] is False
    assert summary["new_known_cost_micro_usd"] == (137 if phase == "failed_pair" else 0)
    if phase == "failed_pair":
        assert json.loads((fixture.output / "case-1.json").read_bytes())["status"] == "PARTIAL"
    with pytest.raises(ValueError, match="EXECUTION_ALREADY_STARTED_USE_REPLAY"):
        pilot.main(args)
    assert len(observed) == 1


@pytest.mark.parametrize("reservation_change,halted", [(1, False), (0, True), (0, False), ("reset", False)])
def test_preflight_preserves_exact_legacy_reservations_and_blocks_new_or_reset_values(
        tmp_path, monkeypatch, reservation_change, halted):
    routes = ExternalRoutes(account_model_metadata_sha256="0" * 64)
    counters = {name: {"halted": halted,
        "reserved_micro_usd": 0 if reservation_change == "reset" else reserved + reservation_change,
        "settled_micro_usd": 100, "cap_micro_usd": 500000}
        for name, reserved in pilot.PRESERVED_RESERVATIONS.items()}
    original = copy.deepcopy(counters)
    reads = []

    def get(key):
        reads.append(key)
        return copy.deepcopy(counters[key[0]])

    ledger = SimpleNamespace(store=SimpleNamespace(get=get), run_key=lambda run: ("run", run),
        external_funding_key=lambda project: ("funding", project), month_key=lambda project, month: ("month", month))
    raw = SimpleNamespace(ledger=ledger)
    monkeypatch.setattr(pilot, "load_private_key", lambda *args: "SYNTHETIC_NOT_A_REAL_KEY")
    monkeypatch.setattr(pilot, "OfficialHttpsTransport", lambda *args: object())
    monkeypatch.setattr(pilot, "deepseek_preflight", lambda *args: {"available": True, "metadata_sha256": "0" * 64})
    monkeypatch.setattr(pilot, "managed_gateway", lambda **kwargs: raw)
    if reservation_change or halted:
        with pytest.raises(ValueError, match="LEDGER_NOT_READY_OR_PENDING_RESERVATION"):
            pilot._preflight(routes, tmp_path / "prices.json")
    else:
        assert pilot._preflight(routes, tmp_path / "prices.json")[0] is raw
    assert len(reads) == 3 and counters == original
