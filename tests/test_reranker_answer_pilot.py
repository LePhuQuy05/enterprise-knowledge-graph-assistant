import copy
import json
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalRoutes
from app.rag_methods import RunInputV1
from app.rag_runner import run_methods
from scripts import run_reranker_answer_pilot as pilot
from tests.test_collection_answering import FixtureReranker
from tests.test_collection_review import review_context as review_context
from tests.test_rag_runner import (
    fixture_gateway,
    install_provider,
)
from tests.test_rag_runner import (
    setup_runner as setup_runner,
)


def case_data():
    return {"purpose": "PREREGISTERED_DEVELOPMENT_ANSWER_PILOT_NOT_GOLD_OR_HOLDOUT",
        "human_reviewed": False, "gold": False, "holdout": False,
        "cases": [{"id": "synthetic-a", "question": "Service A?", "language": "en",
                   "methods": ["hybrid", "hybrid_reranked"]},
                  {"id": "synthetic-b", "question": "Service B?", "language": "en",
                   "methods": ["hybrid", "hybrid_reranked"]},
                  {"id": "synthetic-c", "question": "Dịch vụ C?", "language": "vi",
                   "methods": ["hybrid", "hybrid_reranked"]}]}


def case_file(tmp_path, data=None):
    path = tmp_path / "cases.json"
    path.write_bytes(canonical_json(case_data() if data is None else data))
    return path


def test_load_cases_has_exact_three_pairs_one_vi_and_no_review_targets_in_runtime_input(tmp_path):
    path = case_file(tmp_path)
    metadata, requests = pilot.load_cases(path)
    assert metadata == case_data() and len(requests) == 3
    assert [request.language for request in requests] == ["en", "en", "vi"]
    assert all(request.method_ids == ("hybrid", "hybrid_reranked") and request.run_kind == "answer" for request in requests)
    assert len({request.idempotency_key for request in requests}) == 3
    assert all(sha256_bytes(path.read_bytes())[:16] in request.idempotency_key for request in requests)
    for request in requests:
        dumped = request.model_dump(mode="json")
        assert not set(dumped).intersection({"id", "gold", "expected_facts", "target_ids", "review_focus", "answer"})
        assert dumped["original_question"] in {case["question"] for case in metadata["cases"]}
    assert [request.model_dump() for request in pilot.load_cases(path)[1]] == [request.model_dump() for request in requests]


@pytest.mark.parametrize("extra", ["expected_facts", "target_ids", "gold_answer", "review_focus"])
def test_load_cases_rejects_expected_answers_and_targets_on_a_runtime_case(tmp_path, extra):
    data = case_data()
    data["cases"][0][extra] = ["DO_NOT_LEAK_TO_RANKER_OR_GENERATOR"]
    with pytest.raises(ValueError, match="RUNTIME_INPUT_FIELDS_INVALID"):
        pilot.load_cases(case_file(tmp_path, data))


@pytest.mark.parametrize("change", ["two", "four", "no_vi", "two_vi", "auto", "human", "gold", "holdout", "methods"])
def test_preregistration_requires_exact_count_languages_and_not_gold_or_holdout(tmp_path, change):
    data = case_data()
    if change == "two":
        data["cases"].pop()
    elif change == "four":
        data["cases"].append(copy.deepcopy(data["cases"][0]) | {"id": "synthetic-d"})
    elif change == "no_vi":
        data["cases"][2]["language"] = "en"
    elif change == "two_vi":
        data["cases"][0]["language"] = "vi"
    elif change == "auto":
        data["cases"][0]["language"] = "auto"
    elif change == "human":
        data["human_reviewed"] = True
    elif change in {"gold", "holdout"}:
        data[change] = True
    else:
        data["cases"][0]["methods"] = ["hybrid_reranked", "hybrid"]
    with pytest.raises(ValueError):
        pilot.load_cases(case_file(tmp_path, data))


@pytest.mark.parametrize("identifier", ["", "   ", "synthetic-b", None])
def test_preregistration_rejects_empty_or_duplicate_case_ids(tmp_path, identifier):
    data = case_data()
    data["cases"][0]["id"] = identifier
    with pytest.raises(ValueError, match="CASE_ID_INVALID"):
        pilot.load_cases(case_file(tmp_path, data))


@pytest.fixture
def paired_report(setup_runner):
    collection, lexical, kwargs = setup_runner
    gateway = fixture_gateway(kwargs)
    install_provider(gateway, [])
    request = RunInputV1(original_question="Service B", method_ids=("hybrid", "hybrid_reranked"),
                         run_kind="answer", idempotency_key="pilot-driver-fixture")
    return run_methods(collection, lexical, request, reranker=FixtureReranker(), gateway=gateway, **kwargs)


def test_pair_check_accepts_actual_common_runner_h_r_fixture_contract(paired_report):
    pilot.check_pair(paired_report)


def test_pair_check_rejects_a_different_candidate_pool(paired_report):
    paired_report["methods"][1]["rerank_trace"]["baseline_candidate_ids"] = ["chk_foreign_pool"]
    with pytest.raises(ValueError, match="SAME_POOL_MISMATCH"):
        pilot.check_pair(paired_report)


@pytest.mark.parametrize("field", ["collection_manifest_sha256", "lexical_index_version", "document_recipe_sha256",
    "query_recipe_sha256", "generation_recipe_sha256", "max_context_chunks", "context_proxy_token_budget",
    "max_output_tokens", "candidates_per_branch", "fusion"])
def test_pair_check_rejects_changed_common_configuration(paired_report, field):
    paired_report["methods"][1]["config"][field] = "FORGED_DIFFERENT_CONFIGURATION"
    with pytest.raises(ValueError, match="COMMON_CONFIGURATION_MISMATCH"):
        pilot.check_pair(paired_report)


@pytest.mark.parametrize("guard", ["execution-started.json", "summary.json"])
def test_existing_execution_guard_blocks_before_models_gateway_or_cases_are_loaded(tmp_path, monkeypatch, guard):
    output = tmp_path / "pilot"
    output.mkdir()
    (output / guard).write_text("{}", encoding="utf-8")

    def forbidden(*args, **kwargs):
        pytest.fail("Existing execution must be blocked before local setup or managed gateway.")

    for name in ("load_cases", "BgeEncoder", "OnnxCrossEncoder", "load_collection", "managed_gateway"):
        monkeypatch.setattr(pilot, name, forbidden)
    with pytest.raises(ValueError, match="EXECUTION_ALREADY_STARTED_USE_REPLAY"):
        pilot.main(["--execute", "--cases", str(tmp_path / "does-not-exist.json"), "--output-directory", str(output)])


def test_write_once_never_overwrites_an_existing_case_or_summary(tmp_path):
    path = tmp_path / "case.json"
    pilot.write_once(path, {"status": "FAILED", "raw_receipt_preserved": True})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        pilot.write_once(path, {"status": "COMPLETED"})
    assert path.read_bytes() == original


@pytest.mark.parametrize("failure_phase", ["method", "gateway_setup"])
def test_failed_pair_is_persisted_and_stops_later_cases_without_automatic_resume(tmp_path, monkeypatch, failure_phase):
    # Exercise driver ordering with synthetic dependencies only. Neither model
    # constructors nor AWS/provider clients can be reached through these fakes.
    monkeypatch.setattr(pilot, "ROOT", tmp_path)
    paths = {"collection_path": str(tmp_path / "collection.json"),
             "lexical_manifest_path": str(tmp_path / "lexical.json")}
    pipeline = tmp_path / "results/corpus-expansion-v2/pipeline.json"
    pipeline.parent.mkdir(parents=True)
    pipeline.write_bytes(canonical_json(paths))
    prices = tmp_path / "configs/prices/external-direct-2026-10-24.json"
    prices.parent.mkdir(parents=True)
    prices.write_bytes(b"SYNTHETIC_PRICES_HASH_ONLY")
    preflight = tmp_path / "results/reranker-answering/preflight-summary.json"
    preflight.parent.mkdir(parents=True)
    preflight.write_bytes(canonical_json({"status": "READY_FOR_ROOT_SCOPED_SEQUENTIAL_PILOT_DECISION",
                                          "price_sheet_sha256": sha256_bytes(prices.read_bytes())}))
    routes_path = tmp_path / "configs/models/external-direct-v1.json"
    routes_path.parent.mkdir(parents=True)
    routes_path.write_text(ExternalRoutes(account_model_metadata_sha256="0" * 64).model_dump_json(), encoding="utf-8")
    fake_encoder = SimpleNamespace(inference_calls=0, recipe=lambda role: SimpleNamespace(model_id="synthetic", fingerprint="q" * 64))
    fake_reranker = SimpleNamespace(inference_calls=0, recipe=SimpleNamespace(fingerprint="r" * 64))
    monkeypatch.setattr(pilot, "BgeEncoder", lambda *args, **kwargs: fake_encoder)
    monkeypatch.setattr(pilot, "OnnxCrossEncoder", lambda *args, **kwargs: fake_reranker)
    monkeypatch.setattr(pilot, "load_collection", lambda *args, **kwargs: SimpleNamespace(manifest_sha256="c" * 64))
    monkeypatch.setattr(pilot, "load_global_lexical_index", lambda *args, **kwargs: nullcontext(object()))
    for name in ("EmbeddingCache", "AnswerCache", "RewriteCache"):
        monkeypatch.setattr(pilot, name, lambda *args, **kwargs: object())
    provider = SimpleNamespace(dispatch_attempts=0)
    observed = []

    def managed(*, prices_path):
        assert prices_path == prices
        if failure_phase == "gateway_setup":
            raise ValueError("Synthetic authentication failure before model dispatch")
        return SimpleNamespace(provider=provider)

    def failed(*args, **kwargs):
        provider.dispatch_attempts += 1
        observed.append(args[2])
        return {"status": "PARTIAL", "cost_complete": False, "methods": [], "new_known_cost_micro_usd": 137}

    monkeypatch.setattr(pilot, "managed_gateway", managed)
    monkeypatch.setattr(pilot, "run_methods", failed)
    output = tmp_path / "pilot-output"
    arguments = ["--execute", "--cases", str(case_file(tmp_path)), "--output-directory", str(output)]
    assert pilot.main(arguments) == 2
    if failure_phase == "gateway_setup":
        assert not observed and provider.dispatch_attempts == 0
        assert not (output / "execution-started.json").exists() and not (output / "summary.json").exists()
        failures = list(output.glob("setup-failure-*.json"))
        assert len(failures) == 1
        failure = json.loads(failures[0].read_bytes())
        assert failure["status"] == "BLOCKED_BEFORE_EXECUTION" and failure["model_dispatch_attempts"] == 0
        assert failure["ledger_writes"] == 0 and failure["execution_started"] is False
        return
    assert len(observed) == 1 and provider.dispatch_attempts == 1
    assert (output / "execution-started.json").exists() and (output / "case-1.json").exists()
    assert not (output / "case-2.json").exists()
    saved = json.loads((output / "case-1.json").read_bytes())
    summary = json.loads((output / "summary.json").read_bytes())
    assert saved["status"] == "PARTIAL" and saved["new_known_cost_micro_usd"] == 137
    assert summary["status"] == "PARTIAL_STOPPED_NO_RETRY" and summary["new_known_cost_micro_usd"] == 137
    assert not summary["cost_complete"]
    with pytest.raises(ValueError, match="EXECUTION_ALREADY_STARTED_USE_REPLAY"):
        pilot.main(arguments)
    assert len(observed) == provider.dispatch_attempts == 1
