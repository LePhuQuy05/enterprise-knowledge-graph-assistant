import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.external_gateway import ExternalModelGateway
from app.ledger import Change, Ledger
from app.owned_runs import OwnedRunRepository
from app.query_auth import ApiError, AuthContext
from app.query_state import QueryRepository
from app.run_answer_executor import CollectionOwnedAnswerExecutor
from app.run_artifacts import LocalRunBlobs, RunArtifactRepository
from app.run_budget import RunBudgetController
from app.run_execution import OwnedRunWorker
from app.run_service import OwnedRunService
from app.run_storage import SQLiteRunStore
from tests.test_collection_answering import FixtureReranker
from tests.test_collection_review import review_context as review_context
from tests.test_external_pilot import configured
from tests.test_rag_runner import setup_runner as setup_runner


@pytest.fixture
def answers(setup_runner, tmp_path):
    collection, lexical, runtime = setup_runner
    runtime = {k: v for k, v in runtime.items() if k not in {"tenant_id", "owner_id", "run_id"}}
    original = configured()
    prices = original.prices.model_copy(update={"rates": tuple(rate.model_copy(update={
        "input_micro_usd_per_unit": 10_000, "output_micro_usd_per_unit": 10_000}) for rate in original.prices.rates)})
    clock = [prices.checked_at + timedelta(seconds=1)]
    store = SQLiteRunStore(tmp_path / "all-state.sqlite3")
    ledger = Ledger(store, clock=lambda: clock[0])
    ledger.initialize_month("kg-r72", clock[0].strftime("%Y-%m"))
    ledger.initialize_run("kg-r72", collection.manifest.tenant_id, "owned-fixture", cap_micro_usd=500_000)
    ledger.initialize_external_funding("kg-r72", cap_micro_usd=2_000_000)
    gateway = ExternalModelGateway(ledger, original.provider, original.routes, prices)
    controller = RunBudgetController(gateway, budget_run_id="owned-fixture")
    repo = OwnedRunRepository(store, clock=lambda: clock[0], answer_budget=controller)
    auth = AuthContext(collection.manifest.tenant_id, "11111111-1111-1111-1111-111111111111")
    flags = {"disabled": False, "invalid": False, "timeout": False, "rewrite_invalid": False}
    def authorize(candidate):
        if candidate != auth or flags["disabled"]:
            raise ApiError(403, "ACCESS_DENIED")
    runtime["routes"] = gateway.routes
    runtime["reranker"] = FixtureReranker()
    executor = CollectionOwnedAnswerExecutor(collection, lexical, controller=controller, repository=repo,
        authorize=authorize, **runtime)
    artifacts = RunArtifactRepository(LocalRunBlobs(tmp_path / "private"))
    service = OwnedRunService(repo, artifacts, lambda: executor.release,
        authorize=authorize, release_check=executor.check_release)
    worker = OwnedRunWorker(repo, artifacts, executor, authorize=authorize, release_check=executor.check_release)
    seen = []
    def provider(provider, method, path, body):
        gateway.provider.calls += 1
        request = json.loads(body)
        system, prompt = request["messages"]
        prompt = json.loads(prompt["content"])
        seen.append(prompt)
        if flags["timeout"]:
            raise TimeoutError("SYNTHETIC_PRIVATE_ERROR")
        if "untrusted_sources" in prompt:
            first = prompt["untrusted_sources"][0]
            text = "Dịch vụ B." if "Vietnamese" in system["content"] else "Synthetic Service B."
            content = json.dumps({"answer_status": "ANSWERED", "reason": "NONE", "claims": [{
                "text": text, "evidence": [{"source_id": first["source_id"], "span_id": first["spans"][0]["span_id"]}]}]})
            if flags["invalid"]:
                content = "not valid JSON"
        else:
            content = json.dumps({"search_query": "Service B"}) if not flags["rewrite_invalid"] else "invalid rewrite"
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}
    gateway.provider.request = provider
    return SimpleNamespace(repo=repo, ledger=ledger, controller=controller, gateway=gateway, executor=executor,
        service=service, worker=worker, artifacts=artifacts, auth=auth, clock=clock, flags=flags, seen=seen)


def submit(s, methods=("hybrid",), *, language="en", key=None):
    return s.service.submit(s.auth, {"original_question": "Dịch vụ B là gì?" if language == "vi" else "Service B",
        "method_ids": list(methods), "run_kind": "answer", "language": language}, key or str(uuid4()))


def messages(s, run_id):
    return [r["message"] for r in s.repo.outboxes(s.auth, run_id)]


def test_parent_hold_is_atomic_and_allocates_without_double_reservation(answers):
    s = answers
    row = submit(s, ("bm25", "vector", "hybrid"))
    quoted = row["budget"]["max_cost_micro_usd"]
    assert s.gateway.provider.calls == 0
    run_key = s.ledger.run_key("owned-fixture")
    assert s.repo.store.get(run_key)["reserved_micro_usd"] == quoted
    assert len(row["budget"]["stages"]) == 3
    for i, msg in enumerate(messages(s, row["run_id"]), 1):
        assert s.worker.process(msg) == "ACK"
        assert s.gateway.provider.calls == i
    final = s.service.detail(s.auth, row["run_id"])
    assert final["status"] == "COMPLETED" and final["budget"]["cost_complete"]
    assert final["budget"]["reserved_micro_usd"] == s.repo.store.get(run_key)["reserved_micro_usd"] == 0
    assert final["budget"]["settled_micro_usd"] == s.repo.store.get(run_key)["settled_micro_usd"] == 6
    assert QueryRepository(s.repo.store, clock=s.repo.clock).quota_remaining() == 297


def test_insufficient_group_budget_admits_nothing_and_preserves_existing_obligations(answers):
    s = answers
    key = s.ledger.run_key("owned-fixture")
    counter = s.repo.store.get(key)
    s.repo.store.commit([Change(key, counter["version"], counter | {"version": counter["version"] + 1,
        "reserved_micro_usd": 490_000})])
    with pytest.raises(ApiError, match="BUDGET_LIMIT"):
        submit(s, ("bm25", "vector", "hybrid"))
    assert s.repo.store.get(key)["reserved_micro_usd"] == 490_000
    assert QueryRepository(s.repo.store, clock=s.repo.clock).quota_remaining() == 300
    assert s.repo.store.run_history_keys(s.auth.partition, limit=20) == ([], None)
    assert s.gateway.provider.calls == 0


@pytest.mark.parametrize("methods,language", [(("vector", "hybrid"), "en"), (("hybrid",), "vi")])
def test_published_rate_bounds_do_not_fit_existing_half_dollar_allowance(answers, methods, language):
    s = answers
    # Same numeric Flash rates as the verified live sheet, explicitly still a fixture provider/sheet.
    s.gateway.prices = s.gateway.prices.model_copy(update={"rates": tuple(rate.model_copy(update={
        "input_micro_usd_per_unit": 300_000, "output_micro_usd_per_unit": 1_200_000})
        if rate.provider == "DEEPSEEK" else rate for rate in s.gateway.prices.rates)})
    before = s.repo.store.get(s.ledger.run_key("owned-fixture"))
    with pytest.raises(ApiError, match="BUDGET_LIMIT"):
        submit(s, methods, language=language)
    assert s.repo.store.get(s.ledger.run_key("owned-fixture")) == before
    assert QueryRepository(s.repo.store, clock=s.repo.clock).quota_remaining() == 300
    assert s.gateway.provider.calls == 0


def test_vi_rewrite_is_shared_once_and_receipts_precede_generation_validation(answers):
    s = answers
    row = submit(s, ("bm25", "vector", "hybrid"), language="vi")
    assert len(row["budget"]["stages"]) == 4
    for msg in messages(s, row["run_id"]):
        assert s.worker.process(msg) == "ACK"
    assert s.gateway.provider.calls == 4
    assert sum("untrusted_sources" not in p for p in s.seen) == 1
    for method in ("bm25", "vector", "hybrid"):
        result = s.service.public_result(s.auth, row["run_id"], method)
        assert result["response"]["answer_language"] == "vi" and result["response"]["citations"]
        assert "raw_output" not in json.dumps(result)
        stage = s.repo.store.get((s.auth.partition, f"RUNSTAGE#{row['run_id']}#GENERATE:{method}"))
        assert stage["status"] == "RECEIVED" and stage["receipt"]["raw_output"]
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["budget"]["settled_micro_usd"] == 8 and detail["query_transform"] == "SHARED_REWRITE_SETTLED"
    assert s.worker.process(messages(s, row["run_id"])[0]) == "ACK" and s.gateway.provider.calls == 4


def test_replay_and_concurrent_duplicate_never_rehold_or_regenerate(answers):
    s = answers
    key = str(uuid4())
    with ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda _: submit(s, key=key), range(6)))
    assert sum(not r["idempotent_replay"] for r in rows) == 1
    row = rows[0]
    s.worker.process(messages(s, row["run_id"])[0])
    before = s.repo.store.get(s.ledger.run_key("owned-fixture"))
    assert submit(s, key=key)["idempotent_replay"]
    assert s.worker.process(messages(s, row["run_id"])[0]) == "ACK"
    assert s.repo.store.get(s.ledger.run_key("owned-fixture")) == before and s.gateway.provider.calls == 1


@pytest.mark.parametrize("shared", [False, True])
def test_invalid_json_is_chargeable_receipted_failed_and_never_retried(answers, shared):
    s = answers
    s.flags["rewrite_invalid" if shared else "invalid"] = True
    row = submit(s, language="vi" if shared else "en")
    msg = messages(s, row["run_id"])[0]
    assert s.worker.process(msg) == "ACK"
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["status"] == "FAILED" and detail["budget"]["settled_micro_usd"] == 2
    assert detail["budget"]["cost_complete"] and detail["budget"]["reserved_micro_usd"] == 0
    assert s.worker.process(msg) == "ACK" and s.gateway.provider.calls == 1
    slot = "REWRITE:shared" if shared else "GENERATE:hybrid"
    assert s.repo.store.get((s.auth.partition, f"RUNSTAGE#{row['run_id']}#{slot}"))["status"] == "RECEIVED"


def test_uncertain_transport_retains_spend_and_active_obligations(answers):
    s = answers
    s.flags["timeout"] = True
    row = submit(s)
    msg = messages(s, row["run_id"])[0]
    assert s.worker.process(msg) == "ACK"
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["status"] == "UNKNOWN" and not detail["budget"]["cost_complete"]
    assert detail["budget"]["reserved_micro_usd"] == detail["budget"]["max_cost_micro_usd"]
    assert s.repo.store.get(s.repo.active_keys(s.auth)[0])["count"] == 1
    assert "SYNTHETIC_PRIVATE_ERROR" not in json.dumps(detail)
    assert s.worker.process(msg) == "ACK" and s.gateway.provider.calls == 1


def test_cancel_queued_parent_releases_only_provably_unused_holds(answers):
    s = answers
    row = submit(s, ("bm25", "vector", "hybrid"), language="vi")
    result = s.service.cancel(s.auth, row["run_id"])
    assert result["status"] == "CANCELLED" and result["budget"]["reserved_micro_usd"] == 0
    assert s.repo.store.get(s.ledger.run_key("owned-fixture"))["reserved_micro_usd"] == 0
    assert s.gateway.provider.calls == 0


def test_halt_after_admission_blocks_provider_and_releases_uncalled_holds(answers):
    s = answers
    row = submit(s)
    key = s.ledger.run_key("owned-fixture")
    counter = s.repo.store.get(key)
    s.repo.store.commit([Change(key, counter["version"], counter | {"version": counter["version"] + 1, "halted": True})])
    assert s.worker.process(messages(s, row["run_id"])[0]) == "ACK"
    assert s.gateway.provider.calls == 0
    assert s.repo.store.get(key)["halted"] and s.repo.store.get(key)["reserved_micro_usd"] == 0


def test_shared_rewrite_in_progress_returns_retry_without_starting_child(answers):
    s = answers
    row = submit(s, ("bm25", "hybrid"), language="vi")
    entered, release = threading.Event(), threading.Event()
    original = s.gateway.provider.request
    def delayed(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)
    s.gateway.provider.request = delayed
    msgs = messages(s, row["run_id"])
    with ThreadPoolExecutor(max_workers=2) as pool:
        # Each real worker owns its lexical connection. Exercise the shared stage
        # concurrently, then use this fixture's lexical connection on its owning thread.
        first = pool.submit(s.executor.prepare, s.auth, s.repo.get(s.auth, row["run_id"]))
        assert entered.wait(5)
        assert s.worker.process(msgs[1]) == "RETRY"
        assert s.repo.child(s.auth, row["run_id"], "hybrid")["status"] == "QUEUED"
        release.set()
        assert first.result() == ("Service B", "vi")
    assert s.worker.process(msgs[0]) == "ACK"
    assert s.worker.process(msgs[1]) == "ACK" and s.gateway.provider.calls == 3


def test_crash_in_shared_call_becomes_unknown_without_regeneration(answers):
    s = answers
    row = submit(s, language="vi")
    def crash(*args):
        s.gateway.provider.calls += 1
        raise SystemExit("SIMULATED_PROCESS_DEATH")
    s.gateway.provider.request = crash
    msg = messages(s, row["run_id"])[0]
    with pytest.raises(SystemExit):
        s.worker.process(msg)
    assert s.repo.child(s.auth, row["run_id"], "hybrid")["status"] == "QUEUED"
    s.clock[0] += timedelta(seconds=121)
    assert s.worker.process(msg) == "ACK"
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["status"] == "UNKNOWN" and not detail["budget"]["cost_complete"]
    assert detail["budget"]["reserved_micro_usd"] < detail["budget"]["max_cost_micro_usd"]
    assert s.worker.process(msg) == "ACK" and s.gateway.provider.calls == 1


def test_terminal_redelivery_repairs_interrupted_hold_cleanup(answers, monkeypatch):
    s = answers
    row = submit(s, language="vi")
    msg = messages(s, row["run_id"])[0]
    original = s.controller.release_unused
    def unavailable(*args, **kwargs):
        raise OSError("SIMULATED_CLEANUP_OUTAGE")
    monkeypatch.setattr(s.controller, "release_unused", unavailable)
    with pytest.raises(OSError):
        s.repo.finish(s.auth, row["run_id"], "hybrid", "CANCELLED")
    assert s.repo.summary(s.auth, row["run_id"])["status"] == "CANCELLED"
    monkeypatch.setattr(s.controller, "release_unused", original)
    assert s.worker.process(msg) == "ACK"
    assert s.service.detail(s.auth, row["run_id"])["budget"]["reserved_micro_usd"] == 0
    assert s.gateway.provider.calls == 0


def test_revocation_between_hold_transfer_and_dispatch_never_calls_provider(answers, monkeypatch):
    s = answers
    row = submit(s)
    original = s.controller.allocate
    def revoke(*args):
        operation = original(*args)
        s.flags["disabled"] = True
        return operation
    monkeypatch.setattr(s.controller, "allocate", revoke)
    assert s.worker.process(messages(s, row["run_id"])[0]) == "ACK"
    assert s.gateway.provider.calls == 0
    assert s.repo.store.get(s.ledger.run_key("owned-fixture"))["reserved_micro_usd"] == 0
    with pytest.raises(ApiError, match="ACCESS_DENIED"):
        s.service.detail(s.auth, row["run_id"])


def test_cancel_during_final_call_keeps_paid_result_without_starting_queued_child(answers):
    s = answers
    row = submit(s, ("bm25", "hybrid"))
    original = s.gateway.provider.request
    def cancel_during_call(*args):
        s.service.cancel(s.auth, row["run_id"])
        return original(*args)
    s.gateway.provider.request = cancel_during_call
    for msg in messages(s, row["run_id"]):
        assert s.worker.process(msg) == "ACK"
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["status"] == "PARTIAL"
    assert [r["status"] for r in detail["methods"]] == ["COMPLETED", "CANCELLED"]
    assert detail["budget"]["settled_micro_usd"] == 2 and detail["budget"]["reserved_micro_usd"] == 0
    assert s.service.public_result(s.auth, row["run_id"], "bm25")["response"]["citations"]
    assert s.gateway.provider.calls == 1


@pytest.mark.parametrize("corruption", ["checksum", "shape", "ledger_binding"])
def test_shared_receipt_corruption_blocks_remaining_methods_without_new_calls(answers, corruption):
    from app.contracts import canonical_json, sha256_bytes
    from app.owned_runs import transition

    s = answers
    row = submit(s, ("bm25", "hybrid"), language="vi")
    s.executor.prepare(s.auth, s.repo.get(s.auth, row["run_id"]))
    key = s.auth.partition, f"RUNSTAGE#{row['run_id']}#REWRITE:shared"
    stage = s.repo.store.get(key)
    payload = stage["receipt"] | {"raw_output": "tampered"}
    if corruption == "shape":
        payload["usage"] = {"input_tokens": True, "output_tokens": 10}
    if corruption == "ledger_binding":
        payload["cost_micro_usd"] += 1
    s.repo.store.commit([transition(stage, receipt=payload, receipt_sha256=(
        stage["receipt_sha256"] if corruption == "checksum" else sha256_bytes(canonical_json(payload))))])
    for msg in messages(s, row["run_id"]):
        assert s.worker.process(msg) == "ACK"
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["status"] == "FAILED" and detail["budget"]["cost_complete"]
    assert s.gateway.provider.calls == 1


def test_month_rollover_blocks_new_dispatch_but_refunds_original_period(answers):
    s = answers
    s.clock[0] = s.clock[0].replace(day=31, hour=23, minute=59, second=50)
    s.gateway.prices = s.gateway.prices.model_copy(update={
        "checked_at": s.clock[0] - timedelta(days=1), "expires_at": s.clock[0] + timedelta(days=1)})
    row = submit(s)
    month = s.ledger._now().strftime("%Y-%m")
    s.clock[0] += timedelta(seconds=20)
    assert s.worker.process(messages(s, row["run_id"])[0]) == "ACK"
    assert s.repo.store.get(s.ledger.month_key("kg-r72", month))["reserved_micro_usd"] == 0
    assert s.gateway.provider.calls == 0


def test_no_evidence_localizes_abstention_and_refunds_uncalled_generation(answers):
    s = answers
    original = s.gateway.provider.request
    def unknown_rewrite(provider, method, path, body):
        reply = original(provider, method, path, body)
        reply["choices"][0]["message"]["content"] = '{"search_query":"zzzz_missing_term"}'
        return reply
    s.gateway.provider.request = unknown_rewrite
    row = submit(s, ("bm25",), language="vi")
    assert s.worker.process(messages(s, row["run_id"])[0]) == "ACK"
    result = s.service.public_result(s.auth, row["run_id"], "bm25")
    assert result["response"]["answer_status"] == "ABSTAINED"
    assert result["response"]["answer"].startswith("Tài liệu")
    assert not result["response"]["citations"] and s.gateway.provider.calls == 1
    assert s.service.detail(s.auth, row["run_id"])["budget"]["settled_micro_usd"] == 2


def test_expired_stage_lease_between_allocation_and_dispatch_is_fenced(answers, monkeypatch):
    s = answers
    row = submit(s, language="vi")
    original = s.controller.allocate
    def expire_after_allocation(*args):
        operation = original(*args)
        s.clock[0] += timedelta(seconds=121)
        return operation
    monkeypatch.setattr(s.controller, "allocate", expire_after_allocation)
    assert s.worker.process(messages(s, row["run_id"])[0]) == "ACK"
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["status"] == "FAILED" and detail["budget"]["reserved_micro_usd"] == 0
    assert s.gateway.provider.calls == 0


def test_receipt_replay_needs_no_current_price_for_completed_rewrite(answers):
    s = answers
    row = submit(s, language="vi")
    parent = s.repo.get(s.auth, row["run_id"])
    assert s.executor.prepare(s.auth, parent) == ("Service B", "vi")
    # A saved settled stage is readable even when no new stage can pass the current rate guard.
    s.gateway.prices = s.gateway.prices.model_copy(update={"expires_at": s.clock[0] - timedelta(seconds=1)})
    assert s.executor.prepare(s.auth, parent) == ("Service B", "vi")
    assert s.gateway.provider.calls == 1


def test_lexical_connection_is_not_shared_across_worker_threads(answers):
    import sqlite3

    with ThreadPoolExecutor(max_workers=1) as pool:
        with pytest.raises(sqlite3.ProgrammingError, match="same thread"):
            pool.submit(answers.executor.lexical.search, "Service B").result()


def test_owned_hybrid_and_reranked_answers_use_common_runner_context_and_pins(answers):
    from app.rag_methods import RunInputV1
    from app.rag_runner import run_methods

    s = answers
    methods = ("hybrid", "hybrid_reranked")
    baseline = run_methods(s.executor.collection, s.executor.lexical,
        RunInputV1(original_question="Service B", method_ids=methods, run_kind="retrieval", idempotency_key="fixture-fairness"),
        tenant_id=s.auth.tenant_id, owner_id=s.auth.user_id, run_id="owned-fixture", **s.executor.runtime)
    row = submit(s, methods)
    for method, msg, expected in zip(methods, messages(s, row["run_id"]), baseline["methods"], strict=True):
        assert s.worker.process(msg) == "ACK"
        result = s.service.result(s.auth, row["run_id"], method).model_dump(mode="json")
        for field in ("config_sha256", "ranked_chunk_ids", "context_chunk_ids", "context_sources", "context_proxy_tokens"):
            assert result[field] == expected[field]
        assert result["response"]["mode_used"] == method and not result["response"]["used_graph"]
    assert s.gateway.provider.calls == 2
