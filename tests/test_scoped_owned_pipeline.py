"""Real local API/queue/ledger/retrieval/graph flow, synthetic provider/embeddings.

Graph fixtures retain the existing source-validated authorization facts. These
tests are protocol/security evidence, not measured AWS answer quality.
"""

import json
from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.contracts import sha256_bytes
from app.external_gateway import ExternalModelGateway
from app.fixtures import fixture_vector
from app.generation import GenerationRecipeV9
from app.ledger import Change, Ledger, MemoryLedgerStore
from app.owned_runs import OwnedRunRepository
from app.query_auth import ApiError, AuthContext, require_current_membership
from app.query_state import QueryRepository
from app.run_artifacts import LocalRunBlobs
from app.run_service import OwnedRunApi
from app.run_storage import SQLiteRunQueue, SQLiteRunStore
from app.scoped_method_flow import METHODS
from app.scoped_owned_runs import (
    SCOPED_RUN_CAP_MICRO_USD,
    ScopedOutboxDispatcher,
    ScopedRunRepository,
)
from app.scoped_run_budget import PLAN_SLOT, ScopedRunBudgetController
from app.scoped_run_contracts import validate_scoped_result
from app.scoped_run_executor import ScopedOwnedAnswerExecutor
from app.scoped_run_service import ScopedArtifactRepository, ScopedRunService, ScopedRunWorker
from tests.test_authorization_evidence import auth_case as auth_case  # noqa: F401
from tests.test_collection_answering import FixtureReranker
from tests.test_external_pilot import configured
from tests.test_query_api import PUBLIC, USER_A, USER_B, event

QUESTION = "Explain Lambda execution role and Amazon S3 access permissions."
QUESTION_VI = "Giải thích execution role của Lambda và quyền truy cập Amazon S3."


class FixtureEncoder:
    fixture_only = True

    def __init__(self, recipe):
        self.value = recipe

    def recipe(self, role):
        return self.value


class FixtureVectorCache:
    """Fixture 1024-vector component; production uses the validated BGE cache."""

    def resolve_local(self, tenant, *, text, recipe, encoder):
        return SimpleNamespace(record=SimpleNamespace(vector=fixture_vector(text)))


@pytest.fixture
def scoped_stack(auth_case, tmp_path):
    original = configured()
    prices = original.prices.model_copy(update={"rates": tuple(rate.model_copy(update={
        "input_micro_usd_per_unit": 10000, "output_micro_usd_per_unit": 10000}) for rate in original.prices.rates)})
    ledger = Ledger(SQLiteRunStore(tmp_path / "owned-state.sqlite3"), clock=original.ledger.clock)
    ledger.initialize_month("kg-r72", ledger._now().strftime("%Y-%m"))
    ledger.initialize_external_funding("kg-r72", cap_micro_usd=2000000)
    gateway = ExternalModelGateway(ledger, original.provider, original.routes, prices)
    ledger.initialize_run("kg-r72", "demo", "scoped-pipeline-fixture", cap_micro_usd=500000)
    controller = ScopedRunBudgetController(gateway, budget_run_id="scoped-pipeline-fixture")
    repo = ScopedRunRepository(ledger.store, clock=ledger.clock, answer_budget=controller)
    metadata = MemoryLedgerStore()
    for user in (USER_A, USER_B):
        row = {"PK": "USER#"+user, "SK": "MEMBERSHIP", "version": 1, "record_type": "MEMBERSHIP",
            "schema_version": 1, "tenant_id": "demo", "status": "ACTIVE", "roles": ["reviewer"]}
        metadata.commit([Change((row["PK"], row["SK"]), None, row)])

    def authorize(auth):
        return require_current_membership(auth, metadata)

    recipe = auth_case.collection.manifest.embedding_recipe
    runtime = SimpleNamespace(recipe=GenerationRecipeV9(model_metadata_sha256=gateway.routes.account_model_metadata_sha256),
        encoder=FixtureEncoder(recipe), query_recipe=recipe, cache=FixtureVectorCache(), reranker=FixtureReranker())
    executor = ScopedOwnedAnswerExecutor(auth_case.collection, auth_case.lexical, auth_case.graph, runtime,
        controller=controller, repository=repo, authorize=authorize)
    artifacts = ScopedArtifactRepository(LocalRunBlobs(tmp_path / "private"))
    service = ScopedRunService(repo, artifacts, lambda: executor.release, authorize=authorize, release_check=executor.check_release)
    failures = []

    class RecordingExecutor:
        prepare, safe_failure = executor.prepare, executor.safe_failure

        def __call__(self, *args):
            try:
                return executor(*args)
            except Exception as exc:
                failures.append(f"{type(exc).__name__}: {exc}")
                raise

    worker = ScopedRunWorker(repo, artifacts, RecordingExecutor(), authorize=authorize, release_check=executor.check_release)
    queue = SQLiteRunQueue(tmp_path / "queue.sqlite3")
    flags = {"empty_facets": False, "held": False, "timeout": False, "invalid_facets": False, "after_call": None}
    seen = []

    def provider(provider, method, path, body):
        gateway.provider.calls += 1
        request = json.loads(body)
        system, prompt = request["messages"]
        prompt = json.loads(prompt["content"])
        seen.append(request)
        if flags["timeout"]:
            raise TimeoutError("SYNTHETIC provider uncertainty")
        if "untrusted_sources" in prompt:
            first = prompt["untrusted_sources"][0]
            span = first["spans"][0]
            text = span["text"]
            if flags["held"]:
                first = next(source for source in prompt["untrusted_sources"] if
                    "invoke" not in source["spans"][0]["text"].lower())
                span = first["spans"][0]
                text = "Permission to invoke Lambda is guaranteed."
            if prompt["untrusted_question"] == QUESTION_VI and not flags["held"]:
                text = "Đoạn nguồn cung cấp thông tin về quyền truy cập."
            content = {"answer_status": "PARTIAL", "reason": "INSUFFICIENT_EVIDENCE",
                "limitations": [{"code": "MISSING_SUPPLIED_EVIDENCE", "question_part": prompt["untrusted_question"]}],
                "claims": [{"text": text, "evidence": [{"source_id": first["source_id"], "span_id": span["span_id"]}]}]}
        elif "facet_1" in system["content"]:
            content = {"facet_1": None, "facet_2": None, "facet_3": None} if flags["empty_facets"] else {
                "facet_1": "Lambda execution role", "facet_2": "Amazon S3 access permissions", "facet_3": None}
            if flags["invalid_facets"]:
                content = {"facet_1": "Invented task not in the search", "facet_2": None, "facet_3": None}
        else:
            content = {"search_query": QUESTION}
        if flags["after_call"]:
            flags["after_call"]()
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(content)}}]}

    gateway.provider.request = provider
    return SimpleNamespace(repo=repo, controller=controller, gateway=gateway, ledger=ledger, metadata=metadata,
        executor=executor, artifacts=artifacts, service=service, worker=worker, queue=queue,
        dispatcher=ScopedOutboxDispatcher(repo, queue.send), api=OwnedRunApi(service, metadata, PUBLIC),
        auth=AuthContext("demo", USER_A), other=AuthContext("demo", USER_B), flags=flags, seen=seen, failures=failures)


def submit(s, methods=("hybrid",), language="en", key=None):
    return s.service.submit(s.auth, {"original_question": QUESTION_VI if language == "vi" else QUESTION,
        "method_ids": list(methods), "language": language, "run_kind": "answer"}, key or str(uuid4()))


def deliver(s, run):
    s.dispatcher.publish(s.auth, run)
    for identifier, message in s.queue.receive():
        assert s.worker.process(message) == "ACK"
        s.queue.ack(identifier)


@pytest.mark.parametrize("method", METHODS)
def test_all_eight_methods_execute_their_real_retrieval_mechanism_and_return_owned_answer(scoped_stack, method):
    s = scoped_stack
    row = submit(s, (method,))
    assert s.gateway.provider.calls == 0
    deliver(s, row["run_id"])
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["status"] == "COMPLETED", s.failures or detail
    result = s.service.public_result(s.auth, row["run_id"], method)
    assert result["schema_version"] == 2 and result["requested_method"] == result["effective_method"] == method
    assert result["qualification"] == "NO_MECHANICAL_FLAGS" and result["response"]["answer_status"] == "PARTIAL"
    assert result["semantic_verified"] is False and result["quality_metrics"] is None
    assert all("text" not in source for source in result["context_sources"])
    if method.startswith("graph_"):
        assert result["used_graph"] and any(bundle["assertion_ids"] for bundle in result["retrieval"]["bundles"])
        assert result["graph_evidence"]
        by_chunk = {source.chunk_id: source for source in s.service.result(s.auth, row["run_id"], method).context_sources}
        for fact in result["graph_evidence"]:
            assert fact["clause"] in by_chunk[fact["chunk_id"]].text and fact["permission_granted"] is False
            nodes = {node["node_id"] for node in fact["nodes"]}
            assert all(left in nodes and right in nodes for left, right, _ in fact["links"])
    if method in {"hybrid_multi_query", "hybrid_multi_query_balanced"}:
        assert result["retrieval"]["planner_origin"] == "FIXTURE" and len(result["retrieval"]["queries"]) == 3
        assert s.gateway.provider.calls == 2
    else:
        assert s.gateway.provider.calls == 1
    assert detail["budget"]["reserved_micro_usd"] == 0
    assert s.service.source(s.auth, row["run_id"], method, result["context_chunk_ids"][0])["text"]


def test_q_qb_share_one_native_plan_and_pool_but_balance_differently(scoped_stack):
    s = scoped_stack
    methods = ("hybrid", "hybrid_multi_query", "hybrid_multi_query_balanced")
    row = submit(s, methods, "vi")
    deliver(s, row["run_id"])
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["status"] == "COMPLETED", detail
    q, qb = [s.service.public_result(s.auth, row["run_id"], method) for method in methods[1:]]
    assert q["answer_language"] == qb["answer_language"] == "vi"
    assert q["retrieval"]["pool_fingerprint"] == qb["retrieval"]["pool_fingerprint"]
    assert q["retrieval"]["planner_operation_id"] == qb["retrieval"]["planner_operation_id"]
    assert not q["retrieval"]["balanced_allocator_used"] and qb["retrieval"]["balanced_allocator_used"]
    assert s.gateway.provider.calls == 5  # rewrite + shared planner + three generations
    assert detail["budget"]["cost_complete"] and detail["budget"]["reserved_micro_usd"] == 0


def test_empty_facets_are_explicit_not_ready_without_any_generation_or_hybrid(scoped_stack):
    s = scoped_stack
    s.flags["empty_facets"] = True
    methods = ("hybrid_multi_query", "hybrid_multi_query_balanced")
    row = submit(s, methods)
    deliver(s, row["run_id"])
    for method in methods:
        result = s.service.public_result(s.auth, row["run_id"], method)
        assert result["qualification"] == "NOT_READY" and result["response"] is None
        assert result["effective_method"] == method and result["answer_origin"] == "NOT_READY"
    assert s.gateway.provider.calls == 1 and s.service.detail(s.auth, row["run_id"])["budget"]["reserved_micro_usd"] == 0


@pytest.mark.parametrize("method", ["graph_1hop", "graph_2hop"])
def test_graph_zero_keeps_graph_method_and_releases_generation_hold_without_dispatch(scoped_stack, method):
    s = scoped_stack
    empty = replace(s.executor.graph, facts=(), catalog_nodes=(), release_sha256=sha256_bytes(b"fixture-empty-graph"))
    executor = ScopedOwnedAnswerExecutor(s.executor.collection, s.executor.lexical, empty, s.executor.runtime,
        controller=s.controller, repository=s.repo, authorize=s.service.authorize)
    s.service.release_provider = lambda: executor.release
    s.service.release_check = s.worker.release_check = executor.check_release
    s.worker.executor = executor
    row = submit(s, (method,))
    deliver(s, row["run_id"])
    result = s.service.public_result(s.auth, row["run_id"], method)
    assert result["qualification"] == "NOT_READY" and result["effective_method"] == method
    assert result["response"] is None and not result["used_graph"] and s.gateway.provider.calls == 0


def test_held_answer_never_appears_in_result_history_trace_or_artifact_export(scoped_stack):
    s = scoped_stack
    s.flags["held"] = True
    row = submit(s, ("vector",))
    deliver(s, row["run_id"])
    result = s.service.public_result(s.auth, row["run_id"], "vector")
    assert result["qualification"] == "REVIEW_REQUIRED" and result["response"] is None
    for value in (result, s.service.history(s.auth), s.service.trace(s.auth, row["run_id"]),
            s.service.result(s.auth, row["run_id"], "vector").model_dump(mode="json")):
        assert "Permission to invoke Lambda is guaranteed." not in json.dumps(value)
    assert s.gateway.provider.calls == 1


def test_duplicate_idempotent_submit_and_queue_delivery_do_not_charge_again(scoped_stack):
    s = scoped_stack
    key = str(uuid4())
    row = submit(s, ("graph_2hop",), key=key)
    deliver(s, row["run_id"])
    message = s.repo.outboxes(s.auth, row["run_id"])[0]["message"]
    assert s.worker.process(message) == "ACK"
    again = submit(s, ("graph_2hop",), key=key)
    assert again["idempotent_replay"] and again["run_id"] == row["run_id"] and s.gateway.provider.calls == 1
    assert QueryRepository(s.repo.store, clock=s.repo.clock).quota_remaining() == 299


def test_invalid_native_plan_is_failed_and_unknown_plan_stays_unknown(scoped_stack):
    s = scoped_stack
    s.flags["invalid_facets"] = True
    row = submit(s, ("hybrid_multi_query",))
    deliver(s, row["run_id"])
    assert s.service.detail(s.auth, row["run_id"])["status"] == "FAILED"
    s.flags["invalid_facets"], s.flags["timeout"] = False, True
    second = submit(s, ("hybrid_multi_query",))
    deliver(s, second["run_id"])
    detail = s.service.detail(s.auth, second["run_id"])
    assert detail["status"] == "UNKNOWN" and not detail["budget"]["cost_complete"]
    plan = next(stage for stage in detail["budget"]["stages"] if stage["slot"] == PLAN_SLOT)
    assert plan["status"] == "UNKNOWN_SPEND" and s.gateway.provider.calls == 2


def test_owned_api_graph_post_result_history_sources_and_cross_owner_denial(scoped_stack):
    s = scoped_stack
    raw = s.api.handler(event("POST", "/v1/runs", body={"schema_version": 2,
        "original_question": QUESTION, "method_ids": ["graph_2hop"], "run_kind": "answer"}))
    assert raw["statusCode"] == 202, raw
    row = json.loads(raw["body"])
    deliver(s, row["run_id"])
    path = "/v1/runs/"+row["run_id"]+"/methods/graph_2hop"
    result = s.api.handler(event("GET", path))
    assert result["statusCode"] == 200, result
    data = json.loads(result["body"])
    source_path = path+"/sources/"+data["context_chunk_ids"][0]
    assert s.api.handler(event("GET", source_path))["statusCode"] == 200
    assert s.api.handler(event("GET", source_path, user=USER_B))["statusCode"] == 404
    assert s.api.handler(event("GET", path, user=USER_B))["statusCode"] == 404
    assert len(s.service.history(s.auth)["items"]) == 1 and s.service.history(s.other)["items"] == []
    # Legacy repository must not interpret the new outer protocol as V1 results.
    with pytest.raises(ApiError, match="RUN_NOT_FOUND"):
        OwnedRunRepository(s.repo.store, clock=s.repo.clock).get(s.auth, row["run_id"])


def test_revoked_membership_before_worker_prevents_all_provider_calls(scoped_stack):
    s = scoped_stack
    row = submit(s, ("hybrid_multi_query",))
    key = "USER#"+USER_A, "MEMBERSHIP"
    member = s.metadata.get(key)
    s.metadata.commit([Change(key, member["version"], member | {"version": member["version"]+1, "status": "DISABLED"})])
    deliver(s, row["run_id"])
    assert s.repo.summary(s.auth, row["run_id"])["status"] == "CANCELLED" and s.gateway.provider.calls == 0


@pytest.mark.parametrize("mutation", ["owner", "request", "source", "response", "raw_field", "usage_field", "graph_method"])
def test_serialized_result_tampering_never_becomes_public(scoped_stack, mutation):
    s = scoped_stack
    row = submit(s, ("graph_2hop",))
    deliver(s, row["run_id"])
    parent = s.repo.get(s.auth, row["run_id"])
    raw = deepcopy(s.service.result(s.auth, row["run_id"], "graph_2hop").model_dump(mode="json"))
    if mutation == "owner":
        raw["owner_id"] = USER_B
    elif mutation == "request":
        raw["request_id"] = "wrong-request"
    elif mutation == "source":
        cited = raw["response"]["citations"][0]["chunk_id"]
        next(source for source in raw["context_sources"] if source["chunk_id"] == cited)["source_uri"] += "?tampered=1"
    elif mutation == "response":
        raw["qualification"] = "REVIEW_REQUIRED"  # Cannot retain public claims or omit flags.
    elif mutation == "raw_field":
        raw["native_response_for_review"] = {"text": "unsafe private output"}
    elif mutation == "usage_field":
        raw["usage"]["raw_output"] = "unsafe private output"
    else:
        raw["effective_method"] = "hybrid"
    with pytest.raises(ValueError):
        validate_scoped_result(parent, "graph_2hop", raw)


def test_native_generation_uncertainty_and_stale_policy_are_not_answer_abstention(scoped_stack, monkeypatch):
    s = scoped_stack
    s.flags["timeout"] = True
    row = submit(s, ("hybrid",))
    deliver(s, row["run_id"])
    detail = s.service.detail(s.auth, row["run_id"])
    assert detail["status"] == "UNKNOWN" and not detail["budget"]["cost_complete"]
    with pytest.raises(ApiError, match="RESULT_UNAVAILABLE"):
        s.service.public_result(s.auth, row["run_id"], "hybrid")
    s.flags["timeout"] = False
    next_run = submit(s, ("vector",))
    deliver(s, next_run["run_id"])
    import app.answer_qualification as qualification

    monkeypatch.setattr(qualification, "POLICY_VERSION", "changed-policy-test")
    with pytest.raises(ValueError, match="RELEASE_CHANGED"):
        s.service.public_result(s.auth, next_run["run_id"], "vector")


def test_pinned_price_worst_case_five_stage_quote_fits_internal_run_cap():
    from datetime import timedelta
    from pathlib import Path

    from app.cost import TokenUsage
    from app.external_gateway import ExternalPriceSheet, ExternalRoutes

    root = Path(__file__).resolve().parents[1]
    prices = ExternalPriceSheet.model_validate_json((root/"configs/prices/external-direct-2026-11-05.json").read_bytes())
    routes = ExternalRoutes.model_validate_json((root/"configs/models/external-direct-v1.json").read_bytes())
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, "external-default",
        now=prices.checked_at+timedelta(seconds=1), live=True)
    g = rate.quote(TokenUsage(input_tokens=routes.generation_context_ceiling, output_tokens=1600))
    p = rate.quote(TokenUsage(input_tokens=routes.generation_context_ceiling, output_tokens=256))
    assert 750000 < g+2*p < 3*g+2*p <= SCOPED_RUN_CAP_MICRO_USD
    assert SCOPED_RUN_CAP_MICRO_USD < 8000000  # Nested run cap does not raise shared allocation.
