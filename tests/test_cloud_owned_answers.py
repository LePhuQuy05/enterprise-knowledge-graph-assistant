from types import SimpleNamespace
from uuid import uuid4

import pytest

from app.cloud_owned_answers import CloudAnswerDispatch, CollectionAnswerPolicy
from app.query_auth import ApiError
from app.run_budget import ReceiptedStageGateway, RunBudgetController, RunStageBlocked
from app.run_cloud import CloudRunService
from tests.test_collection_review import review_context as review_context
from tests.test_owned_answers import answers as answers
from tests.test_rag_runner import setup_runner as setup_runner


def quote_controller(s):
    return RunBudgetController(ledger=s.ledger, routes=s.gateway.routes, prices=s.gateway.prices,
                              quote_live=False, budget_run_id="owned-fixture")


def test_admission_only_quotes_without_key_or_dispatch_then_binds_exact_worker_policy(answers):
    s = answers
    controller = quote_controller(s)
    assert controller.gateway is None
    parent = {"tenant_id": s.auth.tenant_id, "owner_id": s.auth.user_id, "run_id": "run_" + "3" * 64,
              "request": {"original_question": "Service B", "language": "en", "method_ids": ["hybrid"]},
              "admission_token": "4" * 32}
    changes = controller.admission_changes(parent)
    assert len(changes) == 4 and s.gateway.provider.calls == 0
    stage = ReceiptedStageGateway(controller, s.auth, parent, "GENERATE:hybrid", lambda: None)
    with pytest.raises(RunStageBlocked, match="ADMISSION_CANNOT_INVOKE"):
        stage.generate(controller.context(parent, "GENERATE:hybrid"), system="fixture", prompt="{}")
    assert s.gateway.provider.calls == 0
    controller.bind_gateway(s.gateway)
    assert controller.gateway is s.gateway


def test_binding_rejects_different_transaction_store_or_rates(answers):
    s = answers
    other = quote_controller(s)
    other.store = object()
    with pytest.raises(ValueError, match="POLICY_BINDING"):
        other.bind_gateway(s.gateway)
    other = quote_controller(s)
    other._prices = other.prices.model_copy(update={"version": "different"})
    with pytest.raises(ValueError, match="POLICY_BINDING"):
        other.bind_gateway(s.gateway)


def test_answer_policy_limits_methods_and_requires_preexisting_exact_caps(answers):
    s, policy = answers, CollectionAnswerPolicy()
    controller = RunBudgetController(s.gateway, budget_run_id=policy.budget_run_id)
    with pytest.raises(ApiError, match="BUDGET_LIMIT"):
        policy.check_admission(controller, {"method_ids": ["hybrid"]})
    # A distinct fixture allowance, not resetting/increasing the old experiment.
    s.ledger.initialize_run("kg-r72", s.auth.tenant_id, policy.budget_run_id, cap_micro_usd=750000)
    policy.check_admission(controller, {"method_ids": ["hybrid"]})
    for methods in (["hybrid", "vector"], ["hybrid_reranked"], ["bm25"], []):
        with pytest.raises(ApiError, match="ANSWER_METHOD_UNAVAILABLE"):
            policy.check_admission(controller, {"method_ids": methods})
    assert s.gateway.provider.calls == 0


def test_cloud_policy_rejects_unreleased_answer_before_enqueue_or_quota(answers):
    s = answers
    service = CloudRunService(s.repo, s.artifacts, lambda: s.executor.release,
        authorize=s.service.authorize, release_check=s.service.release_check,
        coordinator=SimpleNamespace(notify=lambda *args: pytest.fail("must not enqueue")),
        answer_policy=CollectionAnswerPolicy())
    with pytest.raises(ApiError, match="ANSWER_METHOD_UNAVAILABLE"):
        service.submit(s.auth, {"original_question": "Service B", "run_kind": "answer", "method_ids": ["hybrid_reranked"]}, str(uuid4()))
    assert s.service.history(s.auth)["items"] == []


def test_lazy_dispatch_failure_is_safe_without_initializing_models_or_reading_key(answers):
    s = answers
    dispatch = CloudAnswerDispatch(service=s.service, config={}, context=None,
        retrieval_factory=lambda *args: pytest.fail("no models"), answer_factory=lambda *args: pytest.fail("no key"))
    assert dispatch.assets is None and dispatch.answer is None
    dispatch.close()
    assert s.gateway.provider.calls == 0
