"""Opt-in collection answering policy. API quotes need no provider credentials."""

from contextlib import ExitStack
from pathlib import Path
from typing import Literal

from app.contracts import Record, canonical_json
from app.funding_policy import FundingPolicyError, validate_external_funding
from app.query_auth import ApiError

ROOT = Path(__file__).resolve().parents[1]


class CollectionAnswerPolicy(Record):
    version: Literal["collection316-answer-serving-v1"] = "collection316-answer-serving-v1"
    budget_run_id: Literal["collection316-serving-v1"] = "collection316-serving-v1"
    run_cap_micro_usd: Literal[750000] = 750000
    funding_cap_micro_usd: Literal[2000000] = 2000000
    month_cap_micro_usd: Literal[12000000] = 12000000
    methods: tuple[Literal["hybrid"], Literal["vector"]] = ("hybrid", "vector")
    maximum_methods: Literal[1] = 1

    def check_admission(self, controller, payload):
        if len(payload.get("method_ids", ("hybrid",))) != 1 or any(
            method not in self.methods for method in payload.get("method_ids", ("hybrid",))
        ):
            raise ApiError(400, "ANSWER_METHOD_UNAVAILABLE")
        self.check_budget_profile(controller)

    def check_budget_profile(self, controller):
        if controller is None or controller.budget_run_id != self.budget_run_id:
            raise ApiError(503, "ANSWER_POLICY_BINDING")
        ledger = controller.ledger
        for key, cap in ((ledger.run_key(self.budget_run_id), self.run_cap_micro_usd),
                         (ledger.month_key("kg-r72", ledger._now().strftime("%Y-%m")), self.month_cap_micro_usd)):
            row = controller.store.get(key)
            if not row or row.get("cap_micro_usd") != cap or row.get("halted"):
                raise ApiError(429, "BUDGET_LIMIT")
        try:
            validate_external_funding(ledger, profile=getattr(self, "funding_reader_profile", "LEGACY_2_USD"))
        except FundingPolicyError:
            raise ApiError(429, "BUDGET_LIMIT") from None


class CollectionAnswerPolicyV2(CollectionAnswerPolicy):
    version: Literal["collection316-answer-serving-v2-audited-funding"] = "collection316-answer-serving-v2-audited-funding"
    funding_cap_micro_usd: Literal[8000000] = 8000000
    funding_reader_profile: Literal["AUDITED_2_TO_8_USD_V2"] = "AUDITED_2_TO_8_USD_V2"


def load_collection_answer_policy(value):
    model = CollectionAnswerPolicyV2 if value.get("version") == "collection316-answer-serving-v2-audited-funding" else CollectionAnswerPolicy
    return model.model_validate_json(canonical_json(value))


class CloudAnswerDispatch:
    """Lazy worker composition; duplicate terminal messages never load models or secrets."""

    def __init__(self, *, service, config, context, retrieval_factory, answer_factory):
        self.service, self.config, self.context = service, config, context
        self.retrieval_factory, self.answer_factory = retrieval_factory, answer_factory
        self.stack, self.assets, self.answer = ExitStack(), None, None

    def close(self):
        self.stack.close()

    def _assets(self):
        if self.assets is None:
            import boto3
            from botocore.config import Config

            from app.workbench_assets import load_s3_workbench_assets

            s3 = boto3.client("s3", region_name="us-east-1", config=Config(retries={"total_max_attempts": 1},
                connect_timeout=2, read_timeout=5))
            self.assets = self.stack.enter_context(load_s3_workbench_assets(s3,
                **self.config["assets"], cache_root=Path("/tmp/kg-workbench-assets")))
        return self.assets

    def _answer(self):
        if self.answer is None:
            self.answer = self.answer_factory(self._assets(), self.service, self.config, self.context)
        return self.answer

    def prepare(self, auth, parent):
        return self._answer().prepare(auth, parent)

    def safe_failure(self, auth, parent, method):
        # Also works when asset/secret initialization failed before a provider dispatch.
        return self.service.repo.answer_budget.safe_failure(auth, parent, method)

    def __call__(self, auth, parent, method, guard):
        guard()
        executor = self._answer() if parent["request"]["run_kind"] == "answer" else self.retrieval_factory(self._assets())
        executor.check_release(parent["release"])
        return executor(auth, parent, method, guard)


def make_quote_controller(store, policy):
    from app.external_gateway import ExternalPriceSheet, ExternalRoutes
    from app.ledger import Ledger
    from app.run_budget import RunBudgetController

    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    routes = routes.model_copy(update={"live_inference_enabled": True,
                                      "live_inference_gate": "OWNED_COLLECTION_ANSWER_RELEASE_V1"})
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    return RunBudgetController(ledger=Ledger(store), routes=routes, prices=prices, budget_run_id=policy.budget_run_id)


def make_cloud_answer_executor(assets, service, config, context):
    import os
    import time

    import boto3
    from botocore.config import Config

    from app.cloud_assets import read_deepseek_secret
    from app.external_gateway import ExternalModelGateway
    from app.external_http import OfficialHttpsTransport
    from app.run_answer_executor import CollectionOwnedAnswerExecutor
    from app.run_cloud import make_retrieval_executor

    if os.environ.get("MODEL_ENABLED") != "true" or service.answer_policy is None:
        raise ApiError(503, "MODEL_DISABLED")
    controller = service.repo.answer_budget
    service.answer_policy.check_budget_profile(controller)
    remaining = context.get_remaining_time_in_millis() / 1000
    if remaining < 40:
        raise ApiError(503, "ANSWER_TIME_BUDGET")
    key = read_deepseek_secret(boto3.client("secretsmanager", region_name="us-east-1",
        config=Config(retries={"total_max_attempts": 1}, connect_timeout=2, read_timeout=3)),
        secret_arn=os.environ["KG_DEEPSEEK_SECRET_ARN"])
    transport = OfficialHttpsTransport({"DEEPSEEK": key}, timeout_seconds=25,
        deadline_monotonic=time.monotonic() + remaining - 12)
    controller.bind_gateway(ExternalModelGateway(controller.ledger, transport, controller.routes, controller.prices))
    base = make_retrieval_executor(assets)
    runtime = base.runtime | {"routes": controller.routes, "run_id": service.answer_policy.budget_run_id}
    return CollectionOwnedAnswerExecutor(assets.collection, assets.lexical, controller=controller,
        repository=service.repo, authorize=service.authorize, **runtime)
