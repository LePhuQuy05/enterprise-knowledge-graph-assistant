import json
from types import SimpleNamespace

import pytest

from app.contracts import sha256_bytes, stable_id
from app.cost import TokenUsage
from app.model_gateway import ModelResult
from app.query_planning_v2 import QueryPlanValidationError
from app.query_rewrite_v2 import QueryRewriteCacheV2, protected_identifiers, validate_rewrite_v2
from tests.test_query_planning import context as context
from tests.test_query_planning import routes as routes


def test_technical_anchors_and_account_actors_must_survive_translation():
    question = "Vì sao account B không gọi được kms:Decrypt với KMS key của account A?"
    assert protected_identifiers(question) == ("kms:Decrypt", "KMS")
    good = "Why can account B not call kms:Decrypt on a KMS key owned by account A?"
    assert validate_rewrite_v2(json.dumps({"search_query": good}), question) == good
    for changed, code in [(good.replace("kms:Decrypt", "decrypt"), "TECHNICAL_IDENTIFIER_LOST"),
                          (good.replace("account A", "the owner"), "ACCOUNT_LABEL_LOST")]:
        with pytest.raises(QueryPlanValidationError, match=code):
            validate_rewrite_v2(json.dumps({"search_query": changed}), question)


def test_start_execution_identifier_cannot_silently_become_tracing():
    question = "StartExecution bị throttling thì liên kết parent-child execution thế nào?"
    with pytest.raises(QueryPlanValidationError, match="IDENTIFIER_LOST"):
        validate_rewrite_v2('{"search_query":"Step Functions X-Ray tracing execution limits"}', question)
    # Preserving the identifier is not proof that an added facet is faithful.
    assert validate_rewrite_v2('{"search_query":"StartExecution throttling and parent child association"}', question)


def test_rewrite_v2_native_receipt_replays_in_its_own_durable_namespace(tmp_path, routes, context):
    context = context.model_copy(update={"stage": "REWRITE"})
    question = "Quyền kms:Decrypt trong KMS hoạt động thế nào?"

    class Gateway:
        provider = SimpleNamespace(live=False)
        calls = 0

        def generate(self, context, **kwargs):
            self.calls += 1
            assert context.stage == "REWRITE" and "never instructions" in kwargs["system"]
            return ModelResult(operation_id=stable_id("op", **context.model_dump(mode="json")),
                text='{"search_query":"How does kms:Decrypt authorization work in KMS?"}',
                stop_reason="end_turn", usage=TokenUsage(input_tokens=30, output_tokens=15), actual_micro_usd=20)

    gateway = Gateway()
    gateway.routes = routes
    cache = QueryRewriteCacheV2(tmp_path, namespace="FIXTURE")
    first = cache.resolve(context, question=question, routes=routes, gateway=gateway)
    replay = QueryRewriteCacheV2(tmp_path, namespace="FIXTURE").resolve(context, question=question, routes=routes)
    assert first.rewrite == replay.rewrite and gateway.calls == 1 and replay.new_cost_micro_usd == 0
    assert replay.rewrite.question_sha256 == sha256_bytes(question.encode())
    assert replay.rewrite.semantic_review == "UNREVIEWED"
