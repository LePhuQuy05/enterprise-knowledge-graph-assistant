import pytest

from app.query_planning_v2 import QueryPlanValidationError
from app.query_preparation_v3 import QueryPlanCacheV3, validate_no_introduced_names_v3
from tests.test_query_planning import Gateway
from tests.test_query_planning import context as context
from tests.test_query_planning import routes as routes


def test_v2_canary_drift_cannot_be_silently_accepted_as_faithful_v3():
    original = "Step Functions starts another execution; keep a parent child association"
    for query in ("Step Functions parent child X-Ray tracing", "Step Functions StartExecution API"):
        with pytest.raises(QueryPlanValidationError, match="INTRODUCED_TECHNICAL_NAME"):
            validate_no_introduced_names_v3((query,), original)
    validate_no_introduced_names_v3(("AWS Step Functions execution-start throttling", "Step Functions parent child association"), original)


def test_explicitly_requested_named_solution_is_not_blanket_banned():
    original = "How do StartExecution, IAM and X-Ray work together?"
    validate_no_introduced_names_v3(("IAM permissions StartExecution", "X-Ray tracing"), original)


def test_native_v3_facets_replay_with_their_own_recipe_and_provider_proof(tmp_path, routes, context):
    question = "How does Step Functions start another execution and preserve the parent child association?"
    cache = QueryPlanCacheV3(tmp_path, namespace="FIXTURE")
    gateway = Gateway(routes, text='{"subqueries":["Step Functions nested execution throttling","Step Functions parent child association"]}')
    first = cache.resolve(context, question=question, search_query=question, routes=routes, gateway=gateway)
    replay = cache.resolve(context, question=question, search_query=question, routes=routes)
    assert first.plan.origin == "NATIVE_V3" and first.plan == replay.plan and gateway.calls == 1
    assert first.plan.recipe_sha256 == cache.recipe.fingerprint
