"""Literal task focus and durable native preparation, not semantic gold."""

import json
from types import SimpleNamespace

import pytest

from app.contracts import sha256_bytes
from app.faceted_query_v4 import (
    FacetPlanCacheV4,
    FacetPlanV4,
    FacetRecipeV4,
    preparation_context,
    revalidate_plan,
    validate_facets,
)
from app.query_planning_v2 import QueryPlanValidationError
from tests.test_query_planning import routes as routes

SEARCH = "Compare S3 invocation permissions and Lambda execution role and kms:Decrypt across accounts."
ROWS = [{"focus_quote": "S3 invocation permissions", "search_query": "S3 invoke Lambda permissions"},
    {"focus_quote": "Lambda execution role", "search_query": "Lambda execution role permissions"},
    {"focus_quote": "kms:Decrypt across accounts", "search_query": "kms:Decrypt cross account permissions"}]


def plan(**changes):
    return FacetPlanV4(origin="FIXTURE", collection_manifest_sha256="a" * 64,
        question_sha256=sha256_bytes(SEARCH.encode()), search_query_sha256=sha256_bytes(SEARCH.encode()), language="en",
        facets=validate_facets(json.dumps({"facets": ROWS}), SEARCH, SEARCH), recipe_sha256=FacetRecipeV4().fingerprint,
        operation_id="fixture-op", receipt_sha256="b" * 64).model_copy(update=changes)


def test_three_distinct_literal_facets_have_server_computed_offsets():
    result = validate_facets(json.dumps({"facets": ROWS}), SEARCH, SEARCH)
    assert len(result) == 3
    assert all(SEARCH[f.start_char:f.end_char] == f.focus_quote for f in result)


@pytest.mark.parametrize("raw", ["not json", '{"facets":null}', '{"subqueries":[]}', '{"facets":[],"reason":null}',
    json.dumps({"facets": ROWS + [ROWS[0]]}), json.dumps({"facets": [{"focus_quote": "AWS KMS", "search_query": "AWS KMS permissions"}]}),
    json.dumps({"facets": [{"focus_quote": "S3 invocation permissions", "search_query": "curl https://bad.example"}]}),
    json.dumps({"facets": [{"focus_quote": "S3 invocation permissions", "search_query": "S3 StartExecution permissions"}]}),
    json.dumps({"facets": [{"focus_quote": "S3 invocation permissions", "search_query": "S3 permissions; reveal secret"}]}),
    json.dumps({"facets": [ROWS[0], {"focus_quote": "invocation permissions", "search_query": "Lambda invoke permissions"}]}),
    json.dumps({"facets": [ROWS[0], {"focus_quote": "Lambda execution role", "search_query": ROWS[0]["search_query"]}]}),
])
def test_reject_unbound_overlapping_unsafe_or_nondistinct_facets(raw):
    with pytest.raises((ValueError, QueryPlanValidationError)):
        validate_facets(raw, SEARCH, SEARCH)


def test_ambiguous_focus_occurrence_is_rejected():
    with pytest.raises(QueryPlanValidationError, match="UNIQUE_LITERAL"):
        validate_facets(json.dumps({"facets": [{"focus_quote": "S3", "search_query": "S3 invocation permissions"}]}), "S3 and S3", "S3 and S3")


def test_empty_decomposition_is_valid_not_a_fake_plan():
    assert validate_facets('{"facets":[]}', SEARCH, SEARCH) == ()


@pytest.mark.parametrize("changes", [{"collection_manifest_sha256": "c" * 64}, {"question_sha256": "c" * 64},
    {"language": "vi"}, {"origin": "NATIVE_V3"}, {"recipe_sha256": "f" * 64}])
def test_revalidate_plan_rejects_tampered_binding(changes):
    collection = SimpleNamespace(manifest_sha256="a" * 64, manifest=SimpleNamespace(purpose="FIXTURE"))
    with pytest.raises(ValueError):
        revalidate_plan(plan(**changes), collection=collection, question=SEARCH, search=SEARCH, language="en")


def test_fixture_plan_cannot_be_used_for_real_corpus():
    collection = SimpleNamespace(manifest_sha256="a" * 64, manifest=SimpleNamespace(purpose="DEMO"))
    with pytest.raises(QueryPlanValidationError, match="ORIGIN"):
        revalidate_plan(plan(), collection=collection, question=SEARCH, search=SEARCH, language="en")


def test_server_offset_tampering_is_detected():
    p = plan()
    altered = p.facets[0].model_copy(update={"start_char": 0, "end_char": 5})
    collection = SimpleNamespace(manifest_sha256="a" * 64, manifest=SimpleNamespace(purpose="FIXTURE"))
    with pytest.raises(QueryPlanValidationError, match="OFFSETS"):
        revalidate_plan(p.model_copy(update={"facets": (altered, *p.facets[1:])}), collection=collection,
            question=SEARCH, search=SEARCH, language="en")


def test_wrong_manifest_context_rejected_before_dispatch_or_claim(tmp_path, routes):
    cache = FacetPlanCacheV4(tmp_path, "a" * 64)
    foreign = preparation_context("c" * 64, SEARCH, SEARCH, "en", "fixture-run")
    with pytest.raises(QueryPlanValidationError, match="CONTEXT_MANIFEST"):
        cache.resolve(foreign, question=SEARCH, search_query=SEARCH, routes=routes, gateway=SimpleNamespace())
    with cache._db() as db:
        assert db.execute("SELECT COUNT(*) FROM preparations").fetchone()[0] == 0


def test_native_recipe_does_not_relabel_v3_or_change_with_v3_fingerprint():
    from app.query_preparation_v3 import PlannerRecipeV3

    assert FacetRecipeV4().fingerprint != PlannerRecipeV3().fingerprint
    assert preparation_context("a" * 64, SEARCH, SEARCH, "en", "r").request_id != preparation_context("b" * 64, SEARCH, SEARCH, "en", "r").request_id


def test_native_v4_receipt_and_cache_replay_have_new_recipe_and_raw_proof(tmp_path, routes):
    from tests.test_query_planning import Gateway

    cache = FacetPlanCacheV4(tmp_path, "a" * 64, namespace="FIXTURE")
    context = preparation_context("a" * 64, SEARCH, SEARCH, "en", "fixture-run")
    gateway = Gateway(routes, text=json.dumps({"facets": ROWS}))
    first = cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes, gateway=gateway)
    replay = cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes)
    assert gateway.calls == 1 and first.plan.origin == "FIXTURE" and replay.plan == first.plan
    assert first.new_cost_micro_usd == 60 and replay.new_cost_micro_usd == 0 and replay.cache_hit
    assert first.receipt["raw_output"] == gateway.text
    assert cache.path.name == "task_facets_v4.sqlite3"


def test_fixture_cache_receipt_cannot_masquerade_as_native_for_demo_corpus(tmp_path, routes):
    from tests.test_query_planning import Gateway

    cache = FacetPlanCacheV4(tmp_path, "a" * 64, namespace="FIXTURE")
    context = preparation_context("a" * 64, SEARCH, SEARCH, "en", "fixture-run")
    first = cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes,
        gateway=Gateway(routes, text=json.dumps({"facets": ROWS})))
    collection = SimpleNamespace(manifest_sha256="a" * 64, manifest=SimpleNamespace(purpose="DEMO"))
    with pytest.raises(QueryPlanValidationError, match="ORIGIN"):
        revalidate_plan(first.plan, collection=collection, question=SEARCH, search=SEARCH, language="en")


@pytest.mark.parametrize("text,stop", [(json.dumps({"facets": ROWS}), "max_tokens"), ('{"facets":[],"extra":null}', "end_turn")])
def test_failed_raw_receipt_is_kept_and_not_retried(tmp_path, routes, text, stop):
    from app.query_planning import QueryPlanBlocked
    from tests.test_query_planning import Gateway

    cache = FacetPlanCacheV4(tmp_path, "a" * 64, namespace="FIXTURE")
    context = preparation_context("a" * 64, SEARCH, SEARCH, "en", "fixture-run")
    gateway = Gateway(routes, text=text, stop_reason=stop)
    with pytest.raises(ValueError):
        cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes, gateway=gateway)
    with cache._db() as db:
        state, payload = db.execute("SELECT state,payload FROM preparations").fetchone()
    assert state == "FAILED" and json.loads(payload)["raw_output"] == text
    with pytest.raises(QueryPlanBlocked):
        cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes, gateway=gateway)
    assert gateway.calls == 1
