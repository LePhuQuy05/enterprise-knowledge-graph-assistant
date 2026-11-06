"""Fixed-slot regression against native V4 over-count; fixtures are not LIVE."""

import json
from types import SimpleNamespace as NS

import pytest

from app.contracts import sha256_bytes
from app.faceted_query_v5 import (
    FacetPlanCacheV5,
    FacetPlanV5,
    FacetRecipeV5,
    preparation_context,
    revalidate_plan,
    validate_facets,
)
from app.query_planning import QueryPlanBlocked
from app.query_planning_v2 import QueryPlanValidationError
from tests.test_faceted_query_v4 import ROWS, SEARCH
from tests.test_query_planning import Gateway
from tests.test_query_planning import routes as routes


def slots(rows):
    return json.dumps({f"facet_{i+1}": rows[i] if i < len(rows) else None for i in range(3)})


@pytest.mark.parametrize("n", [0, 1, 2, 3])
def test_null_trailing_slots_and_literal_offsets(n):
    facets = validate_facets(slots(ROWS[:n]), SEARCH, SEARCH)
    assert len(facets) == n
    assert all(SEARCH[f.start_char:f.end_char] == f.focus_quote for f in facets)


@pytest.mark.parametrize("raw", [
    "[]", '{"facets":[]}', json.dumps({"facets": ROWS + [ROWS[0]]}),
    '{"facet_1":null,"facet_2":null}',
    json.dumps({"facet_1": ROWS[0], "facet_2": None, "facet_3": ROWS[2]}),
    json.dumps({"facet_1": None, "facet_2": ROWS[1], "facet_3": None}),
    json.dumps({**json.loads(slots(ROWS)), "facet_4": ROWS[0]}),
    json.dumps({**json.loads(slots(ROWS)), "reason": None}),
    slots([{**ROWS[0], "extra": "text"}]), slots([False]),
    slots([{**ROWS[0], "focus_quote": "AWS KMS"}]),
    slots([{**ROWS[0], "search_query": "curl https://bad.example"}]),
    slots([{**ROWS[0], "search_query": "S3 StartExecution permissions"}]),
    slots([ROWS[0], {"focus_quote": "invocation permissions", "search_query": "S3 invoke permissions"}]),
    slots([ROWS[0], {**ROWS[1], "search_query": ROWS[0]["search_query"]}]),
])
def test_strict_slots_no_repair_or_safety_relaxation(raw):
    with pytest.raises(ValueError):
        validate_facets(raw, SEARCH, SEARCH)


def test_actual_v4_failure_is_not_silently_clipped():
    from pathlib import Path

    raw = Path("tests/fixtures/facet_v4_native_overcount.json").read_text(encoding="utf-8")
    with pytest.raises(QueryPlanValidationError, match="FIXED_SLOT_SCHEMA"):
        validate_facets(raw, SEARCH, SEARCH)


def make_plan(**changes):
    return FacetPlanV5(origin="FIXTURE", collection_manifest_sha256="a"*64,
        question_sha256=sha256_bytes(SEARCH.encode()), search_query_sha256=sha256_bytes(SEARCH.encode()),
        language="en", facets=validate_facets(slots(ROWS), SEARCH, SEARCH),
        recipe_sha256=FacetRecipeV5().fingerprint, operation_id="fixture", receipt_sha256="b"*64).model_copy(update=changes)


@pytest.mark.parametrize("changes", [{"question_sha256": "c"*64}, {"language": "vi"}, {"recipe_sha256": "c"*64}])
def test_binding_is_revalidated(changes):
    with pytest.raises(ValueError):
        revalidate_plan(make_plan(**changes), collection=NS(manifest_sha256="a"*64, manifest=NS(purpose="FIXTURE")),
            question=SEARCH, search=SEARCH, language="en")


def test_foreign_v4_typed_plan_is_not_relabelled():
    from tests.test_faceted_query_v4 import plan

    with pytest.raises(QueryPlanValidationError, match="TYPED_PLAN"):
        revalidate_plan(plan(), collection=NS(manifest_sha256="a"*64), question=SEARCH, search=SEARCH, language="en")


def test_fixture_cache_keeps_namespace_and_exact_replay(tmp_path, routes):
    cache = FacetPlanCacheV5(tmp_path, "a"*64, namespace="FIXTURE")
    context = preparation_context("a"*64, SEARCH, SEARCH, "en", "fixture")
    gateway = Gateway(routes, text=slots(ROWS))
    first = cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes, gateway=gateway)
    second = cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes)
    assert first.plan == second.plan and first.plan.origin == "FIXTURE"
    assert first.receipt["raw_output"] == gateway.text and gateway.calls == 1 and second.new_cost_micro_usd == 0
    assert cache.path.name == "task_facets_v5.sqlite3"
    with pytest.raises(QueryPlanValidationError, match="ORIGIN"):
        revalidate_plan(first.plan, collection=NS(manifest_sha256="a"*64, manifest=NS(purpose="DEMO")),
            question=SEARCH, search=SEARCH, language="en")


def test_wrong_context_prevents_dispatch_and_claim(tmp_path, routes):
    cache = FacetPlanCacheV5(tmp_path, "a"*64)
    context = preparation_context("c"*64, SEARCH, SEARCH, "en", "fixture")
    with pytest.raises(QueryPlanValidationError, match="CONTEXT_MANIFEST"):
        cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes, gateway=NS())
    with cache._db() as db:
        assert db.execute("SELECT COUNT(*) FROM preparations").fetchone()[0] == 0


def test_invalid_native_shape_keeps_raw_failed_and_never_retries(tmp_path, routes):
    cache = FacetPlanCacheV5(tmp_path, "a"*64, namespace="FIXTURE")
    context = preparation_context("a"*64, SEARCH, SEARCH, "en", "fixture")
    raw = json.dumps({"facets": ROWS + [ROWS[0]]})
    gateway = Gateway(routes, text=raw)
    with pytest.raises(ValueError):
        cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes, gateway=gateway)
    with cache._db() as db:
        state, payload = db.execute("SELECT state,payload FROM preparations").fetchone()
    assert state == "FAILED" and json.loads(payload)["raw_output"] == raw
    with pytest.raises(QueryPlanBlocked):
        cache.resolve(context, question=SEARCH, search_query=SEARCH, routes=routes, gateway=gateway)
    assert gateway.calls == 1
