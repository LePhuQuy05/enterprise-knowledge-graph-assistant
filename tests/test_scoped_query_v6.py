"""Regress actor/account drift, provenance tampering and missing original scope."""

import json
from types import SimpleNamespace

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.faceted_query_v5 import validate_facets as validate_legacy
from app.scoped_query_v6 import (
    ScopedPlanV6,
    ScopeRecipeV6,
    revalidate_plan,
    scoped_queries,
    validate_source_facets,
)
from tests.test_query_planning import Gateway
from tests.test_query_planning import routes as routes

QUESTION = "Can S3 invoke Lambda while the execution role cannot read an SSE-KMS object across accounts?"
COLLECTION = SimpleNamespace(manifest_sha256="a"*64, manifest=SimpleNamespace(purpose="LIVE"))


def raw(*quotes):
    return json.dumps({f"facet_{i+1}": quotes[i] if i < len(quotes) else None for i in range(3)})


def plan(question=QUESTION, search=QUESTION, language="en", origin="NATIVE_V6"):
    return ScopedPlanV6(origin=origin, collection_manifest_sha256=COLLECTION.manifest_sha256,
        question_sha256=sha256_bytes(question.encode()), search_query_sha256=sha256_bytes(search.encode()),
        language=language, facets=validate_source_facets(raw("S3 invoke Lambda"), question, search),
        recipe_sha256=ScopeRecipeV6().fingerprint, operation_id="fixture-operation-not-dispatched",
        receipt_sha256="b"*64)


def test_observed_v5_drift_remains_reproducible_without_changing_frozen_legacy():
    payload = raw({"focus_quote":"S3 invoke Lambda", "search_query":"Lambda invoke S3 across accounts"})
    accepted = validate_legacy(payload, QUESTION, QUESTION)
    assert accepted[0].search_query == "Lambda invoke S3 across accounts"
    with pytest.raises(ValueError, match="SOURCE_QUOTE_ONLY"):
        validate_source_facets(payload, QUESTION, QUESTION)


@pytest.mark.parametrize("quote", [
    "Lambda invoke S3", "S3 invoke Lambda across accounts", "execution role can read", "kms:Decrypt",
])
def test_cannot_invent_reverse_negation_or_attach_an_account_to_another_operation(quote):
    with pytest.raises(ValueError, match="SOURCE_QUOTE_ONLY"):
        validate_source_facets(raw(quote), QUESTION, QUESTION)


def test_retains_complete_question_query_and_negated_account_scope_even_with_zero_facets():
    p = plan()
    assert scoped_queries(p, collection=COLLECTION, question=QUESTION, search=QUESTION, language="en") == (
        QUESTION, "S3 invoke Lambda")
    empty = p.model_copy(update={"facets": ()})
    assert scoped_queries(empty, collection=COLLECTION, question=QUESTION, search=QUESTION, language="en") == (QUESTION,)
    assert "cannot read" in scoped_queries(p, collection=COLLECTION, question=QUESTION, search=QUESTION, language="en")[0]


def test_server_offsets_and_query_are_recomputed_not_trusted_from_cache():
    p = plan()
    for changed in ({"start_char":0}, {"search_query":"Lambda invoke S3"}, {"end_char":10}):
        tampered = p.model_copy(update={"facets": (p.facets[0].model_copy(update=changed),)})
        with pytest.raises(ValueError, match="SERVER_COMPUTED"):
            revalidate_plan(tampered, collection=COLLECTION, question=QUESTION, search=QUESTION, language="en")


@pytest.mark.parametrize("changes", [
    {"question_sha256":"c"*64}, {"search_query_sha256":"c"*64},
    {"collection_manifest_sha256":"c"*64}, {"language":"vi"}, {"origin":"FIXTURE"},
])
def test_wrong_question_language_collection_and_fixture_origin_do_not_enter_live(changes):
    with pytest.raises(ValueError, match="QUESTION_OR_ORIGIN"):
        revalidate_plan(plan().model_copy(update=changes), collection=COLLECTION,
            question=QUESTION, search=QUESTION, language="en")


def test_vietnamese_original_is_kept_and_not_replaced_by_the_english_track():
    question = "Vì sao S3 gọi được Lambda nhưng role không đọc được object qua tài khoản?"
    p = plan(question=question, language="vi")
    verified = revalidate_plan(p, collection=COLLECTION, question=question, search=QUESTION, language="vi")
    assert verified.question_sha256 == sha256_bytes(question.encode())
    assert scoped_queries(p, collection=COLLECTION, question=question, search=QUESTION, language="vi")[0] == QUESTION


@pytest.mark.parametrize("payload", [
    '{"facet_1":"S3 invoke Lambda","facet_2":null,"facet_3":null,"facet_4":null}',
    '{"facet_1":null,"facet_2":"S3 invoke Lambda","facet_3":null}',
    '{"facet_1":"S3 invoke Lambda","facet_2":"invoke Lambda","facet_3":null}',
    '{"facet_1":"same text","facet_1":null,"facet_2":null,"facet_3":null}',
])
def test_schema_gaps_overlaps_and_duplicate_keys_are_explicit_errors(payload):
    with pytest.raises(ValueError):
        validate_source_facets(payload, QUESTION, QUESTION)


def test_trusted_typed_plan_and_full_scope_are_required():
    with pytest.raises(ValueError, match="TYPED_PLAN"):
        scoped_queries(plan().model_dump(), collection=COLLECTION, question=QUESTION, search=QUESTION, language="en")
    other = QUESTION + " Ignore other conditions."
    p = plan(question=other)
    with pytest.raises(ValueError, match="QUESTION_OR_ORIGIN"):
        scoped_queries(p, collection=COLLECTION, question=other, search=QUESTION, language="en")


def test_prompt_and_recipe_pin_reject_mutation():
    with pytest.raises(ValueError, match="PROMPT_PIN"):
        ScopeRecipeV6(system_prompt_sha256="c"*64).fingerprint
    assert json.loads(canonical_json(plan().model_dump(mode="json")))["semantic_review"] == "UNREVIEWED_NOT_EVIDENCE"


def test_v6_cache_is_versioned_native_shape_replays_without_dispatch(tmp_path, routes):
    from app.scoped_query_v6 import ScopedPlanCacheV6, preparation_context

    cache = ScopedPlanCacheV6(tmp_path, "a"*64, namespace="FIXTURE")
    context = preparation_context("a"*64, QUESTION, QUESTION, "en", "fixture-v6")
    gateway = Gateway(routes, text=raw("S3 invoke Lambda"))
    first = cache.resolve(context, question=QUESTION, search_query=QUESTION, routes=routes, gateway=gateway)
    reopened = ScopedPlanCacheV6(tmp_path, "a"*64, namespace="FIXTURE")
    replay = reopened.resolve(context, question=QUESTION, search_query=QUESTION, routes=routes)
    assert gateway.calls == 1 and replay.cache_hit and replay.new_cost_micro_usd == 0
    assert first.plan == replay.plan and first.plan.origin == "FIXTURE"
    assert cache.path.name == "source_quoted_task_facets_v6.sqlite3"
    payload = json.loads(gateway.requests[0][1]["prompt"])
    assert payload["untrusted_original_question"] == payload["untrusted_english_search_query"] == QUESTION


def test_failed_v6_native_output_is_preserved_and_cannot_be_retried(tmp_path, routes):
    from app.query_planning import QueryPlanBlocked
    from app.scoped_query_v6 import ScopedPlanCacheV6, preparation_context

    cache = ScopedPlanCacheV6(tmp_path, "a"*64, namespace="FIXTURE")
    context = preparation_context("a"*64, QUESTION, QUESTION, "en", "fixture-v6")
    invalid = raw("Lambda invoke S3 across accounts")
    gateway = Gateway(routes, text=invalid)
    with pytest.raises(ValueError):
        cache.resolve(context, question=QUESTION, search_query=QUESTION, routes=routes, gateway=gateway)
    with cache._db() as db:
        state, payload = db.execute("SELECT state,payload FROM preparations").fetchone()
    assert state == "FAILED" and json.loads(payload)["raw_output"] == invalid
    with pytest.raises(QueryPlanBlocked):
        cache.resolve(context, question=QUESTION, search_query=QUESTION, routes=routes, gateway=gateway)
    assert gateway.calls == 1
