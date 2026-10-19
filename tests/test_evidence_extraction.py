"""V3 regression cases derived from failed v2 responses, with synthetic source text."""

import copy
import json

import pytest

from app.contracts import canonical_json
from app.extraction import (
    ExtractionError,
    deepseek_evidence_recipe,
    deepseek_quote_recipe,
    extraction_request,
    raw_cache_key,
    validate_output,
)
from app.extraction_cache import ExtractionCache
from app.model_gateway import CallContext
from scripts.run_bge_extraction_pilot import RUN, settled_rejection
from tests.test_external_pilot import configured
from tests.test_extraction import aliases, source

TEXT = "Earlier Service A. Service A does not support Service B in region demo-1."
QUOTE = "Service A does not support Service B in region demo-1."


def response():
    return {"standalone_entities": [], "relationships": [{
        "evidence_quote": QUOTE, "subject": {"name": "Service A", "entity_type": "CONCEPT"},
        "predicate": "SUPPORTS", "object": {"name": "Service B", "entity_type": "CONCEPT"},
        "polarity": "NEGATIVE", "qualifiers": [{"kind": "REGION", "text": "region demo-1"}],
    }]}


def check(data, text=TEXT):
    doc, chunk = source(text)
    return validate_output(json.dumps(data), chunk=chunk, document=doc, stop_reason="end_turn",
                           recipe=deepseek_evidence_recipe(aliases(), metadata_sha256="0" * 64))


def test_inline_names_derive_ids_and_preserve_negation_qualifier_and_exact_offsets():
    result = check(response())
    assert [e.local_id for e in result.output.entities] == ["e0", "e1"]
    relation = result.output.relationships[0]
    assert relation.polarity == "NEGATIVE" and relation.qualifiers[0].text == "region demo-1"
    assert result.output.entities[0].mentions[0].start_char == TEXT.index(QUOTE)


@pytest.mark.parametrize("kind", ["endpoint_outside", "invented_quote", "ambiguous_evidence", "ambiguous_endpoint",
                                  "missing_qualifier", "self_loop", "type_conflict", "old_wire_shape"])
def test_fail_closed_instead_of_inventing_edges(kind):
    data, text = response(), TEXT
    rel = data["relationships"][0]
    if kind == "endpoint_outside":
        rel["evidence_quote"] = "Service B in region demo-1."
    elif kind == "invented_quote":
        rel["evidence_quote"] = "Service A supports Service B."
    elif kind == "ambiguous_evidence":
        text += " " + QUOTE
    elif kind == "ambiguous_endpoint":
        rel["evidence_quote"] = text
    elif kind == "missing_qualifier":
        rel["qualifiers"][0]["text"] = "region imaginary-2"
    elif kind == "self_loop":
        rel["object"] = copy.deepcopy(rel["subject"])
    elif kind == "type_conflict":
        data["standalone_entities"] = [{"name": "Service A", "entity_type": "RESOURCE", "context_quote": QUOTE}]
    else:
        rel["subject"] = "Service A"
    with pytest.raises(ExtractionError):
        check(data, text)


def test_no_relation_is_valid_and_standalone_mentions_need_grounding():
    result = check({"relationships": [], "standalone_entities": [{
        "name": "Service A", "entity_type": "CONCEPT", "context_quote": "Earlier Service A."}]})
    assert len(result.output.entities) == 1 and not result.output.relationships
    assert not check({"relationships": [], "standalone_entities": []}).output.entities


def test_v2_identity_is_unchanged_and_v3_replays_without_paid_call(tmp_path):
    table, gateway = aliases(), configured()
    recipe = deepseek_evidence_recipe(table, metadata_sha256="0" * 64)
    old = deepseek_quote_recipe(table, metadata_sha256="0" * 64)
    assert raw_cache_key("demo", TEXT, recipe) != raw_cache_key("demo", TEXT, old)
    assert "unique-evidence-context-v1" == old.parameters["span_resolver_version"]
    assert "OMIT" in extraction_request(TEXT, recipe)[0]
    original = gateway.provider.request

    def request(*args):
        result = copy.deepcopy(original(*args))
        result["choices"][0]["message"]["content"] = canonical_json(response()).decode()
        return result

    gateway.provider.request = request
    cache = ExtractionCache(tmp_path, namespace="FIXTURE")
    doc, chunk = source(TEXT)
    kwargs = dict(chunk=chunk, document=doc, recipe=recipe, aliases=table)
    context = CallContext(tenant_id="demo", run_id="pilot-test", request_id="evidence-v3", stage="EXTRACT")
    first = cache.resolve(context, gateway=gateway, **kwargs)
    second = cache.resolve(context, **kwargs)
    assert len(first.canonical.assertions) == 1
    assert first.canonical == second.canonical and second.cache_hit and gateway.provider.calls == 1


def test_quarantine_skips_settled_invalid_output_and_never_retries_it(tmp_path):
    gateway, table = configured(), aliases()
    gateway.ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500_000)
    recipe = deepseek_evidence_recipe(table, metadata_sha256="0" * 64)
    original = gateway.provider.request

    def request(*args):
        result = copy.deepcopy(original(*args))
        bad = response()
        bad["relationships"][0]["qualifiers"][0]["text"] = "invented condition"
        result["choices"][0]["message"]["content"] = canonical_json(bad).decode()
        return result

    gateway.provider.request = request
    cache = ExtractionCache(tmp_path, namespace="FIXTURE")
    doc, chunk = source(TEXT)
    context = CallContext(tenant_id="demo", run_id=RUN, request_id="extract-bge-v3-" + chunk.chunk_id, stage="EXTRACT")
    with pytest.raises(ExtractionError):
        cache.resolve(context, chunk=chunk, document=doc, recipe=recipe, aliases=table, gateway=gateway)
    rejected = settled_rejection(cache, chunk, doc, recipe, gateway.ledger)
    assert rejected["validation_error"] == "EXTRACTION_QUALIFIER_NOT_IN_EVIDENCE"
    assert rejected["settled_attempts"] == 1 and rejected["paid_retry_calls"] == 0
    assert gateway.provider.calls == 1
    with pytest.raises(Exception, match="settlement verification"):
        settled_rejection(cache, chunk, doc, recipe, None)
