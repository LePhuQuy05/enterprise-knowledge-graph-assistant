"""Regression tests for the real pilot's long IDs and incorrect model offsets."""

import copy
import json

import pytest

from app.contracts import canonical_json
from app.extraction import (
    ExtractionError,
    deepseek_extraction_recipe,
    deepseek_quote_recipe,
    raw_cache_key,
    validate_output,
)
from app.extraction_cache import ExtractionCache
from app.model_gateway import CallContext
from tests.test_external_pilot import configured
from tests.test_extraction import aliases, source

TEXT = "🚀 Service A is mentioned earlier. Service A does not support Service B in region demo-1."
QUOTE = "Service A does not support Service B in region demo-1."


def quoted():
    return {"entities": [
        {"local_id": "an identifier longer than thirty two characters", "name": "Service A", "entity_type": "CONCEPT"},
        {"local_id": "b", "name": "Service B", "entity_type": "CONCEPT"},
    ], "relationships": [{"subject": "an identifier longer than thirty two characters", "object": "b",
        "predicate": "SUPPORTS", "polarity": "NEGATIVE", "evidence_quote": QUOTE,
        "qualifiers": [{"kind": "REGION", "text": "region demo-1"}]}]}


def validate(data=None, *, text=TEXT):
    doc, chunk = source(text)
    recipe = deepseek_quote_recipe(aliases(), metadata_sha256="0" * 64)
    return validate_output(json.dumps(quoted() if data is None else data), chunk=chunk, document=doc,
                           stop_reason="end_turn", recipe=recipe)


def test_quote_context_resolves_repeated_names_unicode_and_long_local_ids():
    result = validate()
    assert result.offset_repairs == ()
    assert [e.local_id for e in result.output.entities] == ["e0", "e1"]
    mention = result.output.entities[0].mentions[0]
    assert mention.start_char == TEXT.index(QUOTE)
    assert TEXT[mention.start_char:mention.end_char] == "Service A"
    assert result.output.relationships[0].polarity == "NEGATIVE"


@pytest.mark.parametrize("kind", ["duplicate_id", "unknown_endpoint", "missing_quote", "missing_qualifier", "repeated_endpoint", "repeated_quote"])
def test_resolver_rejects_ambiguity_and_keeps_existing_source_invariants(kind):
    data, text = quoted(), TEXT
    if kind == "duplicate_id":
        data["entities"][1]["local_id"] = data["entities"][0]["local_id"]
    elif kind == "unknown_endpoint":
        data["relationships"][0]["object"] = "not-defined"
    elif kind == "missing_quote":
        data["relationships"][0]["evidence_quote"] = "An invented statement."
    elif kind == "missing_qualifier":
        data["relationships"][0]["qualifiers"][0]["text"] = "region not-in-source"
    elif kind == "repeated_endpoint":
        data["relationships"][0]["evidence_quote"] = text
    else:
        text += " " + QUOTE
    with pytest.raises(ExtractionError):
        validate(data, text=text)


def test_entity_only_mentions_need_unique_context_when_surface_repeats():
    data = quoted()
    data["relationships"] = []
    with pytest.raises(ExtractionError, match="AMBIGUOUS"):
        validate(data)
    data["entities"][0]["mention_contexts"] = ["Service A is mentioned earlier."]
    assert validate(data).output.entities[0].mentions[0].start_char == TEXT.index("Service A")


def test_v2_cache_retains_raw_quotes_and_replays_without_calls(tmp_path):
    gateway, table = configured(), aliases()
    recipe = deepseek_quote_recipe(table, metadata_sha256="0" * 64)
    legacy = deepseek_extraction_recipe(table, metadata_sha256="0" * 64)
    assert raw_cache_key("demo", TEXT, recipe) != raw_cache_key("demo", TEXT, legacy)
    original = gateway.provider.request

    def request(*args):
        response = copy.deepcopy(original(*args))
        response["choices"][0]["message"]["content"] = canonical_json(quoted()).decode()
        return response

    gateway.provider.request = request
    cache = ExtractionCache(tmp_path, namespace="FIXTURE")
    doc, chunk = source(TEXT)
    kwargs = dict(chunk=chunk, document=doc, recipe=recipe, aliases=table)
    context = CallContext(tenant_id="demo", run_id="pilot-test", request_id="quotes", stage="EXTRACT")
    first = cache.resolve(context, gateway=gateway, **kwargs)
    second = cache.resolve(context, **kwargs)
    assert not first.cache_hit and second.cache_hit and gateway.provider.calls == 1
    assert first.canonical == second.canonical
    assert json.loads(first.record.attempts[0].text) == quoted()
    assert len(first.canonical.assertions) == 1
