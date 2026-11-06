"""V4 changes occurrence addressing, never the source/qualifier or V3 cache rules."""

import copy
import json

import pytest

from app.contracts import canonical_json
from app.extraction import (
    ExtractionError,
    deepseek_anchored_recipe,
    deepseek_evidence_recipe,
    extraction_request,
    raw_cache_key,
    validate_output,
)
from app.extraction_cache import ExtractionCache
from app.model_gateway import CallContext
from tests.test_external_pilot import configured
from tests.test_extraction import aliases, source

QUOTE = "CloudFront can use CloudFront Origin Shield when configured."
TEXT = "Tiếng Việt 🚀\n" + QUOTE


def response():
    return {"relationships": [{"evidence_quote": QUOTE,
        "subject": {"name": "CloudFront", "entity_type": "AWS_SERVICE", "occurrence": 1},
        "object": {"name": "CloudFront Origin Shield", "entity_type": "FEATURE", "occurrence": 1},
        "predicate": "USES", "polarity": "POSITIVE", "qualifiers": [{"kind": "CONFIGURATION", "text": "when configured"}]}]}


def validate(data=None, *, text=TEXT, stop_reason="end_turn"):
    doc, chunk = source(text)
    return validate_output(json.dumps(response() if data is None else data), chunk=chunk, document=doc,
        stop_reason=stop_reason, recipe=deepseek_anchored_recipe(aliases(), metadata_sha256="0" * 64))


def test_explicit_nonoverlapping_occurrences_resolve_nested_names_and_unicode():
    result = validate()
    subject, target = result.output.entities
    assert subject.mentions[0].start_char == TEXT.index("CloudFront")
    assert target.mentions[0].start_char == TEXT.index("CloudFront Origin Shield")
    assert subject.mentions[0].end_char <= target.mentions[0].start_char
    assert not result.offset_repairs
    assert result.output.relationships[0].qualifiers[0].text == "when configured"


@pytest.mark.parametrize("mutation,code", [
    ("overlap", "OVERLAP"), ("missing_occurrence", "OCCURRENCE_MISSING"),
    ("outside_evidence", "OCCURRENCE_MISSING"), ("invented_quote", "EVIDENCE_MISSING"),
    ("missing_qualifier", "QUALIFIER_NOT_IN_EVIDENCE"), ("extra_field", "SCHEMA_INVALID"),
    ("boolean_occurrence", "SCHEMA_INVALID"), ("float_occurrence", "SCHEMA_INVALID"),
    ("zero_occurrence", "SCHEMA_INVALID"), ("v3_shape", "SCHEMA_INVALID"),
])
def test_unsafe_or_ambiguous_model_records_fail_closed(mutation, code):
    data = response()
    relation = data["relationships"][0]
    if mutation == "overlap":
        relation["subject"]["occurrence"] = 2
    elif mutation == "missing_occurrence":
        relation["subject"]["occurrence"] = 3
    elif mutation == "outside_evidence":
        relation["subject"]["name"] = "Tiếng Việt"
    elif mutation == "invented_quote":
        relation["evidence_quote"] = "CloudFront uses Shield."
    elif mutation == "missing_qualifier":
        relation["qualifiers"][0]["text"] = "in every region"
    elif mutation == "extra_field":
        relation["inferred"] = True
    elif mutation == "v3_shape":
        del relation["subject"]["occurrence"]
    else:
        relation["subject"]["occurrence"] = {"boolean_occurrence": True, "float_occurrence": 1.0, "zero_occurrence": 0}[mutation]
    with pytest.raises(ExtractionError, match=code):
        validate(data)


def test_evidence_itself_must_still_be_unique_and_truncation_never_passes():
    with pytest.raises(ExtractionError, match="EVIDENCE_MISSING_OR_AMBIGUOUS"):
        validate(text=TEXT + "\n" + QUOTE)
    with pytest.raises(ExtractionError, match="INCOMPLETE_STOP"):
        validate(stop_reason="max_tokens")
    assert not validate({"relationships": []}).output.entities


def test_v3_still_rejects_the_ambiguous_name_and_uses_a_distinct_cache_identity():
    v4 = deepseek_anchored_recipe(aliases(), metadata_sha256="0" * 64)
    v3 = deepseek_evidence_recipe(aliases(), metadata_sha256="0" * 64)
    assert raw_cache_key("demo", TEXT, v4) != raw_cache_key("demo", TEXT, v3)
    old = response()
    for endpoint in ("subject", "object"):
        del old["relationships"][0][endpoint]["occurrence"]
    old["standalone_entities"] = []
    doc, chunk = source(TEXT)
    with pytest.raises(ExtractionError, match="CONTEXT_MISSING_OR_AMBIGUOUS"):
        validate_output(json.dumps(old), chunk=chunk, document=doc, stop_reason="end_turn", recipe=v3)
    assert '"occurrence"' not in extraction_request(TEXT, v3)[0]
    assert "1-based" in extraction_request(TEXT, v4)[0]


def test_negation_and_conditions_survive_the_new_resolver_without_rewording():
    quote = "Service A does not support Service B in region demo-1."
    data = {"relationships": [{"evidence_quote": quote,
        "subject": {"name": "Service A", "entity_type": "CONCEPT", "occurrence": 1},
        "object": {"name": "Service B", "entity_type": "CONCEPT", "occurrence": 1},
        "predicate": "SUPPORTS", "polarity": "NEGATIVE", "qualifiers": [{"kind": "REGION", "text": "region demo-1"}]}]}
    relation = validate(data, text=quote).output.relationships[0]
    assert relation.polarity == "NEGATIVE" and relation.qualifiers[0].text == "region demo-1"
    assert relation.evidence.quote == quote


def test_new_recipe_keeps_raw_receipt_replays_and_does_not_invoke_provider_twice(tmp_path):
    gateway, table = configured(), aliases()
    recipe = deepseek_anchored_recipe(table, metadata_sha256="0" * 64)
    original = gateway.provider.request

    def request(*args):
        payload = copy.deepcopy(original(*args))
        payload["choices"][0]["message"]["content"] = canonical_json(response()).decode()
        return payload

    gateway.provider.request = request
    cache = ExtractionCache(tmp_path, namespace="FIXTURE")
    doc, chunk = source(TEXT)
    context = CallContext(tenant_id="demo", run_id="pilot-test", request_id="anchored", stage="EXTRACT")
    kwargs = dict(chunk=chunk, document=doc, recipe=recipe, aliases=table)
    first = cache.resolve(context, gateway=gateway, **kwargs)
    replay = cache.resolve(context, **kwargs)
    assert replay.cache_hit and first.canonical == replay.canonical and gateway.provider.calls == 1
    assert json.loads(first.record.attempts[0].text) == response()
