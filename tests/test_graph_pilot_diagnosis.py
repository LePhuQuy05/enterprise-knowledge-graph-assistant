import json

from scripts.summarize_collection_graph_pilot import anchor_errors, occurrences


def test_occurrence_diagnostics_count_overlaps_like_the_resolver():
    assert occurrences("banana", "ana") == 2
    assert occurrences("anything", "") == 0


def test_realistic_nested_name_failure_is_not_a_missing_source_quote():
    quote = "CloudFront can use CloudFront Origin Shield."
    raw = json.dumps({"relationships": [{"evidence_quote": quote,
        "subject": {"name": "CloudFront", "entity_type": "AWS_SERVICE"},
        "object": {"name": "CloudFront Origin Shield", "entity_type": "FEATURE"},
        "predicate": "USES", "polarity": "POSITIVE", "qualifiers": []}], "standalone_entities": []})
    assert anchor_errors(raw, quote) == ["RELATION_ENDPOINT_REPEATED"]
    assert anchor_errors('{"relationships": [', quote) == ["OUTPUT_NOT_COMPLETE_V3_SCHEMA"]
