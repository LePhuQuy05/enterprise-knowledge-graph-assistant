import time

import pytest

from app.collection_graph import graph_fingerprint
from app.collection_graph_v2 import CollectionGraphOverlayV2, load_collection_graph_v2
from app.contracts import Entity, RelationAssertion, canonical_json, sha256_bytes
from tests.test_collection_graph_runtime import graph_case as graph_case


def convert(case, *, extra_entities=0):
    collection, _, graph, _ = case
    old = graph.overlay
    extra = tuple(Entity.create(tenant_id=old.tenant_id, canonical_key=f"capacity/{i}",
        canonical_name=f"Capacity Entity {i}", entity_type="CONCEPT", aliases=(), canonicalization_version="fixture-v2")
        for i in range(extra_entities))
    data = {k: getattr(old, k) for k in ("tenant_id", "collection_version", "collection_manifest_sha256", "curation_sha256",
        "upstream_review_queue_sha256", "mentions", "assertions")}
    data.update(entities=old.entities + extra, graph_sha256=graph_fingerprint(collection.manifest_sha256,
        old.entities + extra, old.mentions, old.assertions), source_inventory_sha256="2" * 64,
        extraction_recipe_sha256="3" * 64, aliases_sha256="4" * 64, inventory_units=48, processed_units=24)
    return CollectionGraphOverlayV2(**data)


def load(case, overlay):
    path = case[3].with_name("graph-v2.json")
    raw = canonical_json(overlay.model_dump(mode="json")) + b"\n"
    path.write_bytes(raw)
    return load_collection_graph_v2(path, case[0], trusted_sha256=sha256_bytes(raw), allow_experimental=True)


def test_v2_more_than_v1_entity_bound_and_indexed_source_validation(graph_case):
    overlay = convert(graph_case, extra_entities=1500)
    start = time.monotonic()
    graph = load(graph_case, overlay)
    assert len(graph.overlay.entities) > 1000
    assert time.monotonic() - start < 5  # Generous regression bound on tiny fixture, not a cloud SLA.


@pytest.mark.parametrize("change", ["endpoint", "quote", "qualifier", "coverage", "tenant"])
def test_v2_rejects_broken_provenance_or_scope(graph_case, change):
    overlay = convert(graph_case)
    if change == "coverage":
        with pytest.raises(ValueError, match="COVERAGE"):
            CollectionGraphOverlayV2.model_validate(overlay.model_dump() | {"complete_source_processing": True})
        return
    if change == "endpoint":
        overlay = overlay.model_copy(update={"mentions": overlay.mentions[:1]})
    elif change == "quote":
        a = overlay.assertions[0]
        overlay = overlay.model_copy(update={"assertions": (a.model_copy(update={"evidence": a.evidence.model_copy(update={"quote": "X" * len(a.evidence.quote)})}),)})
    elif change == "qualifier":
        from app.contracts import Qualifier

        a = overlay.assertions[0]
        overlay = overlay.model_copy(update={"assertions": (a.model_copy(update={"qualifiers": (Qualifier(kind="OTHER", text="Invented condition"),)}),)})
    else:
        overlay = overlay.model_copy(update={"tenant_id": "other"})
    overlay = overlay.model_copy(update={"assertions": tuple(a.model_copy(update={"assertion_id": RelationAssertion.expected_id(a.model_dump())}) for a in overlay.assertions)})
    overlay = overlay.model_copy(update={"graph_sha256": graph_fingerprint(overlay.collection_manifest_sha256,
        overlay.entities, overlay.mentions, overlay.assertions)})
    with pytest.raises(ValueError, match="GRAPH_V2"):
        load(graph_case, overlay)
