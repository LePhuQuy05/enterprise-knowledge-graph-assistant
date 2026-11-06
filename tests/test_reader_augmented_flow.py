import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.answer_cache import AnswerReceipt
from app.contracts import canonical_json, sha256_bytes
from app.cost import TokenUsage
from app.literal_source_catalog import SourceCatalog
from app.reader_augmented_flow import augment_with_reader, validate_selection
from app.routed_reader_json_v2 import READ_SYSTEM
from tests.test_scoped_method_flow import QUESTION, fixture, prepare


def selection(collection, docs, chunks, *, question=QUESTION):
    keys = tuple(chunks)
    message = canonical_json({"untrusted_question": question, "untrusted_english_search": question, "language": "en",
        "untrusted_sources": [{"source_id": f"r{i}", "untrusted_title": docs[chunks[k].document_version_id].title,
            "untrusted_text": chunks[k].text} for i, k in enumerate(keys, 1)]}).decode()
    catalog = SourceCatalog(collection.manifest_sha256, question, question, "en", keys, message, {"fixture": True})
    recipe = sha256_bytes(READ_SYSTEM.encode())
    identity = sha256_bytes(canonical_json({"binding": catalog.fingerprint, "recipe": recipe,
        "system": READ_SYSTEM, "prompt": catalog.message, "max_output_tokens": 768}))
    receipt = AnswerReceipt(namespace="FIXTURE", cache_key="fixture", operation_id="fixture",
        recipe_sha256=recipe, binding_sha256=identity, raw_output='{"source_ids":["r11","r12"]}',
        stop_reason="end_turn", usage=TokenUsage(input_tokens=100, output_tokens=12), original_cost_micro_usd=50)
    return validate_selection(catalog, receipt, collection)


def test_q_qb_keep_shared_pool_and_native_plan_while_disclosing_reader_augmentation(monkeypatch):
    collection, docs, chunks, _, plan = fixture(monkeypatch)
    reader = selection(collection, docs, chunks)
    graph = SimpleNamespace(facts=[])
    q, qb = [augment_with_reader(prepare(collection, mode, plan), reader, collection, graph)
        for mode in ("hybrid_multi_query", "hybrid_multi_query_balanced")]
    assert q.trace["pool_fingerprint"] == qb.trace["pool_fingerprint"]
    assert q.trace["bundles"] == qb.trace["bundles"]
    assert q.trace["native_v6_plan"] == qb.trace["native_v6_plan"]
    assert q.trace["variant"].endswith("NOT_PURE_BASELINE")
    assert not q.trace["balanced_allocator_used"] and qb.trace["balanced_allocator_used"]


def test_reader_selection_cannot_cross_task_or_live_namespace(monkeypatch):
    collection, docs, chunks, _, plan = fixture(monkeypatch)
    reader = selection(collection, docs, chunks, question="A different question?")
    with pytest.raises(ValueError, match="ORIGINAL_TASK_CHANGED"):
        augment_with_reader(prepare(collection, "hybrid_multi_query", plan), reader, collection, SimpleNamespace(facts=[]))
    live = SimpleNamespace(manifest_sha256=collection.manifest_sha256, manifest=SimpleNamespace(purpose="LIVE"))
    with pytest.raises(ValueError, match="NAMESPACE"):
        validate_selection(reader.catalog, reader.receipt, live)


def test_reader_does_not_make_native_q_without_facets_ready(monkeypatch):
    collection, docs, chunks, _, plan = fixture(monkeypatch)
    flow = prepare(collection, "hybrid_multi_query", plan.model_copy(update={"facets": ()}))
    augmented = augment_with_reader(flow, selection(collection, docs, chunks), collection, SimpleNamespace(facts=[]))
    assert not augmented.ready and augmented.status == "NOT_READY_NO_DISTINCT_NATIVE_FACETS"


def test_forged_source_text_is_rejected_even_with_a_rebound_fixture_receipt(monkeypatch):
    collection, docs, chunks, _, plan = fixture(monkeypatch)
    reader = selection(collection, docs, chunks)
    payload = json.loads(reader.catalog.message)
    payload["untrusted_sources"][0]["untrusted_text"] = "forged"
    catalog = replace(reader.catalog, message=canonical_json(payload).decode())
    receipt = reader.receipt.model_copy(update={"binding_sha256": sha256_bytes(canonical_json({"binding": catalog.fingerprint,
        "recipe": reader.receipt.recipe_sha256, "system": READ_SYSTEM, "prompt": catalog.message, "max_output_tokens": 768}))})
    forged = validate_selection(catalog, receipt, collection)
    with pytest.raises(ValueError, match="SOURCE_BYTES"):
        augment_with_reader(prepare(collection, "hybrid_multi_query", plan), forged, collection, SimpleNamespace(facts=[]))
