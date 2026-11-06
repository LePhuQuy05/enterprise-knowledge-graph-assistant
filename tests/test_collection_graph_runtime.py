import dataclasses

import pytest

from app.collection_graph import graph_fingerprint
from app.collection_graph_runtime import (
    CollectionGraphOverlayV1,
    load_collection_graph,
    prepare_collection_graph_answer,
    validate_graph_sources,
)
from app.collection_lexical import build_global_lexical_index, load_global_lexical_index
from app.collections import build_collection, load_collection
from app.contracts import RelationAssertion, canonical_json, sha256_bytes
from app.fixtures import FIXTURE_EMBEDDING_MODEL, FIXTURE_TENANT, fixture_vector
from app.generation import GenerationError, GenerationRecipeV9, generation_request


@pytest.fixture
def graph_case(snapshot_dir, loaded, tmp_path):
    path = build_collection(snapshot_dir.parent, [snapshot_dir], tenant_id=FIXTURE_TENANT,
        embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    collection = load_collection(path, tenant_id=FIXTURE_TENANT, embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    lexical_path = build_global_lexical_index(collection, tmp_path / "lexical")
    lexical = load_global_lexical_index(lexical_path, collection)
    content = loaded.content
    overlay = CollectionGraphOverlayV1(tenant_id=FIXTURE_TENANT, collection_version=collection.manifest.collection_version,
        collection_manifest_sha256=collection.manifest_sha256, curation_sha256="0" * 64, upstream_review_queue_sha256="1" * 64,
        entities=content.entities, mentions=content.mentions, assertions=content.assertions,
        graph_sha256=graph_fingerprint(collection.manifest_sha256, content.entities, content.mentions, content.assertions))
    raw = canonical_json(overlay.model_dump(mode="json")) + b"\n"
    graph_path = tmp_path / "graph.json"
    graph_path.write_bytes(raw)
    graph = load_collection_graph(graph_path, collection, trusted_sha256=sha256_bytes(raw), allow_experimental=True)
    yield collection, lexical, graph, graph_path
    lexical.close()


def prepare(case, **overrides):
    collection, lexical, graph, _ = case
    arguments = dict(tenant_id=FIXTURE_TENANT, owner_id="reviewer-a", run_id="graph-fixture",
        recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64), allow_experimental=True)
    return prepare_collection_graph_answer(collection, lexical, graph, "Which services need Permission D?",
        fixture_vector("Resource C Permission D"), **(arguments | overrides))


def test_graph_pipeline_preserves_source_paths_and_independent_three_list_fusion(graph_case):
    view, prepared, trace = prepare(graph_case, hops=2)
    assert prepared.binding.used_graph and prepared.binding.mode_used == "graph_2hop"
    assert prepared.packed.assertion_ids
    assert trace["vector_ranking"] and trace["lexical_ranking"] and trace["baseline_ranking"]
    assert trace["policy"] == "COLLECTION_REVIEWED_GRAPH_V3_BOUNDED_TIME"
    negative = {a.assertion_id for a in graph_case[2].overlay.assertions if a.polarity == "NEGATIVE"}
    assert not negative.intersection(prepared.packed.assertion_ids)
    for key in prepared.packed.assertion_ids:
        assertion = next(a for a in view.content.assertions if a.assertion_id == key)
        assert assertion.chunk_id in prepared.packed.chunk_ids
        doc = view.documents[assertion.document_version_id]
        assert doc.normalized_text[assertion.evidence.start_char:assertion.evidence.end_char] == assertion.evidence.quote
    assert "GRAPH_AI_REVIEWED_EXPERIMENTAL" in prepared.binding.warnings
    _, prompt = generation_request(prepared.question, view, prepared.packed, FIXTURE_TENANT, prepared.recipe)
    assert "untrusted_sources" in prompt


def test_timeout_returns_exact_hybrid_and_reports_the_requested_graph_method(graph_case):
    times = iter((0, 1))
    _, prepared, trace = prepare(graph_case, clock=lambda: next(times), time_budget_ms=1)
    assert trace["timed_out"] and "GRAPH_TIMEOUT" in trace["warnings"]
    assert prepared.binding.mode_requested == "graph_1hop" and prepared.binding.mode_used == "hybrid"
    assert not prepared.binding.used_graph and not prepared.packed.assertion_ids
    assert prepared.retrieved_chunk_ids == trace["baseline_ranking"]


def test_mentions_without_any_relation_never_masquerade_as_graph_success(graph_case):
    graph = graph_case[2]
    overlay = graph.overlay.model_copy(update={"assertions": ()})
    overlay = overlay.model_copy(update={"graph_sha256": graph_fingerprint(overlay.collection_manifest_sha256,
        overlay.entities, overlay.mentions, ())})
    case = (graph_case[0], graph_case[1], dataclasses.replace(graph, overlay=overlay), graph_case[3])
    _, prepared, trace = prepare(case)
    assert trace["graph_candidates"] and not any(c["assertion_path"] for c in trace["graph_candidates"])
    assert prepared.retrieved_chunk_ids == trace["baseline_ranking"]
    assert not prepared.binding.used_graph and "GRAPH_NO_RELATION_PATH" in trace["warnings"]


def test_graph_context_without_packed_path_falls_back_to_unmodified_hybrid(graph_case, monkeypatch):
    from app import collection_graph_runtime as runtime

    real = runtime.pack_context
    calls = []

    def no_path(view, result):
        packed = real(view, result)
        calls.append(result)
        return dataclasses.replace(packed, assertion_ids=()) if len(calls) == 1 else packed

    monkeypatch.setattr(runtime, "pack_context", no_path)
    _, prepared, trace = prepare(graph_case)
    assert len(calls) == 2 and prepared.retrieved_chunk_ids == trace["baseline_ranking"]
    assert not prepared.binding.used_graph and "GRAPH_PATH_NOT_PACKED" in prepared.binding.warnings


@pytest.mark.parametrize("change", ["checksum", "collection", "source", "tenant", "duplicate", "opt_in"])
def test_loader_fails_closed_on_corruption_scope_source_and_missing_opt_in(graph_case, change):
    collection, _, graph, path = graph_case
    overlay, digest, opt_in = graph.overlay, graph.file_sha256, True
    if change == "checksum":
        digest = "f" * 64
    elif change == "collection":
        overlay = overlay.model_copy(update={"collection_manifest_sha256": "f" * 64})
    elif change == "source":
        assertion = overlay.assertions[0]
        bad = assertion.model_dump()
        bad["evidence"]["quote"] = "x" + assertion.evidence.quote[1:]
        bad["assertion_id"] = RelationAssertion.expected_id(bad)
        overlay = overlay.model_copy(update={"assertions": (RelationAssertion.model_validate(bad), *overlay.assertions[1:])})
    elif change == "tenant":
        overlay = overlay.model_copy(update={"tenant_id": "other"})
    elif change == "duplicate":
        overlay = overlay.model_copy(update={"entities": (*overlay.entities, overlay.entities[0])})
    else:
        opt_in = False
    if overlay != graph.overlay:
        overlay = overlay.model_copy(update={"graph_sha256": graph_fingerprint(overlay.collection_manifest_sha256,
            overlay.entities, overlay.mentions, overlay.assertions)})
        raw = canonical_json(overlay.model_dump(mode="json"))
        path.write_bytes(raw)
        digest = sha256_bytes(raw)
    with pytest.raises((GenerationError, ValueError)):
        load_collection_graph(path, collection, trusted_sha256=digest, allow_experimental=opt_in)


def test_graph_preparation_does_not_cross_tenants_or_enable_itself(graph_case):
    with pytest.raises(GenerationError, match="OPT_IN"):
        prepare(graph_case, allow_experimental=False)
    with pytest.raises(GenerationError, match="RELEASE_BINDING"):
        prepare(graph_case, tenant_id="other")
    with pytest.raises(GenerationError, match="HOP_BOUND"):
        prepare(graph_case, hops=3)
    validate_graph_sources(graph_case[2].overlay, graph_case[0])


def test_real_source_quote_cannot_be_attached_to_an_unrelated_entity(graph_case):
    collection, _, graph, _ = graph_case
    original = graph.overlay.entities[0]
    unrelated = original.model_copy(update={"canonical_name": "Unrelated entity", "aliases": ("Unrelated entity",)})
    entities = (unrelated, *graph.overlay.entities[1:])
    overlay = graph.overlay.model_copy(update={"entities": entities})
    overlay = overlay.model_copy(update={"graph_sha256": graph_fingerprint(collection.manifest_sha256,
        entities, overlay.mentions, overlay.assertions)})
    with pytest.raises(GenerationError, match="MENTION_ALIAS"):
        validate_graph_sources(overlay, collection)
