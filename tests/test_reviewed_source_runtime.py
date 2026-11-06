"""Source-bound integration and failure tests using offline, labelled fixtures."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest

from app.collection_graph import graph_fingerprint
from app.collection_graph_v2 import CollectionGraphOverlayV2
from app.collection_lexical import build_global_lexical_index, load_global_lexical_index
from app.collections import build_collection, load_collection
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.fixtures import FIXTURE_EMBEDDING_MODEL, fixture_recipes, fixture_vector
from app.generation import GenerationRecipeV9, generation_request
from app.method_execution import ReviewedGraphExecutionV3, answer_strict_prepared
from app.query_preparation_v3 import QueryPlanCacheV3
from app.reviewed_source_graph import (
    RELEASE_VERSION,
    evidence_adjacency,
    load_reviewed_source_graph,
)
from app.snapshot import SnapshotContent, build_snapshot
from app.source_action_proposals import propose_actions
from app.strict_collection_methods import prepare_strict_graph_answer
from app.syntax_graph_records import canonical_recipe, canonicalize
from tests.test_extraction import source
from tests.test_query_planning import context as context
from tests.test_query_planning import routes as routes
from tests.test_source_syntax import FIXTURE, parse
from tests.test_syntax_graph_records import fixture


@pytest.fixture(scope="module")
def reviewed_case(tmp_path_factory):
    root = tmp_path_factory.mktemp("reviewed-source-graph")
    documents, chunks, entities, mentions, relations, actions, sidecar = {}, {}, {}, {}, [], [], []
    candidates = []
    for kind in ("RELATION", "ACTION"):
        candidate, judgment, _, _, recipe, parsed = fixture(kind)
        candidates.append((kind, candidate, judgment, recipe, parsed))
    # Identical resource wording in a DIFFERENT document must retain its own ID.
    candidates.append(deepcopy(candidates[-1]))
    for text in ("CloudTrail does not deliver log files to an S3 bucket.",):
        parsed = parse(text)
        proposals, _ = propose_actions(parsed, text)
        assert len(proposals) == 1
        candidate = proposals[0]
        judgment = {"status": "AI_SOURCE_SUPPORTED_SOURCE_SCOPED", "candidate_sha256": sha256_bytes(canonical_json(candidate)),
            "reason_vi": "Offline mechanism fixture, not an independent semantic label."}
        candidates.append(("ACTION", candidate, judgment, recipe, parsed))
    for i, (kind, candidate, judgment, native_recipe, parsed) in enumerate(candidates):
        text = candidate["evidence"]["quote"]
        judgment = judgment | {"source_sha256": sha256_bytes(text.encode())}
        doc, chunk = source(text, suffix=str(i))
        es, ms, assertion, metadata = canonicalize(candidate, judgment, chunk=chunk, document=doc, parsed=parsed,
            native_recipe=native_recipe, installation=FIXTURE["installation"])
        documents[doc.document_version_id], chunks[chunk.chunk_id] = doc, chunk
        entities.update((e.entity_id, e) for e in es)
        mentions.update((m.mention_id, m) for m in ms)
        (relations if kind == "RELATION" else actions).append(assertion)
        sidecar.append(metadata)
    recipe = canonical_recipe(native_recipe, FIXTURE["installation"])
    _, embedding = fixture_recipes()
    base = build_snapshot(root / "snapshots", SnapshotContent(tuple(documents.values()), tuple(chunks.values())),
        {k: fixture_vector(c.text) for k, c in chunks.items()}, tenant_id="demo", purpose="FIXTURE",
        extraction_recipe=recipe, embedding_recipe=embedding, corpus_version="reviewed-runtime-fixture-v1",
        normalizer_version="fixture-v1", chunking_version="fixture-v1", tokenizer_version="fixture-v1")
    path = build_collection(base.parent, [base], tenant_id="demo", embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    collection = load_collection(path, tenant_id="demo", embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    overlay = CollectionGraphOverlayV2(tenant_id="demo", collection_version=collection.manifest.collection_version,
        collection_manifest_sha256=collection.manifest_sha256,
        graph_sha256=graph_fingerprint(collection.manifest_sha256, tuple(entities.values()), tuple(mentions.values()), tuple(relations)),
        source_inventory_sha256="1" * 64, extraction_recipe_sha256=recipe.fingerprint,
        aliases_sha256="2" * 64, curation_sha256="3" * 64, upstream_review_queue_sha256="4" * 64,
        inventory_units=10, processed_units=len(candidates), entities=tuple(entities.values()),
        mentions=tuple(mentions.values()), assertions=tuple(relations))
    artifact = {"version": "source-actions-reviewed-composite-v1", "recipe": recipe.model_dump(mode="json"),
        "source_manifest_sha256": collection.manifest_sha256, "assertions": actions,
        "serving_published": False, "human_reviewed": 0}
    lexical_path = build_global_lexical_index(collection, root / "lexical")
    with load_global_lexical_index(lexical_path, collection) as lexical:
        yield collection, lexical, {"relations-v2.json": overlay.model_dump(mode="json"), "actions.json": artifact, "sidecar.json": sidecar}


def write_release(tmp_path, case, payloads=None, change=None):
    data = deepcopy(case[2] if payloads is None else payloads)
    for name, value in data.items():
        (tmp_path / name).write_bytes(canonical_json(value) + b"\n")
    release = {"version": RELEASE_VERSION, "source_manifest_sha256": case[0].manifest_sha256,
        "collection_version": case[0].manifest.collection_version,
        "artifacts": {name: sha256_bytes((tmp_path / name).read_bytes()) for name in data},
        "human_reviewed": 0, "serving_published": False}
    if change:
        release.update(change)
    path = tmp_path / "release.json"
    path.write_bytes(canonical_json(release) + b"\n")
    return path, sha256_bytes(path.read_bytes())


def load(tmp_path, case, payloads=None, change=None):
    path, digest = write_release(tmp_path, case, payloads, change)
    return load_reviewed_source_graph(path, case[0], trusted_sha256=digest, allow_experimental=True)


def prepare(case, graph, question="When CloudTrail logging is turned on, where are log files delivered?", **overrides):
    return prepare_strict_graph_answer(case[0], case[1], graph, question, fixture_vector(question),
        **(dict(tenant_id="demo", owner_id="fixture-reviewer", run_id="fixture-runtime",
            recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64), allow_experimental=True, clock=lambda: 0.0) | overrides))


def test_real_source_scopes_actions_and_full_records_reach_context_and_trace(reviewed_case, tmp_path):
    graph = load(tmp_path, reviewed_case)
    result = prepare(reviewed_case, graph)
    assert result.outcome.schema_version == 3 and result.outcome.mechanism_status == "USED_GRAPH"
    records = json.loads(result.prepared.packed.serialized)["untrusted_graph_assertions"]
    assert records and all(r["kind"] == "ACTION" and "predicate" not in r for r in records)
    scoped = next(r for r in records if "logging is turned on" in r["record"]["clause_quote"])
    assert scoped["record"]["target_role"] == "DESTINATION"
    assert scoped["nodes"][1]["identity_scope"] == "SOURCE_LOCAL"
    assert all(x["metadata"]["origin"] for x in result.trace["source_records"])
    _, prompt = generation_request(result.prepared.question, result.view, result.prepared.packed, "demo", result.prepared.recipe)
    assert "logging is turned on" in prompt and all(k in result.view.chunks for k in result.outcome.context_chunk_ids)
    assert result.trace["hidden_fallbacks"] == 0


def test_negative_actions_are_inspectable_but_cannot_expand_and_actions_are_directed(reviewed_case, tmp_path):
    graph = load(tmp_path, reviewed_case)
    adjacent = evidence_adjacency(graph)
    negative = [e for e in graph.edges if e.polarity == "NEGATIVE"]
    assert negative
    assert not {e.assertion_id for e in negative} & {e.assertion_id for rows in adjacent.values() for e, _, _ in rows}
    for edge in (e for e in graph.edges if e.kind == "ACTION" and e.polarity == "POSITIVE"):
        assert (edge, edge.object_id, "FORWARD") in adjacent[edge.subject_id]
        assert all(row[0].assertion_id != edge.assertion_id for row in adjacent[edge.object_id])


def test_same_generic_resource_name_does_not_merge_across_sources(reviewed_case, tmp_path):
    graph = load(tmp_path, reviewed_case)
    nodes = [n for e in graph.edges for n in e.metadata()["nodes"] if n["identity_scope"] == "SOURCE_LOCAL"]
    a, b = [n for n in nodes if n["name"] == "Amazon S3 bucket that you specify"]
    assert a["node_id"] != b["node_id"]
    assert not any(link["entity_id"] in {a["node_id"], b["node_id"]}
        for link in prepare(reviewed_case, graph, question="How does the Amazon S3 bucket that you specify work?").trace["links"])


def test_zero_expansion_is_completed_explicit_and_can_generate_without_claiming_graph(reviewed_case, tmp_path):
    graph = load(tmp_path, reviewed_case)
    graph = replace(graph, edges=())  # Empty reviewed graph is a mechanism fixture, not a new release.
    result = prepare(reviewed_case, graph, question="How do quantum circuits work?")
    assert result.outcome.mechanism_status == "GRAPH_ZERO_EXPANSION"
    assert result.outcome.zero_expansion_reason == "NO_LINKED_ENTITY"
    assert result.outcome.effective_method == "graph_1hop" and not result.outcome.graph_path_used
    assert not result.prepared.packed.assertion_ids and not result.prepared.binding.used_graph
    assert result.trace["context_chunk_ids"] == result.trace["baseline_context_chunk_ids"]
    class Cache:
        def resolve(self, *args, **kwargs):
            return kwargs["binding"].used_graph
    assert answer_strict_prepared(result, Cache()) is False


def test_timeout_and_foreign_tenant_fail_without_baseline_answer(reviewed_case, tmp_path):
    graph = load(tmp_path, reviewed_case)
    result = prepare(reviewed_case, graph, time_budget_ms=1, traversal_clock=lambda: 1.0)
    assert result.outcome.mechanism_status == "BUDGET_EXCEEDED" and result.prepared is None
    class ForbiddenCache:
        def resolve(self, *args, **kwargs):
            pytest.fail("failure must not generate")
    with pytest.raises(ValueError, match="NOT_READY"):
        answer_strict_prepared(result, ForbiddenCache())
    with pytest.raises(ValueError, match="COLLECTION_BINDING"):
        prepare(reviewed_case, graph, tenant_id="foreign")


def test_path_that_cannot_fit_is_explicit_zero_expansion_with_original_sources(reviewed_case, tmp_path, monkeypatch):
    from app import reviewed_graph_runtime as runtime

    graph = load(tmp_path, reviewed_case)
    real = runtime.pack_context
    def restricted_first_pack(view, retrieval, **kwargs):
        if kwargs.get("graph_records"):
            return real(view, retrieval, proxy_token_budget=100, **kwargs)
        return real(view, retrieval, **kwargs)
    monkeypatch.setattr(runtime, "pack_context", restricted_first_pack)
    result = prepare(reviewed_case, graph)
    assert result.outcome.mechanism_status == "GRAPH_ZERO_EXPANSION"
    assert result.outcome.zero_expansion_reason == "NO_PATH_FITS_CONTEXT"
    assert "CONTEXT_BUNDLE_SKIPPED" in result.trace["attempted_path_packing_warnings"]
    assert not result.outcome.context_assertion_ids and result.prepared.packed.chunk_ids


def test_generic_verb_overlap_does_not_make_delivery_an_invocation_path(reviewed_case, tmp_path):
    graph = load(tmp_path, reviewed_case)
    result = prepare(reviewed_case, graph, question="Does CloudTrail invoke a target function?")
    assert result.outcome.mechanism_status == "GRAPH_ZERO_EXPANSION"
    assert result.outcome.zero_expansion_reason == "NO_RELEVANT_PATH"
    assert result.trace["paths"] and not result.trace["packed_paths"]


@pytest.mark.parametrize("change", ["version", "collection", "missing-actions", "missing-sidecar", "checksum", "opt-in"])
def test_release_failures_do_not_dispatch(reviewed_case, tmp_path, change):
    updates = {"version": {"version": "invented"}, "collection": {"source_manifest_sha256": "f" * 64}}.get(change)
    path, digest = write_release(tmp_path, reviewed_case, change=updates)
    if change.startswith("missing-"):
        (tmp_path / ("actions.json" if change == "missing-actions" else "sidecar.json")).unlink()
    with pytest.raises(ValueError, match="REVIEWED_GRAPH"):
        load_reviewed_source_graph(path, reviewed_case[0], trusted_sha256="f" * 64 if change == "checksum" else digest,
            allow_experimental=change != "opt-in")


@pytest.mark.parametrize("change", ["source", "role", "node", "scope", "merge", "candidate", "review", "recipe", "duplicate", "sidecar"])
def test_repinned_but_internally_inconsistent_records_are_rejected(reviewed_case, tmp_path, change):
    data = deepcopy(reviewed_case[2])
    row = data["actions.json"]["assertions"][0]
    metadata = next(m for m in data["sidecar.json"] if m["assertion_id"] == row["assertion_id"])
    if change == "source":
        row["evidence"]["quote"] = "X" + row["evidence"]["quote"][1:]
    elif change == "role":
        row["record"]["target_role"] = "KEY"
    elif change == "node":
        metadata["nodes"][1]["source_span"]["start_char"] += 1
    elif change == "scope":
        metadata["nodes"][1]["identity_scope"] = "CATALOG_TYPE"
    elif change == "merge":
        other = next(r for r in data["actions.json"]["assertions"] if r["assertion_id"] != row["assertion_id"]
            and r["record"]["target"]["name"] == row["record"]["target"]["name"])
        metadata["nodes"][1]["node_id"] = other["nodes"][1]["node_id"]
    elif change == "candidate":
        metadata["syntax_witness"]["record"]["modality"] = "REQUIRED"
    elif change == "review":
        metadata["source_review"]["status"] = "PENDING"
    elif change == "recipe":
        data["actions.json"]["recipe"]["version"] = "invented"
    elif change == "duplicate":
        data["actions.json"]["assertions"].append(deepcopy(row))
    else:
        data["sidecar.json"].remove(metadata)
    if change in {"source", "role"}:
        old = row["assertion_id"]
        row["assertion_id"] = stable_id("source-action", **{k: v for k, v in row.items() if k != "assertion_id"})
        next(m for m in data["sidecar.json"] if m["assertion_id"] == old)["assertion_id"] = row["assertion_id"]
    with pytest.raises(ValueError, match="REVIEWED_GRAPH"):
        load(tmp_path, reviewed_case, data)


def test_zero_graph_contribution_cannot_be_relabelled_as_used_graph():
    with pytest.raises(ValueError, match="EXECUTION_BINDING"):
        ReviewedGraphExecutionV3(requested_method="graph_1hop", effective_method="graph_1hop",
            mechanism_status="USED_GRAPH", executed_stages=("TRAVERSE_GRAPH",), graph_path_used=False,
            context_chunk_ids=(), context_assertion_ids=(), release_sha256="0" * 64)


def test_native_prep_cache_miss_has_no_synthetic_or_legacy_plan(tmp_path, routes, context):
    with pytest.raises(ValueError, match="CACHE_MISS_NO_GATEWAY"):
        QueryPlanCacheV3(tmp_path, namespace="FIXTURE").resolve(context, question="How does Step Functions execute tasks?",
            search_query="Step Functions execute tasks", routes=routes)
