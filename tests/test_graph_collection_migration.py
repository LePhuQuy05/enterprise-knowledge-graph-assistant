"""Offline adversarial migration checks; synthetic sources, no quality labels."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest

from app.authorization_evidence import load_authorization_facts
from app.authorization_retrieval import combine_evidence_graph
from app.collections import build_collection, load_collection
from app.contracts import canonical_json, sha256_bytes
from app.fixtures import FIXTURE_EMBEDDING_MODEL, fixture_vector
from app.generation import GenerationError
from app.graph_collection_migration import (
    collection_records,
    migrate_graph_payloads,
    prove_append_only,
)
from app.reviewed_source_graph import load_reviewed_source_graph
from app.snapshot import SnapshotContent, build_snapshot
from tests.test_authorization_evidence import auth_case as auth_case  # noqa: F401
from tests.test_authorization_evidence import write_auth
from tests.test_extraction import source
from tests.test_reviewed_source_runtime import write_release


@pytest.fixture(scope="module")
def migration_case(auth_case, tmp_path_factory):
    root = tmp_path_factory.mktemp("migration-fixture")
    parent = auth_case.collection
    docs, chunks, vectors = collection_records(parent)
    extra_doc, extra_chunk = source("Additional synthetic document without graph facts.", suffix="new-migration")
    docs[extra_doc.document_version_id], chunks[extra_chunk.chunk_id] = extra_doc, extra_chunk
    vectors[extra_chunk.chunk_id] = fixture_vector(extra_chunk.text)
    manifest = parent.manifest
    path = build_snapshot(root / "snapshots", SnapshotContent(tuple(docs.values()), tuple(chunks.values())), vectors,
        tenant_id="demo", purpose="FIXTURE", extraction_recipe=manifest.extraction_recipe,
        embedding_recipe=manifest.embedding_recipe, corpus_version="migration-appended-v1",
        normalizer_version=manifest.normalizer_version, chunking_version=manifest.chunking_version,
        tokenizer_version=manifest.tokenizer_version)
    target_path = build_collection(path.parent, [path], tenant_id="demo", embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    target = load_collection(target_path, tenant_id="demo", embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    old = auth_case.reviewed
    # Recover exact envelopes from validated graph records/recipe rather than
    # depending on a fixture's directory layout.
    action_rows = [edge.source_record() for edge in old.edges if edge.kind == "ACTION"]
    recipe = auth_case.collection.manifest.extraction_recipe
    actions = {"version": "source-actions-reviewed-composite-v1", "recipe": recipe.model_dump(mode="json"),
        "source_manifest_sha256": parent.manifest_sha256, "assertions": action_rows,
        "serving_published": False, "human_reviewed": 0}
    return SimpleNamespace(parent=parent, target=target, reviewed=old, auth=auth_case.authorization,
        actions=actions, sidecar=[e.metadata() for e in old.edges])


def migrate(case, **changes):
    args = dict(action_envelope=case.actions, sidecar=case.sidecar) | changes
    return migrate_graph_payloads(case.parent, case.target, case.reviewed, case.auth, **args)


def test_append_only_proof_covers_all_old_sources_and_vectors(migration_case):
    case = migration_case
    proof = prove_append_only(case.parent, case.target)
    assert proof["shared_chunks"] == case.parent.manifest.counts.chunks
    assert proof["shared_vectors_bit_exact"] == proof["shared_chunks"]
    assert len(proof["added_document_version_ids"]) == 1
    assert not proof["added_document_graph_coverage_claimed"]


def test_migrated_graph_reloads_with_existing_guards_and_keeps_semantics(migration_case, tmp_path):
    case = migration_case
    result = migrate(case)
    (tmp_path / "reviewed").mkdir()
    path, digest = write_release(tmp_path / "reviewed", (case.target, None, result["reviewed_payloads"]))
    reviewed = load_reviewed_source_graph(path, case.target, trusted_sha256=digest, allow_experimental=True)
    rows = result["authorization_payload"]["records"]
    path, digest = write_auth(tmp_path / "authorization", case.target, rows)
    docs, chunks, _ = collection_records(case.target)
    auth = load_authorization_facts(path, case.target, chunks, docs, trusted_sha256=digest, allow_experimental=True)
    graph = combine_evidence_graph(case.target, reviewed, auth)
    assert len(graph.facts) == len(combine_evidence_graph(case.parent, case.reviewed, case.auth).facts)
    assert reviewed.edges == case.reviewed.edges
    for old, new in zip(case.auth.records(), auth.records(), strict=True):
        assert old["assertion_id"] != new["assertion_id"]
        assert old["nodes"] == new["nodes"] and old["evidence"] == new["evidence"]
        assert old["candidate"] | {"collection_manifest_sha256": case.target.manifest_sha256} == new["candidate"]
        assert new["permission_granted"] is False and new["native_extraction_claimed"] is False
    assert len(result["lineage"]["authorization_id_mappings"]) == len(rows)
    assert result["lineage"]["new_semantic_reviews"] == 0


def test_old_release_does_not_load_on_new_collection(migration_case, tmp_path):
    case = migration_case
    original = {"relations-v2.json": case.reviewed.overlay.model_dump(mode="json"),
        "actions.json": case.actions, "sidecar.json": case.sidecar}
    path, digest = write_release(tmp_path, (case.parent, None, original))
    with pytest.raises(GenerationError, match="RELEASE_BINDING"):
        load_reviewed_source_graph(path, case.target, trusted_sha256=digest, allow_experimental=True)


@pytest.mark.parametrize("field,value", [("tenant_id", "other"), ("normalizer_version", "different"),
    ("chunking_version", "different"), ("tokenizer_version", "different")])
def test_migration_rejects_tenant_and_recipe_changes(migration_case, field, value):
    case = migration_case
    target = SimpleNamespace(manifest=case.target.manifest.model_copy(update={field: value}),
        manifest_sha256=case.target.manifest_sha256)
    with pytest.raises(GenerationError, match="TENANT_RECIPE"):
        prove_append_only(case.parent, target)


def test_no_change_is_not_a_migration(migration_case):
    with pytest.raises(GenerationError, match="NO_CHANGE"):
        prove_append_only(migration_case.parent, migration_case.parent)


@pytest.mark.parametrize("tamper", ["document", "chunk", "missing", "vector"])
def test_source_identity_is_not_trusted_without_exact_records(migration_case, tamper):
    case = migration_case
    first_ref = case.target.manifest.shards[0]
    original = case.target.get_shard(first_ref.shard_id)
    docs, chunks = dict(original.documents), dict(original.chunks)
    vectors = original.vectors.copy()
    if tamper == "document":
        key = next(iter(case.parent.get_shard(case.parent.manifest.shards[0].shard_id).documents))
        docs[key] = docs[key].model_copy(update={"title": "changed metadata"})
    elif tamper == "chunk":
        key = next(iter(case.parent.get_shard(case.parent.manifest.shards[0].shard_id).chunks))
        chunks[key] = chunks[key].model_copy(update={"section_path": ("changed",)})
    elif tamper == "missing":
        chunks.pop(next(iter(case.parent.get_shard(case.parent.manifest.shards[0].shard_id).chunks)))
    else:
        vectors[0, 0] += 0.01
    changed = replace(original, documents=docs, chunks=chunks, vectors=vectors)
    target = SimpleNamespace(manifest=case.target.manifest, manifest_sha256=case.target.manifest_sha256,
        validate_all=lambda: None, get_shard=lambda _: changed)
    with pytest.raises(GenerationError, match="APPEND_ONLY|VECTOR_CHANGED"):
        prove_append_only(case.parent, target)


def test_wrong_parent_auth_binding_is_rejected(migration_case):
    case = migration_case
    with pytest.raises(GenerationError, match="VALIDATED_PARENT"):
        migrate_graph_payloads(case.parent, case.target, case.reviewed,
            replace(case.auth, collection_manifest_sha256="f" * 64), action_envelope=case.actions, sidecar=case.sidecar)


@pytest.mark.parametrize("tamper", ["action", "sidecar", "duplicate"])
def test_payloads_must_be_exact_validated_parent_artifacts(migration_case, tamper):
    case = migration_case
    actions, sidecar = deepcopy(case.actions), deepcopy(case.sidecar)
    if tamper == "action":
        actions["assertions"][0]["polarity"] = "NEGATIVE"
    elif tamper == "sidecar":
        sidecar[0]["human_reviewed"] = True
    else:
        sidecar.append(deepcopy(sidecar[0]))
    with pytest.raises(GenerationError, match="ARTIFACT_MISMATCH"):
        migrate(case, action_envelope=actions, sidecar=sidecar)


def test_migration_is_deterministic_and_does_not_mutate_parent(migration_case):
    case = migration_case
    before = canonical_json({"actions": case.actions, "sidecar": case.sidecar, "auth": case.auth.records()})
    assert canonical_json(migrate(case)) == canonical_json(migrate(case))
    assert before == canonical_json({"actions": case.actions, "sidecar": case.sidecar, "auth": case.auth.records()})
    assert sha256_bytes(before)
