"""Explicit local graph inheritance after exact append-only source/vector proof.

This is a technical re-attestation of existing source reviews, not new native
extraction, independent adjudication, or graph coverage of added documents.
Existing loaders and their manifest/checksum guards remain unchanged.
"""

from copy import deepcopy

import numpy as np

from app.authorization_evidence import (
    AuthorizationCandidate,
    ValidatedAuthorizationFacts,
    materialize,
    policy_hashes,
)
from app.collection_graph import graph_fingerprint
from app.collection_graph_v2 import CollectionGraphOverlayV2
from app.contracts import canonical_json, sha256_bytes
from app.external_http import strict_json
from app.generation import GenerationError
from app.reviewed_source_graph import ValidatedReviewedSourceGraph

POLICY = "APPEND_ONLY_EXACT_SOURCES_AND_VECTORS_GRAPH_INHERITANCE_V1"


def collection_records(collection):
    documents, chunks, vectors = {}, {}, {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        if documents.keys() & shard.documents.keys() or chunks.keys() & shard.chunks.keys():
            raise GenerationError("GRAPH_MIGRATION_DUPLICATE_SOURCE")
        documents.update(shard.documents)
        chunks.update(shard.chunks)
        vectors.update((chunk.chunk_id, shard.vectors[i]) for i, chunk in enumerate(shard.content.chunks))
    return documents, chunks, vectors


def prove_append_only(parent, target):
    if (parent.manifest_sha256 == target.manifest_sha256
            or parent.manifest.tenant_id != target.manifest.tenant_id
            or any(getattr(parent.manifest, field) != getattr(target.manifest, field) for field in
                ("embedding_recipe", "normalizer_version", "chunking_version", "tokenizer_version"))):
        raise GenerationError("GRAPH_MIGRATION_TENANT_RECIPE_OR_NO_CHANGE")
    parent.validate_all()
    target.validate_all()
    old_docs, old_chunks, old_vectors = collection_records(parent)
    new_docs, new_chunks, new_vectors = collection_records(target)
    if (not old_docs.keys() < new_docs.keys() or not old_chunks.keys() < new_chunks.keys()
            or any(new_docs[k] != v for k, v in old_docs.items())
            or any(new_chunks[k] != v for k, v in old_chunks.items())):
        raise GenerationError("GRAPH_MIGRATION_NOT_EXACT_APPEND_ONLY")
    if any(not np.array_equal(vector, new_vectors[key]) for key, vector in old_vectors.items()):
        raise GenerationError("GRAPH_MIGRATION_VECTOR_CHANGED")
    # Byte-level source records and vectors, not merely IDs or body hashes.
    proof = {"policy": POLICY, "parent_manifest_sha256": parent.manifest_sha256,
        "target_manifest_sha256": target.manifest_sha256,
        "shared_documents": len(old_docs), "shared_chunks": len(old_chunks),
        "shared_vectors_bit_exact": len(old_vectors),
        "shared_document_records_sha256": sha256_bytes(canonical_json(
            [old_docs[k].model_dump(mode="json") for k in sorted(old_docs)])),
        "shared_chunk_records_sha256": sha256_bytes(canonical_json(
            [old_chunks[k].model_dump(mode="json") for k in sorted(old_chunks)])),
        "shared_vectors_sha256": sha256_bytes(b"".join(
            k.encode() + old_vectors[k].tobytes(order="C") for k in sorted(old_vectors))),
        "added_document_version_ids": sorted(new_docs.keys() - old_docs.keys()),
        "added_chunk_ids": sorted(new_chunks.keys() - old_chunks.keys()),
        "new_native_extraction_calls": 0, "new_semantic_reviews": 0,
        "added_document_graph_coverage_claimed": False, "serving_published": False}
    return proof


def migrate_graph_payloads(parent, target, reviewed, authorization, *, action_envelope, sidecar):
    if (type(reviewed) is not ValidatedReviewedSourceGraph
            or type(authorization) is not ValidatedAuthorizationFacts
            or reviewed.overlay.collection_manifest_sha256 != parent.manifest_sha256
            or authorization.collection_manifest_sha256 != parent.manifest_sha256):
        raise GenerationError("GRAPH_MIGRATION_VALIDATED_PARENT_REQUIRED")
    proof = prove_append_only(parent, target)
    overlay = reviewed.overlay
    # These artifacts must be exactly those the parent loader validated.
    expected_actions = [edge.source_record() for edge in reviewed.edges if edge.kind == "ACTION"]
    expected_metadata = {edge.assertion_id: edge.metadata() for edge in reviewed.edges}
    parent_artifacts = strict_json(reviewed.release_json.encode())["artifacts"]
    if (action_envelope.get("source_manifest_sha256") != parent.manifest_sha256
            or sha256_bytes(canonical_json(action_envelope) + b"\n") != parent_artifacts["actions.json"]
            or sha256_bytes(canonical_json(sidecar) + b"\n") != parent_artifacts["sidecar.json"]
            or canonical_json(action_envelope.get("assertions")) != canonical_json(expected_actions)
            or len(sidecar) != len(expected_metadata)
            or {row["assertion_id"]: row for row in sidecar} != expected_metadata):
        raise GenerationError("GRAPH_MIGRATION_PARENT_ARTIFACT_MISMATCH")
    data = overlay.model_dump(mode="json") | {"collection_manifest_sha256": target.manifest_sha256,
        "collection_version": target.manifest.collection_version,
        "graph_sha256": graph_fingerprint(target.manifest_sha256, overlay.entities, overlay.mentions, overlay.assertions)}
    updated = CollectionGraphOverlayV2.model_validate_json(canonical_json(data))
    # inventory/processed counts intentionally retain their PARENT scope.
    actions = deepcopy(action_envelope) | {"source_manifest_sha256": target.manifest_sha256}
    docs, chunks, _ = collection_records(target)
    rows, mappings = [], []
    for old in authorization.records():
        candidate = AuthorizationCandidate.model_validate_json(canonical_json(old["candidate"] |
            {"collection_manifest_sha256": target.manifest_sha256}))
        new_digest = sha256_bytes(canonical_json(candidate.model_dump(mode="json")))
        review = deepcopy(old["review"]) | {"candidate_sha256": new_digest}
        new = materialize(candidate, review, target, chunks, docs)
        if canonical_json(new["nodes"]) != canonical_json(old["nodes"]) or new["evidence"] != old["evidence"]:
            raise GenerationError("GRAPH_MIGRATION_AUTH_MEANING_CHANGED")
        rows.append(new)
        mappings.append({"parent_assertion_id": old["assertion_id"], "target_assertion_id": new["assertion_id"],
            "parent_candidate_sha256": old["candidate_sha256"], "target_candidate_sha256": new_digest,
            "parent_review_sha256": sha256_bytes(canonical_json(old["review"])),
            "review_treatment": "TECHNICAL_REBIND_SAME_SOURCE_NOT_NEW_SEMANTIC_REVIEW"})
    return {"proof": proof, "reviewed_payloads": {"relations-v2.json": updated.model_dump(mode="json"),
        "actions.json": actions, "sidecar.json": deepcopy(sidecar)},
        "authorization_payload": {"records": rows}, "authorization_release_base": {
            "version": "authorization-evidence-release-v1", "policy": "SOURCE_SCOPED_AUTHORIZATION_NARY_V1",
            "collection_manifest_sha256": target.manifest_sha256, "policy_sha256": policy_hashes(),
            "records_count": len(rows), "human_reviewed": 0, "serving_published": False},
        "lineage": {"policy": POLICY, "source_equivalence_proof_sha256": sha256_bytes(canonical_json(proof)),
            "parent_reviewed_release_sha256": reviewed.release_sha256,
            "parent_authorization_release_sha256": authorization.release_sha256,
            "unchanged_reviewed_assertion_ids": [edge.assertion_id for edge in reviewed.edges],
            "authorization_id_mappings": mappings,
            "inventory_count_scope": "PARENT_COLLECTION_SOURCE_REVIEW_INVENTORY_NOT_TARGET_FULL_COVERAGE",
            "parent_inventory_units": overlay.inventory_units, "parent_processed_units": overlay.processed_units,
            "new_native_extraction_calls": 0, "new_semantic_reviews": 0, "human_reviewed": 0,
            "serving_published": False}}
