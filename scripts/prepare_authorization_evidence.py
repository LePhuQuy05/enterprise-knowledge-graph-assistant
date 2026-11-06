"""Prepare six exposed source controls, then export only after bound AI review.

No model call, benchmark question, synthetic corpus chunk or serving write.
These are deliberately authored development examples, not native extraction.
"""

import argparse
import json
from pathlib import Path

from app.authorization_evidence import (
    CONCLUSIONS,
    POLICY,
    AuthorizationCandidate,
    AuthorizationParticipant,
    LiteralSpan,
    load_authorization_facts,
    materialize,
    policy_hashes,
    source_span,
    validate_candidate,
)
from app.contracts import canonical_json, sha256_bytes
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

CONTROLS = ROOT / "results/graph-source-context-v2/authorization-example.json"
REVIEW = ROOT / "configs/extraction/authorization-source-review-v1.json"


def load_sources():
    collection, chunks, _ = load_collection_inputs()
    documents = {}
    for ref in collection.manifest.shards:
        documents.update(collection.get_shard(ref.shard_id).documents)
    return collection, chunks, documents


def candidates_from_controls(collection, chunks, documents):
    controls = json.loads(CONTROLS.read_bytes())["controls"]
    result = []
    specs = {
        "E0_ASSUME_ROLE": ("CONDITIONAL_ACTION", "assumes", "assumes", ("when you invoke your function",), (
            ("ACTOR", "Lambda", "CATALOG_TYPE"), ("RESOURCE", "your execution role", "SOURCE_LOCAL_RESOURCE"))),
        "E0_TRUST": ("TRUST_REQUIREMENT", "must specify", "assume", (
            "In order for Lambda to properly assume your execution role",), (
            ("ACTOR", "Lambda", "CATALOG_TYPE"), ("RESOURCE", "your execution role", "SOURCE_LOCAL_RESOURCE"),
            ("POLICY", "the role's trust policy", "SOURCE_LOCAL_RESOURCE"),
            ("PRINCIPAL", "lambda.amazonaws.com", "SOURCE_LOCAL_PRINCIPAL"))),
        "E1_INVOKE": ("PERMISSION_REQUIREMENT", "needs permission", "invoke", ("To invoke your function",), (
            ("PRINCIPAL", "Amazon S3", "CATALOG_TYPE"), ("RESOURCE", "your function", "SOURCE_LOCAL_RESOURCE"),
            ("POLICY", "the function's resource-based policy", "SOURCE_LOCAL_RESOURCE"),
            ("CONTEXT_SERVICE", "Lambda", "CATALOG_TYPE"))),
        "E2_S3_ACCESS": ("PERMISSION_REQUIREMENT", "needs", "manage", (
            "If your function uses the AWS SDK to manage Amazon S3 resources",), (
            ("PRINCIPAL", "it", "UNRESOLVED_REFERENCE"), ("RESOURCE", "Amazon S3 resources", "SOURCE_LOCAL_RESOURCE"),
            ("POLICY", "its execution role", "SOURCE_LOCAL_RESOURCE"),
            ("CONTEXT_SERVICE", "Amazon S3", "CATALOG_TYPE"), ("CONTEXT_SERVICE", "Lambda", "CATALOG_TYPE"))),
        "E3_DECRYPT": ("PERMISSION_REQUIREMENT", "need", "kms:Decrypt", (
            "To download an object encrypted with an AWS KMS key",), (
            ("PRINCIPAL", "you", "UNRESOLVED_REFERENCE"), ("RESOURCE", "an AWS KMS key", "SOURCE_LOCAL_RESOURCE"),
            ("CONTEXT_SERVICE", "AWS KMS", "CATALOG_TYPE"))),
        "E3_CROSS_ACCOUNT": ("PERMISSION_REQUIREMENT", "requires permission", "Cross-account access", ("Cross-account access",), (
            ("POLICY", "the key policy of the KMS key", "SOURCE_LOCAL_RESOURCE"),
            ("POLICY", "an IAM policy in the external user's account", "SOURCE_LOCAL_RESOURCE"),
            ("RESOURCE", "the KMS key", "SOURCE_LOCAL_RESOURCE"),
            ("CONTEXT_SERVICE", "AWS KMS", "CATALOG_TYPE"), ("CONTEXT_SERVICE", "IAM", "CATALOG_TYPE"))),
    }
    for control in controls:
        evidence = control["evidence"]
        doc = documents[evidence["document_version_id"]]
        # The statement must fit entirely inside an existing frozen chunk.
        possible = [chunks[k] for k in control["overlapping_frozen_chunk_ids"] if
            chunks[k].start_char <= evidence["start_char"] < evidence["end_char"] <= chunks[k].end_char]
        if not possible:
            raise ValueError("AUTH_CONTROL_REQUIRES_NEW_SOURCE_BUNDLE_NOT_A_FABRICATED_CHUNK")
        chunk = min(possible, key=lambda c: c.chunk_id)
        clause = LiteralSpan(
            start_char=evidence["start_char"], end_char=evidence["end_char"], quote=evidence["quote"])
        full = LiteralSpan(start_char=chunk.start_char, end_char=chunk.end_char, quote=chunk.text)
        kind, trigger, operation, conditions, participants = specs[control["id"]]
        roles = tuple(AuthorizationParticipant(role=role, identity_scope=scope,
            span=source_span(doc, text, within=full if role == "CONTEXT_SERVICE" and text not in clause.quote else clause))
            for role, text, scope in participants)
        candidate = AuthorizationCandidate(tenant_id=chunk.tenant_id, collection_manifest_sha256=collection.manifest_sha256,
            document_version_id=doc.document_version_id, document_normalized_sha256=doc.normalized_sha256,
            chunk_id=chunk.chunk_id, chunk_sha256=chunk.text_sha256, source_uri=doc.source_uri, kind=kind, clause=clause,
            trigger=source_span(doc, trigger, within=clause), operation=source_span(doc, operation, within=clause),
            conditions=tuple(source_span(doc, c, within=clause) for c in conditions), participants=roles,
            requirement_combination="NOT_A_REQUIREMENT" if kind == "CONDITIONAL_ACTION" else "ALL_STATED_REQUIREMENTS_NECESSARY",
            conclusion=CONCLUSIONS[kind])
        validate_candidate(candidate, collection, chunks, documents)
        result.append({"control_id": control["id"], "candidate": candidate.model_dump(mode="json"),
            "candidate_sha256": sha256_bytes(canonical_json(candidate.model_dump(mode="json")))})
    return result


def prepare(output, *, export=False):
    collection, chunks, documents = load_sources()
    rows = candidates_from_controls(collection, chunks, documents)
    raw = canonical_json({"origin": "AI_AUTHORED_SOURCE_CONTROLS_NOT_NATIVE_EXTRACTION", "candidates": rows}) + b"\n"
    if not export:
        output.mkdir(parents=True, exist_ok=False)
        (output / "candidates.json").write_bytes(raw)
        print(json.dumps({"status": "CANDIDATES_REQUIRE_BOUND_SOURCE_REVIEW", "candidates": len(rows),
            "new_model_calls": 0, "aws_calls": 0}))
        return
    if (output / "candidates.json").read_bytes() != raw:
        raise ValueError("AUTH_CANDIDATE_PREPARATION_CHANGED")
    review = json.loads(REVIEW.read_bytes())
    if (review["source_controls_sha256"] != sha256_bytes(CONTROLS.read_bytes())
            or set(review["reviews"]) != {r["control_id"] for r in rows}):
        raise ValueError("AUTH_REVIEW_CONTROLS_BINDING")
    records = [materialize(AuthorizationCandidate.model_validate_json(canonical_json(row["candidate"])),
        review["reviews"][row["control_id"]], collection, chunks, documents) for row in rows]
    if (output / "facts.json").exists() or (output / "release.json").exists():
        raise ValueError("AUTH_RELEASE_ALREADY_EXISTS_NO_OVERWRITE")
    records_raw = canonical_json({"records": records}) + b"\n"
    (output / "facts.json").write_bytes(records_raw)
    release = {"version": "authorization-evidence-release-v1", "policy": POLICY,
        "collection_manifest_sha256": collection.manifest_sha256, "records_sha256": sha256_bytes(records_raw),
        "policy_sha256": policy_hashes(), "records_count": len(records), "human_reviewed": 0, "serving_published": False}
    path = output / "release.json"
    path.write_bytes(canonical_json(release) + b"\n")
    loaded = load_authorization_facts(path, collection, chunks, documents,
        trusted_sha256=sha256_bytes(path.read_bytes()), allow_experimental=True)
    report = {"status": "SIX_AI_REVIEWED_SOURCE_FACTS_LOCAL_ONLY", "collection_documents": 316,
        "collection_manifest_sha256": collection.manifest_sha256, "facts": len(loaded.records_json),
        "kind_counts": {kind: sum(r["candidate"]["kind"] == kind for r in records) for kind in CONCLUSIONS},
        "source_documents": len({r["candidate"]["document_version_id"] for r in records}),
        "source_chunks": len({r["candidate"]["chunk_id"] for r in records}),
        "release_sha256": loaded.release_sha256, "review_config_sha256": sha256_bytes(REVIEW.read_bytes()),
        "new_model_calls": 0, "aws_calls": 0, "native_extraction_calls": 0, "human_reviewed": 0,
        "permission_grants_asserted": 0, "configuration_checks": 0, "serving_published": False}
    (output / "report.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--export", action="store_true")
    args = parser.parse_args()
    prepare(args.output_dir, export=args.export)
