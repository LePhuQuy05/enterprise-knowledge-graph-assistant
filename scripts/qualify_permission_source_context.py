"""Source-only qualification and explicit qualifier-dependency diagnostics.

Annotations are AI-authored after inspecting the sources, never independent
benchmark labels. They do not change retrieval ranking, graph facts or serving.
"""

import argparse
import json
from pathlib import Path

from app.authorization_evidence import LiteralSpan
from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes
from app.source_dependency_audit import audit_source_dependencies, make_source_dependency
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT, load_sources

OUTPUT = ROOT / "results/permission-context-qualification-v1"
SUPPLEMENT = ROOT / "results/permission-source-supplement-v1"
CODE = ("app/source_dependency_audit.py", "scripts/qualify_permission_source_context.py")


def literal(document, quote):
    if document.normalized_text.count(quote) != 1:
        raise ValueError("PERMISSION_SOURCE_CONTROL_AMBIGUOUS_OR_MISSING")
    start = document.normalized_text.index(quote)
    return LiteralSpan(start_char=start, end_char=start + len(quote), quote=quote)


def paragraph(document, starts):
    text = document.normalized_text
    if text.count(starts) != 1:
        raise ValueError("PERMISSION_SOURCE_PARAGRAPH_AMBIGUOUS_OR_MISSING")
    start = text.index(starts)
    end = text.find("\n\n", start)
    return LiteralSpan(start_char=start, end_char=len(text) if end < 0 else end,
        quote=text[start:] if end < 0 else text[start:end])


def covering_chunks(document, spans, chunks):
    own = sorted((c for c in chunks.values() if c.document_version_id == document.document_version_id),
        key=lambda c: (c.start_char, c.end_char, c.chunk_id))
    keys = []
    for span in spans:
        cursor = span.start_char
        while cursor < span.end_char:
            available = [c for c in own if c.start_char <= cursor < c.end_char]
            if not available:
                raise ValueError("PERMISSION_SOURCE_NO_CONTIGUOUS_CHUNK_COVERAGE")
            chosen = max(available, key=lambda c: (c.end_char, -c.start_char, c.chunk_id))
            if chosen.chunk_id not in keys:
                keys.append(chosen.chunk_id)
            cursor = chosen.end_char
    if len(keys) > 8:
        raise ValueError("PERMISSION_SOURCE_REVIEW_BUNDLE_EXCEEDS_SHARED_CHUNK_CAP")
    return tuple(keys)


def qualify(*, replay=False):
    report = json.loads((SUPPLEMENT / "report.json").read_bytes())
    corpus = Path(report["corpus_directory"])
    for name, digest in report["artifact_sha256"].items():
        if sha256_bytes((corpus / name).read_bytes()) != digest:
            raise ValueError("PERMISSION_SOURCE_SUPPLEMENT_ARTIFACT_CHANGED")
    documents = {d.document_version_id: d for d in (DocumentVersion.model_validate_json(line)
        for line in (corpus / "documents.jsonl").read_bytes().splitlines())}
    chunks = {c.chunk_id: c for c in (Chunk.model_validate_json(line) for line in (corpus / "chunks.jsonl").read_bytes().splitlines())}
    collection, base_chunks, base_docs = load_sources()
    by_uri = {d.source_uri: d for d in documents.values()}
    prefix = "https://docs.aws.amazon.com/lambda/latest/dg/"
    managing, tutorial, asynchronous = [by_uri[prefix + name] for name in ("lambda-permissions.html", "with-s3-example.html", "invocation-async.html")]
    policy_groups = paragraph(managing, "You can use AWS Identity and Access Management (IAM)")
    group_function = paragraph(managing, "Permissions that your Lambda functions need")
    group_caller = paragraph(managing, "Permissions that other AWS users and entities need")
    execution = paragraph(managing, "You define the permissions that your Lambda function needs")
    caller = paragraph(managing, "You can use resource-based policies")
    tutorial_get = literal(tutorial, '"s3:GetObject"')
    tutorial_purpose = paragraph(tutorial, "Create a permissions policy that allows Lambda to get objects")
    tutorial_role = paragraph(tutorial, "An execution role is an AWS Identity and Access Management (IAM) role")
    async_accepted = paragraph(asynchronous, "For asynchronous invocation, Lambda places the event")
    async_type = paragraph(asynchronous, "To invoke a Lambda Event Function asynchronously using")
    async_code = literal(asynchronous, '"StatusCode": 202')
    specifications = (
        ("LAMBDA_TWO_PERMISSION_CATEGORIES", managing, caller, (execution, group_function, group_caller),
            "Distinguish caller-to-function access from function-to-resource access; this is a documentation distinction, not account verification."),
        ("TUTORIAL_GETOBJECT_ROLE_ANTECEDENT", tutorial, tutorial_get, (tutorial_purpose, tutorial_role),
            "A policy action alone does not establish the actor; retain the tutorial purpose and attachment to its Lambda execution role. Wildcard example resources are not a least-privilege recommendation."),
        ("ASYNC_202_ACCEPTANCE_NOT_CODE_COMPLETION", asynchronous, async_code, (async_accepted, async_type),
            "The 202 example is scoped to asynchronous Event invocation. Its success response follows queueing before the separate process dispatches the event to code."),
    )
    dependencies = {name: make_source_dependency(doc, anchor, required, review_reason=reason)
        for name, doc, anchor, required, reason in specifications}
    plan = {"version": "permission-source-context-qualification-v1", "supplement_inventory_sha256": sha256_bytes((SUPPLEMENT / "inventory.json").read_bytes()),
        "parent_collection_manifest_sha256": collection.manifest_sha256,
        "dependency_review_sha256": sha256_bytes(canonical_json({k: d.model_dump(mode="json") for k, d in dependencies.items()})),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "new_model_calls": 0, "aws_calls": 0, "changes_parent_collection": False, "human_reviewed": 0,
        "review_origin": "AI_SOURCE_DEPENDENCY_REVIEW_NOT_INDEPENDENT_GOLD", "serving_published": False}
    immutable(OUTPUT / "plan.json", plan)
    rows = []
    for name, doc, anchor, required, reason in specifications:
        dependency = dependencies[name]
        isolated = covering_chunks(doc, (anchor,), chunks)
        complete = covering_chunks(doc, (anchor, *required), chunks)
        minimal_audit = audit_source_dependencies(documents, chunks, isolated, (dependency,), tenant_id="demo")
        full_audit = audit_source_dependencies(documents, chunks, complete, (dependency,), tenant_id="demo")
        if full_audit["incomplete_count"]:
            raise ValueError("PERMISSION_SOURCE_FULL_DEPENDENCY_NOT_COVERED")
        rows.append({"control_id": name, "dependency": dependency.model_dump(mode="json"),
            "source_scope_review_vi": reason, "isolated_anchor_audit": minimal_audit, "complete_source_audit": full_audit,
            "new_native_fact_extracted": False, "source_review_status": "AI_SOURCE_SCOPED_REVIEW_NOT_HUMAN_GOLD"})
    # A real parent-corpus condition: an unqualified operation-coverage answer
    # must retain the limits following the cross-account policy requirement.
    kms = next(d for d in base_docs.values() if d.source_uri.endswith("/key-policy-modifying-external-accounts.html"))
    anchor = literal(kms, "Cross-account access requires permission in the key policy of the KMS key and in an IAM policy in the external user's account.")
    text = kms.normalized_text
    start, end = text.index("Cross-account permission is effective only"), text.index("For details about using KMS keys")
    required = LiteralSpan(start_char=start, end_char=end, quote=text[start:end])
    dependency = make_source_dependency(kms, anchor, (required,),
        review_reason="Development operation-coverage questions require the list and ineffectiveness of unsupported cross-account operations after the policy requirement.")
    native_plan = json.loads((ROOT / "results/graph-document-answers-v1/answer-plan.json").read_bytes())
    native_audits = [{"id": row["id"], "audit": audit_source_dependencies(base_docs, base_chunks,
        tuple(row["trace"]["context_chunk_ids"]), (dependency,), tenant_id="demo")} for row in native_plan["cases"]]
    raw = canonical_json({"source_controls": rows, "permission_category_intro": policy_groups.model_dump(mode="json"),
        "parent_kms_dependency": dependency.model_dump(mode="json"), "parent_native_context_audits": native_audits}) + b"\n"
    summary = {"status": "SOURCE_QUALIFICATION_AND_REVIEWED_DEPENDENCIES_CHECKED", "supplement_documents": 3,
        "supplement_chunks": len(chunks), "source_controls": len(rows),
        "isolated_anchor_incomplete_controls": [r["control_id"] for r in rows if r["isolated_anchor_audit"]["incomplete_count"]],
        "complete_bundles_all_covered": True, "parent_native_contexts_audited": len(native_audits),
        "parent_contexts_incomplete_for_this_one_reviewed_dependency": sum(r["audit"]["incomplete_count"] for r in native_audits),
        "receipts_sha256": sha256_bytes(raw), "native_extraction_calls": 0, "new_model_calls": 0, "aws_calls": 0,
        "human_reviewed": 0, "all_qualifiers_discovered": False, "parent_collection_unchanged": True, "serving_published": False}
    if replay:
        if (OUTPUT / "receipts.json").read_bytes() != raw:
            raise ValueError("PERMISSION_SOURCE_QUALIFICATION_REPLAY_CHANGED")
        immutable(OUTPUT / "replay.json", summary | {"status": "SOURCE_DEPENDENCY_QUALIFICATION_EXACT_REPLAY"})
    else:
        (OUTPUT / "receipts.json").write_bytes(raw)
        immutable(OUTPUT / "report.json", summary)
    print(json.dumps(summary))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    qualify(replay=parser.parse_args().replay)
