"""Freeze and replay local, nonblind question/source-equivalence review.

No model, retrieval, graph mutation, account access or serving deployment.
Semantic witness annotations below are authored review, not automatic gold.
"""

import argparse
import csv
import io
import json
from contextlib import ExitStack
from unittest.mock import patch

from app.authorization_evidence import LiteralSpan, source_span
from app.contracts import canonical_json, sha256_bytes
from app.graph_collection_migration import collection_records
from app.ingestion.text import ProxyTokenizer
from app.question_coverage_review import (
    PRESENT,
    ReviewedSpan,
    ReviewedWitness,
    audit_question_coverage,
    load_rubric,
    make_witness,
)
from scripts.audit_facet_recall_v5 import rendered_sources
from scripts.audit_section_recall import FAMILIES
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT

OUTPUT = ROOT / "results/question-coverage-v1"
SCOPE = "scope-v2.json"  # Pre-run import lint amendment; scope.json was never executed.
RUBRIC = "configs/evaluation/question-coverage-v1.json"
RUBRIC_SHA = "ef7811f81f6505a96385aed5d6f8e03d8e9c7d17c5b40ef47fb569460fc8079a"
NATIVE = "results/facet-recall-v5-canary-v1/receipts.json"
CANONICAL = "results/permission-context-qualification-v1/receipts.json"
CODE = ("app/question_coverage_review.py", "scripts/review_question_coverage.py",
    "app/source_dependency_audit.py", "app/authorization_evidence.py", "app/contracts.py",
    "scripts/audit_facet_recall_v5.py", "scripts/audit_section_recall.py")


def author_catalog(documents):
    """Exact frozen source review; OR alternatives, AND spans per witness."""
    by_uri = {d.source_uri: d for d in documents.values()}
    witnesses = []

    def span(path, *, quote=None, bounds=None):
        d = by_uri["https://docs.aws.amazon.com" + path]
        s = source_span(d, quote) if quote is not None else LiteralSpan(
            start_char=bounds[0], end_char=bounds[1], quote=d.normalized_text[bounds[0]:bounds[1]])
        return ReviewedSpan(document_version_id=d.document_version_id,
            document_normalized_sha256=d.normalized_sha256, source_uri=d.source_uri, span=s)

    def add(key, spans, why, kind="EXPLICIT_SUPPORT"):
        witnesses.append(make_witness(key, kind, tuple(spans), why))

    permissions = "/lambda/latest/dg/lambda-permissions.html"
    tutorial = "/lambda/latest/dg/with-s3-example.html"
    role = "/lambda/latest/dg/lambda-intro-execution-role.html"
    s3 = "/AmazonS3/latest/userguide/UsingKMSEncryption.html"
    kms = "/kms/latest/developerguide/key-policy-modifying-external-accounts.html"
    iam = "/IAM/latest/UserGuide/access_policies-cross-account-resource-access.html"
    resource = "/lambda/latest/dg/access-control-resource-based.html"
    add("S1", [span(permissions, bounds=(1762, 2175))],
        "Directly distinguishes resource-policy treatment of S3 invoking Lambda. This is not object-read permission.")
    add("S1", [span("/lambda/latest/dg/with-s3.html", bounds=(49, 383))],
        "Alternative S3 integration paragraph states granting S3 invocation permission on the function resource policy; no special URI is required.")
    add("S1", [span(resource, bounds=(0, 1097)), span(resource, bounds=(8386, 9697))],
        "Resource-policy definition and complete S3 InvokeFunction example including its conditions/deny. Does not establish a user's real topology.")
    add("S2", [span(permissions, bounds=(883, 1434))],
        "Execution-role definition and permissions for function access to other resources.")
    add("S2", [span(role, bounds=(65, 1096))],
        "Alternative role overview retains both runtime resource permissions and the separate trust-policy requirement for assuming the role.")
    add("S2", [span(tutorial, bounds=(3036, 3325))],
        "Tutorial explicitly defines the role and its use of the preceding permissions policy.")
    add("S3", [span(tutorial, bounds=(2081, 2238)), span(tutorial, bounds=(2238, 3036)),
        span(tutorial, bounds=(3036, 3325))],
        "Purpose paragraph, GetObject policy with object ARN and role linkage. Wildcard example is evidence of purpose, not a least-privilege recommendation.")
    add("S3", [span(tutorial, bounds=(2238, 3036)), span(tutorial, bounds=(3036, 3325)),
        span(tutorial, bounds=(3325, 3816))],
        "Alternative explicit linkage: full policy, role definition using the previous policy, and named-policy attachment steps. Does not require the earlier purpose paragraph if these are all present.")
    add("S4", [span(s3, quote="To successfully make a PutObject request to encrypt an object with an AWS KMS key to Amazon S3, you need kms:GenerateDataKey permissions on the key. To download an object encrypted with an AWS KMS key, you need kms:Decrypt permissions for the key. To perform a multipart upload to encrypt an object with an AWS KMS key, you must have the kms:GenerateDataKey and kms:Decrypt permissions for the key.")],
        "Entire permissions paragraph distinguishes downloading/Decrypt from upload/GenerateDataKey; does not certify role configuration.")
    both = span(iam, quote="In cross account access, a principal needs an Allow in the identity policy and the resource-based policy.")
    s3_resource = span(iam, quote="Amazon S3 buckets — The policy is attached to the bucket, but the policy controls access to both the bucket and the objects in it. For more information, see Bucket policies for Amazon S3 in the Amazon Simple Storage Service User Guide. In some cases, it may be best to use roles for cross-account access to Amazon S3. For more information, see the example walkthroughs in the Amazon Simple Storage Service User Guide.")
    add("S5", [both, s3_resource],
        "Cross-account resource-policy path needs both sides; the same page explicitly includes S3 bucket/object policy scope. Assumed-role alternatives and object ownership still require care.")
    key_intro = span(kms, bounds=(49, 267))
    key_types = span(kms, quote="To give permission to use a KMS key to users and roles in another account, you must use two different types of policies:\n\nThe key policy for the KMS key must give the external account (or users and roles in the external account) permission to use the KMS key. The key policy is in the account that owns the KMS key.\n\nIAM policies in the external account must delegate the key policy permissions to its users and roles. These policies are set in the external account and give permissions to users and roles in that account.")
    managed = span(s3, quote="Objects encrypted using SSE-KMS with AWS managed keys can't be shared cross-account. If you need to share SSE-KMS data cross-account, you must use a customer managed key from AWS KMS.")
    add("S6", [key_intro, managed], "KMS's explicit two-account policy requirement AND S3's customer-managed-key sharing limitation; both sources are necessary.")
    add("S6", [key_types, managed], "Equivalent detailed KMS policy explanation AND the same S3 key-type limitation; partial alternatives must not be mixed into a full witness.")
    add("K1", [key_intro], "KMS introduction explicitly names both policy sides and the external account.")
    add("K1", [key_types], "Alternative detailed explanation retains key-owner and external-account IAM ownership.")
    add("K2", [span(kms, quote="The key policy determines who can have access to the KMS key. The IAM policy determines who does have access to the KMS key. Neither the key policy nor the IAM policy alone is sufficient—you must change both.")],
        "Direct neither-policy-alone statement in the cross-account page, not a claim of sufficient real-world authorization.")
    d = by_uri["https://docs.aws.amazon.com" + kms]
    start = d.normalized_text.index("Cross-account permission is effective only")
    end = d.normalized_text.index("If you give a user in a different account")
    add("K3", [span(kms, bounds=(start, end))],
        "Complete frozen list: cryptographic operations; CreateGrant, DescribeKey, GetKeyRotationStatus, GetPublicKey, ListGrants, RetireGrant and RevokeGrant. Category members are not invented.")
    add("K4", [span(kms, quote="If you give a user in a different account permission for other operations, those permissions have no effect. For example, if you give a principal in a different account kms:ListKeys permission in an IAM policy, or kms:ScheduleKeyDeletion permission on a KMS key in a key policy, the user's attempts to call those operations on your resources still fail.")],
        "Explicit unsupported-operation limitation with ListKeys/ScheduleKeyDeletion examples.")
    add("S4", [span("/vpc/latest/userguide/flow-logs-s3-cmk-policy.html", bounds=(0, 895))],
        "This is VPC flow-log delivery writing to S3, not evidence that the Lambda reader has Decrypt permission.", "MIS_SCOPED_IF_GENERALIZED")
    add("S3", [span("/AmazonCloudFront/latest/DeveloperGuide/lambda-edge-permissions.html", bounds=(3520, 4065))],
        "Lambda@Edge example explains role permissions for S3 access, but does not explicitly establish GetObject action and cross-account conditions for this task.", "MIS_SCOPED_IF_GENERALIZED")
    add("S2", [span("/AmazonECS/latest/developerguide/task_execution_IAM_role.html", bounds=(354, 1586))],
        "An ECS task execution role is not the Lambda function execution role in this question.", "MIS_SCOPED_IF_GENERALIZED")
    add("K3", [span("/kms/latest/developerguide/key-policy-default.html", bounds=(11838, 12987))],
        "Default key-user policy examples do not enumerate the complete cross-account supported operation set.", "MIS_SCOPED_IF_GENERALIZED")
    return tuple(witnesses)


def read_rows():
    inputs = {NATIVE: sha256_bytes((ROOT / NATIVE).read_bytes()),
        CANONICAL: sha256_bytes((ROOT / CANONICAL).read_bytes())}
    native = json.loads((ROOT / NATIVE).read_bytes())
    rows = [("v5", r) for r in native["recall_cells"]]
    for family in FAMILIES:
        path = f"results/{family}/receipts.json"
        inputs[path] = sha256_bytes((ROOT / path).read_bytes())
        rows.extend((family, r) for r in json.loads((ROOT / path).read_bytes())["cells"])
    return inputs, rows, native


def plan_review(native):
    rows = []
    for p in native["native_preparations"]:
        plan = p["plan"]
        s3 = p["intent"] == "s3-permission-layers"
        issue = ("Actor/action conflation: invocation permission is qualified as object reading." if s3 and p["language"] == "en"
            else "Qualifier attachment: across accounts is attached to invoking Lambda without a confirmed invocation topology." if s3
            else "Policy-set narrowing: which policies becomes key policy; external-account IAM policy is not named in this facet.")
        rows.append({"intent": p["intent"], "language": p["language"], "question": p["question"],
            "plan_sha256": sha256_bytes(canonical_json(plan)), "receipt_sha256": plan["receipt_sha256"],
            "facet_index": 0, "focus_quote": plan["facets"][0]["focus_quote"],
            "search_query": plan["facets"][0]["search_query"], "review": "SEMANTIC_RISK_NOT_APPROVED",
            "rationale": issue, "origin": "AI_AUTHORED_NONBLIND_DEVELOPMENT_REVIEW", "human_reviewers": 0,
            "raw_plan_edited": False, "literal_probes_solve_this": False,
            "full_question_search_retained": True})
    return rows


def freeze():
    collection, _, _ = load_migrated()
    docs, _, _ = collection_records(collection)
    rubric = load_rubric((ROOT / RUBRIC).read_bytes(), trusted_sha256=RUBRIC_SHA)
    catalog = {"version": "question-witness-catalog-v1", "rubric_sha256": RUBRIC_SHA,
        "collection_manifest_sha256": collection.manifest_sha256,
        "witnesses": [w.model_dump(mode="json") for w in author_catalog(docs)],
        "annotation_coverage": "Selected-source review, not exhaustive equivalence discovery; no human adjudication."}
    immutable(OUTPUT / "witnesses.json", catalog)
    pins, rows, native = read_rows()
    immutable(OUTPUT / "native-plan-review.json", plan_review(native))
    immutable(OUTPUT / SCOPE, {"version": "offline-question-coverage-v1", "rubric_path": RUBRIC,
        "supersedes_unexecuted_pre_run_scope": "scope.json", "amendment": "Import lint only before the first evaluation run; pre-run code preserved in pre-run-code-v1.",
        "rubric_sha256": RUBRIC_SHA, "witnesses_sha256": sha256_bytes((OUTPUT / "witnesses.json").read_bytes()),
        "native_plan_review_sha256": sha256_bytes((OUTPUT / "native-plan-review.json").read_bytes()),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}, "input_sha256": pins,
        "collection_manifest_sha256": collection.manifest_sha256, "cells": len(rows),
        "underlying_intents": 2, "language_inputs": 4, "new_control_groups_not_retrieved": len(rubric.controls_not_yet_retrieved),
        "witness_semantics": "OR alternatives, AND original spans; review assertions, not automatic entailment",
        "new_model_calls": 0, "new_aws_calls": 0, "new_retrieval_experiments": 0, "serving_published": False})


def evaluate():
    frozen = json.loads((OUTPUT / SCOPE).read_bytes())
    for path, expected in frozen["code_sha256"].items():
        if sha256_bytes((ROOT / path).read_bytes()) != expected:
            raise ValueError("QUESTION_COVERAGE_CODE_PIN")
    pins, rows, native = read_rows()
    if pins != frozen["input_sha256"] or json.loads((OUTPUT / "native-plan-review.json").read_bytes()) != plan_review(native):
        raise ValueError("QUESTION_COVERAGE_INPUT_PIN")
    if sha256_bytes((OUTPUT / "native-plan-review.json").read_bytes()) != frozen["native_plan_review_sha256"]:
        raise ValueError("QUESTION_COVERAGE_NATIVE_REVIEW_PIN")
    collection, _, _ = load_migrated()
    if collection.manifest_sha256 != frozen["collection_manifest_sha256"]:
        raise ValueError("QUESTION_COVERAGE_COLLECTION_PIN")
    docs, chunks, _ = collection_records(collection)
    rubric = load_rubric((ROOT / RUBRIC).read_bytes(), trusted_sha256=frozen["rubric_sha256"])
    raw = (OUTPUT / "witnesses.json").read_bytes()
    if sha256_bytes(raw) != frozen["witnesses_sha256"]:
        raise ValueError("QUESTION_COVERAGE_WITNESS_PIN")
    catalog = json.loads(raw)
    if catalog["rubric_sha256"] != frozen["rubric_sha256"] or catalog["collection_manifest_sha256"] != collection.manifest_sha256:
        raise ValueError("QUESTION_COVERAGE_CATALOG_BINDING")
    witnesses = tuple(ReviewedWitness.model_validate_json(canonical_json(w)) for w in catalog["witnesses"])
    tokenizer = ProxyTokenizer()
    cells, source_count, span_count = [], 0, 0
    for family, row in rows:
        n, s = rendered_sources(row, docs, chunks, tokenizer)
        source_count += n
        span_count += s
        review = audit_question_coverage(rubric, witnesses, docs, chunks, row["context_chunk_ids"], intent=row["intent"], tenant_id="demo")
        cells.append({"family": family, "intent": row["intent"], "language": row["language"], "arm": row["arm"],
            "source_message_sha256": sha256_bytes(row["source_message"].encode()), "question_sha256": sha256_bytes(row["question"].encode()),
            "canonical_checks_unchanged": row["context_audit"]["checks"], "review": review})
    losses = []
    for cell in cells:
        if cell["family"] == "v5":
            continue
        row = next(r for f, r in rows if f == cell["family"] and r["intent"] == cell["intent"]
            and r["language"] == cell["language"] and r["arm"] == cell["arm"])
        base_arm = f"V5_WINDOW_R{row['pool_trace']['recipe']['radius']}_" + ("QB" if cell["arm"].endswith("_QB") else "Q")
        base = next(c for c in cells if c["family"] == "v5" and c["intent"] == cell["intent"]
            and c["language"] == cell["language"] and c["arm"] == base_arm)

        def present(c):
            return {r["requirement_id"] for r in c["review"]["requirements"] if r["status"] in PRESENT}

        losses.append({"family": cell["family"], "arm": cell["arm"], "intent": cell["intent"], "language": cell["language"],
            "baseline_arm": base_arm, "gained": sorted(present(cell) - present(base)), "lost": sorted(present(base) - present(cell))})
    return {"version": "question-coverage-review-receipts-v1", "scope_sha256": sha256_bytes((OUTPUT / SCOPE).read_bytes()),
        "cells": cells, "losses": losses, "rendered_source_appearances": source_count, "original_spans_checked": span_count,
        "new_model_calls": 0, "new_aws_calls": 0, "human_reviewers": 0, "independent_accuracy_measured": False}


def summary(payload):
    arms = []
    for family, arm in sorted({(c["family"], c["arm"]) for c in payload["cells"]}):
        own = [c for c in payload["cells"] if c["family"] == family and c["arm"] == arm]
        losses = [r for r in payload["losses"] if r["family"] == family and r["arm"] == arm]
        arms.append({"family": family, "arm": arm, "inputs": len(own),
            "reviewed_requirements_present": sum(c["review"]["reviewed_requirements_present"] for c in own),
            "requirements_total": sum(c["review"]["requirements_total"] for c in own),
            "all_requirements_present_inputs": sum(c["review"]["all_reviewed_requirements_present"] for c in own),
            "gained_vs_matched_v5": sum(len(r["gained"]) for r in losses), "lost_vs_matched_v5": sum(len(r["lost"]) for r in losses)})
    return {"version": "question-coverage-review-summary-v1", "status": "LOCAL_SOURCE_REVIEW_ONLY_NO_ANSWER_PROMOTION",
        "arms": arms, "cells": len(payload["cells"]), "underlying_intents": 2, "language_inputs": 4,
        "rubric_sha256": RUBRIC_SHA, "human_reviewers": 0, "new_controls_executed": 0,
        "warnings": ["AI authored nonblind development annotations", "Witness completeness is not answer correctness",
            "Equivalent-source discovery is not exhaustive", "Native V5 semantic risks remain unapproved",
            "Old ancestor prototype has known child-loss and is not a promotion candidate", "No winner selected, generation, extraction or serving change"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if sum((args.freeze, args.run, args.replay)) != 1:
        parser.error("Select exactly one of --freeze, --run or --replay")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.create_connection", "socket.getaddrinfo"):
            stack.enter_context(patch(target, side_effect=AssertionError("QUESTION_REVIEW_NETWORK_FORBIDDEN")))
        if args.freeze:
            freeze()
            print("Question rubric/source witnesses and input/code pins frozen; no inference.")
            return
        payload = evaluate()
        if args.run:
            immutable(OUTPUT / "receipts.json", payload)
            immutable(OUTPUT / "report.json", summary(payload))
            data = summary(payload)["arms"]
            stream = io.StringIO(newline="")
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
            path = OUTPUT / "arm-summary.csv"
            if path.exists() and path.read_bytes() != stream.getvalue().encode():
                raise ValueError("QUESTION_REVIEW_CSV_IMMUTABLE")
            path.write_bytes(stream.getvalue().encode())
        else:
            if (json.loads((OUTPUT / "receipts.json").read_bytes()) != json.loads(canonical_json(payload))
                    or json.loads((OUTPUT / "report.json").read_bytes()) != summary(payload)):
                raise ValueError("QUESTION_REVIEW_REPLAY_MISMATCH")
            immutable(OUTPUT / "replay.json", {"status": "EXACT_OFFLINE_REPLAY_PASS",
                "receipts_sha256": sha256_bytes((OUTPUT / "receipts.json").read_bytes()), "cells": len(payload["cells"]),
                "socket_access_blocked": True, "new_model_calls": 0, "new_aws_calls": 0})
        print(json.dumps({k: v for k, v in summary(payload).items() if k != "arms"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
