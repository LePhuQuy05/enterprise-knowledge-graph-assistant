"""Explicit post-review amendment: necessary A AND B implies neither alone.

Preserve the original review receipts. The same source/rubric can support a
qualified inference without the exact canonical 'neither' wording. No retrieval
or model calls; this is an annotation correction, not a system improvement.
"""

import argparse
import json
from contextlib import ExitStack
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes
from app.graph_collection_migration import collection_records
from app.question_coverage_review import (
    PRESENT,
    ReviewedWitness,
    audit_question_coverage,
    load_rubric,
    make_witness,
)
from scripts import review_question_coverage as parent
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable

OUTPUT = parent.OUTPUT / "semantic-amendment-v2"
CODE = (*parent.CODE, "scripts/amend_question_coverage.py")


def freeze():
    raw = (parent.OUTPUT / "witnesses.json").read_bytes()
    original = json.loads(raw)
    witnesses = [ReviewedWitness.model_validate_json(canonical_json(w)) for w in original["witnesses"]]
    for w in tuple(witnesses):
        if w.requirement_id == "K1" and w.evidence_kind == "EXPLICIT_SUPPORT":
            witnesses.append(make_witness("K2", "QUALIFIED_INFERENCE", w.spans,
                "Qualified logical decomposition of the documented necessary conjunction: if authorization requires key policy AND external IAM policy, either policy alone does not establish authorization. This does not claim both are sufficient or account access is verified."))
    catalog = original | {"version": "question-witness-catalog-v2",
        "parent_witnesses_sha256": sha256_bytes(raw),
        "amendment": "Add two qualified alternatives for K2; original explicit witness, all other annotations and rubric unchanged. Added after inspecting review-v1 omissions; nonblind, not an improved retriever.",
        "witnesses": [w.model_dump(mode="json") for w in witnesses]}
    immutable(OUTPUT / "witnesses.json", catalog)
    inputs, _, _ = parent.read_rows()
    for name in ("scope-v2.json", "receipts.json", "witnesses.json", "native-plan-review.json"):
        path = parent.OUTPUT / name
        inputs[str(path.relative_to(parent.ROOT)).replace("\\", "/")] = sha256_bytes(path.read_bytes())
    immutable(OUTPUT / "scope.json", {"version": "question-coverage-semantic-amendment-v2",
        "input_sha256": inputs, "rubric_sha256": parent.RUBRIC_SHA,
        "witnesses_sha256": sha256_bytes((OUTPUT / "witnesses.json").read_bytes()),
        "code_sha256": {p: sha256_bytes((parent.ROOT / p).read_bytes()) for p in CODE},
        "new_contexts_retrieved": 0, "new_questions_executed": 0, "new_model_calls": 0,
        "new_aws_calls": 0, "human_reviewers": 0, "serving_published": False})


def evaluate():
    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    for paths in (scope["input_sha256"], scope["code_sha256"]):
        if any(sha256_bytes((parent.ROOT / p).read_bytes()) != h for p, h in paths.items()):
            raise ValueError("QUESTION_COVERAGE_AMENDMENT_PIN")
    original = parent.evaluate()  # Revalidates original rendering, corpus, plans and code.
    if canonical_json(original) != canonical_json(json.loads((parent.OUTPUT / "receipts.json").read_bytes())):
        raise ValueError("QUESTION_COVERAGE_PARENT_REPLAY")
    collection, _, _ = load_migrated()
    docs, chunks, _ = collection_records(collection)
    rubric = load_rubric((parent.ROOT / parent.RUBRIC).read_bytes(), trusted_sha256=scope["rubric_sha256"])
    raw = (OUTPUT / "witnesses.json").read_bytes()
    if sha256_bytes(raw) != scope["witnesses_sha256"]:
        raise ValueError("QUESTION_COVERAGE_AMENDMENT_WITNESS_PIN")
    witnesses = tuple(ReviewedWitness.model_validate_json(canonical_json(w)) for w in json.loads(raw)["witnesses"])
    result = json.loads(canonical_json(original))
    changed = []
    for cell in result["cells"]:
        previous = cell["review"]
        cell["review"] = audit_question_coverage(rubric, witnesses, docs, chunks,
            previous["selected_chunk_ids_unchanged"], intent=cell["intent"], tenant_id="demo")
        before = {r["requirement_id"]: r["status"] for r in previous["requirements"]}
        after = {r["requirement_id"]: r["status"] for r in cell["review"]["requirements"]}
        changed.extend({"family": cell["family"], "arm": cell["arm"], "intent": cell["intent"],
            "language": cell["language"], "requirement_id": key, "before": before[key], "after": after[key]}
            for key in before if before[key] != after[key])
    for loss in result["losses"]:
        candidate = next(c for c in result["cells"] if c["family"] == loss["family"] and c["arm"] == loss["arm"]
            and c["intent"] == loss["intent"] and c["language"] == loss["language"])
        baseline = next(c for c in result["cells"] if c["family"] == "v5" and c["arm"] == loss["baseline_arm"]
            and c["intent"] == loss["intent"] and c["language"] == loss["language"])
        a = {r["requirement_id"] for r in candidate["review"]["requirements"] if r["status"] in PRESENT}
        b = {r["requirement_id"] for r in baseline["review"]["requirements"] if r["status"] in PRESENT}
        loss.update(gained=sorted(a-b), lost=sorted(b-a))
    result.update(version="question-coverage-review-receipts-v2",
        scope_sha256=sha256_bytes((OUTPUT / "scope.json").read_bytes()), annotation_changes=changed,
        annotation_amendment_is_not_system_improvement=True)
    if any(r["requirement_id"] != "K2" for r in changed):
        raise ValueError("QUESTION_COVERAGE_UNEXPECTED_REQUIREMENT_CHANGE")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--run", action="store_true")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.create_connection", "socket.getaddrinfo"):
            stack.enter_context(patch(target, side_effect=AssertionError("QUESTION_REVIEW_NETWORK_FORBIDDEN")))
        if args.freeze:
            freeze()
            print("Qualified-inference annotation amendment frozen; original receipts preserved.")
            return
        result = evaluate()
        report = parent.summary(result) | {"version": "question-coverage-review-summary-v2",
            "annotation_status_changes": len(result["annotation_changes"]),
            "annotation_amendment_is_not_system_improvement": True}
        if args.run:
            immutable(OUTPUT / "receipts.json", result)
            immutable(OUTPUT / "report.json", report)
        else:
            if (canonical_json(result) != canonical_json(json.loads((OUTPUT / "receipts.json").read_bytes()))
                    or report != json.loads((OUTPUT / "report.json").read_bytes())):
                raise ValueError("QUESTION_COVERAGE_AMENDMENT_REPLAY")
            immutable(OUTPUT / "replay.json", {"status": "EXACT_OFFLINE_REPLAY_PASS", "socket_access_blocked": True,
                "receipts_sha256": sha256_bytes((OUTPUT / "receipts.json").read_bytes()),
                "cells": len(result["cells"]), "annotation_status_changes": len(result["annotation_changes"]),
                "new_model_calls": 0, "new_aws_calls": 0})
        print(json.dumps({k: v for k, v in report.items() if k != "arms"}))


if __name__ == "__main__":
    main()
