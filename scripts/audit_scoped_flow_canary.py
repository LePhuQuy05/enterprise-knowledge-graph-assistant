"""Evaluator-only source/coverage audit; never imported by retrieval or serving."""

import json
from collections import Counter, defaultdict
from contextlib import ExitStack
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes
from app.graph_collection_migration import collection_records
from app.profiled_source_context import SourceContextProfile, render_profiled_context
from app.question_coverage_review import PRESENT
from app.source_dependency_audit import missing_intervals
from scripts import run_scoped_flow_canary as experiment
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.profiled_coverage_audit import audit_profiled_coverage
from scripts.run_context_profiles import load_review


def pool_missing(rubric, witnesses, chunks, ids, intent):
    intervals = defaultdict(list)
    for key in ids:
        c = chunks[key]
        intervals[c.document_version_id].append((c.start_char, c.end_char))
    present = {}
    for req in (r for r in rubric.requirements if r.intent == intent):
        present[req.id] = (all(present[key] for key in req.requires) if req.kind == "QUALIFIED_INFERENCE" else
            any(w.requirement_id == req.id and w.evidence_kind in {"EXPLICIT_SUPPORT", "QUALIFIED_INFERENCE"}
                and all(not missing_intervals(s.span.start_char, s.span.end_char, intervals[s.document_version_id])
                    for s in w.spans) for w in witnesses))
    return [key for key, exists in present.items() if not exists]


def audit():
    experiment.checked_scope()
    collection, _, graph = load_migrated()
    documents, chunks, _ = collection_records(collection)
    facts = {f.assertion_id: f for f in graph.facts}
    rubric, witnesses, _ = load_review()
    rows, appearances, graph_bundles = [], 0, 0
    semantic = ["# Native canary: unscored claims for semantic review", "",
        "AI/nonblind source review; 0 independent human reviewers. Not answer accuracy.", ""]
    for path in sorted((experiment.OUT / "cells").glob("*.json")):
        cell = json.loads(path.read_bytes())
        task, trace = cell["task"], cell["trace"]
        context = render_profiled_context(documents, chunks, tuple(cell["context_chunk_ids"]),
            question=task["question"], language=task["language"], tenant_id=collection.manifest.tenant_id,
            collection_sha256=collection.manifest_sha256,
            profile=SourceContextProfile.model_validate_json(canonical_json(trace["profile"])))
        if (context.fingerprint != cell["context_fingerprint"]
                or trace["method_requested"] != trace["method_used"] or trace["hidden_fallbacks"]
                or not set(context.chunk_ids) <= set(cell["pool_ids"])):
            raise ValueError("SCOPED_AUDIT_TASK_SOURCE_OR_METHOD")
        review = audit_profiled_coverage(rubric, witnesses, documents, chunks, context, intent=task["intent"])
        missing = [r["requirement_id"] for r in review["requirements"] if r["status"] not in PRESENT]
        for decision in trace["packing"]["decisions"]:
            if decision["reason"] != "SELECTED" or not decision["assertion_ids"]:
                continue
            if ({facts[key].chunk_id for key in decision["assertion_ids"]} != set(decision["chunk_ids"])
                    or not set(decision["chunk_ids"]) <= set(context.chunk_ids)):
                raise ValueError("SCOPED_AUDIT_ATOMIC_GRAPH_SUPPORT")
            graph_bundles += 1
        response = cell.get("response")
        if response:
            if (response["mode_requested"] != cell["method"] or response["mode_used"] != cell["method"]
                    or response["used_graph"] != trace["packing"]["used_graph"]
                    or response["answer_language"] != task["language"]):
                raise ValueError("SCOPED_AUDIT_ANSWER_METHOD_LANGUAGE")
            citations = {c["citation_id"]: c for c in response["citations"]}
            for citation in citations.values():
                chunk = chunks[citation["chunk_id"]]
                doc = documents[chunk.document_version_id]
                if (chunk.chunk_id not in context.chunk_ids or citation["document_version_id"] != doc.document_version_id
                        or citation["source_uri"] != doc.source_uri or citation["quote"] not in chunk.text):
                    raise ValueError("SCOPED_AUDIT_CITATION_NOT_ORIGINAL_SOURCE")
                appearances += 1
            semantic += ["## " + path.stem, "", "Question: " + task["question"], "",
                "Status: " + response["answer_status"], ""]
            for claim in response["claims"]:
                semantic += ["**" + claim["claim_id"] + ":** " + claim["text"], ""]
                for key in claim["citation_ids"]:
                    cite = citations[key]
                    semantic += ["Source: " + cite["source_uri"], "", "> " + cite["quote"].replace("\n", "\n> "), ""]
            semantic += ["Limitations: " + json.dumps(response["assessment"]["limitations"], ensure_ascii=False), ""]
        rows.append({"cell": path.name, "method": cell["method"], "task": task,
            "cell_sha256": sha256_bytes(path.read_bytes()), "status": cell["status"],
            "answer_status": None if response is None else response["answer_status"],
            "missing_reviewed_requirements_in_context": missing,
            "missing_reviewed_requirements_in_pool": pool_missing(rubric, witnesses, chunks, cell["pool_ids"], task["intent"]),
            "review": review, "source_count": len(context.chunk_ids),
            "distinct_source_documents": len({chunks[k].document_version_id for k in context.chunk_ids})})
    if len(rows) != 20:
        raise ValueError("SCOPED_AUDIT_CANARY_INCOMPLETE")
    summary = {"status": "ORIGINAL_SOURCES_METHOD_AND_ATOMIC_PATHS_CHECKED", "cells": len(rows),
        "answer_statuses": dict(Counter(r["answer_status"] for r in rows)),
        "citation_appearances": appearances, "selected_graph_bundles": graph_bundles,
        "contexts_with_all_reviewed_requirements": sum(not r["missing_reviewed_requirements_in_context"] for r in rows),
        "independent_intents": 2, "human_reviewers": 0, "answer_accuracy_measured": False,
        "new_model_calls": 0, "new_aws_calls": 0, "serving_published": False}
    immutable(experiment.OUT / "source-audit.json", summary | {"cells": rows})
    target = experiment.OUT / "semantic-review-input.md"
    content = "\n".join(semantic)
    if target.exists() and target.read_text(encoding="utf-8") != content:
        raise ValueError("SCOPED_AUDIT_SEMANTIC_INPUT_CHANGED")
    target.write_text(content, encoding="utf-8")
    print(json.dumps(summary))
    for row in rows:
        print(json.dumps({k: row[k] for k in ("cell", "answer_status", "missing_reviewed_requirements_in_context",
            "missing_reviewed_requirements_in_pool", "distinct_source_documents")}))


if __name__ == "__main__":
    with ExitStack() as stack:
        for call in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(call, side_effect=AssertionError("SCOPED_AUDIT_NETWORK_FORBIDDEN")))
        audit()
