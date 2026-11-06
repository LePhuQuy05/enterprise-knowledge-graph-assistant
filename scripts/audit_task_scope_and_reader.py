"""Read-only source/antecedent/coverage audit, no automatic answer rewriting."""

import json
from collections import Counter
from contextlib import ExitStack
from unittest.mock import patch

from app.citation_antecedents import audit_claim_antecedents
from app.contracts import QueryResponse, canonical_json, sha256_bytes
from app.graph_collection_migration import collection_records
from scripts import run_paged_reader_canary as reader
from scripts import run_task_scope_v14_canary as answer
from scripts.audit_scoped_flow_canary import pool_missing
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.run_context_profiles import load_review

OUT = answer.ROOT / "results/reader-and-task-scope-audit-2026-11-05-v1"


def main():
    reader.checked_scope()
    answer.checked_scope()
    c, _, _ = load_migrated()
    docs, chunks, _ = collection_records(c)
    rubric, witnesses, _ = load_review()
    reader_rows, answer_rows, citations_checked = [], [], 0
    for path in sorted((reader.OUT / "cells").glob("*.json")):
        row = json.loads(path.read_bytes())
        original = tuple(k for p in row["pages"] for k in p["chunk_ids"])
        selected_by_page = tuple(dict.fromkeys(k for p in row["pages"] for k in p["selected_ids"]))
        intent = row["task"]["intent"]
        reader_rows.append({"cell": path.name, "cell_sha256": sha256_bytes(path.read_bytes()),
            "routed_pages_missing_reviewed_witnesses": pool_missing(rubric, witnesses, chunks, original, intent),
            "page_selections_missing_reviewed_witnesses": pool_missing(rubric, witnesses, chunks, selected_by_page, intent),
            "final_missing_reviewed_witnesses": pool_missing(rubric, witnesses, chunks, row["selected_ids"], intent),
            "new_answers": 0, "whole_source_truncations": row["whole_chunk_truncations"]})
    for path in sorted((answer.OUT / "cells").glob("*.json")):
        row = json.loads(path.read_bytes())
        response = QueryResponse.model_validate_json(canonical_json(row["response_v14"]))
        baseline = row["baseline_v12"]
        by_id = {v.citation_id: v for v in response.citations}
        for cite in response.citations:
            ch, doc = chunks[cite.chunk_id], docs[cite.document_version_id]
            if (ch.chunk_id not in row["context_chunk_ids"] or cite.quote not in ch.text
                    or cite.source_uri != doc.source_uri or doc.normalized_text[ch.start_char:ch.end_char] != ch.text):
                raise ValueError("SCOPE_AUDIT_SOURCE_BINDING")
            citations_checked += 1
        dependencies = []
        for claim in response.claims:
            for dependency in audit_claim_antecedents([by_id[k] for k in claim.citation_ids], docs, chunks, row["context_chunk_ids"]):
                dependencies.append({"claim_id": claim.claim_id, **dependency})
        answer_rows.append({"cell": path.name, "cell_sha256": sha256_bytes(path.read_bytes()),
            "baseline_status": baseline["answer_status"], "new_status": response.answer_status,
            "context_unchanged": row["context_unchanged"], "claim_count": len(response.claims),
            "antecedent_diagnostics": dependencies,
            "manual_nonblind_flags": ([{"claim_id": "c1", "flag": "INVOKER_ROLE_DISTINCTION_NEEDS_SEPARATE_PREMISE",
                "note": "Role-definition citation alone does not establish the additional invocation-permission clause."}]
                if path.name == "s3-permission-layers-vi-hybrid_multi_query.json" else []),
            "human_reviewers": 0, "semantic_entailment_automatically_verified": False})
    report = {"reader_cells": reader_rows, "answer_cells": answer_rows,
        "native_answer_statuses": dict(Counter(r["new_status"] for r in answer_rows)),
        "citation_appearances_checked": citations_checked, "human_reviewers": 0,
        "answer_accuracy_measured": False, "new_model_calls": 0, "new_aws_calls": 0,
        "reader_promoted": False, "v14_globally_qualified": False, "serving_changed": False}
    immutable(OUT / "audit.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in {"reader_cells", "answer_cells"}}))
    for r in reader_rows:
        print(json.dumps(r))
    for r in answer_rows:
        print(json.dumps({k: r[k] for k in ("cell", "baseline_status", "new_status", "antecedent_diagnostics", "manual_nonblind_flags")}))


if __name__ == "__main__":
    with ExitStack() as stack:
        for call in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(call, side_effect=AssertionError("SCOPE_READER_AUDIT_NETWORK_FORBIDDEN")))
        main()
