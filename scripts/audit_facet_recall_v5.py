"""Offline source-exact audit of native V5 recall, including recorded V3 cells."""

import ast
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes
from app.faceted_query_v5 import FacetPlanV5, revalidate_plan
from app.graph_collection_migration import collection_records
from app.ingestion.text import ProxyTokenizer
from app.source_dependency_audit import SourceDependency, audit_source_dependencies
from scripts.audit_collection319_methods import source_reachability
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.run_facet_recall_v5 import CODE, OUTPUT, PARENT, ROOT


def algorithm_equivalence():
    original = (ROOT / "app/document_recall_v1.py").read_text(encoding="utf-8")
    current = (ROOT / "app/document_recall_v2.py").read_text(encoding="utf-8")
    for old, new in (("app.faceted_query_v4", "app.faceted_query_v5"),
            ("DOCUMENT_WINDOWS_V1", "DOCUMENT_WINDOWS_V2"),
            ("bounded-document-recall-v1", "bounded-document-recall-v2"),
            ("faceted-document-recall-answer-v1", "faceted-document-recall-answer-v2"),
            ("FACETS_V4", "FACETS_V5"), ("READY_DOCUMENT_RECALL_V1", "READY_DOCUMENT_RECALL_V2")):
        current = current.replace(new, old)
    trees = [ast.parse(text) for text in (original, current)]
    for tree in trees:
        tree.body = tree.body[1:]  # only the module docstring differs additionally
    if ast.dump(trees[0]) != ast.dump(trees[1]):
        raise ValueError("FACET_V5_WINDOW_ALGORITHM_CHANGED_BEYOND_BINDING_AND_IDENTITY")
    return True


def rendered_sources(row, docs, chunks, tokenizer):
    keys = row["context_chunk_ids"]
    pool = row.get("pool_ids")
    if (len(keys) > 8 or len(set(keys)) != len(keys) or pool is not None
            and (len(pool) > 40 or not set(keys) <= set(pool))):
        raise ValueError("FACET_AUDIT_POOL_CONTEXT_BOUNDS")
    prompt = json.loads(row["source_message"])
    if prompt["untrusted_question"] != row["question"]:
        raise ValueError("FACET_AUDIT_QUESTION_BINDING")
    if (tokenizer.count(row["source_message"]) != row["user_message_proxy_tokens"]
            or row["user_message_proxy_tokens"] > 4096):
        raise ValueError("FACET_AUDIT_TOKEN_BUDGET")
    sources = prompt["untrusted_sources"]
    if [s["source_id"] for s in sources] != [f"s{i+1}" for i in range(len(keys))]:
        raise ValueError("FACET_AUDIT_RENDERED_SOURCE_IDS")
    span_count = 0
    for key, rendered in zip(keys, sources, strict=True):
        chunk = chunks[key]
        doc = docs[chunk.document_version_id]
        if (doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text
                or sha256_bytes(chunk.text.encode()) != chunk.text_sha256
                or not doc.source_uri.startswith("https://docs.aws.amazon.com/")
                or rendered["untrusted_document_title"] != doc.title):
            raise ValueError("FACET_AUDIT_ORIGINAL_SOURCE_BINDING")
        cursor = 0
        for span in rendered["spans"]:
            start = chunk.text.find(span["text"], cursor)
            if start < 0 or chunk.text[cursor:start].strip():
                raise ValueError("FACET_AUDIT_INTERIOR_LOSS_OR_INVENTED_TEXT")
            cursor = start + len(span["text"])
            span_count += 1
        if chunk.text[cursor:].strip():
            raise ValueError("FACET_AUDIT_RENDERED_SOURCE_TRUNCATED")
    return len(keys), span_count


def audit():
    collection, _, _ = load_migrated()
    docs, chunks, _ = collection_records(collection)
    payload = json.loads((OUTPUT / "receipts.json").read_bytes())
    report = json.loads((OUTPUT / "report.json").read_bytes())
    frozen = json.loads((OUTPUT / "scope.json").read_bytes())
    if (report["receipts_sha256"] != sha256_bytes((OUTPUT / "receipts.json").read_bytes())
            or frozen["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}
            or frozen["parent_receipts_sha256"] != sha256_bytes((PARENT / "receipts.json").read_bytes())):
        raise ValueError("FACET_AUDIT_FROZEN_PINS")
    controls = json.loads((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())
    deps = [SourceDependency.model_validate_json(canonical_json(r["dependency"])) for r in controls["source_controls"]]
    deps.append(SourceDependency.model_validate_json(canonical_json(controls["parent_kms_dependency"])))
    labels = {d.dependency_id: name for d, name in zip(deps,
        ("lambda-permission-categories", "tutorial-getobject-role", "async-event-202", "kms-cross-account-operations"), strict=True)}
    groups = {}
    for prep in payload["native_preparations"]:
        plan = FacetPlanV5.model_validate_json(canonical_json(prep["plan"]))
        revalidate_plan(plan, collection=collection, question=prep["question"], search=prep["search"], language=prep["language"])
        if plan.origin != "NATIVE_V5" or plan.receipt_sha256 != sha256_bytes(canonical_json(prep["receipt"])):
            raise ValueError("FACET_AUDIT_NATIVE_LINEAGE")
    tokenizer = ProxyTokenizer()
    details, source_count, span_count = [], 0, 0
    new_rows = payload["recall_cells"]
    recorded = payload["recorded_v3_q_qb_baseline"]
    if len(new_rows) != 24 or len(recorded) != 8:
        raise ValueError("FACET_AUDIT_MATRIX_SIZE")
    for row in new_rows:
        groups.setdefault((row["intent"], row["language"]), {})[row["arm"]] = row
    paired = []
    for (intent, language), group in groups.items():
        if set(group) != set(frozen["arms"]):
            raise ValueError("FACET_AUDIT_ARMS")
        for prefix in ("V5_CHUNK", "V5_WINDOW_R1", "V5_WINDOW_R2"):
            q, qb = group[prefix + "_Q"], group[prefix + "_QB"]
            if (q["pool_ids"] != qb["pool_ids"] or prefix != "V5_CHUNK" and
                    (q["pool_fingerprint"] != qb["pool_fingerprint"] or q["bundles"] != qb["bundles"]
                    or q["pool_trace"] != qb["pool_trace"])):
                raise ValueError("FACET_AUDIT_Q_QB_SHARED_POOL")
            paired.append({"intent": intent, "language": language, "prefix": prefix,
                "same_native_plan_and_pool": True, "selected_context_changed": q["context_chunk_ids"] != qb["context_chunk_ids"]})
    for original in [*new_rows, *recorded]:
        baseline = "method" in original
        row = dict(original)
        if baseline:
            row.update(intent=row["case"], question=next(t["question"] for t in frozen["tasks"]
                if t["intent"] == row["case"] and t["language"] == row["language"]),
                arm="V3_" + ("QB" if row["method"].endswith("balanced") else "Q"),
                pool_ids=row["trace"]["multi_query"]["ranked_chunk_ids"])
        n, spans = rendered_sources(row, docs, chunks, tokenizer)
        source_count += n
        span_count += spans
        audited = audit_source_dependencies(docs, chunks, row["context_chunk_ids"], deps, tenant_id="demo")
        if not baseline and canonical_json(audited) != canonical_json(row["context_audit"]):
            raise ValueError("FACET_AUDIT_ANNOTATION_RECOMPUTE")
        reach = [source_reachability(d, row["pool_ids"], row["context_chunk_ids"], chunks) for d in deps]
        if not baseline and canonical_json(reach) != canonical_json(row["source_reachability"]):
            raise ValueError("FACET_AUDIT_REACHABILITY_RECOMPUTE")
        if "packing_audit" in row:
            pack = row["packing_audit"]
            if (pack["used_graph"] or pack["hidden_fallbacks"] or pack["context_assertion_ids"]
                    or pack["source_message_sha256"] != sha256_bytes(row["source_message"].encode())):
                raise ValueError("FACET_AUDIT_METHOD_IDENTITY")
            for decision in pack["decisions"]:
                if decision["reason"] == "SELECTED" and not set(decision["chunk_ids"]) <= set(row["context_chunk_ids"]):
                    raise ValueError("FACET_AUDIT_PARTIAL_ADMITTED_WINDOW")
        details.append({"intent": row["intent"], "language": row["language"], "arm": row["arm"],
            "pool_size": len(row["pool_ids"]), "context_size": n,
            "user_message_proxy_tokens": row["user_message_proxy_tokens"],
            "source_message_sha256": sha256_bytes(row["source_message"].encode()),
            "controls": [{"label": labels[c["dependency_id"]], "status": c["status"], "reachability": r["stage"]}
                for c, r in zip(audited["checks"], reach, strict=True)]})
    summary = []
    for arm in ["V3_Q", "V3_QB", *frozen["arms"]]:
        own = [r for r in details if r["arm"] == arm]
        summary.append({"arm": arm, "inputs": len(own),
            "annotation_statuses": dict(Counter(c["status"] for r in own for c in r["controls"])),
            "reachability": dict(Counter(c["reachability"] for r in own for c in r["controls"])),
            "context_chunks": [r["context_size"] for r in own],
            "proxy_tokens": [r["user_message_proxy_tokens"] for r in own]})
    result = {"status": "SOURCE_EXACT_PAIRED_POOL_AND_CONTEXT_AUDIT_PASS_NOT_QUALITY_ACCURACY",
        "receipts_sha256": report["receipts_sha256"], "cells": len(details), "new_cells": 24,
        "rendered_source_appearances": source_count, "exact_source_spans": span_count,
        "source_truncations": 0, "q_qb_pairs": paired, "arm_summary": summary, "cell_details": details,
        "window_v1_v2_ast_equivalent_except_plan_binding_and_identity": algorithm_equivalence(),
        "quality_accuracy_measured": False, "independent_benchmark": False, "human_reviewed": 0,
        "new_model_calls": 0, "aws_calls": 0, "serving_published": False,
        "audit_code_sha256": sha256_bytes(Path(__file__).read_bytes())}
    immutable(OUTPUT / "source-recall-audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"q_qb_pairs", "cell_details"}}))


if __name__ == "__main__":
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=AssertionError("FACET_AUDIT_NETWORK_BLOCKED")))
        audit()
