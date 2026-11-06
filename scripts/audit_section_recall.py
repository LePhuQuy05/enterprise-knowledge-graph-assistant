"""Read-only source/loss audit across section, parent and reranker experiments."""

import json
import math
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.ancestor_section_recall import ancestor_lead
from app.bounded_ancestor_recall import bounded_anchor
from app.contracts import canonical_json, sha256_bytes
from app.document_recall_v2 import document_window
from app.faceted_query_v5 import FacetPlanV5
from app.graph_collection_migration import collection_records
from app.ingestion.text import ProxyTokenizer
from app.section_recall import literal_probes
from app.source_dependency_audit import (
    SourceDependency,
    audit_source_dependencies,
    missing_intervals,
)
from scripts.audit_collection319_methods import source_reachability
from scripts.audit_facet_recall_v5 import rendered_sources
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT

FAMILIES = ("section-recall-v1", "parent-section-recall-v1", "reranked-parent-recall-v1", "ancestor-section-recall-v1", "bounded-ancestor-recall-v1")
OUTPUT = ROOT / "results/section-recall-audit-v2"
LABELS = ("lambda-permission-categories", "tutorial-getobject-role", "async-event-202", "kms-cross-account-operations")


def audit_reranker(trace, chunks):
    record = trace.get("reranking")
    if record is None:
        return 0
    anchors = record["candidate_anchor_ids"]
    ranked = record["ranked_anchor_ids"]
    if (len(anchors) > 40 or len(set(anchors)) != len(anchors) or len(anchors) != len(ranked)
            or set(anchors) != set(ranked) or record["fallback_executed"] or len(record["windows"]) > 160):
        raise ValueError("SECTION_AUDIT_RERANK_BOUNDS_OR_FALLBACK")
    for key in anchors:
        windows = [w for w in record["windows"] if w["chunk_id"] == key]
        if not 1 <= len(windows) <= 8 or missing_intervals(0, len(chunks[key].text),
                [(w["start_char"], w["end_char"]) for w in windows]):
            raise ValueError("SECTION_AUDIT_RERANK_WINDOW_COVERAGE")
        if any(not 0 <= w["start_char"] < w["end_char"] <= len(chunks[key].text)
                or not 1 <= w["input_tokens"] <= 512 or not math.isfinite(w["raw_logit"]) for w in windows):
            raise ValueError("SECTION_AUDIT_RERANK_WINDOW_INVALID")
    if any(w["chunk_id"] not in anchors for w in record["windows"]):
        raise ValueError("SECTION_AUDIT_RERANK_UNKNOWN_SOURCE")
    return len(record["windows"])


def audit():
    collection, _, _ = load_migrated()
    docs, chunks, _ = collection_records(collection)
    raw = json.loads((ROOT / "results/facet-recall-v5-canary-v1/receipts.json").read_bytes())
    baselines = raw["recall_cells"]
    annotations = json.loads((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())
    deps = [SourceDependency.model_validate_json(canonical_json(r["dependency"])) for r in annotations["source_controls"]]
    deps.append(SourceDependency.model_validate_json(canonical_json(annotations["parent_kms_dependency"])))
    tokenizer = ProxyTokenizer()
    families, details, pairs = [], [], []
    source_count, spans, rerank_windows = 0, 0, 0
    for family in FAMILIES:
        directory = ROOT / "results" / family
        payload = json.loads((directory / "receipts.json").read_bytes())
        scope = json.loads((directory / "scope.json").read_bytes())
        report = json.loads((directory / "report.json").read_bytes())
        replay = json.loads((directory / "replay.json").read_bytes())
        digest = sha256_bytes((directory / "receipts.json").read_bytes())
        if (report["receipts_sha256"] != digest or replay["receipts_sha256"] != digest
                or any(sha256_bytes((ROOT / p).read_bytes()) != h for p, h in scope["code_sha256"].items())
                or canonical_json(payload["recorded_v5_baseline"]) != canonical_json(baselines)
                or len(payload["cells"]) != scope["new_cells"]):
            raise ValueError("SECTION_AUDIT_PINS_REPLAY_OR_BASELINE")
        families.append({"family": family, "cells": len(payload["cells"]), "receipts_sha256": digest})
        grouped = {}
        for row in payload["cells"]:
            key = (row["intent"], row["language"], row["arm"].removesuffix("B"))
            grouped.setdefault(key, []).append(row)
            trace = row["pool_trace"]
            plan = FacetPlanV5.model_validate_json(canonical_json(trace["native_v5_plan"]))
            queries = literal_probes(collection, row["question"], row["search"], row["language"], plan)
            if tuple(trace["queries"]) != queries or trace["source_annotations_used_for_ranking"]:
                raise ValueError("SECTION_AUDIT_QUERY_TRACK")
            base = next(r for r in baselines if r["intent"] == row["intent"] and r["language"] == row["language"]
                and r["arm"] == "V5_CHUNK_Q")["pool_ids"]
            if not set(base[:16]) <= set(row["pool_ids"]):
                raise ValueError("SECTION_AUDIT_PROTECTED_BASE_PREFIX")
            n, s = rendered_sources(row, docs, chunks, tokenizer)
            source_count += n
            spans += s
            actual = audit_source_dependencies(docs, chunks, row["context_chunk_ids"], deps, tenant_id="demo")
            reach = [source_reachability(d, row["pool_ids"], row["context_chunk_ids"], chunks) for d in deps]
            if canonical_json(actual) != canonical_json(row["context_audit"]) or canonical_json(reach) != canonical_json(row["source_reachability"]):
                raise ValueError("SECTION_AUDIT_SOURCE_CONTROL_MISMATCH")
            pack = row["packing_audit"]
            if (pack["used_graph"] or pack["hidden_fallbacks"] or row["graph_records_used"]
                    or pack["context_assertion_ids"] or row["binding"]["used_graph"]
                    or row["generation_executed"] or pack["source_message_sha256"] != sha256_bytes(row["source_message"].encode())):
                raise ValueError("SECTION_AUDIT_METHOD_OR_MESSAGE")
            windows = [b for b in row["bundles"] if b["kind"] == "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT"]
            if len(windows) > 6:
                raise ValueError("SECTION_AUDIT_WINDOW_CAP")
            members = {key for b in windows for key in b["chunk_ids"]}
            for b in row["bundles"]:
                if b["kind"] == "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT":
                    if tuple(b["chunk_ids"]) != document_window(chunks, b["anchor_chunk_id"], trace["recipe"]["radius"]):
                        raise ValueError("SECTION_AUDIT_NEIGHBOR_SOURCE_MISMATCH")
                    if "retrieved_child_anchor_id" in b:
                        if trace["recipe"]["version"] == "bounded-ancestor-recall-v1":
                            expected, _ = bounded_anchor(chunks, b["retrieved_child_anchor_id"], trace["recipe"]["radius"])
                            if b["retrieved_child_anchor_id"] not in b["chunk_ids"]:
                                raise ValueError("SECTION_AUDIT_RETRIEVED_CHILD_DROPPED")
                        else:
                            expected = ancestor_lead(chunks, b["retrieved_child_anchor_id"])
                        if b["anchor_chunk_id"] != expected:
                            raise ValueError("SECTION_AUDIT_ANCESTOR_CONTEXT_MISMATCH")
                elif len(b["chunk_ids"]) != 1 or b["chunk_ids"][0] in members:
                    raise ValueError("SECTION_AUDIT_PARTIAL_SINGLETON")
            for d in pack["decisions"]:
                if d["reason"] == "SELECTED" and not set(d["chunk_ids"]) <= set(row["context_chunk_ids"]):
                    raise ValueError("SECTION_AUDIT_NONATOMIC_PACK")
            if row["arm"].endswith("_Q"):
                rerank_windows += audit_reranker(trace, chunks)
            balanced = row["arm"].endswith("_QB")
            baseline = next(r for r in baselines if r["intent"] == row["intent"] and r["language"] == row["language"]
                and r["arm"] == f"V5_WINDOW_R{trace['recipe']['radius']}_" + ("QB" if balanced else "Q"))
            if (row["context_removed_vs_v5"] != sorted(set(baseline["context_chunk_ids"])-set(row["context_chunk_ids"]))
                    or row["context_added_vs_v5"] != sorted(set(row["context_chunk_ids"])-set(baseline["context_chunk_ids"]))):
                raise ValueError("SECTION_AUDIT_LOSS_ACCOUNTING")
            related = (0, 1) if row["intent"] == "s3-permission-layers" else (3,)
            checks = actual["checks"]
            baseline_checks = baseline["context_audit"]["checks"]
            details.append({"family": family, "intent": row["intent"], "language": row["language"], "arm": row["arm"],
                "context_size": n, "proxy_tokens": row["user_message_proxy_tokens"], "sources_removed": len(row["context_removed_vs_v5"]),
                "related_checks": [{"label": LABELS[i], "status": checks[i]["status"], "baseline_status": baseline_checks[i]["status"],
                    "reachability": reach[i]["stage"]} for i in related],
                "all_checks": [{"label": label, "status": c["status"]} for label, c in zip(LABELS, checks, strict=True)]})
        for key, values in grouped.items():
            if (len(values) != 2 or not any(r["arm"].endswith("_Q") for r in values)
                    or values[0]["pool_trace"] != values[1]["pool_trace"] or values[0]["pool_ids"] != values[1]["pool_ids"]
                    or values[0]["pool_fingerprint"] != values[1]["pool_fingerprint"] or values[0]["bundles"] != values[1]["bundles"]):
                raise ValueError("SECTION_AUDIT_PAIRED_METHODS")
            pairs.append({"family": family, "intent": key[0], "language": key[1], "arm": key[2],
                "same_plan_pool_bundles": True, "context_changed": values[0]["context_chunk_ids"] != values[1]["context_chunk_ids"]})
    summaries = []
    complete = "COMPLETE_REVIEWED_SOURCE_CONTEXT"
    incomplete = "INCOMPLETE_REVIEWED_SOURCE_CONTEXT"
    for family in FAMILIES:
        for arm in sorted({r["arm"] for r in details if r["family"] == family}):
            own = [r for r in details if r["family"] == family and r["arm"] == arm]
            checks = [c for r in own for c in r["related_checks"]]
            summaries.append({"family": family, "arm": arm, "inputs": len(own), "related_checks": len(checks),
                "complete": sum(c["status"] == complete for c in checks), "incomplete": sum(c["status"] == incomplete for c in checks),
                "recovered_vs_v5": sum(c["status"] == complete and c["baseline_status"] != complete for c in checks),
                "regressed_vs_v5": sum(c["status"] != complete and c["baseline_status"] == complete for c in checks),
                "proxy_tokens_max": max(r["proxy_tokens"] for r in own), "sources_removed": [r["sources_removed"] for r in own]})
    result = {"status": "SOURCE_INTEGRITY_AND_REPLAY_PASS_DEVELOPMENT_COVERAGE_REMAINS_PARTIAL",
        "families": families, "new_cells": len(details), "underlying_intents": 2, "language_inputs": 4,
        "paired_q_qb": len(pairs), "rendered_source_appearances": source_count, "exact_source_spans": spans,
        "reranker_token_windows_checked": rerank_windows, "arm_summary": summaries, "cell_details": details, "pairs": pairs,
        "no_independent_benchmark_or_answer_accuracy": True, "source_annotations_are_not_required_gold_uris": True,
        "human_reviewed": 0, "model_api_calls": 0, "aws_calls": 0, "generation_calls": 0, "extraction_calls": 0,
        "serving_published": False, "audit_code_sha256": sha256_bytes(Path(__file__).read_bytes())}
    immutable(OUTPUT / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k not in {"arm_summary", "cell_details", "pairs"}}))


if __name__ == "__main__":
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=AssertionError("SECTION_AUDIT_NETWORK_BLOCKED")))
        audit()
