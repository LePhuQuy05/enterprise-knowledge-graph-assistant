"""Read-only trace, rendered-source and supplemental-source reachability audit."""

import json
from collections import Counter
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.graph_collection_migration import collection_records
from app.source_dependency_audit import SourceDependency, missing_intervals
from scripts.check_collection319_methods import CODE, METHODS, OUTPUT
from scripts.migrate_collection319_graph import OUTPUT as MIGRATION
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT


def source_reachability(dependency, pool, selected, chunks):
    own = [chunks[key] for key in pool if chunks[key].document_version_id == dependency.document_version_id]
    intervals = tuple((chunk.start_char, chunk.end_char) for chunk in own)
    anchor_missing = missing_intervals(dependency.anchor.start_char, dependency.anchor.end_char, intervals)
    required_missing = [missing_intervals(span.start_char, span.end_char, intervals) for span in dependency.required_spans]
    selected_intervals = tuple((chunks[key].start_char, chunks[key].end_char) for key in selected
        if chunks[key].document_version_id == dependency.document_version_id)
    selected_missing = [missing_intervals(span.start_char, span.end_char, selected_intervals)
        for span in (dependency.anchor, *dependency.required_spans)]
    stage = ("DOCUMENT_ABSENT_FROM_POOL" if not own else "ANCHOR_OUTSIDE_POOL" if anchor_missing
        else "REQUIRED_SPANS_OUTSIDE_POOL" if any(required_missing)
        else "ELIGIBLE_BUNDLE_NOT_SELECTED" if any(selected_missing) else "COMPLETE_REVIEWED_BUNDLE_SELECTED")
    return {"dependency_id": dependency.dependency_id, "document_version_id": dependency.document_version_id,
        "stage": stage, "pool_document_chunk_ids": [chunk.chunk_id for chunk in own],
        "anchor_missing_in_pool": anchor_missing, "required_missing_in_pool": required_missing,
        "missing_in_selected": selected_missing,
        "not_a_required_gold_source_or_semantic_quality_label": True}


def audit():
    collection, _, graph = load_migrated()
    docs, chunks, _ = collection_records(collection)
    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    report = json.loads((OUTPUT / "report.json").read_bytes())
    payload = json.loads((OUTPUT / "receipts.json").read_bytes())
    if (scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}
            or report["receipts_sha256"] != sha256_bytes((OUTPUT / "receipts.json").read_bytes())):
        raise ValueError("COLLECTION319_AUDIT_PINS")
    dependencies = [SourceDependency.model_validate_json(canonical_json(row["dependency"])) for row in json.loads(
        (ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())["source_controls"]]
    added = set(json.loads((MIGRATION / "source-equivalence.json").read_bytes())["added_document_version_ids"])
    by_fact = {fact.assertion_id: fact for fact in graph.facts}
    cells = payload["cells"]
    groups = {}
    for row in cells[:32]:
        groups.setdefault((row["case"], row["language"]), {})[row["method"]] = row
    details, source_count, span_count = [], 0, 0
    for row in cells:
        trace = row["trace"]
        packing = trace["source_message_packing"]
        audit = packing["audit"]
        context = tuple(row["context_chunk_ids"])
        prompt = json.loads(row["source_message"])
        if (sha256_bytes(row["source_message"].encode()) != row["prompt_sha256"]
                or packing["user_message_sha256"] != row["prompt_sha256"]
                or sha256_bytes(canonical_json(audit)) != packing["audit_sha256"]
                or audit["manifest_sha256"] != collection.manifest_sha256
                or tuple(s["chunk_id"] for s in audit["sources"]) != context
                or row["binding"]["mode_requested"] != row["method"]
                or row["binding"]["mode_used"] != row["method"]):
            raise ValueError("COLLECTION319_AUDIT_PROMPT_OR_BINDING")
        rendered = {source["source_id"]: source for source in prompt["untrusted_sources"]}
        if set(rendered) != {source["source_id"] for source in audit["sources"]}:
            raise ValueError("COLLECTION319_AUDIT_SOURCE_OMITTED_FROM_MODEL_INPUT")
        for source in audit["sources"]:
            chunk = chunks[source["chunk_id"]]
            document = docs[chunk.document_version_id]
            if (source["text_sha256"] != chunk.text_sha256 or source["source_uri"] != document.source_uri
                    or source["start_char"] != chunk.start_char or source["end_char"] != chunk.end_char
                    or source["document_version_id"] != chunk.document_version_id
                    or document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
                raise ValueError("COLLECTION319_AUDIT_ORIGINAL_SOURCE")
            cursor = 0
            for span in rendered[source["source_id"]]["spans"]:
                start = chunk.text.find(span["text"], cursor)
                if start < 0 or chunk.text[cursor:start].strip():
                    raise ValueError("COLLECTION319_AUDIT_NON_LITERAL_OR_INTERIOR_LOSS")
                cursor = start + len(span["text"])
                span_count += 1
            if chunk.text[cursor:].strip():
                raise ValueError("COLLECTION319_AUDIT_RENDERED_SOURCE_TRUNCATED")
            source_count += 1
        if row["method"].startswith("graph_"):
            used = row["outcome"]["mechanism_status"] == "USED_GRAPH"
            if (used != row["binding"]["used_graph"] or used != bool(trace["packed_paths"])
                    or row["outcome"]["effective_method"] != row["method"]):
                raise ValueError("COLLECTION319_AUDIT_GRAPH_STATUS")
            for path in trace["packed_paths"]:
                support = tuple(dict.fromkeys(by_fact[key].chunk_id for key in path["assertion_ids"]))
                if (support != tuple(path["supporting_chunk_ids"]) or not set(support) <= set(context)
                        or not set(path["assertion_ids"]) <= set(row["context_assertion_ids"])
                        or path["condition_compatibility"] != "NOT_INFERRED"):
                    raise ValueError("COLLECTION319_AUDIT_PATH_SUPPORT")
        if not row["case"].startswith("control-"):
            group = groups[(row["case"], row["language"])]
            method = row["method"]
            graph_trace = group["graph_1hop"]["trace"]
            if method == "bm25":
                pool = graph_trace["lexical_ranking"]
            elif method == "vector":
                pool = graph_trace["vector_ranking"]
            elif method in {"hybrid", "hybrid_reranked"}:
                pool = group["hybrid_reranked"]["trace"]["reranking"][0]["baseline_candidate_ids"]
            elif method.startswith("hybrid_multi_query"):
                pool = trace["multi_query"]["ranked_chunk_ids"]
            else:
                pool = trace["document_allocation"]["final_order"]
            details.append({"case": row["case"], "language": row["language"], "method": method,
                "pool_size": len(pool), "selected_size": len(context),
                "supplemental_sources_selected": sum(chunks[key].document_version_id in added for key in context),
                "canonical_source_controls": [source_reachability(dep, pool, context, chunks) for dep in dependencies]})
    paired = []
    for key, group in groups.items():
        if set(group) != set(METHODS):
            raise ValueError("COLLECTION319_AUDIT_METHOD_MATRIX")
        q, qb = (group[method] for method in ("hybrid_multi_query", "hybrid_multi_query_balanced"))
        if (q["trace"]["multi_query"] != qb["trace"]["multi_query"] or not qb["outcome"]["context_selection_used"]
                or q["trace"]["plan"] != qb["trace"]["plan"]):
            raise ValueError("COLLECTION319_AUDIT_Q_QB_POOL")
        paired.append({"case": key[0], "language": key[1], "same_native_plan_and_pool": True,
            "qb_order_changed": qb["trace"]["selection_order_changed"],
            "qb_allocator_decisions": len(qb["trace"]["context_selection"]["decisions"])})
    result = {"status": "EXACT_SOURCES_AND_METHODS_PASS_CANONICAL_COVERAGE_NOT_ESTABLISHED",
        "receipts_sha256": report["receipts_sha256"], "audit_implementation_sha256": sha256_bytes(Path(__file__).read_bytes())}
    result.update(cells=len(cells), rendered_source_appearances=source_count, exact_source_spans=span_count,
        source_truncations=0, q_qb_pairs=paired,
        dependency_audit_statuses=dict(Counter(check["status"] for row in cells for check in row["dependency_audit"]["checks"])),
        source_reachability_statuses=dict(Counter(item["stage"] for row in details for item in row["canonical_source_controls"])),
        canonical_source_reachability=details, all_qualifiers_discovered=False, quality_accuracy_measured=False,
        new_model_calls=0, aws_calls=0, serving_published=False,
        caveat="NOT_TRIGGERED is not COMPLETE. Canonical-source absence does not prove no alternative supporting evidence exists.")
    immutable(OUTPUT / "source-method-audit.json", result)
    print(json.dumps({key: value for key, value in result.items() if key not in {"canonical_source_reachability", "q_qb_pairs"}}))


if __name__ == "__main__":
    audit()
