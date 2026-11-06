"""Publish only source-verified public examples, never raw provider or owner records."""

import json
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_bytes())


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def project():
    bundle = Path(read("results/workbench-cloud/assets-build-v1.json")["bundle_directory"])
    docs, chunks = {}, {}
    for shard in (bundle / "collection").iterdir():
        if shard.is_dir():
            docs.update((d["document_version_id"], d) for d in rows(shard / "documents.jsonl"))
            chunks.update((c["chunk_id"], c) for c in rows(shard / "chunks.jsonl"))
    examples, sources, pins = [], {}, []
    cases = [
        ("results/collection-answering/ecs-en-controlled-live.json", ("bm25", "vector")),
        ("results/reranker-answering/pilot-v1/case-1.json", ("hybrid", "hybrid_reranked")),
        ("results/multi-query/pilot-v1/case-1.json", ("hybrid_multi_query",)),
        ("results/context-allocation/answer-pilot-v1/case-1.json", ("hybrid_multi_query_balanced",)),
    ]

    def source(key, documents, passages):
        c = passages[key]
        d = documents[c["document_version_id"]]
        text = d["normalized_text"][c["start_char"]:c["end_char"]]
        if text != c["text"] or sha256_bytes(text.encode()) != c["text_sha256"]:
            raise ValueError("METHOD_SOURCE_PIN_CHANGED")
        if not d["source_uri"].startswith("https://docs.aws.amazon.com/"):
            raise ValueError("METHOD_SOURCE_NOT_OFFICIAL")
        return {k: c[k] for k in ("chunk_id", "document_version_id", "start_char", "end_char", "text_sha256")} | {
            "title": d["title"], "url": d["source_uri"], "text": text, "section": c["section_path"]}

    for path, methods in cases:
        raw = (ROOT / path).read_bytes()
        case = json.loads(raw)
        pins.append({"artifact": path, "sha256": sha256_bytes(raw)})
        for method in methods:
            m = next(m for m in case["methods"] if m["requested_method"] == method)
            if m["status"] != "COMPLETED" or m["response"] is None:
                raise ValueError("METHOD_RECORDED_ANSWER_REQUIRED")
            for key in set(m["ranked_chunk_ids"] + m["context_chunk_ids"]):
                sources[key] = source(key, docs, chunks)
            for citation in m["response"]["citations"]:
                s = sources[citation["chunk_id"]]
                if citation["quote"] not in s["text"] or citation["document_version_id"] != s["document_version_id"]:
                    raise ValueError("METHOD_CITATION_NOT_IN_SOURCE")
            example = {"id": method, "question": case["input"]["original_question"],
                "search_query": case["search_query"], "collection_version": case["collection_version"],
                "documents": case["documents"], "chunks": case["chunks"], "recorded_at": case["created_at"],
                "artifact_sha256": sha256_bytes(raw), "config": m["config"], "config_sha256": m["config_sha256"],
                "ranked_chunk_ids": m["ranked_chunk_ids"], "context_chunk_ids": m["context_chunk_ids"],
                "context_proxy_tokens": m["context_proxy_tokens"], "response": {
                    k: m["response"][k] for k in ("answer_status", "answer_language", "claims", "citations")},
                "events": m["trace"], "subqueries": m.get("planning_trace", {}).get("subqueries", []),
                "query_rankings": m.get("multi_query_trace", {}).get("queries", []),
                "review": "NOT_REVIEWED", "kind": "RECORDED_LOCAL_EXECUTION"}
            if m.get("rerank_trace"):
                example["rerank"] = m["rerank_trace"]
            if m.get("context_selection_trace"):
                example["selection"] = m["context_selection_trace"]
            examples.append(example)

    receipt = read("results/collection-graph/curated-subset-v1.json")
    graph_path = Path(receipt["graph_path"])
    if sha256_bytes(graph_path.read_bytes()) != receipt["graph_file_sha256"]:
        raise ValueError("METHOD_GRAPH_FILE_CHANGED")
    overlay = json.loads(graph_path.read_bytes())
    if overlay["collection_version"] != examples[0]["collection_version"] or receipt["documents"] != 316 or receipt["chunks"] != 2978:
        raise ValueError("METHOD_GRAPH_MUST_USE_COLLECTION316")
    curation_raw = Path(receipt["curation_path"]).read_bytes()
    if sha256_bytes(curation_raw) != overlay["curation_sha256"]:
        raise ValueError("METHOD_GRAPH_CURATION_CHANGED")
    curation = json.loads(curation_raw)
    selected = {r["assertion_id"]: r for r in curation["decisions"] if r["decision"] == "AI_SUPPORTED_FOR_EXPERIMENT"}
    audit_raw = (ROOT / "results/collection-graph/answer-pilot-audit-v1.json").read_bytes()
    audit = json.loads(audit_raw)
    if audit["status"] != "PASS_SOURCE_RECEIPT_INPUT_USAGE_LEDGER_AND_NETWORK_FREE_REPLAY" or audit["graph_file_sha256"] != receipt["graph_file_sha256"]:
        raise ValueError("METHOD_GRAPH_ANSWER_AUDIT_REQUIRED")
    marker = read("results/collection-graph/answer-pilot-started-v1.json")
    modes = []
    for i, case in enumerate(audit["cases"], 1):
        path = "results/collection-graph/answer-" + case["id"] + "-v1.json"
        raw = (ROOT / path).read_bytes()
        value = json.loads(raw)
        if sha256_bytes(raw) != case["receipt_sha256"] or value["status"] != "COMPLETED" or value["requested_method"] != f"graph_{i}hop":
            raise ValueError("METHOD_GRAPH_RECORDED_ANSWER_CHANGED")
        modes.append(value["trace"] | {"response": {k: value["response"][k] for k in
            ("answer_status", "answer_language", "claims", "citations")}, "artifact_sha256": sha256_bytes(raw),
            "execution_identity_sha256": value["execution_identity_sha256"], "recorded_at": marker["started_at"],
            "events": value["events"], "answer_review_notes_vi": [
                "Cần review citation cho phát biểu 3: quote [2] dùng ‘By using this’, nên đọc cả câu đứng trước để xác định ‘this’ là NAT64. Quote được chọn riêng chưa tự làm rõ tiền ngữ."
                if i == 1 else
                "Câu trả lời tiếng Việt này chưa được người chấm độc lập xác nhận. Lần chạy cho phép tối đa 2 hops; chưa chứng minh cần đường hai cạnh hoặc tốt hơn 1-hop."
            ]})
        pins.append({"artifact": path, "sha256": sha256_bytes(raw)})
    pins.append({"artifact": "results/collection-graph/answer-pilot-audit-v1.json", "sha256": sha256_bytes(audit_raw)})
    edges = []
    for a in overlay["assertions"]:
        sources[a["chunk_id"]] = source(a["chunk_id"], docs, chunks)
        ev = a["evidence"]
        if docs[a["document_version_id"]]["normalized_text"][ev["start_char"]:ev["end_char"]] != ev["quote"]:
            raise ValueError("METHOD_GRAPH_EVIDENCE_OFFSET_CHANGED")
        decision = selected.get(a["assertion_id"])
        if decision is None or decision["assertion_payload_sha256"] != sha256_bytes(canonical_json(a)):
            raise ValueError("METHOD_GRAPH_UNREVIEWED_ASSERTION")
        edges.append({k: a[k] for k in ("assertion_id", "subject_id", "object_id", "predicate", "polarity", "qualifiers", "chunk_id", "evidence")} |
            {"review_status": "AI_SUPPORTED_FOR_EXPERIMENT", "review_note_vi": decision["reason_vi"]})
    graph = {"snapshot_version": overlay["collection_version"], "manifest_sha256": overlay["collection_manifest_sha256"],
        "graph_sha256": overlay["graph_sha256"], "documents": 316, "chunks": 2978, "review": "PENDING",
        "nodes": [{k: e[k] for k in ("entity_id", "canonical_name", "entity_type", "aliases")} for e in overlay["entities"]],
        "edges": edges, "recorded_question": modes[0]["question"], "modes": modes,
        "trace_sha256": receipt["graph_file_sha256"], "full_relation_extraction": False,
        "relation_source_chunks": receipt["relation_source_chunks"], "relation_source_documents": receipt["relation_source_documents"],
        "mention_covered_chunks": receipt["mention_covered_chunks"], "mentions": len(overlay["mentions"]),
        "curation_sha256": overlay["curation_sha256"], "source_review": "AI_PROVISIONAL_NOT_HUMAN_GOLD", "human_reviewed": False,
        "selection": {"selected": curation["selected"], "rejected": curation["rejected"], "pending": curation["pending"]},
        "legacy_graph_reused": False}
    for mode in graph["modes"]:
        for key in set(mode["final_ranking"] + mode["context_chunk_ids"]):
            sources[key] = source(key, docs, chunks)
        for citation in mode["response"]["citations"]:
            s = sources[citation["chunk_id"]]
            if citation["chunk_id"] not in mode["context_chunk_ids"] or citation["document_version_id"] != s["document_version_id"] or citation["quote"] not in s["text"]:
                raise ValueError("METHOD_GRAPH_CITATION_NOT_IN_CONTEXT_SOURCE")
    return {"schema_version": 1, "scope": "METHOD_WALKTHROUGHS_NOT_QUALITY_SCORES", "examples": examples,
            "sources": sources, "graph": graph, "artifact_pins": pins}


def main():
    result = project()
    raw = canonical_json(result) + b"\n"
    if len(raw) > 512 * 1024:
        raise ValueError("METHOD_PROJECTION_TOO_LARGE")
    output = ROOT / "frontend/tracer/methods-data.json"
    output.write_bytes(raw)
    print(json.dumps({"status": "SOURCE_VERIFIED_LOCAL_PROJECTION", "bytes": len(raw),
        "examples": len(result["examples"]), "sources": len(result["sources"]),
        "graph_nodes": len(result["graph"]["nodes"]), "graph_edges": len(result["graph"]["edges"]),
        "sha256": sha256_bytes(raw), "model_calls": 0, "aws_calls": 0}))


if __name__ == "__main__":
    main()
