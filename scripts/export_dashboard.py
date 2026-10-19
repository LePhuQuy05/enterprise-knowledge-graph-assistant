"""Export a bounded public dashboard projection from verified, frozen local artifacts.

No provider/AWS calls; never copies raw results, local paths, identities or credentials.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

from app.contracts import canonical_json, sha256_bytes
from app.evaluation import evidence_coverage, score_retrieval
from scripts.evaluate_development import SYSTEMS, load_inputs, read_dataset, summarize

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results/evaluation-dev/20261017T214800Z-6197bdfe"
OUTPUT = ROOT / "frontend/tracer/dashboard-data.json"
MAX_BYTES = 400_000


def official_url(value):
    url = urlsplit(value)
    if (url.scheme != "https" or url.hostname not in {"docs.aws.amazon.com", "aws.amazon.com"}
            or url.username or url.password or url.port not in {None, 443}):
        raise ValueError("DASHBOARD_UNAPPROVED_SOURCE_URL")
    return value


def project(dataset, baseline, graph, manifest, rows):
    if (manifest["status"] != "COMPLETE_DIAGNOSTIC_ONLY" or manifest["k"] != 5
            or manifest["source_manifest_sha256"] != baseline.manifest_sha256
            or manifest["graph_manifest_sha256"] != graph.manifest_sha256
            or manifest["graph_release_approved"] is not False):
        raise ValueError("DASHBOARD_RUN_IDENTITY_MISMATCH")
    if any(item.review_status != "AI_DRAFT" for item in dataset.items):
        raise ValueError("DASHBOARD_REVIEW_STATUS_REQUIRES_PROJECTION_VERSION_REVIEW")
    counts = Counter(c.document_version_id for c in baseline.chunks.values())
    documents = []
    for doc in sorted(baseline.documents.values(), key=lambda d: d.title):
        service = urlsplit(doc.source_uri).path.split("/")[1].upper()
        documents.append({"id": doc.document_version_id, "title": doc.title,
            "url": official_url(doc.source_uri), "service": service, "fetched_at": doc.fetched_at.isoformat(),
            "chunks": counts[doc.document_version_id]})
    chunks = [{"id": c.chunk_id, "document": c.document_version_id,
               "section": list(c.section_path), "preview": c.text[:280],
               "truncated": len(c.text) > 280, "start_char": c.start_char, "end_char": c.end_char}
              for c in sorted(baseline.chunks.values(), key=lambda c: c.chunk_id)]
    positions = {c["id"]: i for i, c in enumerate(chunks)}
    expected = {(item.question_id, system) for item in dataset.items for system in SYSTEMS}
    actual = [(r["question_id"], r["system"]) for r in rows]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("DASHBOARD_INCOMPLETE_OR_DUPLICATE_ROWS")
    cases = []
    for item in dataset.items:
        methods = {}
        for row in (r for r in rows if r["question_id"] == item.question_id):
            ranked, context = row["ranked_chunk_ids"], row["context_chunk_ids"]
            metrics = score_retrieval(item, ranked, context, graph, k=manifest["k"])
            if metrics != row["metrics"]:
                raise ValueError("DASHBOARD_REPLAY_METRIC_MISMATCH")
            sets = {"corpus": list(graph.chunks.values()),
                "candidates": [graph.chunks[key] for key in ranked],
                "top_k": [graph.chunks[key] for key in ranked[:manifest["k"]]],
                "context": [graph.chunks[key] for key in context]}
            methods[row["system"]] = {"ranked": [positions[key] for key in ranked],
                "context": [positions[key] for key in context],
                "coverage": {key: list(evidence_coverage(item, values)) if item.answerable else None
                             for key, values in sets.items()},
                "complete_at_k": metrics["complete_evidence_at_k"],
                "context_complete": metrics["context_complete"],
                "context_proxy_tokens": row["context_proxy_tokens"],
                "used_graph": row["used_graph"], "warnings": row["warnings"]}
        cases.append({"id": item.question_id, "question": item.question, "category": item.category,
            "answerable": item.answerable, "review_status": item.review_status,
            "unanswerable_reason": item.unanswerable_reason,
            "groups": [{"id": g.group_id, "fact": g.required_fact,
                        "alternatives": [{"document": s.document_version_id, "quote": s.quote,
                                          "start_char": s.start_char, "end_char": s.end_char}
                                         for s in g.alternatives]} for g in item.evidence_groups],
            "methods": methods})
    summary = summarize(rows, dataset)
    if sha256_bytes(canonical_json(summary)) != manifest["summary_sha256"]:
        raise ValueError("DASHBOARD_SUMMARY_MISMATCH")
    return {"schema_version": 1, "scope": "PUBLIC_RECORDED_DEVELOPMENT",
        "snapshot": {"id": baseline.manifest.snapshot_version, "manifest_sha256": baseline.manifest_sha256,
                     "created_at": baseline.manifest.created_at.isoformat(),
                     "documents": len(documents), "chunks": len(chunks),
                     "embedding_model": baseline.manifest.embedding_recipe.model_id,
                     "dimension": baseline.manifest.embedding_recipe.dimension},
        "run": {"id": manifest["run_id"], "completed_at": manifest["completed_at"],
                "dataset_sha256": manifest["dataset_sha256"], "rankings_sha256": manifest["retrieval_sha256"],
                "graph_snapshot_sha256": manifest["graph_manifest_sha256"],
                "source_vector_parity_verified": True, "k": manifest["k"],
                "context_max_chunks": manifest["context_max_chunks"],
                "context_proxy_tokens": manifest["context_proxy_tokens"], "rrf_constant": manifest["rrf_constant"],
                "label_status": "AI_DRAFT", "generation_calls": 0, "graph_serving": False},
        "documents": documents, "chunks": chunks, "cases": cases,
        "summaries": {s: {"draft": summary["systems"][s]["provisional_ai_draft"],
                          "approved": summary["systems"][s]["human_approved_only"]} for s in SYSTEMS}}


def build_projection(run=RUN):
    dataset, dataset_sha = read_dataset(run / "dataset.json")
    baseline, graph = load_inputs(dataset)
    manifest = json.loads((run / "manifest.json").read_bytes())
    raw = (run / "retrieval.jsonl").read_bytes()
    if dataset_sha != manifest["dataset_sha256"] or sha256_bytes(raw) != manifest["retrieval_sha256"]:
        raise ValueError("DASHBOARD_FROZEN_INPUT_HASH_MISMATCH")
    value = project(dataset, baseline, graph, manifest, [json.loads(line) for line in raw.splitlines()])
    raw = canonical_json(value) + b"\n"
    if len(raw) > MAX_BYTES:
        raise ValueError("DASHBOARD_PUBLIC_PAYLOAD_TOO_LARGE")
    return raw


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify the checked-in projection matches pinned artifacts")
    args = parser.parse_args()
    raw = build_projection()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_bytes() != raw:
            raise ValueError("DASHBOARD_PROJECTION_STALE")
    else:
        OUTPUT.write_bytes(raw)
    print(json.dumps({"status": "PASS", "path": OUTPUT.relative_to(ROOT).as_posix(),
                      "bytes": len(raw), "sha256": sha256_bytes(raw), "model_calls": 0, "aws_calls": 0}))


if __name__ == "__main__":
    main()
