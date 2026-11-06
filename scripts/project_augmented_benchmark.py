"""Project complete paired results; retain the previous public projection locally."""

import argparse
import json
from pathlib import Path

from app.benchmark_queries import AugmentedRetrievalReceipt
from app.collection_benchmark import ALL_METHODS, score_receipts, summarize_receipts
from app.contracts import canonical_json, sha256_bytes
from scripts.run_augmented_collection_benchmark import QUERY_DIR, load_shared_queries
from scripts.run_collection_benchmark import ROOT, load_inputs


def project(run_dir):
    protocol_raw, dataset, collection, index, _ = load_inputs()
    bundle, bundle_sha, preparation = load_shared_queries(dataset, protocol_raw)
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    if (manifest["protocol_sha256"] != sha256_bytes(protocol_raw) or manifest["shared_bundle_sha256"] != bundle_sha
            or manifest["languages"] != ["en", "vi"] or manifest["receipts"] != 1600):
        raise ValueError("PAIRED_PUBLIC_BENCHMARK_INPUT_MISMATCH")
    for name, expected in manifest["artifacts"].items():
        if Path(name).name != name or sha256_bytes((run_dir / name).read_bytes()) != expected:
            raise ValueError("PAIRED_PUBLIC_BENCHMARK_ARTIFACT_CHANGED")
    receipts = [AugmentedRetrievalReceipt.model_validate_json(line) for line in (run_dir / "receipts.jsonl").read_bytes().splitlines()]
    scored = score_receipts(dataset, receipts, index)
    expected = {(s.scenario_id, lang, method) for s in dataset.scenarios for lang in ("en", "vi") for method in ALL_METHODS}
    if len(scored) != 1600 or {(r["scenario_id"], r["language"], r["requested_method"]) for r in scored} != expected:
        raise ValueError("PAIRED_PUBLIC_BENCHMARK_MATRIX_INCOMPLETE")
    summary = summarize_receipts(dataset, scored, languages=("en", "vi"), methods=ALL_METHODS)
    if canonical_json(summary) != canonical_json(json.loads((run_dir / "summary.json").read_bytes())["methods"]):
        raise ValueError("PAIRED_PUBLIC_BENCHMARK_REPLAY_MISMATCH")
    queries = {(r.scenario_id, r.language): r for r in bundle.rows}
    files, cases = {}, []
    for scenario in dataset.scenarios:
        rows = [r for r in scored if r["scenario_id"] == scenario.scenario_id]
        chunk_ids = {key for r in rows for key in [*r["ranked_chunk_ids"][:5], *r["context_chunk_ids"]]}
        chunks = {}
        for key in sorted(chunk_ids):
            chunk = index.chunks[key]
            doc = index.documents[chunk.document_version_id]
            chunks[key] = {"title": doc.title, "uri": doc.source_uri, "start_char": chunk.start_char,
                "end_char": chunk.end_char, "preview": chunk.text[:800], "preview_truncated": len(chunk.text) > 800}
        groups = [{"id": g.group_id, "fact_en": g.required_fact, "fact_vi": g.required_fact_vi,
            "alternatives": [{"title": index.documents[a.document_version_id].title,
                "uri": index.documents[a.document_version_id].source_uri, "quote": a.quote,
                "start_char": a.start_char, "end_char": a.end_char} for a in g.alternatives]}
                  for g in scenario.evidence_groups]
        methods = {lang: {r["requested_method"]: {key: r[key] for key in (
            "status", "effective_method", "ranked_chunk_ids", "context_chunk_ids", "context_assertion_ids",
            "used_graph", "warnings", "metrics", "subquery_count", "context_selection_used", "execution_source", "preparation_error")}
            for r in rows if r["language"] == lang} for lang in ("en", "vi")}
        for lang in ("en", "vi"):
            q, qb = (methods[lang][m] for m in ("hybrid_multi_query", "hybrid_multi_query_balanced"))
            if q["status"] == qb["status"] == "COMPLETED" and set(q["ranked_chunk_ids"]) != set(qb["ranked_chunk_ids"]):
                raise ValueError("PAIRED_Q_QB_POOL_MISMATCH")
        detail = {"schema_version": 2, "scenario_id": scenario.scenario_id, "review_status": "AI_DRAFT",
            "question_en": scenario.question_en, "question_vi": scenario.question_vi,
            "split": scenario.split, "answerable": scenario.answerable, "groups": groups,
            "expected_behavior": scenario.expected_behavior, "chunks": chunks, "methods_by_language": methods,
            "queries": {lang: {"track": "EN_ORIGINAL" if lang == "en" else "VI_SHARED_REWRITE",
                "search_query": queries[(scenario.scenario_id, lang)].search_query,
                "subqueries": queries[(scenario.scenario_id, lang)].subqueries,
                "preparation_status": queries[(scenario.scenario_id, lang)].preparation_status} for lang in ("en", "vi")}}
        raw = canonical_json(detail) + b"\n"
        if len(raw) > 256 * 1024:
            raise ValueError("PAIRED_PUBLIC_CASE_TOO_LARGE")
        files[f"benchmark316-cases/{scenario.scenario_id}.json"] = raw
        cases.append({"id": scenario.scenario_id, "question_en": scenario.question_en,
            "question_vi": scenario.question_vi, "split": scenario.split, "kind": scenario.kind,
            "services": scenario.services, "answerable": scenario.answerable, "sha256": sha256_bytes(raw)})
    public = {"schema_version": 3, "scope": "PUBLIC_RECORDED_COLLECTION316_PAIRED_RETRIEVAL_AI_DRAFT",
        "checked_at": manifest["completed_at"], "source_manifest_sha256": collection.manifest_sha256,
        "protocol_sha256": manifest["protocol_sha256"], "receipt_sha256": manifest["receipts_sha256"],
        "documents": 316, "chunks": 2978, "service_groups": 24, "graph_assertions": 17,
        "independent_holdout": False, "human_approved": 0, "answer_quality_measured": False,
        "model_calls": 0, "aws_calls": 0, "shared_preparation_calls": preparation["provider_operations"],
        "shared_preparation_cost_micro_usd": preparation["preparation_cost_micro_usd"],
        "baseline_receipts_reused": 600, "query_tracks": {"en": "EN_ORIGINAL", "vi": "VI_SHARED_REWRITE"},
        "methods": summary, "cases": cases}
    files["benchmark316-summary.json"] = canonical_json(public) + b"\n"
    if len(files["benchmark316-summary.json"]) > 128 * 1024:
        raise ValueError("PAIRED_PUBLIC_SUMMARY_TOO_LARGE")
    # A projection is replaceable; the source run and its older public view are not.
    frontend = ROOT / "frontend/tracer"
    current = json.loads((frontend / "benchmark316-summary.json").read_bytes())
    archive = ROOT / "results/benchmark316/public-projection-v2"
    if current["schema_version"] == 2 and not archive.exists():
        archive.mkdir()
        for name in files:
            old = frontend / name
            target = archive / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(old.read_bytes())
    for name, raw in files.items():
        target = frontend / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    receipt = {"status": "PAIRED_PUBLIC_PROJECTION_REPLAY_VALIDATED", "summary_bytes": len(files["benchmark316-summary.json"]),
        "max_case_bytes": max(len(v) for k, v in files.items() if "cases/" in k),
        "cases": 100, "retrieval_receipts": 1600, "retrieval_model_calls": 0,
        "shared_preparation_operations": preparation["provider_operations"],
        "source_run_manifest_sha256": sha256_bytes((run_dir / "manifest.json").read_bytes()),
        "shared_completion_sha256": sha256_bytes((QUERY_DIR / "completion.json").read_bytes())}
    print(json.dumps(receipt))
    (ROOT / "results/benchmark316/paired-projection-v1.json").write_bytes(canonical_json(receipt) + b"\n")
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/benchmark316/all-eight-paired-v1")
    project(parser.parse_args(argv).run_dir)


if __name__ == "__main__":
    main()
