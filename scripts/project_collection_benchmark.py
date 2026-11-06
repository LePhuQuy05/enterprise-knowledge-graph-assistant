"""Project only curated public questions/sources and validated offline scores."""

import argparse
import json
from pathlib import Path

from app.collection_benchmark import RetrievalReceipt, score_receipts, summarize_receipts
from app.contracts import canonical_json, sha256_bytes
from scripts.run_collection_benchmark import ALL_METHODS, ROOT, load_inputs


def project(run_dir, output):
    protocol_raw, dataset, collection, index, _ = load_inputs()
    manifest = json.loads((run_dir / "manifest.json").read_bytes())
    if manifest["protocol_sha256"] != sha256_bytes(protocol_raw) or manifest["languages"] != ["en"]:
        raise ValueError("PUBLIC_BENCHMARK_PROTOCOL_OR_TRACK_MISMATCH")
    if set(manifest["scenario_ids"]) != {s.scenario_id for s in dataset.scenarios}:
        raise ValueError("PUBLIC_BENCHMARK_REQUIRES_ALL_100")
    for name, expected in manifest["artifacts"].items():
        if Path(name).name != name or sha256_bytes((run_dir / name).read_bytes()) != expected:
            raise ValueError("PUBLIC_BENCHMARK_ARTIFACT_CHANGED")
    receipts = [RetrievalReceipt.model_validate_json(line) for line in (run_dir / "receipts.jsonl").read_bytes().splitlines()]
    scored = score_receipts(dataset, receipts, index)
    summary = summarize_receipts(dataset, scored, languages=("en",), methods=ALL_METHODS)
    if canonical_json(summary) != canonical_json(json.loads((run_dir / "summary.json").read_bytes())["methods"]):
        raise ValueError("PUBLIC_BENCHMARK_SCORE_REPLAY_MISMATCH")
    files, case_index = {}, []
    for scenario in dataset.scenarios:
        rows = [r for r in scored if r["scenario_id"] == scenario.scenario_id]
        chunk_ids = {key for r in rows for key in [*r["ranked_chunk_ids"][:5], *r["context_chunk_ids"]]}
        chunks = {}
        for key in sorted(chunk_ids):
            chunk = index.chunks[key]
            doc = index.documents[chunk.document_version_id]
            chunks[key] = {"title": doc.title, "uri": doc.source_uri, "start_char": chunk.start_char,
                "end_char": chunk.end_char, "preview": chunk.text[:800], "preview_truncated": len(chunk.text) > 800}
        groups = []
        for group in scenario.evidence_groups:
            groups.append({"id": group.group_id, "fact_en": group.required_fact, "fact_vi": group.required_fact_vi,
                "alternatives": [{"title": index.documents[a.document_version_id].title,
                    "uri": index.documents[a.document_version_id].source_uri, "quote": a.quote,
                    "start_char": a.start_char, "end_char": a.end_char} for a in group.alternatives]})
        detail = {"schema_version": 1, "scenario_id": scenario.scenario_id, "review_status": "AI_DRAFT",
            "question_en": scenario.question_en, "question_vi": scenario.question_vi, "split": scenario.split,
            "answerable": scenario.answerable, "groups": groups, "expected_behavior": scenario.expected_behavior,
            "chunks": chunks, "methods": {r["requested_method"]: {key: r[key] for key in (
                "status", "effective_method", "ranked_chunk_ids", "context_chunk_ids", "context_assertion_ids",
                "used_graph", "warnings", "metrics")} for r in rows}}
        raw = canonical_json(detail) + b"\n"
        if len(raw) > 256 * 1024:
            raise ValueError("BENCHMARK_CASE_ASSET_LIMIT")
        files[f"benchmark316-cases/{scenario.scenario_id}.json"] = raw
        case_index.append({"id": scenario.scenario_id, "question_en": scenario.question_en,
            "question_vi": scenario.question_vi, "split": scenario.split, "kind": scenario.kind,
            "services": scenario.services, "answerable": scenario.answerable, "sha256": sha256_bytes(raw)})
    configuration = json.loads((run_dir / "configuration.json").read_bytes())
    public = {"schema_version": 2, "scope": "PUBLIC_RECORDED_COLLECTION316_RETRIEVAL_AI_DRAFT",
        "checked_at": manifest["completed_at"], "source_manifest_sha256": collection.manifest_sha256,
        "protocol_sha256": manifest["protocol_sha256"], "receipt_sha256": manifest["receipts_sha256"],
        "documents": 316, "chunks": 2978, "service_groups": 24,
        "graph_assertions": configuration["graph_assertions"], "independent_holdout": False,
        "human_approved": 0, "answer_quality_measured": False, "model_calls": 0, "aws_calls": 0,
        "methods": summary, "cases": case_index}
    files["benchmark316-summary.json"] = canonical_json(public) + b"\n"
    if len(files["benchmark316-summary.json"]) > 128 * 1024:
        raise ValueError("BENCHMARK_SUMMARY_ASSET_LIMIT")
    output = output.resolve()
    if not output.is_relative_to(ROOT / "frontend/tracer"):
        raise ValueError("BENCHMARK_PUBLIC_OUTPUT_MUST_STAY_IN_FRONTEND")
    for name, raw in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(raw)
    print(json.dumps({"status": "PUBLIC_PROJECTION_REPLAY_VALIDATED", "files": len(files),
        "max_case_bytes": max(len(v) for k, v in files.items() if "cases/" in k),
        "summary_bytes": len(files["benchmark316-summary.json"]), "model_calls": 0, "aws_calls": 0}))
    return public


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=ROOT / "results/benchmark316/core-six-en-v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "frontend/tracer")
    args = parser.parse_args(argv)
    project(args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
