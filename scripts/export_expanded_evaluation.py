"""Export public-only experiment receipts and audit scores without AWS or model assets."""

import argparse
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

from app.contracts import canonical_json, sha256_bytes
from app.evaluation import score_retrieval
from scripts.evaluate_development import load_inputs, save
from scripts.evaluate_expanded import ROOT, read_pair, replay, summarize
from scripts.evaluate_graph_components import evaluate as replay_components
from scripts.evaluate_graph_components import summarize as summarize_components

PUBLIC = ROOT / "benchmarks/expanded-development-v1"


def audit(directory):
    manifest = json.loads((directory / "manifest.json").read_bytes())
    datasets, hashes = read_pair()
    raw = (directory / "retrieval.jsonl").read_bytes()
    index_raw = (directory / "source-index.json").read_bytes()
    index = json.loads(index_raw)
    if (hashes != manifest["dataset_sha256"] or sha256_bytes(raw) != manifest["retrieval_sha256"]
            or sha256_bytes(index_raw) != manifest["public_source_index_sha256"]):
        raise ValueError("PUBLIC_AUDIT_INPUT_HASH_MISMATCH")
    # Range-only audit can be run on a fresh checkout without redistributing the
    # source corpus/model. It cannot establish document text integrity or entailment.
    chunks = {c["chunk_id"]: SimpleNamespace(**c) for c in index["chunks"]}
    if len(chunks) != len(index["chunks"]) or any(c.start_char < 0 or c.end_char <= c.start_char for c in chunks.values()):
        raise ValueError("PUBLIC_SOURCE_INDEX_INVALID")
    snapshot = SimpleNamespace(chunks=chunks)
    items = {(lang, item.question_id): item for lang, ds in datasets.items() for item in ds.items}
    rows = [json.loads(line) for line in raw.splitlines()]
    for row in rows:
        item = items[(row["language"], row["question_id"])]
        if row["metrics"] != score_retrieval(item, row["ranked_chunk_ids"], row["context_chunk_ids"], snapshot):
            raise ValueError("PUBLIC_AUDIT_METRIC_MISMATCH")
    calculated = summarize(rows, datasets)
    if (sha256_bytes(canonical_json(calculated)) != manifest["summary_sha256"] or
            canonical_json(calculated) != canonical_json(json.loads((directory / "summary.json").read_bytes()))):
        raise ValueError("PUBLIC_AUDIT_SUMMARY_MISMATCH")
    component_rows = 0
    components = directory / "graph-components"
    if components.exists():
        component_manifest = json.loads((components / "manifest.json").read_bytes())
        component_raw = (components / "retrieval.jsonl").read_bytes()
        if (component_manifest["dataset_sha256"] != hashes or component_manifest["graph_manifest_sha256"] != manifest["graph_manifest_sha256"]
                or component_manifest["retrieval_sha256"] != sha256_bytes(component_raw)
                or component_manifest["prior_control_sha256"] != sha256_bytes(raw)):
            raise ValueError("PUBLIC_COMPONENT_BINDING_INVALID")
        own = [json.loads(line) for line in component_raw.splitlines()]
        for row in own:
            item = items[(row["language"], row["question_id"])]
            if row["metrics"] != score_retrieval(item, row["ranked_chunk_ids"], row["context_chunk_ids"], snapshot):
                raise ValueError("PUBLIC_COMPONENT_METRIC_MISMATCH")
        calculated = summarize_components(own, rows, datasets)
        if (sha256_bytes(canonical_json(calculated)) != component_manifest["summary_sha256"] or
                canonical_json(calculated) != canonical_json(json.loads((components / "summary.json").read_bytes()))):
            raise ValueError("PUBLIC_COMPONENT_SUMMARY_MISMATCH")
        component_rows = len(own)
    answer_rows = 0
    answer_manifest_path = directory / "answer-manifest.json"
    if answer_manifest_path.exists():
        answer_manifest = json.loads(answer_manifest_path.read_bytes())
        answer_raw = (directory / "answers.jsonl").read_bytes()
        report_raw = (directory / "answer-summary.json").read_bytes()
        if (sha256_bytes(answer_raw) != answer_manifest["answers_sha256"] or
                sha256_bytes(report_raw) != answer_manifest["report_sha256"]):
            raise ValueError("PUBLIC_ANSWER_HASH_MISMATCH")
        answers = [json.loads(line) for line in answer_raw.splitlines()]
        report = json.loads(report_raw)
        expected = {(lang, item.question_id) for lang, ds in datasets.items() for item in ds.items}
        actual = [(r["language"], r["question_id"]) for r in answers]
        if len(set(actual)) != len(actual) or not set(actual).issubset(expected) or len(answers) != report["attempted"]:
            raise ValueError("PUBLIC_ANSWER_ROW_SET_INVALID")
        for lang in ("en", "vi"):
            own = [r for r in answers if r["language"] == lang]
            counts = Counter(r["structural_status"] for r in own)
            if (report["languages"][lang]["structural_pass"] != counts["PASS"] or
                    report["languages"][lang]["structural_fail"] != counts["FAIL"] or
                    report["languages"][lang]["attempted"] != len(own)):
                raise ValueError("PUBLIC_ANSWER_SUMMARY_INVALID")
        if sum(r["cost_upper_bound_micro_usd"] for r in answers) != report["settled_published_peak_cost_upper_bound_micro_usd"]:
            raise ValueError("PUBLIC_ANSWER_COST_SUMMARY_INVALID")
        answer_rows = len(answers)
    return {"status": "PASS_RANGE_AUDIT", "rows": len(rows), "component_rows": component_rows,
        "answer_rows": answer_rows, "model_api_calls": 0, "aws_calls": 0,
        "scope": "Stored source ranges and metrics only; original text integrity requires the private frozen corpus; semantic labels require human review."}


def export(directory):
    datasets, hashes = read_pair(directory)
    _, graph = load_inputs(datasets["en"])
    replay(directory, datasets, hashes, graph)
    PUBLIC.mkdir(parents=True, exist_ok=True)
    source_index = {"graph_manifest_sha256": graph.manifest_sha256,
        "documents": [{"document_version_id": doc.document_version_id, "title": doc.title,
                       "source_uri": doc.source_uri, "normalized_text_sha256": sha256_bytes(doc.normalized_text.encode()),
                       "raw_sha256": doc.raw_sha256, "fetched_at": doc.fetched_at.isoformat(),
                       "normalizer_version": doc.normalizer_version,
                       "normalized_characters": len(doc.normalized_text)} for doc in graph.content.documents],
        "chunks": [{"chunk_id": c.chunk_id, "document_version_id": c.document_version_id,
                    "start_char": c.start_char, "end_char": c.end_char} for c in graph.content.chunks]}
    save(PUBLIC / "source-index.json", source_index)
    for name in ("retrieval.jsonl", "summary.json"):
        (PUBLIC / name).write_bytes((directory / name).read_bytes())
    manifest = json.loads((directory / "manifest.json").read_bytes())
    manifest["public_source_index_sha256"] = sha256_bytes((PUBLIC / "source-index.json").read_bytes())
    manifest["public_projection"] = "No credentials, private account records, source documents or model weights. Recorded public-doc question IDs and retrieval ranges."
    save(PUBLIC / "manifest.json", manifest)
    components = directory / "graph-components-v1"
    if components.exists():
        replay_components(directory, replay_only=True)
        target = PUBLIC / "graph-components"
        target.mkdir(exist_ok=True)
        for name in ("manifest.json", "summary.json", "retrieval.jsonl"):
            (target / name).write_bytes((components / name).read_bytes())
    answers = directory / "answers-v1"
    report_path = answers / "report.json"
    if report_path.exists():
        from scripts.evaluate_expanded_answers import evaluate

        evaluate(directory)  # Revalidate saved receipts, no provider/ledger call.
        report = json.loads(report_path.read_bytes())
        projected = []
        for row in report["cases"]:
            scored = json.loads((answers / ("expanded-v1-" + row["question_id"] + "-scored.json")).read_bytes())
            for citation in scored.get("response", {}).get("citations", []):
                if not citation["source_uri"].startswith("https://docs.aws.amazon.com/"):
                    raise ValueError("ANSWER_EXPORT_NONPUBLIC_SOURCE")
            projected.append(scored)
        raw = b"".join(canonical_json(row) + b"\n" for row in projected)
        (PUBLIC / "answers.jsonl").write_bytes(raw)
        save(PUBLIC / "answer-summary.json", {
            **{k: v for k, v in report.items() if k not in {"cases", "new_model_operations"}},
            "replay_new_model_operations": report["new_model_operations"],
            "recorded_generation_receipts": len(projected),
        })
        save(PUBLIC / "answer-manifest.json", {
            "plan": json.loads((answers / "plan.json").read_bytes()),
            "answers_sha256": sha256_bytes(raw), "report_sha256": sha256_bytes((PUBLIC / "answer-summary.json").read_bytes()),
            "scope": "Public documentation claims/citations only; no prompts, keys or private user history. Semantic review pending."})
    return audit(PUBLIC)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export", type=Path)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if bool(args.export) == bool(args.audit):
        raise ValueError("CHOOSE_EXPORT_OR_AUDIT")
    print(json.dumps(export(args.export.resolve()) if args.export else audit(PUBLIC)))


if __name__ == "__main__":
    main()
