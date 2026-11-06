"""Freeze source-only context queues for missing graph actions and permission evidence."""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.source_context import build_source_window, classify_source, validate_source_window
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

CUES = {
    "READ_RESOURCE": r"\b(?:reads?|reading)\b.{0,80}\b(?:objects?|bucket|Amazon S3)\b|\b(?:objects?|bucket)\b.{0,80}\b(?:reads?|reading)\b",
    "ASSUME_ROLE": r"\bassumes?\b.{0,65}\brole\b",
    "INVOKE_PERMISSION": r"(?:Amazon S3|resource.based policy).{0,100}(?:permission|invoke)|execution role.{0,100}permission",
    "KMS_AUTHORIZATION": r"kms:Decrypt|cross.account.{0,100}(?:permission|key policy)|(?:key policy|IAM policy).{0,100}cross.account",
}


def source_priority(cue, flags, text, *, repeated_document=False):
    diagnostic = any(f in flags for f in ("DIAGNOSTIC_REVIEW_REQUIRED", "CONTAINS_CODE_REVIEW_REQUIRED", "CONTAINS_TABLE_REVIEW_REQUIRED"))
    explicit = cue == "ASSUME_ROLE" and bool(re.search(
        r"\b(?:Lambda|Amazon S3|EventBridge|Amazon ECS|CloudTrail)\s+(?:automatically\s+)?assumes?\b", text, re.I))
    return diagnostic, not explicit if cue == "ASSUME_ROLE" else False, repeated_document


def prepare(output):
    collection, chunks, units = load_collection_inputs()
    documents = {}
    for ref in collection.manifest.shards:
        documents.update(collection.get_shard(ref.shard_id).documents)
    candidates, inventory_flags = defaultdict(list), Counter()
    for unit in units:
        occurrence = unit["occurrences"][0]
        chunk = chunks[occurrence["chunk_id"]]
        document = documents[chunk.document_version_id]
        flags = classify_source(chunk.text, document.title)
        inventory_flags.update(flags)
        for cue, pattern in CUES.items():
            if re.search(pattern, chunk.text, re.I | re.S):
                candidates[cue].append((flags, unit, chunk))
    selected, seen = [], set()
    for cue in CUES:
        # Prefer prose over tables/errors, then different docs, then source hash.
        used_docs = set()
        ordered = sorted(candidates[cue], key=lambda row: (*source_priority(cue, row[0], row[2].text), row[1]["text_sha256"]))
        for _ in range(4):
            available = [r for r in ordered if r[1]["text_sha256"] not in seen]
            if not available:
                break
            flags, unit, chunk = min(available, key=lambda r: (*source_priority(cue, r[0], r[2].text,
                repeated_document=r[2].document_version_id in used_docs), r[1]["text_sha256"]))
            used_docs.add(chunk.document_version_id)
            seen.add(unit["text_sha256"])
            selected.append((cue, chunk, "SOURCE_CUE_SELECTED_NOT_QUESTION_OR_BENCHMARK_SELECTED"))
    review_path = ROOT / "configs/extraction/source-syntax-source-review-v1.json"
    review = json.loads(review_path.read_bytes())
    for judgment in review["reviews"]:
        if judgment["status"] == "AI_SOURCE_SUPPORTED_SOURCE_SCOPED":
            continue
        unit = next(u for u in units if u["text_sha256"] == judgment["source_sha256"])
        chunk = chunks[unit["occurrences"][0]["chunk_id"]]
        selected.append(("KNOWN_QUARANTINE_CONTEXT_REVIEW", chunk, judgment["status"]))
    windows = []
    for cue, chunk, reason in selected:
        document = documents[chunk.document_version_id]
        window = build_source_window(document, chunk, chunks, collection.manifest_sha256)
        validate_source_window(window, document, chunks, collection_manifest_sha256=collection.manifest_sha256)
        windows.append({"cue": cue, "selection_reason": reason, "window": window.model_dump(mode="json")})
    output.mkdir(parents=True, exist_ok=False)
    raw = b"".join(canonical_json(row) + b"\n" for row in windows)
    (output / "windows.jsonl").write_bytes(raw)
    report = {"version": "graph-source-context-queue-v1", "status": "EXACT_SOURCE_WINDOWS_READY_FOR_REVIEW_NOT_EXTRACTED_OR_SERVING",
        "source_manifest_sha256": collection.manifest_sha256, "inventory_unique_texts": len(units),
        "cue_candidate_counts": {key: len(rows) for key, rows in candidates.items()},
        "inventory_flags": dict(inventory_flags), "queued_windows": len(windows),
        "selection_counts": dict(Counter(row["cue"] for row in windows)),
        "unique_window_ids": len({row["window"]["window_id"] for row in windows}),
        "windows_sha256": sha256_bytes(raw), "review_config_sha256": sha256_bytes(review_path.read_bytes()),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in ("app/source_context.py", "scripts/prepare_graph_source_context.py")},
        "new_model_calls": 0, "aws_calls": 0, "new_assertions": 0, "human_reviewed": 0,
        "policy": "WINDOWS_ARE_REVIEW_INPUTS_NOT_FABRICATED_COLLECTION_CHUNKS_OR_EVIDENCE_OUTSIDE_PRIMARY_CHUNK"}
    report["selection_policy"] = "SOURCE_CUES_PROSE_FIRST_EXPLICIT_ASSUME_SUBJECT_DOC_DIVERSITY_HASH_TIEBREAK_V2_NO_QUESTION_LABELS"
    (output / "report.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    prepare(parser.parse_args().output_dir)
