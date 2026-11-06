"""Offline per-item diagnostics over durable extraction receipts; never publish salvaged edges."""

import json
import sqlite3
from pathlib import Path

from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes
from app.evidence_extraction import EvidenceExtractionOutput
from app.extraction import validate_output

ROOT = Path(__file__).resolve().parents[1]


def main():
    report = json.loads((ROOT / "results/collection-graph/pilot-v1.json").read_bytes())
    bundle = Path(json.loads((ROOT / "results/workbench-cloud/assets-build-v1.json").read_bytes())["bundle_directory"])
    docs, chunks = {}, {}
    for shard in (bundle / "collection").iterdir():
        if shard.is_dir():
            docs.update((d.document_version_id, d) for d in map(DocumentVersion.model_validate_json, (shard / "documents.jsonl").read_text(encoding="utf-8").splitlines()))
            chunks.update((c.chunk_id, c) for c in map(Chunk.model_validate_json, (shard / "chunks.jsonl").read_text(encoding="utf-8").splitlines()))
    plan = json.loads((ROOT / "results/collection-graph/extraction-plan-v1.json").read_bytes())
    from app.contracts import ExtractionRecipe

    recipe = ExtractionRecipe.model_validate_json(canonical_json(plan["extraction_recipe"]))
    result = {"status": "OFFLINE_DIAGNOSIS_NOT_SALVAGE_OR_SEMANTIC_APPROVAL", "aws_calls": 0, "model_calls": 0,
        "pilot_status": report["status"], "whole_chunk_outputs_preserved": True, "items": [], "failed_without_receipt": []}
    canonical_path = Path(report["canonical_path"])
    canonical_rows = [json.loads(line) for line in canonical_path.read_text(encoding="utf-8").splitlines()]
    if [row["chunk_id"] for row in canonical_rows] != [row["chunk_id"] for row in report["extractions"]]:
        raise ValueError("GRAPH_PILOT_CANONICAL_PARTITION_CHANGED")
    for row in canonical_rows:
        c = chunks[row["chunk_id"]]
        d = docs[c.document_version_id]
        for record in row["mentions"]:
            if record["chunk_id"] != c.chunk_id or d.normalized_text[record["start_char"]:record["end_char"]] != record["quote"]:
                raise ValueError("GRAPH_PILOT_CANONICAL_MENTION_MISMATCH")
        for record in row["assertions"]:
            evidence = record["evidence"]
            if record["chunk_id"] != c.chunk_id or d.normalized_text[evidence["start_char"]:evidence["end_char"]] != evidence["quote"]:
                raise ValueError("GRAPH_PILOT_CANONICAL_ASSERTION_MISMATCH")
    result.update(canonical_sha256=sha256_bytes(canonical_path.read_bytes()), canonical_rows=len(canonical_rows),
        canonical_assertions=sum(len(row["assertions"]) for row in canonical_rows), current_source_span_check="PASS",
        managed_financial_reconciliation="PENDING_FRESH_AWS_LOGIN")
    with sqlite3.connect("file:" + str(ROOT / "data/cache/external-models/extraction.sqlite3") + "?mode=ro", uri=True) as db:
        for state, claim_raw, attempts_raw, error in db.execute("SELECT state,claim,receipts,error_category FROM extraction WHERE namespace='LIVE'"):
            claim = json.loads(claim_raw)
            context = claim.get("initial_context", claim.get("context", {}))
            if context.get("run_id") != report["run_id"]:
                continue
            key = context["request_id"].removeprefix("graph316-")
            attempts = json.loads(attempts_raw)
            if not attempts:
                result["failed_without_receipt"].append({"chunk_id": key, "state": state, "error_category": error,
                    "managed_status": "REQUIRES_FRESH_RECONCILIATION"})
                continue
            attempt = attempts[-1]
            output = EvidenceExtractionOutput.model_validate_json(attempt["text"])
            for kind, items in (("relationships", output.relationships), ("standalone_entities", output.standalone_entities)):
                for index, item in enumerate(items):
                    one = {"relationships": [], "standalone_entities": []}
                    one[kind] = [item.model_dump(mode="json")]
                    try:
                        validate_output(json.dumps(one), chunk=chunks[key], document=docs[chunks[key].document_version_id],
                            stop_reason=attempt["stop_reason"], recipe=recipe)
                        diagnosis = "STRICT_ITEM_STRUCTURE_PASS_SEMANTICS_NOT_REVIEWED"
                    except Exception as exc:
                        diagnosis = str(exc) if type(exc).__name__ == "ExtractionError" else type(exc).__name__
                    result["items"].append({"chunk_id": key, "whole_chunk_state": state, "kind": kind,
                        "index": index, "raw_output_sha256": sha256_bytes(attempt["text"].encode()), "diagnosis": diagnosis})
    (ROOT / "results/collection-graph/offline-diagnosis-v1.json").write_bytes(canonical_json(result) + b"\n")
    print(json.dumps({"status": result["status"], "items_inspected": len(result["items"]),
        "strict_item_passes": sum(i["diagnosis"].startswith("STRICT_ITEM") for i in result["items"]),
        "receipt_free_failed_inputs": len(result["failed_without_receipt"])}))


if __name__ == "__main__":
    main()
