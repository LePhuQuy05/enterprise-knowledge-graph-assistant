"""Read-only ledger/source reconciliation and pending semantic review queue; never dispatch inference."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from pydantic import TypeAdapter

from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes
from app.corpus import load_corpus
from app.extraction import raw_cache_key
from app.extraction_cache import ExtractionAttempt
from app.ledger import DynamoLedgerStore, Ledger
from app.snapshot import load_snapshot
from scripts.run_bge_extraction_pilot import RUN

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan = json.loads((ROOT / "data/experiments/bge-extraction-pilot-v3/plan.json").read_bytes())
    baseline = json.loads((ROOT / "results/bge_baseline.json").read_bytes())
    pilot = json.loads((ROOT / "results/bge_extraction_pilot_continued.json").read_bytes())
    corpus = load_corpus(Path(baseline["corpus_directory"]), tenant_id="demo")
    recipe = ExtractionRecipe.model_validate_json(json.dumps(plan["extraction_recipe"]))
    chunks = {c.chunk_id: c for c in corpus.chunks}
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=10)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    records = []
    with sqlite3.connect(ROOT / "data/cache/external-models/extraction.sqlite3") as db:
        for selected in plan["selection"]:
            chunk = chunks[selected["chunk_id"]]
            row = db.execute("SELECT state,receipts FROM extraction WHERE namespace='LIVE' AND cache_key=?",
                             (raw_cache_key("demo", chunk.text, recipe),)).fetchone()
            if row is None or row[0] not in {"READY", "FAILED"} or len(row[1]) > 512 * 1024:
                raise ValueError("Extraction outcome is absent, in progress or oversized")
            attempts = TypeAdapter(list[ExtractionAttempt]).validate_json(row[1])
            for attempt in attempts:
                op = ledger.operation(attempt.operation_id)
                if (op.status != "SETTLED" or op.spec.run_id != RUN or op.spec.provider != "DEEPSEEK"
                        or op.spec.request_id != "extract-bge-v3-" + chunk.chunk_id
                        or op.usage != attempt.usage or op.actual_micro_usd != attempt.original_cost_micro_usd):
                    raise ValueError("Recorded output and managed settlement differ")
                records.append({"operation_id": op.spec.operation_id, "status": op.status, "chunk_id": chunk.chunk_id,
                                "raw_output_sha256": sha256_bytes(attempt.text.encode()),
                                "budget_cost_micro_usd": op.actual_micro_usd, "usage": op.usage.model_dump()})
    if len(records) != len({r["operation_id"] for r in records}):
        raise ValueError("Duplicate operation in reconciliation")
    funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
    prior = json.loads((ROOT / "results/external_pilot10_v2.json").read_bytes())["funding_counter"]
    total = sum(r["budget_cost_micro_usd"] for r in records)
    if funding["settled_micro_usd"] - prior["settled_micro_usd"] != total or funding["reserved_micro_usd"] != prior["reserved_micro_usd"]:
        raise ValueError("Funding delta/reservation changed outside reconciled operations")
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "PASS", "paid_inference_calls_this_command": 0,
              "reconciled_dispatches": len(records), "all_reconciled_calls_settled": True,
              "new_v3_total_budget_cost_micro_usd": total, "provider_invoice_micro_usd": None,
              "cost_basis": "PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND", "operations": records,
              "funding_counter": funding, "accepted_chunks": len(pilot["extractions"]),
              "quarantined_chunks": len(pilot["quarantined"]), "old_unknown_reservation_preserved": True,
              "semantic_review": "PENDING", "note": "Accepted-row progress cost excludes rejected calls; this total includes all v3 attempts."}
    (ROOT / "results/extraction_v3_reconciliation.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    receipt = json.loads((ROOT / "results/pilot_graph_snapshot.json").read_bytes())
    snapshot = load_snapshot(Path(receipt["snapshot_directory"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=receipt["manifest_sha256"])
    review = []
    for a in snapshot.content.assertions:
        subject, obj = snapshot.entities[a.subject_id], snapshot.entities[a.object_id]
        review.append({"assertion_id": a.assertion_id, "review_status": "PENDING", "reviewer": None,
            "subject": subject.model_dump(mode="json"), "predicate": a.predicate, "object": obj.model_dump(mode="json"),
            "source_uri": snapshot.documents[a.document_version_id].source_uri,
            "document_version_id": a.document_version_id, "chunk_id": a.chunk_id,
            "evidence_start_char": a.evidence.start_char, "evidence_end_char": a.evidence.end_char,
            "evidence_quote_sha256": sha256_bytes(a.evidence.quote.encode()),
            "polarity": a.polarity, "qualifiers": [q.model_dump(mode="json") for q in a.qualifiers],
            "checks_needed": ["source_entails_predicate_and_direction", "entity_type_and_identity",
                              "all_conditions_preserved", "example_vs_universal_scope", "negation"]})
    target = ROOT / "results/pilot_graph_semantic_review_queue.json"
    if target.exists() and target.read_bytes() != canonical_json(review):
        raise ValueError("Existing review queue differs; preserve reviewer decisions")
    if not target.exists():
        target.write_bytes(canonical_json(review))
    print(json.dumps({k: report[k] for k in ("status", "reconciled_dispatches", "new_v3_total_budget_cost_micro_usd",
                                            "old_unknown_reservation_preserved", "semantic_review")}, indent=2))


if __name__ == "__main__":
    main()
