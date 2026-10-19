"""Frozen ten-chunk DeepSeek v3 pilot; BGE cache-only, shared existing allowance."""

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from pydantic import TypeAdapter

from app.contracts import canonical_json, sha256_bytes, text_hash
from app.corpus import load_corpus
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, load_private_key
from app.extraction import (
    AliasDictionary,
    ExtractionError,
    deepseek_evidence_recipe,
    raw_cache_key,
    validate_output,
)
from app.extraction_cache import ExtractionAttempt, ExtractionCache, ExtractionCacheBlocked
from app.ledger import DynamoLedgerStore, Ledger
from app.model_gateway import CallContext, GatewayError, UncertainModelCall, safe_failure_code
from app.pilot import select_chunks

ROOT = Path(__file__).resolve().parents[1]
RUN = "external-pilot10-r72-v1"  # Reuse, never initialize/reset an allowance.


def settled_rejection(cache, chunk, document, recipe, ledger):
    """A rejected source may be skipped, never retried; all paid attempts must be settled."""
    key = raw_cache_key("demo", chunk.text, recipe)
    with sqlite3.connect(cache.path) as db:
        row = db.execute("SELECT state,claim,length(receipts) FROM extraction WHERE namespace=? AND cache_key=?",
                         (cache.namespace, key)).fetchone()
        if row is None or row[0] != "FAILED" or not 1 <= row[2] <= 512 * 1024:
            raise GatewayError("Only durable failed rows with bounded receipts can be quarantined")
        raw = db.execute("SELECT receipts FROM extraction WHERE namespace=? AND cache_key=?", (cache.namespace, key)).fetchone()[0]
    claim = json.loads(row[1])
    if claim.get("input_sha256") != text_hash(chunk.text):
        raise GatewayError("Rejected source identity changed")
    attempts = TypeAdapter(list[ExtractionAttempt]).validate_json(raw)
    if not 1 <= len(attempts) <= 2 or [a.attempt for a in attempts] != list(range(1, len(attempts) + 1)):
        raise GatewayError("Rejected attempts require reconciliation")
    if ledger is None:
        raise GatewayError("Managed settlement verification required for rejected source")
    for attempt in attempts:
        operation = ledger.operation(attempt.operation_id)
        if (operation.status != "SETTLED" or operation.spec.run_id != RUN
                or operation.spec.request_id != "extract-bge-v3-" + chunk.chunk_id
                or operation.spec.attempt != attempt.attempt or operation.usage != attempt.usage
                or operation.actual_micro_usd != attempt.original_cost_micro_usd):
            raise GatewayError("Rejected call is not fully settled; no continuation")
    try:
        last = attempts[-1]
        validate_output(last.text, chunk=chunk, document=document, stop_reason=last.stop_reason, recipe=recipe)
    except ExtractionError as exc:
        category = str(exc)
    else:
        raise GatewayError("Failed cache row contains valid output; manual reconciliation required")
    return {"chunk_id": chunk.chunk_id, "status": "QUARANTINED_NO_RETRY", "validation_error": category,
            "operation_ids": [a.operation_id for a in attempts],
            "settled_attempts": len(attempts), "paid_retry_calls": 0}


def frozen_plan(corpus, aliases, routes):
    source_manifest = json.loads((ROOT / "configs/corpus/aws-r72-v1.json").read_bytes())
    recipe = deepseek_evidence_recipe(aliases, metadata_sha256=routes.account_model_metadata_sha256)
    plan = {"version": "bge-extraction-pilot-v3", "corpus_version": corpus.version,
            "selection": select_chunks(corpus, source_manifest), "run_id": RUN,
            "run_cap_micro_usd": 500_000, "funding_cap_micro_usd": 2_000_000,
            "allow_one_settled_format_repair": True, "extraction_recipe": recipe.model_dump(mode="json"),
            "embedding_calls": 0, "semantic_review": "PENDING", "max_provider_dispatches": 20}
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return plan, recipe


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    parser.add_argument("--continue-after-settled-rejection", action="store_true",
                        help="Verify managed settlements, skip failed chunks without retry, continue frozen plan")
    args = parser.parse_args()
    if args.continue_after_settled_rejection and not args.execute:
        parser.error("Continuation requires --execute for managed settlement reads")
    filename = ("bge_extraction_pilot_continued.json" if args.continue_after_settled_rejection else
                "bge_extraction_pilot_replay.json" if args.replay else "bge_extraction_pilot.json")
    output = ROOT / "results" / filename
    if args.execute and output.exists():
        print("Existing live receipt retained; no automatic paid rerun.")
        return 1
    baseline = json.loads((ROOT / "results/bge_baseline.json").read_bytes())
    corpus = load_corpus(Path(baseline["corpus_directory"]), tenant_id="demo")
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    plan, recipe = frozen_plan(corpus, aliases, routes)
    artifacts = ROOT / "data/experiments/bge-extraction-pilot-v3"
    artifacts.mkdir(parents=True, exist_ok=True)
    plan_file = artifacts / "plan.json"
    if plan_file.exists() and plan_file.read_bytes() != canonical_json(plan):
        raise GatewayError("Frozen plan changed; inspect without dispatch")
    if not plan_file.exists():
        plan_file.write_bytes(canonical_json(plan))
    if not args.execute and not args.replay:
        print(json.dumps({"status": "PLANNED_NO_CALLS", "chunks": len(plan["selection"]),
                          "plan_sha256": plan["plan_sha256"], "embedding_calls": 0}, indent=2))
        return 0
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "RUNNING", "run_id": RUN,
              "plan_sha256": plan["plan_sha256"], "corpus_version": corpus.version,
              "extraction_recipe": recipe.model_dump(mode="json"), "extractions": [], "quarantined": [],
              "semantic_review": "PENDING", "embedding_calls": 0, "cloud_deployments": 0,
              "provider_invoice_micro_usd": None, "artifacts_directory": str(artifacts)}
    selection = plan["selection"]
    if args.replay:
        prior = json.loads((ROOT / "results/bge_extraction_pilot_continued.json").read_bytes())
        accepted = {r["chunk_id"] for r in prior["extractions"]}
        quarantined = {r["chunk_id"] for r in prior["quarantined"]}
        if (prior["plan_sha256"] != plan["plan_sha256"] or accepted & quarantined
                or accepted | quarantined != {r["chunk_id"] for r in selection}):
            raise GatewayError("Completed frozen pilot partition required for offline replay")
        selection = [r for r in selection if r["chunk_id"] in accepted]
        report["quarantined"] = prior["quarantined"]
        report["quarantine_settlement_evidence"] = "RECORDED_PRIOR_RUN_NOT_FRESH_MANAGED_READ"
    transport, ledger = None, None
    gateway = None

    def checkpoint():
        report["inference_dispatch_attempts_this_invocation"] = transport.dispatch_attempts if transport else 0
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    checkpoint()  # Durable experiment intent before any dispatch.
    try:
        if args.execute:
            session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
            cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=10)
            if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
                raise GatewayError("Known developer identity required")
            ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
            counter = ledger.store.get(ledger.run_key(RUN))
            funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
            if not counter or counter["cap_micro_usd"] != 500_000 or counter["halted"]:
                raise GatewayError("Existing pilot allowance required and must not be halted")
            if not funding or funding["cap_micro_usd"] != 2_000_000 or funding["halted"]:
                raise GatewayError("Existing external funding required and must not be halted")
            report.update(initial_run_counter=counter, initial_funding_counter=funding)
            prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes())
            transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")})
            gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={
                "live_inference_enabled": True, "live_inference_gate": "EXPLICIT_FROZEN_BGE_EXTRACTION_V3_ONLY",
            }), prices)
        cache = ExtractionCache(ROOT / "data/cache/external-models", namespace="LIVE")
        docs = {d.document_version_id: d for d in corpus.documents}
        chunks = {c.chunk_id: c for c in corpus.chunks}
        canonical_rows = []
        for row in selection:
            chunk = chunks[row["chunk_id"]]
            try:
                result = cache.resolve(CallContext(tenant_id="demo", run_id=RUN,
                    request_id="extract-bge-v3-" + chunk.chunk_id, stage="EXTRACT"), chunk=chunk,
                    document=docs[chunk.document_version_id], recipe=recipe, aliases=aliases, gateway=gateway,
                    allow_repair=plan["allow_one_settled_format_repair"])
            except (ExtractionError, ExtractionCacheBlocked):
                if not args.continue_after_settled_rejection:
                    raise
                rejected = settled_rejection(cache, chunk, docs[chunk.document_version_id], recipe, ledger)
                report["quarantined"].append(rejected)
                checkpoint()
                print(json.dumps({"quarantined": len(report["quarantined"]), "error": rejected["validation_error"]}), flush=True)
                continue
            canonical = {name: [r.model_dump(mode="json") for r in getattr(result.canonical, name)]
                         for name in ("entities", "mentions", "assertions")}
            canonical.update(chunk_id=chunk.chunk_id, review_entity_ids=result.canonical.review_entity_ids,
                             semantic_review="PENDING")
            canonical_rows.append(canonical)
            target = artifacts / ("canonical_replay.jsonl" if args.replay else "canonical.jsonl")
            target.write_bytes(b"".join(canonical_json(r) + b"\n" for r in canonical_rows))
            report["extractions"].append({"chunk_id": chunk.chunk_id, "cache_hit": result.cache_hit,
                "entities": len(result.canonical.entities), "assertions": len(result.canonical.assertions),
                "operation_ids": [a.operation_id for a in result.record.attempts],
                "attempts": len(result.record.attempts), "new_budget_cost_micro_usd": result.new_cost_micro_usd})
            checkpoint()
            print(json.dumps({"completed": len(report["extractions"]), "assertions": len(result.canonical.assertions),
                              "cache_hit": result.cache_hit}), flush=True)
        report.update(status="PARTIAL_ACCEPTED_WITH_QUARANTINE" if report["quarantined"] else "STRUCTURAL_PASS_SEMANTIC_PENDING",
                      completed_at=datetime.now(UTC).isoformat(),
                      total_assertions=sum(r["assertions"] for r in report["extractions"]),
                      new_budget_cost_micro_usd=sum(r["new_budget_cost_micro_usd"] for r in report["extractions"]),
                      canonical_sha256=sha256_bytes(target.read_bytes()))
        if args.replay and report["canonical_sha256"] != prior["canonical_sha256"]:
            raise GatewayError("Offline accepted canonical replay differs")
    except Exception as exc:
        report.update(status="STOPPED", error_category=safe_failure_code(exc))
        if isinstance(exc, UncertainModelCall):
            report.update(status="UNKNOWN_STOPPED", operation_id=exc.operation_id)
    finally:
        if ledger is not None:
            try:
                report["run_counter"] = ledger.store.get(ledger.run_key(RUN))
                report["funding_counter"] = ledger.store.get(ledger.external_funding_key("kg-r72"))
                report["invocation_funding_settled_delta_micro_usd"] = (
                    report["funding_counter"]["settled_micro_usd"] - report["initial_funding_counter"]["settled_micro_usd"]
                )
                report["accepted_row_cost_excludes_rejected_attempts"] = True
            except Exception as exc:
                report["counter_read_error_category"] = type(exc).__name__
        checkpoint()
    print(json.dumps({k: report.get(k) for k in ("status", "error_category", "total_assertions",
        "new_budget_cost_micro_usd", "inference_dispatch_attempts_this_invocation")}, indent=2))
    return 0 if report["status"] in {"STRUCTURAL_PASS_SEMANTIC_PENDING", "PARTIAL_ACCEPTED_WITH_QUARANTINE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
