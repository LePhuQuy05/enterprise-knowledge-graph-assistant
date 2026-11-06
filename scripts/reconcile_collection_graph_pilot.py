"""Read-only receipt/source/managed-ledger audit before continuing untouched graph inputs."""

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config

from app.collections import load_collection
from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes, stable_id
from app.external_gateway import ExternalPriceSheet
from app.extraction import (
    AliasDictionary,
    canonicalize,
    extraction_request,
    raw_cache_key,
    validate_output,
)
from app.extraction_cache import ExtractionAttempt
from app.ledger import DynamoLedgerStore, Ledger, Operation
from app.model_gateway import CallContext
from scripts.run_collection_graph_pilot import ROOT, RUN


def require(condition, code):
    if not condition:
        raise ValueError(code)


def load_inputs():
    raw = (ROOT / "results/collection-graph/extraction-plan-v1.json").read_bytes()
    plan = json.loads(raw)
    require(raw == (ROOT / f"data/experiments/{RUN}/plan.json").read_bytes(), "GRAPH_FROZEN_PLAN_CHANGED")
    require(len(plan["selection"]) == plan["max_dispatches"] == 48 and not plan["format_repair"], "GRAPH_PILOT_SCOPE")
    pipeline = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=plan["collection_manifest_sha256"])
    docs, chunks = {}, {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        docs.update(shard.documents)
        chunks.update(shard.chunks)
    for selected in plan["selection"]:
        c = chunks[selected["chunk_id"]]
        require(c.text_sha256 == selected["text_sha256"] and c.document_version_id == selected["document_version_id"]
            and docs[c.document_version_id].source_uri == selected["source_uri"], "GRAPH_CURRENT_SOURCE_CHANGED")
    return raw, plan, docs, chunks


def read_claims(*, run_id=RUN, request_prefix="graph316-"):
    result = {}
    path = ROOT / "data/cache/external-models/extraction.sqlite3"
    with sqlite3.connect("file:" + str(path) + "?mode=ro", uri=True) as db:
        for key, state, claim_raw, receipts_raw, error in db.execute(
            "SELECT cache_key,state,claim,receipts,error_category FROM extraction WHERE namespace='LIVE'"):
            claim = json.loads(claim_raw)
            context = claim.get("initial_context", claim.get("context", {}))
            if context.get("run_id") != run_id:
                continue
            context = CallContext.model_validate(context)
            chunk_id = context.request_id.removeprefix(request_prefix)
            require(chunk_id not in result and context.request_id == request_prefix + chunk_id
                and context.stage == "EXTRACT" and context.attempt == 1 and context.tenant_id == "demo", "GRAPH_CACHE_CONTEXT")
            result[chunk_id] = {"cache_key": key, "state": state, "context": context,
                "receipts": [ExtractionAttempt.model_validate_json(canonical_json(a)) for a in json.loads(receipts_raw)],
                "error_category": error}
    return result


def verify_operation(operation, context, receipt, input_sha256, prices, *, run_id=RUN):
    require(operation.status == "SETTLED" and operation.call_started, "GRAPH_OPERATION_NOT_SETTLED")
    require(operation.operation_id == receipt.operation_id == stable_id("op", **context.model_dump())
        and operation.spec.input_sha256 == input_sha256 and operation.spec.stage == "EXTRACT"
        and operation.spec.run_id == run_id and operation.spec.tenant_id == "demo", "GRAPH_OPERATION_BINDING")
    require(operation.usage == receipt.usage and operation.actual_micro_usd == receipt.original_cost_micro_usd,
        "GRAPH_RECEIPT_USAGE_COST_MISMATCH")
    rate = prices.rate_for(operation.spec.model_id, operation.spec.invocation_id,
        operation.spec.source_region, now=operation.created_at, live=True)
    require(operation.spec.price_sheet_version == prices.version and rate.quote(receipt.usage) == receipt.original_cost_micro_usd,
        "GRAPH_RECEIPT_RATE_MISMATCH")


def audit(ledger, *, loaded=None, include_continuation=False):
    raw, plan, docs, chunks = loaded or load_inputs()
    previous_raw = (ROOT / "results/collection-graph/pilot-v1.json").read_bytes()
    previous = json.loads(previous_raw)
    reports = [previous]
    report_hashes = [sha256_bytes(previous_raw)]
    if include_continuation:
        continuation_raw = (ROOT / "results/collection-graph/pilot-continuation-v1.json").read_bytes()
        continuation = json.loads(continuation_raw)
        require(continuation["plan_sha256"] == sha256_bytes(raw)
            and continuation["reconciliation"]["previous_report_sha256"] == sha256_bytes(previous_raw), "GRAPH_CONTINUATION_BINDING")
        reports.append(continuation)
        report_hashes.append(sha256_bytes(continuation_raw))
    claims = read_claims()
    selected_ids = [r["chunk_id"] for r in plan["selection"]]
    require(set(claims) <= set(selected_ids), "GRAPH_UNPLANNED_CACHE_INPUT")
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    recipe = ExtractionRecipe.model_validate_json(canonical_json(plan["extraction_recipe"]))
    accepted_rows = [r for report in reports for r in report["extractions"]]
    rejected_rows = [r for report in reports for r in report["quarantined"]]
    accepted = {r["chunk_id"] for r in accepted_rows}
    rejected = {r["chunk_id"] for r in rejected_rows}
    require(not accepted & rejected and len(accepted) == len(accepted_rows)
        and len(rejected) == len(rejected_rows), "GRAPH_PREVIOUS_PARTITION_OVERLAP")
    canonical = []
    canonical_hashes = []
    for report in reports:
        content = Path(report["canonical_path"]).read_bytes()
        if "canonical_sha256" in report:
            require(sha256_bytes(content) == report["canonical_sha256"], "GRAPH_CANONICAL_HASH_CHANGED")
        canonical.extend(json.loads(line) for line in content.splitlines())
        canonical_hashes.append(sha256_bytes(content))
    require({r["chunk_id"] for r in canonical} == accepted, "GRAPH_CANONICAL_PARTITION")
    canonical_by_chunk = {r["chunk_id"]: r for r in canonical}
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")
    operations, zero_dispatch = [], []
    for chunk_id in selected_ids:
        claim = claims.get(chunk_id)
        if claim is None:
            continue
        context, receipts = claim["context"], claim["receipts"]
        require(claim["cache_key"] == raw_cache_key("demo", chunks[chunk_id].text, recipe), "GRAPH_CACHE_RECIPE_CHANGED")
        operation_id = stable_id("op", **context.model_dump())
        managed = ledger.store.get(("OP#" + operation_id, "STATE"))
        if not receipts:
            require(claim["state"] == "FAILED" and managed is None and chunk_id not in accepted | rejected,
                "GRAPH_RECEIPT_FREE_INPUT_UNRESOLVED")
            zero_dispatch.append({"chunk_id": chunk_id, "operation_id": operation_id,
                "managed_status": "ABSENT", "state": "FAILED", "decision": "SKIP_KEEP_TOMBSTONE_NO_RETRY"})
            continue
        require(len(receipts) == 1 and managed is not None, "GRAPH_MISSING_OR_REPAIR_RECEIPT")
        operation = Operation.model_validate_json(canonical_json(managed))
        system, prompt = extraction_request(chunks[chunk_id].text, recipe)
        body = canonical_json({"model": "deepseek-flash", "messages": [{"role": "system", "content": system},
            {"role": "user", "content": prompt}], "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": 2048, "stream": False, "response_format": {"type": "json_object"}})
        verify_operation(operation, context, receipts[0], sha256_bytes(body), prices)
        require((claim["state"] == "READY" and chunk_id in accepted) or
            (claim["state"] == "FAILED" and chunk_id in rejected), "GRAPH_SETTLED_LOCAL_PARTITION")
        if chunk_id in accepted:
            validated = validate_output(receipts[0].text, chunk=chunks[chunk_id], document=docs[chunks[chunk_id].document_version_id],
                stop_reason=receipts[0].stop_reason, recipe=recipe)
            restored = canonicalize(validated, chunk=chunks[chunk_id], document=docs[chunks[chunk_id].document_version_id],
                recipe=recipe, aliases=aliases)
            require(all([r.model_dump(mode="json") for r in getattr(restored, name)] == canonical_by_chunk[chunk_id][name]
                for name in ("entities", "mentions", "assertions")), "GRAPH_CANONICAL_RECEIPT_MISMATCH")
        operations.append({"chunk_id": chunk_id, "operation_id": operation_id, "status": operation.status,
            "actual_micro_usd": operation.actual_micro_usd, "input_sha256": operation.spec.input_sha256,
            "receipt_sha256": sha256_bytes(canonical_json(receipts[0].model_dump(mode="json")))})
    total = sum(r["actual_micro_usd"] for r in operations)
    require(len(operations) == sum(r["model_dispatch_attempts"] for r in reports) == len(accepted) + len(rejected)
        and total == sum(r["new_cost_micro_usd"] for r in reports), "GRAPH_PREVIOUS_DISPATCH_COST_MISMATCH")
    run = ledger.store.get(ledger.run_key(RUN))
    require(run and not run["halted"] and run["cap_micro_usd"] == 500000
        and run["settled_micro_usd"] == total and run["reserved_micro_usd"] == 0, "GRAPH_RUN_NOT_RECONCILED")
    funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
    require(funding and not funding["halted"] and funding["cap_micro_usd"] == 2000000, "GRAPH_FUNDING_CHANGED")
    return {"status": "PASS_SOURCE_RECEIPT_INPUT_USAGE_MANAGED_LEDGER", "checked_at": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_bytes(raw), "previous_report_sha256": sha256_bytes(previous_raw),
        "report_sha256s": report_hashes, "canonical_sha256s": canonical_hashes,
        "run_id": RUN, "run": run, "funding": funding,
        "operations": operations, "zero_dispatch_skips": zero_dispatch,
        "remaining_chunk_ids": [key for key in selected_ids if key not in claims],
        "settled_micro_usd": total, "new_model_calls": 0, "aws_reads": True, "aws_writes": False,
        "cache_modified": False, "semantic_review": "PENDING"}


def managed_ledger():
    config = Config(retries={"total_max_attempts": 1}, connect_timeout=10, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    identity = session.client("sts", config=config).get_caller_identity()
    require(identity["Account"] == "357433444130" and identity["Arn"].endswith(":user/kg-dev"), "GRAPH_DEVELOPER_IDENTITY")
    return Ledger(DynamoLedgerStore(session.client("dynamodb", config=config), "kg-r72-query-state"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--read-managed", action="store_true")
    parser.add_argument("--include-continuation", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.read_managed:
        print(json.dumps({"status": "NO_AWS_OR_MODEL_CALLS_USE_READ_MANAGED"}))
        return
    require(args.output is not None and not args.output.exists(), "GRAPH_NEW_AUDIT_PATH_REQUIRED")
    result = audit(managed_ledger(), include_continuation=args.include_continuation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical_json(result) + b"\n")
    print(json.dumps({"status": result["status"], "settled_calls": len(result["operations"]),
        "settled_micro_usd": result["settled_micro_usd"], "untouched_inputs": len(result["remaining_chunk_ids"]),
        "zero_dispatch_skips": len(result["zero_dispatch_skips"]), "new_model_calls": 0}))


if __name__ == "__main__":
    main()
