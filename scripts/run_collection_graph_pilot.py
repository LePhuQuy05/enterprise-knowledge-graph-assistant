"""Execute only the frozen 48-source collection graph pilot within shared funding.

No deployment, no full-corpus batch, no automatic transport/format retries.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config

from app.collections import load_collection
from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes, stable_id
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
from app.extraction import AliasDictionary, ExtractionError
from app.extraction_cache import ExtractionCache, ExtractionCacheBlocked
from app.ledger import DynamoLedgerStore, Ledger
from app.model_gateway import CallContext, safe_failure_code

ROOT = Path(__file__).resolve().parents[1]
RUN = "collection316-graph-pilot-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan_raw = (ROOT / "results/collection-graph/extraction-plan-v1.json").read_bytes()
    plan = json.loads(plan_raw)
    if not args.execute:
        print(json.dumps({"status": "PLANNED_NO_MODEL_OR_AWS_CALL", "inputs": len(plan["selection"]),
                          "plan_sha256": sha256_bytes(plan_raw)}))
        return
    if plan["max_dispatches"] != 48 or len(plan["selection"]) != 48 or plan["format_repair"]:
        raise ValueError("GRAPH_PILOT_PLAN_SCOPE_CHANGED")
    directory = ROOT / "data/experiments/collection316-graph-pilot-v1"
    directory.mkdir(parents=True, exist_ok=True)
    frozen = directory / "plan.json"
    if frozen.exists() and frozen.read_bytes() != plan_raw:
        raise ValueError("GRAPH_PILOT_FROZEN_PLAN_CHANGED")
    frozen.write_bytes(plan_raw)
    report_path = ROOT / "results/collection-graph/pilot-v1.json"
    if report_path.exists():
        raise ValueError("GRAPH_PILOT_ALREADY_STARTED_REVIEW_CHECKPOINT_NO_AUTOMATIC_RETRY")
    report = {"status": "STARTING", "started_at": datetime.now(UTC).isoformat(), "run_id": RUN,
        "plan_sha256": sha256_bytes(plan_raw), "extractions": [], "quarantined": [], "semantic_review": "PENDING",
        "cloud_deployments": 0, "full_corpus_extraction": False, "new_cost_micro_usd": 0,
        "model_dispatch_attempts": 0, "canonical_path": str(directory / "canonical.jsonl")}
    canonical = []
    transport = None
    metadata_attempts = 0
    metadata_complete = False

    def checkpoint():
        report["model_dispatch_attempts"] = max(0, transport.dispatch_attempts - metadata_attempts) if transport and metadata_complete else 0
        report["metadata_read_attempts"] = metadata_attempts
        content = b"".join(canonical_json(r) + b"\n" for r in canonical)
        (directory / "canonical.jsonl").write_bytes(content)
        report["canonical_sha256"] = sha256_bytes(content)
        report_path.write_bytes(canonical_json(report) + b"\n")

    checkpoint()
    try:
        pipeline = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
        collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo", embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=plan["collection_manifest_sha256"])
        docs, chunks = {}, {}
        for ref in collection.manifest.shards:
            shard = collection.get_shard(ref.shard_id)
            docs.update(shard.documents)
            chunks.update(shard.chunks)
        for selected in plan["selection"]:
            chunk = chunks[selected["chunk_id"]]
            if chunk.text_sha256 != selected["text_sha256"] or chunk.document_version_id != selected["document_version_id"]:
                raise ValueError("GRAPH_PILOT_SOURCE_CHANGED")
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=10, read_timeout=15)
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
            raise ValueError("GRAPH_PILOT_DEVELOPER_IDENTITY_REQUIRED")
        ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
        funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
        month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
        if not funding or funding["halted"] or funding["cap_micro_usd"] != 2000000 or not month or month["halted"] or month["cap_micro_usd"] != 12000000:
            raise ValueError("GRAPH_PILOT_EXISTING_FUNDING_REQUIRED")
        if funding["cap_micro_usd"] - funding["settled_micro_usd"] - funding["reserved_micro_usd"] < 500000:
            raise ValueError("GRAPH_PILOT_NO_REMAINING_SHARED_ALLOWANCE")
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
        prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, "external-default", now=datetime.now(UTC), live=True)
        transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
        provider = deepseek_preflight(transport)
        metadata_attempts = transport.dispatch_attempts
        metadata_complete = True
        if provider["metadata_sha256"] != routes.account_model_metadata_sha256:
            raise ValueError("GRAPH_PILOT_MODEL_CHANGED")
        report["provider_balance_observed"] = provider["balances"]
        report["initial_funding"] = funding
        # A distinct small allowance inside existing shared funding. No funding,
        # month or historical run initialization/reset/cap change.
        ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
        report["initial_run"] = ledger.store.get(ledger.run_key(RUN))
        gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={
            "live_inference_enabled": True, "live_inference_gate": "FROZEN_COLLECTION316_GRAPH_48_SOURCES_V1"}), prices)
        aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")
        recipe = ExtractionRecipe.model_validate_json(canonical_json(plan["extraction_recipe"]))
        cache = ExtractionCache(ROOT / "data/cache/external-models", namespace="LIVE")
        report["status"] = "RUNNING"
        checkpoint()
        for selected in plan["selection"]:
            if report["new_cost_micro_usd"] >= 150000 or transport.dispatch_attempts - metadata_attempts >= 48:
                report["status"] = "STOPPED_AT_PILOT_LIMIT"
                break
            chunk = chunks[selected["chunk_id"]]
            context = CallContext(tenant_id="demo", run_id=RUN, request_id="graph316-" + chunk.chunk_id, stage="EXTRACT")
            try:
                result = cache.resolve(context, chunk=chunk, document=docs[chunk.document_version_id], recipe=recipe,
                                       aliases=aliases, gateway=gateway, allow_repair=False)
            except (ExtractionError, ExtractionCacheBlocked) as exc:
                operation = ledger.operation(stable_id("op", **context.model_dump()))
                if operation.status != "SETTLED":
                    raise
                report["quarantined"].append({"chunk_id": chunk.chunk_id, "operation_id": operation.operation_id,
                    "error_category": str(exc)[:120], "cost_micro_usd": operation.actual_micro_usd})
                report["new_cost_micro_usd"] += operation.actual_micro_usd
                checkpoint()
                continue
            row = {name: [r.model_dump(mode="json") for r in getattr(result.canonical, name)] for name in ("entities", "mentions", "assertions")}
            row.update(chunk_id=chunk.chunk_id, semantic_review="PENDING")
            canonical.append(row)
            report["new_cost_micro_usd"] += result.new_cost_micro_usd
            report["extractions"].append({"chunk_id": chunk.chunk_id, "group": selected["group"], "cache_hit": result.cache_hit,
                "assertions": len(result.canonical.assertions), "new_cost_micro_usd": result.new_cost_micro_usd,
                "operation_ids": [a.operation_id for a in result.record.attempts]})
            checkpoint()
            print(json.dumps({"completed": len(report["extractions"]), "quarantined": len(report["quarantined"]),
                              "new_cost_micro_usd": report["new_cost_micro_usd"]}), flush=True)
        else:
            report["status"] = "STRUCTURAL_PILOT_COMPLETE_SEMANTIC_REVIEW_PENDING"
        report["final_run"] = ledger.store.get(ledger.run_key(RUN))
        report["canonical_sha256"] = sha256_bytes((directory / "canonical.jsonl").read_bytes())
    except Exception as exc:
        report["status"] = "STOPPED_REQUIRES_REVIEW_NO_AUTOMATIC_RETRY"
        report["error_category"] = safe_failure_code(exc)
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        checkpoint()
    print(json.dumps({k: report[k] for k in ("status", "model_dispatch_attempts", "new_cost_micro_usd")}))


if __name__ == "__main__":
    main()
