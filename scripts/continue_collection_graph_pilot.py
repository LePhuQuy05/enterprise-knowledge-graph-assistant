"""Continue only untouched frozen pilot inputs after a fresh read-only reconciliation.

Historical failed/received inputs are skipped, never retried or deleted. Default is offline.
"""

import argparse
import json
from datetime import UTC, datetime

from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes, stable_id
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
from app.extraction import AliasDictionary, ExtractionError
from app.extraction_cache import ExtractionCache, ExtractionCacheBlocked
from app.model_gateway import CallContext, safe_failure_code
from scripts.reconcile_collection_graph_pilot import audit, load_inputs, managed_ledger, require
from scripts.run_collection_graph_pilot import ROOT, RUN


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PLANNED_NO_MODEL_OR_AWS_CALL", "skip_all_previously_claimed_inputs": True,
            "retry_failed_inputs": False, "funding_cap_raise": False}))
        return
    path = ROOT / "results/collection-graph/pilot-continuation-v1.json"
    require(not path.exists(), "GRAPH_CONTINUATION_ALREADY_STARTED_NO_AUTOMATIC_RETRY")
    loaded = load_inputs()
    plan_raw, plan, docs, chunks = loaded
    ledger = managed_ledger()
    reconciliation = audit(ledger, loaded=loaded)
    require(reconciliation["remaining_chunk_ids"], "GRAPH_NO_UNTOUCHED_INPUTS")
    report = {"status": "STARTING", "started_at": datetime.now(UTC).isoformat(), "run_id": RUN,
        "plan_sha256": sha256_bytes(plan_raw), "reconciliation": reconciliation,
        "extractions": [], "quarantined": [], "new_cost_micro_usd": 0, "model_dispatch_attempts": 0,
        "cloud_deployments": 0, "full_corpus_extraction": False, "semantic_review": "PENDING",
        "retry_failed_inputs": False, "funding_cap_raise": False}
    directory = ROOT / f"data/experiments/{RUN}/continuation-v1"
    require(not directory.exists(), "GRAPH_CONTINUATION_DIRECTORY_EXISTS")
    directory.mkdir(parents=True)
    report["canonical_path"] = str(directory / "canonical.jsonl")
    canonical = []
    transport = None
    metadata_attempts = 0
    metadata_complete = False

    def checkpoint():
        report["model_dispatch_attempts"] = max(0, transport.dispatch_attempts - metadata_attempts) if transport and metadata_complete else 0
        report["metadata_read_attempts"] = metadata_attempts
        content = b"".join(canonical_json(row) + b"\n" for row in canonical)
        report["canonical_sha256"] = sha256_bytes(content)
        (directory / "canonical.jsonl").write_bytes(content)
        path.write_bytes(canonical_json(report) + b"\n")

    # Establish a durable marker before any provider metadata reads or model dispatch.
    with path.open("xb") as handle:
        handle.write(canonical_json(report) + b"\n")
    checkpoint()
    try:
        month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
        require(month and not month["halted"] and month["cap_micro_usd"] == 12000000, "GRAPH_MONTH_REQUIRED")
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
        prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, "external-default", now=datetime.now(UTC), live=True)
        transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
        provider = deepseek_preflight(transport)
        metadata_attempts = transport.dispatch_attempts
        metadata_complete = True
        require(provider["metadata_sha256"] == routes.account_model_metadata_sha256, "GRAPH_PROVIDER_MODEL_CHANGED")
        report["provider_balance_observed"] = provider["balances"]
        gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={
            "live_inference_enabled": True, "live_inference_gate": "CONTINUE_UNTOUCHED_FROZEN_GRAPH_PILOT_V1"}), prices)
        aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")
        recipe = ExtractionRecipe.model_validate_json(canonical_json(plan["extraction_recipe"]))
        cache = ExtractionCache(ROOT / "data/cache/external-models", namespace="LIVE")
        for key in reconciliation["remaining_chunk_ids"]:
            require(cache.get("demo", chunks[key].text, recipe) is None, "GRAPH_UNTOUCHED_CACHE_NOW_EXISTS")
        by_id = {r["chunk_id"]: r for r in plan["selection"]}
        report["status"] = "RUNNING"
        checkpoint()
        for key in reconciliation["remaining_chunk_ids"]:
            total_cost = reconciliation["settled_micro_usd"] + report["new_cost_micro_usd"]
            total_dispatches = len(reconciliation["operations"]) + transport.dispatch_attempts - metadata_attempts
            if total_cost >= plan["observed_cost_stop_between_calls_micro_usd"] or total_dispatches >= plan["max_dispatches"]:
                report["status"] = "STOPPED_AT_TOTAL_PILOT_LIMIT"
                break
            chunk = chunks[key]
            context = CallContext(tenant_id="demo", run_id=RUN, request_id="graph316-" + key, stage="EXTRACT")
            # The continuation cannot overwrite or adopt another dispatch intent.
            require(ledger.store.get(("OP#" + stable_id("op", **context.model_dump()), "STATE")) is None,
                "GRAPH_UNTOUCHED_OPERATION_NOW_EXISTS")
            try:
                result = cache.resolve(context, chunk=chunk, document=docs[chunk.document_version_id],
                    recipe=recipe, aliases=aliases, gateway=gateway, allow_repair=False)
            except (ExtractionError, ExtractionCacheBlocked) as exc:
                operation = ledger.operation(stable_id("op", **context.model_dump()))
                require(operation.status == "SETTLED", "GRAPH_UNSETTLED_STOP_NO_RETRY")
                report["quarantined"].append({"chunk_id": key, "operation_id": operation.operation_id,
                    "error_category": str(exc)[:120], "cost_micro_usd": operation.actual_micro_usd})
                report["new_cost_micro_usd"] += operation.actual_micro_usd
                checkpoint()
                print(json.dumps({"accepted": len(report["extractions"]), "quarantined": len(report["quarantined"]),
                    "new_cost_micro_usd": report["new_cost_micro_usd"]}), flush=True)
                continue
            canonical.append({"chunk_id": key, "semantic_review": "PENDING", **{
                name: [r.model_dump(mode="json") for r in getattr(result.canonical, name)]
                for name in ("entities", "mentions", "assertions")}})
            report["new_cost_micro_usd"] += result.new_cost_micro_usd
            report["extractions"].append({"chunk_id": key, "group": by_id[key]["group"], "cache_hit": result.cache_hit,
                "assertions": len(result.canonical.assertions), "new_cost_micro_usd": result.new_cost_micro_usd,
                "operation_ids": [a.operation_id for a in result.record.attempts]})
            checkpoint()
            print(json.dumps({"accepted": len(report["extractions"]), "quarantined": len(report["quarantined"]),
                "new_cost_micro_usd": report["new_cost_micro_usd"]}), flush=True)
        else:
            report["status"] = "UNTOUCHED_PILOT_COMPLETE_SEMANTIC_REVIEW_PENDING"
        report["final_run"] = ledger.store.get(ledger.run_key(RUN))
    except Exception as exc:
        report["status"] = "STOPPED_REQUIRES_REVIEW_NO_AUTOMATIC_RETRY"
        report["error_category"] = safe_failure_code(exc)
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        checkpoint()
    print(json.dumps({k: report[k] for k in ("status", "model_dispatch_attempts", "new_cost_micro_usd")}))


if __name__ == "__main__":
    main()
