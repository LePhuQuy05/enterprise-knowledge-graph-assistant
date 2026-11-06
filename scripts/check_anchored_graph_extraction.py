"""Freeze/execute a four-source V4 development check; never retry a V3 cache entry.

Selected known V3 failures are development examples, not an unbiased quality test.
The new run is bounded within existing shared funding; default only writes a plan.
"""

import argparse
import json
from datetime import UTC, datetime

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
from app.extraction import (
    AliasDictionary,
    ExtractionError,
    deepseek_anchored_recipe,
    extraction_request,
)
from app.extraction_cache import ExtractionCache, ExtractionCacheBlocked
from app.model_gateway import CallContext, safe_failure_code
from scripts.reconcile_collection_graph_pilot import (
    audit,
    load_inputs,
    managed_ledger,
    read_claims,
    require,
)
from scripts.run_collection_graph_pilot import ROOT

RUN = "collection316-graph-v4-check-v1"
SELECTED = (
    "chk_477b0ca72fad16452f9f6e1c60777632800d7ca7b79a5ea41aa7c966a347e063",
    "chk_afd230987a863054b8b41550c3c344dc89304ba85bf410308ad5e89b9906956a",
    "chk_89ea9ef32e4eb95694f16cc9f34dd890aec2eb9d7ff0109fa3206ba7201abd6e",
    "chk_fcdbe7f288fecbb28fd308399108674daa73021a6042bc7ae4e402e054b552d0",
)


def freeze(loaded):
    previous_raw, previous, _, chunks = loaded
    claims = read_claims()
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = deepseek_anchored_recipe(aliases, metadata_sha256=routes.account_model_metadata_sha256)
    by_id = {r["chunk_id"]: r for r in previous["selection"]}
    selection = []
    for key in SELECTED:
        claim = claims[key]
        require(claim["state"] == "FAILED" and len(claim["receipts"]) == 1, "GRAPH_V4_REQUIRES_RECEIVED_V3_FAILURE")
        system, prompt = extraction_request(chunks[key].text, recipe)
        selection.append(by_id[key] | {"v3_operation_id": claim["receipts"][0].operation_id,
            "v3_output_sha256": sha256_bytes(claim["receipts"][0].text.encode()),
            "new_system_sha256": sha256_bytes(system.encode()), "new_prompt_sha256": sha256_bytes(prompt.encode())})
    plan = {"status": "FROZEN_DEVELOPMENT_ERROR_CHECK_NOT_QUALITY_BENCHMARK", "run_id": RUN,
        "collection_manifest_sha256": previous["collection_manifest_sha256"], "previous_plan_sha256": sha256_bytes(previous_raw),
        "extraction_recipe": recipe.model_dump(mode="json"), "selection": selection, "maximum_new_model_calls": 4,
        "run_cap_micro_usd": 500000, "shared_funding_cap_micro_usd": 2000000, "observed_stop_between_calls_micro_usd": 20000,
        "conservative_dispatch_bound_micro_usd": 317031, "format_repair": False, "transport_retries": 0,
        "full_batch_authorized": False, "funding_cap_raise": False, "semantic_review": "PENDING",
        "v3_records_modified": False, "implementation_sha256": {name: sha256_bytes((ROOT / name).read_bytes()) for name in
            ("app/extraction.py", "app/anchored_extraction.py", "scripts/check_anchored_graph_extraction.py")}}
    raw = canonical_json(plan) + b"\n"
    path = ROOT / "results/collection-graph/anchored-v4-plan-v2.json"
    if path.exists():
        require(path.read_bytes() == raw, "GRAPH_V4_FROZEN_PLAN_CHANGED")
    else:
        with path.open("xb") as handle:
            handle.write(raw)
    return raw, plan, recipe, aliases, routes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    loaded = load_inputs()
    plan_raw, plan, recipe, aliases, routes = freeze(loaded)
    if not args.execute:
        print(json.dumps({"status": plan["status"], "model_calls": 0, "aws_calls": 0, "planned_inputs": len(SELECTED)}))
        return
    path = ROOT / "results/collection-graph/anchored-v4-check.json"
    require(not path.exists(), "GRAPH_V4_ALREADY_STARTED_NO_AUTOMATIC_RETRY")
    ledger = managed_ledger()
    prior = audit(ledger, loaded=loaded, include_continuation=True)
    require(not prior["remaining_chunk_ids"] and len(prior["operations"]) == 47 and len(prior["zero_dispatch_skips"]) == 1,
        "GRAPH_V3_FINAL_RECONCILIATION_REQUIRED")
    require(ledger.store.get(ledger.run_key(RUN)) is None, "GRAPH_V4_RUN_ALREADY_EXISTS")
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    require(month and not month["halted"] and month["cap_micro_usd"] == 12000000, "GRAPH_V4_MONTH_REQUIRED")
    directory = ROOT / f"data/experiments/{RUN}"
    require(not directory.exists(), "GRAPH_V4_DIRECTORY_EXISTS")
    directory.mkdir(parents=True)
    (directory / "plan.json").write_bytes(plan_raw)
    (directory / "prior-reconciliation.json").write_bytes(canonical_json(prior) + b"\n")
    report = {"status": "STARTING", "started_at": datetime.now(UTC).isoformat(), "run_id": RUN,
        "plan_sha256": sha256_bytes(plan_raw), "prior_reconciliation_sha256": sha256_bytes(canonical_json(prior)),
        "extractions": [], "quarantined": [], "new_cost_micro_usd": 0, "model_dispatch_attempts": 0,
        "canonical_path": str(directory / "canonical.jsonl"), "semantic_review": "PENDING", "cloud_deployments": 0,
        "known_development_errors_only": True, "v3_records_modified": False}
    with path.open("xb") as handle:
        handle.write(canonical_json(report) + b"\n")
    canonical, transport, metadata_attempts, metadata_complete = [], None, 0, False

    def checkpoint():
        report["model_dispatch_attempts"] = max(0, transport.dispatch_attempts - metadata_attempts) if transport and metadata_complete else 0
        report["metadata_read_attempts"] = metadata_attempts
        content = b"".join(canonical_json(row) + b"\n" for row in canonical)
        report["canonical_sha256"] = sha256_bytes(content)
        (directory / "canonical.jsonl").write_bytes(content)
        path.write_bytes(canonical_json(report) + b"\n")

    try:
        prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
        prices.rate_for("deepseek-flash", "deepseek-flash", "external-default", now=datetime.now(UTC), live=True)
        transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
        provider = deepseek_preflight(transport)
        metadata_attempts, metadata_complete = transport.dispatch_attempts, True
        require(provider["metadata_sha256"] == routes.account_model_metadata_sha256, "GRAPH_V4_PROVIDER_CHANGED")
        report["provider_balance_observed"] = provider["balances"]
        ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=plan["run_cap_micro_usd"])
        gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
            "live_inference_gate": "FROZEN_FOUR_SOURCE_ANCHORED_V4_DEVELOPMENT_CHECK"}), prices)
        cache = ExtractionCache(ROOT / "data/cache/external-models", namespace="LIVE")
        _, _, docs, chunks = loaded
        report["status"] = "RUNNING"
        checkpoint()
        for key in SELECTED:
            if report["new_cost_micro_usd"] >= plan["observed_stop_between_calls_micro_usd"]:
                report["status"] = "STOPPED_AT_OBSERVED_BETWEEN_CALL_LIMIT"
                break
            chunk = chunks[key]
            require(cache.get("demo", chunk.text, recipe) is None, "GRAPH_V4_CACHE_ALREADY_CLAIMED")
            context = CallContext(tenant_id="demo", run_id=RUN, request_id="graph316-v4-" + key, stage="EXTRACT")
            try:
                result = cache.resolve(context, chunk=chunk, document=docs[chunk.document_version_id], recipe=recipe,
                    aliases=aliases, gateway=gateway, allow_repair=False)
            except (ExtractionError, ExtractionCacheBlocked) as exc:
                operation = ledger.operation(stable_id("op", **context.model_dump()))
                require(operation.status == "SETTLED", "GRAPH_V4_UNSETTLED_STOP_NO_RETRY")
                report["quarantined"].append({"chunk_id": key, "operation_id": operation.operation_id,
                    "error_category": str(exc)[:120], "cost_micro_usd": operation.actual_micro_usd})
                report["new_cost_micro_usd"] += operation.actual_micro_usd
                checkpoint()
                continue
            canonical.append({"chunk_id": key, "semantic_review": "PENDING", **{name:
                [r.model_dump(mode="json") for r in getattr(result.canonical, name)] for name in ("entities", "mentions", "assertions")}})
            report["extractions"].append({"chunk_id": key, "assertions": len(result.canonical.assertions),
                "new_cost_micro_usd": result.new_cost_micro_usd, "operation_ids": [a.operation_id for a in result.record.attempts]})
            report["new_cost_micro_usd"] += result.new_cost_micro_usd
            checkpoint()
            print(json.dumps({"accepted": len(report["extractions"]), "quarantined": len(report["quarantined"]),
                "new_cost_micro_usd": report["new_cost_micro_usd"]}), flush=True)
        else:
            report["status"] = "DEVELOPMENT_CHECK_COMPLETE_SEMANTIC_REVIEW_PENDING"
        report["final_run"] = ledger.store.get(ledger.run_key(RUN))
    except Exception as exc:
        report["status"], report["error_category"] = "STOPPED_REQUIRES_REVIEW_NO_AUTOMATIC_RETRY", safe_failure_code(exc)
    finally:
        report["finished_at"] = datetime.now(UTC).isoformat()
        checkpoint()
    print(json.dumps({k: report[k] for k in ("status", "model_dispatch_attempts", "new_cost_micro_usd")}))


if __name__ == "__main__":
    main()
