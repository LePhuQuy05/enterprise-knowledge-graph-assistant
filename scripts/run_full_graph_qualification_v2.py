"""Freeze then execute 48 source-stratified V4 inputs under the authorized cap.

Default is local planning. This command cannot launch the remaining full batch.
It never generates answers, retries failed cache entries, or serves candidates.
"""

import argparse
import json
from collections import Counter
from datetime import UTC, datetime

from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes, stable_id
from app.extraction import AliasDictionary, ExtractionError
from app.extraction_cache import ExtractionCache
from app.full_graph_workflow import ScopedExtractionGateway, qualification_units
from app.model_gateway import CallContext, safe_failure_code
from scripts.run_collection_benchmark import ROOT, load_inputs

RUN = "collection316-full-graph-v2-qualification"
DIRECTORY = ROOT / "results/method-correctness-v2/full-graph-qualification-v2"
INVENTORY = ROOT / "results/method-correctness-v2/full-graph-inventory-v2"


def context_for(unit):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=unit["unit_id"], stage="EXTRACT")


def freeze():
    inventory = json.loads((INVENTORY / "plan.json").read_bytes())
    raw = (INVENTORY / "source-units.jsonl").read_bytes()
    if sha256_bytes(raw) != inventory["source_units_sha256"]:
        raise ValueError("FULL_GRAPH_SOURCE_INVENTORY_CHECKSUM")
    units = qualification_units(tuple(json.loads(r) for r in raw.splitlines()))
    if len(units) != 48 or len({u["selected_group"] for u in units}) != 24:
        raise ValueError("FULL_GRAPH_QUALIFICATION_EXPECTED_SCOPE")
    auth_raw = (ROOT / "configs/budgets/full-graph-v2-allocation.json").read_bytes()
    aliases_path = ROOT / "configs/extraction/aws-approved-aliases-v2.json"
    if sha256_bytes(aliases_path.read_bytes()) != inventory["aliases_sha256"]:
        raise ValueError("FULL_GRAPH_ALIASES_CHANGED")
    plan = {"schema_version": 2, "run_id": RUN, "phase": "QUALIFICATION_NOT_FULL_BATCH",
        "inventory_sha256": sha256_bytes((INVENTORY / "plan.json").read_bytes()),
        "source_units_sha256": inventory["source_units_sha256"], "collection_manifest_sha256": inventory["collection_manifest_sha256"],
        "recipe": inventory["extraction_recipe"], "aliases_sha256": inventory["aliases_sha256"],
        "authorization_sha256": sha256_bytes(auth_raw), "selection": units, "maximum_new_calls": 48,
        "run_cap_micro_usd": 7000000, "observed_stop_between_calls_micro_usd": 100000,
        "funding_cap_micro_usd": 8000000, "month_cap_micro_usd": 12000000,
        "selection_policy": "TWO_HASH_ORDERED_UNITS_PER_SOURCE_GROUP_PREFER_DISTINCT_DOCUMENTS_V1",
        "candidate_semantics": "UNREVIEWED_NOT_SERVING", "full_batch_enabled": False,
        "format_repair": False, "transport_retry": False,
        "code_sha256": {name: sha256_bytes((ROOT / name).read_bytes()) for name in (
            "app/full_graph_workflow.py", "app/extraction.py", "app/anchored_extraction.py",
            "app/extraction_cache.py", "scripts/run_full_graph_qualification_v2.py")}}
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    output = DIRECTORY / "plan.json"
    content = canonical_json(plan) + b"\n"
    if output.exists() and output.read_bytes() != content:
        raise ValueError("FULL_GRAPH_FROZEN_PLAN_CHANGED")
    if not output.exists():
        output.write_bytes(content)
    return plan


def execute(plan):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.ledger import DynamoLedgerStore, Ledger

    _, _, collection, _, _ = load_inputs()
    if collection.manifest_sha256 != plan["collection_manifest_sha256"]:
        raise ValueError("FULL_GRAPH_COLLECTION_CHANGED")
    docs, chunks = {}, {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        docs.update(shard.documents)
        chunks.update(shard.chunks)
    for unit in plan["selection"]:
        for occurrence in unit["occurrences"]:
            chunk = chunks[occurrence["chunk_id"]]
            if chunk.text_sha256 != unit["text_sha256"] or chunk.document_version_id != occurrence["document_version_id"]:
                raise ValueError("FULL_GRAPH_SOURCE_CHANGED")
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("FULL_GRAPH_OPERATOR_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    for key, cap in ((ledger.external_funding_key("kg-r72"), plan["funding_cap_micro_usd"]),
                     (ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")), plan["month_cap_micro_usd"])):
        row = ledger.store.get(key)
        if not row or row["halted"] or row["cap_micro_usd"] != cap or cap - row["settled_micro_usd"] - row["reserved_micro_usd"] < 500000:
            raise ValueError("FULL_GRAPH_EXISTING_ALLOWANCE_NOT_READY")
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=plan["run_cap_micro_usd"])
    run = ledger.store.get(ledger.run_key(RUN))
    if run["halted"] or run["reserved_micro_usd"] or run["cap_micro_usd"] != plan["run_cap_micro_usd"]:
        raise ValueError("FULL_GRAPH_RUN_RECONCILIATION_REQUIRED")
    recipe = ExtractionRecipe.model_validate(plan["recipe"])
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v2.json")
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    provider = deepseek_preflight(transport)
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("FULL_GRAPH_PROVIDER_METADATA_CHANGED")
    gateway = ScopedExtractionGateway(ExternalModelGateway(ledger, transport,
        routes.model_copy(update={"live_inference_enabled": True, "live_inference_gate": "SOURCE_STRATIFIED_QUALIFICATION_48_V2"}), prices),
        ((context_for(u), chunks[u["occurrences"][0]["chunk_id"]].text) for u in plan["selection"]), recipe)
    cache = ExtractionCache(ROOT / "data/cache/external-models", namespace="LIVE")
    journal = DIRECTORY / "units"
    journal.mkdir(exist_ok=True)
    rows, status, failure = [], "QUALIFICATION_STRUCTURAL_COMPLETE_SEMANTIC_REVIEW_PENDING", None
    for unit in plan["selection"]:
        path = journal / (unit["unit_id"] + ".json")
        if path.exists():
            raw = path.read_bytes()
            if path.with_suffix(".sha256").read_text("ascii").strip() != sha256_bytes(raw):
                raise ValueError("FULL_GRAPH_JOURNAL_CHECKSUM")
            row = json.loads(raw)
            if row["unit_id"] != unit["unit_id"] or row["text_sha256"] != unit["text_sha256"]:
                raise ValueError("FULL_GRAPH_JOURNAL_BINDING")
            rows.append(row)
            continue
        if gateway.cost >= plan["observed_stop_between_calls_micro_usd"]:
            status = "STOPPED_AT_QUALIFICATION_COST_LIMIT"
            break
        context = context_for(unit)
        row = {"unit_id": unit["unit_id"], "text_sha256": unit["text_sha256"], "selected_group": unit["selected_group"],
            "occurrences": unit["occurrences"], "semantic_review": "UNREVIEWED_NOT_SERVING"}
        try:
            canonical = []
            for occurrence in unit["occurrences"]:
                chunk = chunks[occurrence["chunk_id"]]
                lookup = cache.resolve(context, chunk=chunk, document=docs[chunk.document_version_id], recipe=recipe,
                    aliases=aliases, gateway=gateway, allow_repair=False)
                canonical.append({"chunk_id": chunk.chunk_id, **{name: [r.model_dump(mode="json") for r in getattr(lookup.canonical, name)]
                    for name in ("entities", "mentions", "assertions")}})
            row.update(status="CANDIDATE" if any(r["assertions"] for r in canonical) else "EMPTY",
                canonical=canonical, operation_ids=[a.operation_id for a in lookup.record.attempts],
                receipt_cost_micro_usd=sum(a.original_cost_micro_usd for a in lookup.record.attempts))
        except Exception as exc:
            # A parser/source failure is only continuable after the exact paid
            # operation is SETTLED. Existing FAILED/unknown cache claims stop.
            try:
                operation = ledger.operation(stable_id("op", **context.model_dump()))
            except Exception:
                operation = None
            if not isinstance(exc, ExtractionError) or operation is None or operation.status != "SETTLED":
                status, failure = "STOPPED_RECONCILIATION_REQUIRED_NO_RETRY", safe_failure_code(exc)
                break
            row.update(status="QUARANTINED", error=safe_failure_code(exc),
                operation_ids=[operation.operation_id], receipt_cost_micro_usd=operation.actual_micro_usd)
        with path.open("xb") as handle:
            raw = canonical_json(row) + b"\n"
            handle.write(raw)
        path.with_suffix(".sha256").write_text(sha256_bytes(raw) + "\n", encoding="ascii")
        rows.append(row)
        print(json.dumps({"processed_units": len(rows), "last_status": row["status"], "new_cost_micro_usd": gateway.cost}), flush=True)
    after = ledger.store.get(ledger.run_key(RUN))
    report = {"status": status, "failure": failure, "processed_units": len(rows), "inventory_units": 2897,
        "statuses": dict(Counter(r["status"] for r in rows)), "provider_calls_this_invocation": len(gateway.seen),
        "new_cost_micro_usd": gateway.cost, "run_counter": after, "full_batch_executed": False,
        "semantic_review": "PENDING", "serving_assertions_added": 0, "finished_at": datetime.now(UTC).isoformat()}
    count = len(list(DIRECTORY.glob("attempt-*.json"))) + 1
    (DIRECTORY / f"attempt-{count:03d}.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "processed_units", "statuses", "new_cost_micro_usd")}))
    return 0 if len(rows) == len(plan["selection"]) and not after["reserved_micro_usd"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = freeze()
    if not args.execute:
        print(json.dumps({"status": "PLANNED_ONLY", "units": len(plan["selection"]), "provider_calls": 0, "aws_calls": 0}))
        return 0
    return execute(plan)


if __name__ == "__main__":
    raise SystemExit(main())
