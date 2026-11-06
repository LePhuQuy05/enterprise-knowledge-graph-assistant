"""Freeze/execute V5 development regressions plus 24 source-selected inputs.

Default is local planning. No old cache/output is reinterpreted or retried.
The shared funding counter stays at its compatible 2 USD cap throughout.
"""

import argparse
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from app.collections import load_collection
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.extraction import AliasDictionary
from app.full_graph_workflow import qualification_units
from app.model_gateway import CallContext, safe_failure_code
from app.semantic_extraction import semantic_recipe, semantic_request
from app.semantic_extraction_cache import SemanticExtractionCache

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results/semantic-graph-v5/canary-v1"
RUN = "collection316-semantic-frames-v5-canary-v1"


def context_for(unit):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=unit["v5_unit_id"], stage="EXTRACT")


def load_collection_inputs():
    pipeline = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    inventory_dir = ROOT / "results/method-correctness-v2/full-graph-inventory-v2"
    inventory = json.loads((inventory_dir / "plan.json").read_bytes())
    raw = (inventory_dir / "source-units.jsonl").read_bytes()
    if sha256_bytes(raw) != inventory["source_units_sha256"]:
        raise ValueError("SEMANTIC_CANARY_INVENTORY_CHECKSUM")
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=inventory["collection_manifest_sha256"])
    chunks = {}
    for ref in collection.manifest.shards:
        chunks.update(collection.get_shard(ref.shard_id).chunks)
    return collection, chunks, tuple(json.loads(line) for line in raw.splitlines())


def freeze():
    collection, chunks, units = load_collection_inputs()
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v2.json")
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = semantic_recipe(aliases, metadata_sha256=routes["account_model_metadata_sha256"])
    audit = json.loads((ROOT / "results/method-correctness-v2/qualification-audit-v1/report.json").read_bytes())
    objections = {r["assertion_id"] for r in audit["objections"]}
    development_ids = set()
    for path in (ROOT / "results/method-correctness-v2/full-graph-qualification-v2/units").glob("*.json"):
        row = json.loads(path.read_bytes())
        if any(a["assertion_id"] in objections for c in row.get("canonical", ()) for a in c["assertions"]):
            development_ids.add(row["unit_id"])
        if any(a["assertion_id"].startswith("ast_0148c748") for c in row.get("canonical", ()) for a in c["assertions"]):
            development_ids.add(row["unit_id"])
    # A known positive source from the earlier NAT demo, explicitly development.
    nat = "chk_c8de39f2ee4a39a2a9ec262e50b925e0eeb6de8f478688efd2010d662d65a0f4"
    development_ids.update(u["unit_id"] for u in units if any(o["chunk_id"] == nat for o in u["occurrences"]))
    old_cache = ROOT / "data/cache/external-models/extraction.sqlite3"
    excluded = set()
    with sqlite3.connect(f"file:{old_cache.as_posix()}?mode=ro", uri=True) as db:
        for (raw_claim,) in db.execute("SELECT claim FROM extraction WHERE namespace='LIVE'"):
            claim = json.loads(raw_claim)
            excluded.add(claim["input_sha256"])
    excluded.update(u["text_sha256"] for u in units if u["unit_id"] in development_ids)
    selected = [dict(u, cohort="KNOWN_DEVELOPMENT") for u in units if u["unit_id"] in development_ids]
    selected.extend(dict(u, cohort="SOURCE_SELECTED_NOT_PREVIOUSLY_CLAIMED") for u in
        qualification_units(tuple(u for u in units if u["text_sha256"] not in excluded), per_group=1))
    if not 30 <= len(selected) <= 36 or len({u["text_sha256"] for u in selected}) != len(selected):
        raise ValueError("SEMANTIC_CANARY_EXPECTED_SELECTION_SCOPE")
    for unit in selected:
        unit["v5_unit_id"] = stable_id("semantic-v5-unit", source=collection.manifest_sha256,
            text=unit["text_sha256"], recipe=recipe.fingerprint)
    plan = {"run_id": RUN, "recipe": recipe.model_dump(mode="json"), "selection": selected,
        "source_manifest_sha256": collection.manifest_sha256, "excluded_previous_source_hashes_sha256": sha256_bytes(canonical_json(sorted(excluded))),
        "previous_claimed_unique_texts": len(excluded), "maximum_new_calls": len(selected), "max_output_tokens": 2048,
        "run_cap_micro_usd": 500000, "actual_cost_stop_between_calls_micro_usd": 50000,
        "active_funding_cap_micro_usd": 2000000, "authorized_ceiling_micro_usd": 8000000,
        "funding_counter_changes": False, "full_batch_enabled": False, "semantic_review": "PENDING",
        "notes": ["New source texts have not been claimed by the legacy extraction cache; this is not an independent benchmark holdout.",
            "Development includes the eight observed errors and known positive sources. Guard pass is not entailment certification."],
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in ("app/semantic_extraction.py",
            "app/semantic_extraction_cache.py", "scripts/run_semantic_graph_canary.py")}}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(plan) + b"\n"
    path = OUTPUT / "plan.json"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("SEMANTIC_CANARY_FROZEN_PLAN_CHANGED")
    if not path.exists():
        path.write_bytes(raw)
    return plan, chunks, recipe


class ScopedGateway:
    def __init__(self, gateway, plan, chunks):
        self.gateway, self.provider, self.routes = gateway, gateway.provider, gateway.routes
        self.allowed = {context_for(u).model_dump_json(): semantic_request(chunks[u["occurrences"][0]["chunk_id"]].text)
            for u in plan["selection"]}
        self.seen, self.cost, self.halted = set(), 0, False

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if self.halted or self.cost >= 50000 or key in self.seen or self.allowed.get(key) != (system, prompt) or max_output_tokens != 2048:
            raise ValueError("SEMANTIC_CANARY_DISPATCH_BOUND")
        self.seen.add(key)
        try:
            result = self.gateway.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
        except Exception:
            self.halted = True
            raise
        self.cost += result.actual_micro_usd
        return result


def execute(plan, chunks, recipe, *, replay=False):
    gateway, ledger = None, None
    if not replay:
        import boto3
        from botocore.config import Config

        from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
        from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
        from app.ledger import DynamoLedgerStore, Ledger

        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
        prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
            raise ValueError("SEMANTIC_CANARY_OPERATOR_IDENTITY")
        ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
        for key, cap in ((ledger.external_funding_key("kg-r72"), 2000000),
                         (ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")), 12000000)):
            counter = ledger.store.get(key)
            if not counter or counter["halted"] or counter["cap_micro_usd"] != cap or cap - counter["settled_micro_usd"] - counter["reserved_micro_usd"] < 500000:
                raise ValueError("SEMANTIC_CANARY_COMPATIBLE_EXISTING_ALLOWANCE_REQUIRED")
        ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
        run = ledger.store.get(ledger.run_key(RUN))
        if run["halted"] or run["reserved_micro_usd"] or run["cap_micro_usd"] != 500000:
            raise ValueError("SEMANTIC_CANARY_RUN_RECONCILIATION_REQUIRED")
        transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
        account = deepseek_preflight(transport)
        if not account["available"] or account["metadata_sha256"] != routes.account_model_metadata_sha256:
            raise ValueError("SEMANTIC_CANARY_PROVIDER_CHANGED")
        gateway = ScopedGateway(ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
            "live_inference_gate": "TRIGGER_BOUND_V5_SOURCE_CANARY_ONLY"}), prices), plan, chunks)
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v5", namespace="LIVE")
    directory = OUTPUT / "units"
    directory.mkdir(exist_ok=True)
    rows, status, error = [], "REPLAY_MATCH" if replay else "STRUCTURAL_CANARY_COMPLETE_REVIEW_PENDING", None
    try:
        for unit in plan["selection"]:
            if gateway and gateway.cost >= 50000:
                raise ValueError("SEMANTIC_CANARY_OBSERVED_COST_STOP")
            text = chunks[unit["occurrences"][0]["chunk_id"]].text
            if sha256_bytes(text.encode()) != unit["text_sha256"]:
                raise ValueError("SEMANTIC_CANARY_SOURCE_CHANGED")
            lookup = cache.resolve(context_for(unit), text=text, recipe=recipe, gateway=gateway)
            validation = lookup.validation
            row = {"unit_id": unit["v5_unit_id"], "legacy_unit_id": unit["unit_id"], "text_sha256": unit["text_sha256"],
                "cohort": unit["cohort"], "occurrences": unit["occurrences"], "receipt": lookup.receipt,
                "accepted_relationships": [f.model_dump(mode="json") for f, _ in validation.accepted_relationships],
                "accepted_permissions": [p.model_dump(mode="json") for p in validation.accepted_permissions],
                "rejected_records": validation.rejected_records, "semantic_review": "PENDING_NOT_SERVING"}
            raw = canonical_json(row) + b"\n"
            path = directory / (unit["v5_unit_id"] + ".json")
            if path.exists() and path.read_bytes() != raw:
                raise ValueError("SEMANTIC_CANARY_REPLAY_CHANGED")
            if not path.exists():
                path.write_bytes(raw)
            rows.append(row)
            print(json.dumps({"processed": len(rows), "cohort": unit["cohort"], "relations": len(validation.accepted_relationships),
                "permissions": len(validation.accepted_permissions), "rejected": len(validation.rejected_records),
                "new_cost_micro_usd": gateway.cost if gateway else 0}), flush=True)
    except Exception as exc:
        status, error = "STOPPED_NO_AUTOMATIC_RETRY", safe_failure_code(exc)
    counter = ledger.store.get(ledger.run_key(RUN)) if ledger else None
    if counter and status == "STRUCTURAL_CANARY_COMPLETE_REVIEW_PENDING" and (counter["reserved_micro_usd"]
            or counter["settled_micro_usd"] != sum(r["receipt"]["cost_micro_usd"] for r in rows)):
        status, error = "RECONCILIATION_REQUIRED", "SEMANTIC_CANARY_COST_TOTAL_MISMATCH"
    report = {"status": status, "error": error, "processed": len(rows), "planned": len(plan["selection"]),
        "cohorts": dict(Counter(r["cohort"] for r in rows)), "accepted_relationships": sum(len(r["accepted_relationships"]) for r in rows),
        "accepted_permissions": sum(len(r["accepted_permissions"]) for r in rows),
        "rejected_records": sum(len(r["rejected_records"]) for r in rows), "provider_calls_this_invocation": len(gateway.seen) if gateway else 0,
        "new_cost_micro_usd": gateway.cost if gateway else 0, "run_counter": counter,
        "full_batch_executed": False, "serving_assertions_added": 0, "funding_cap_changed": False,
        "finished_at": datetime.now(UTC).isoformat()}
    attempts = len(list(OUTPUT.glob("attempt-*.json"))) + 1
    (OUTPUT / ("replay.json" if replay else f"attempt-{attempts:03d}.json")).write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))
    return 0 if error is None else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    plan, chunks, recipe = freeze()
    if args.execute or args.replay:
        return execute(plan, chunks, recipe, replay=args.replay)
    print(json.dumps({"status": "PLANNED_ONLY", "maximum_calls": plan["maximum_new_calls"],
        "cohorts": dict(Counter(u["cohort"] for u in plan["selection"])), "provider_calls": 0, "aws_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
