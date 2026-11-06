"""V7 source-only qualification: eight development texts and 48 new texts.

Default freezes local artifacts only. Explicit --execute permits a bounded
canary under the already approved audited 8 USD total funding. No deployment,
serving promotion, inference retries or full-corpus dispatch.
"""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.extraction import AliasDictionary
from app.model_gateway import CallContext, safe_failure_code
from app.semantic_extraction_cache_v7 import SemanticExtractionCache
from app.semantic_extraction_v7 import semantic_recipe, semantic_request
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

OUTPUT = ROOT / "results/semantic-graph-v7/qualification-v1"
RUN = "collection316-semantic-v7-qualification-v1"
CUES = re.compile(r"\b(?:uses?|used by|supports?|provides?|requires?|needs?|integrates? with|part of|belongs? to|replaces?)\b", re.I)
DEV_PREFIXES = ("90a8fa", "973e29", "de24b1", "bb78d0", "167288", "df2025", "264d8b", "406187")
CODE = ("app/semantic_extraction_v7.py", "app/semantic_extraction_cache_v7.py", "scripts/run_typed_graph_qualification.py")


def context_for(unit):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=unit["v7_unit_id"], stage="EXTRACT")


def source_selection(units, chunks, excluded):
    groups = defaultdict(list)
    for unit in units:
        if unit["text_sha256"] in excluded:
            continue
        for group in {o["service_group"] for o in unit["occurrences"]}:
            groups[group].append(unit)
    chosen, seen = [], set()
    for round_number in range(2):
        for group, candidates in sorted(groups.items()):
            used_docs = {o["document_version_id"] for row in chosen if row["selected_group"] == group for o in row["occurrences"]}
            def order(unit):
                text = chunks[unit["occurrences"][0]["chunk_id"]].text
                rich = bool(CUES.search(text)) and len(text) >= 250 and "```" not in text
                repeated_document = bool(used_docs.intersection(o["document_version_id"] for o in unit["occurrences"]))
                return (not rich if round_number == 0 else False, repeated_document, unit["text_sha256"])
            match = next((u for u in sorted(candidates, key=order) if u["text_sha256"] not in seen), None)
            if match is None:
                raise ValueError("V7_QUALIFICATION_GROUP_DISTINCT_SOURCE_REQUIRED")
            seen.add(match["text_sha256"])
            chosen.append(dict(match, selected_group=group, selection_round=round_number,
                cohort="SOURCE_SELECTED_EXTRACTION_UNCLAIMED", selection_stratum="RELATION_CUE_PREFERENCE" if round_number == 0 else "HASH_ORDER_CONTROL"))
    return tuple(chosen)


def freeze():
    collection, chunks, units = load_collection_inputs()
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v2.json")
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = semantic_recipe(aliases, metadata_sha256=routes["account_model_metadata_sha256"])
    old_sources = [json.loads(line) for line in (ROOT / "results/semantic-graph-v7/development-source-audit-v1/sources.jsonl").read_bytes().splitlines()]
    dev_hashes = {r["text_sha256"] for r in old_sources if any(r["unit_id"].split("_")[-1].startswith(prefix) for prefix in DEV_PREFIXES)}
    if len(dev_hashes) != 8:
        raise ValueError("V7_EXPECTED_EIGHT_DEVELOPMENT_SOURCES")
    excluded = {r["text_sha256"] for r in old_sources}
    old_cache = ROOT / "data/cache/external-models/extraction.sqlite3"
    with sqlite3.connect(f"file:{old_cache.as_posix()}?mode=ro", uri=True) as db:
        for (claim,) in db.execute("SELECT claim FROM extraction WHERE namespace='LIVE'"):
            excluded.add(json.loads(claim)["input_sha256"])
    selected = [dict(u, cohort="KNOWN_DEVELOPMENT", selection_stratum="KNOWN_ERROR_OR_MISSED_RELATION")
                for u in units if u["text_sha256"] in dev_hashes]
    selected.extend(source_selection(units, chunks, excluded))
    if len(selected) != 56 or len({u["text_sha256"] for u in selected}) != 56:
        raise ValueError("V7_QUALIFICATION_56_DISTINCT_UNITS_REQUIRED")
    for unit in selected:
        unit["v7_unit_id"] = stable_id("semantic-v7-unit", source=collection.manifest_sha256,
            text=unit["text_sha256"], recipe=recipe.fingerprint)
    input_proxy = sum((len(s) + len(p) + 3) // 4 for s, p in
        (semantic_request(chunks[u["occurrences"][0]["chunk_id"]].text) for u in selected))
    full_proxy = sum((len(s) + len(p) + 3) // 4 for s, p in
        (semantic_request(chunks[u["occurrences"][0]["chunk_id"]].text) for u in units))
    plan = {"version": "semantic-v7-source-qualification-v1", "run_id": RUN,
        "recipe": recipe.model_dump(mode="json"), "selection": selected,
        "source_manifest_sha256": collection.manifest_sha256,
        "excluded_texts": len(excluded), "excluded_source_hashes_sha256": sha256_bytes(canonical_json(sorted(excluded))),
        "source_selection": "24 groups: relation-cue preference plus hash control; prefer a second document; never use benchmark labels/questions/scores",
        "maximum_new_calls": 56, "max_output_tokens": 2048, "run_cap_micro_usd": 500000,
        "actual_cost_stop_between_calls_micro_usd": 100000, "active_funding_cap_micro_usd": 8000000,
        "funding_profile": "AUDITED_2_TO_8_USD_V2", "month_cap_micro_usd": 12000000,
        "funding_counter_changes": False, "full_batch_enabled": False, "serving_promotion": False,
        "input_proxy_tokens": input_proxy, "full_inventory_unique_texts": len(units), "full_inventory_input_proxy_tokens": full_proxy,
        "forecast_micro_usd_by_output_tokens_per_unit": {str(n): (full_proxy * 300000 + len(units) * n * 1200000 + 999999) // 1000000 for n in (256, 512, 1024, 2048)},
        "forecast_notes": "4 characters/token proxy, not provider tokens or admission bound; peak cache-miss rates, excludes repairs/review/answers",
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "evaluation_scope": "DEVELOPMENT_AND_SOURCE_QUALIFICATION_NOT_INDEPENDENT_QUESTION_HOLDOUT", "semantic_review": "PENDING"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(plan) + b"\n"
    path = OUTPUT / "plan.json"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("V7_FROZEN_QUALIFICATION_PLAN_CHANGED")
    path.write_bytes(raw)
    inventory_path = OUTPUT / "source-inventory-v7.jsonl"
    inventory_raw = b"".join(canonical_json(dict(u, v7_unit_id=stable_id("semantic-v7-unit",
        source=collection.manifest_sha256, text=u["text_sha256"], recipe=recipe.fingerprint), recipe_sha256=recipe.fingerprint)) + b"\n" for u in units)
    if inventory_path.exists() and inventory_path.read_bytes() != inventory_raw:
        raise ValueError("V7_FROZEN_INVENTORY_CHANGED")
    inventory_path.write_bytes(inventory_raw)
    return plan, chunks, recipe


class ScopedGateway:
    def __init__(self, gateway, plan, chunks):
        self.gateway, self.provider, self.routes = gateway, gateway.provider, gateway.routes
        self.allowed = {context_for(u).model_dump_json(): semantic_request(chunks[u["occurrences"][0]["chunk_id"]].text) for u in plan["selection"]}
        self.seen, self.cost, self.halted = set(), 0, False

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or self.cost >= 100000 or key in self.seen or self.allowed.get(key) != (system, prompt)
                or max_output_tokens != 2048 or len(self.seen) >= 56):
            raise ValueError("V7_DISPATCH_SCOPE_BOUND")
        self.seen.add(key)
        try:
            result = self.gateway.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
        except Exception:
            self.halted = True
            raise
        self.cost += result.actual_micro_usd
        return result


def execute(plan, chunks, recipe, *, replay=False):
    gateway, ledger, before = None, None, None
    if not replay:
        import boto3
        from botocore.config import Config

        from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
        from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
        from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
        from app.ledger import DynamoLedgerStore, Ledger

        prices_raw = (ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()
        if sha256_bytes(prices_raw) != plan["price_sheet_sha256"]:
            raise ValueError("V7_PRICE_SHEET_CHANGED")
        prices = ExternalPriceSheet.model_validate_json(prices_raw)
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=20)
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
            raise ValueError("V7_OPERATOR_IDENTITY")
        ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
        funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
        if funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 500000:
            raise ValueError("V7_AUDITED_FUNDING_HEADROOM_REQUIRED")
        month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
        if not month or month["halted"] or month["cap_micro_usd"] != 12000000:
            raise ValueError("V7_EXISTING_MONTH_BUDGET_REQUIRED")
        before = funding.model_dump(mode="json")
        ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
        run = ledger.store.get(ledger.run_key(RUN))
        if run["halted"] or run["reserved_micro_usd"] or run["cap_micro_usd"] != 500000:
            raise ValueError("V7_RUN_RECONCILIATION_REQUIRED")
        transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
        account = deepseek_preflight(transport)
        if not account["available"] or account["metadata_sha256"] != routes.account_model_metadata_sha256:
            raise ValueError("V7_PROVIDER_CHANGED")
        gateway = ScopedGateway(ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
            "live_inference_gate": "TYPED_SCOPE_V7_SOURCE_QUALIFICATION_ONLY"}), prices), plan, chunks)
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v7", namespace="LIVE")
    directory = OUTPUT / "units"
    directory.mkdir(exist_ok=True)
    rows, status, error = [], "REPLAY_MATCH" if replay else "NATIVE_V7_QUALIFICATION_COMPLETE_REVIEW_PENDING", None
    try:
        for unit in plan["selection"]:
            text = chunks[unit["occurrences"][0]["chunk_id"]].text
            if sha256_bytes(text.encode()) != unit["text_sha256"]:
                raise ValueError("V7_SOURCE_CHANGED")
            lookup = cache.resolve(context_for(unit), text=text, recipe=recipe, gateway=gateway)
            row = {"unit_id": unit["v7_unit_id"], "text_sha256": unit["text_sha256"], "cohort": unit["cohort"],
                "occurrences": unit["occurrences"], "receipt": lookup.receipt,
                "accepted_relationships": lookup.validation.accepted_relationships,
                "accepted_permissions": lookup.validation.accepted_permissions, "rejected_records": lookup.validation.rejected_records,
                "coverage_note": lookup.validation.output.coverage_note, "semantic_review": "PENDING_NOT_SERVING"}
            raw, path = canonical_json(row) + b"\n", directory / (unit["v7_unit_id"] + ".json")
            if path.exists() and path.read_bytes() != raw:
                raise ValueError("V7_REPLAY_UNIT_CHANGED")
            path.write_bytes(raw)
            rows.append(row)
            print(json.dumps({"processed": len(rows), "cohort": row["cohort"], "relations": len(row["accepted_relationships"]),
                "permissions": len(row["accepted_permissions"]), "rejected": len(row["rejected_records"]),
                "new_cost_micro_usd": gateway.cost if gateway else 0}), flush=True)
    except Exception as exc:
        status, error = "STOPPED_NO_AUTOMATIC_RETRY", safe_failure_code(exc)
    counter = ledger.store.get(ledger.run_key(RUN)) if ledger else None
    if counter and status == "NATIVE_V7_QUALIFICATION_COMPLETE_REVIEW_PENDING" and (counter["reserved_micro_usd"]
            or counter["settled_micro_usd"] != sum(r["receipt"]["cost_micro_usd"] for r in rows)):
        status, error = "RECONCILIATION_REQUIRED", "V7_COST_TOTAL_MISMATCH"
    report = {"status": status, "error": error, "processed": len(rows), "planned": len(plan["selection"]),
        "cohorts": dict(Counter(r["cohort"] for r in rows)),
        "accepted_relationships": sum(len(r["accepted_relationships"]) for r in rows),
        "accepted_permissions": sum(len(r["accepted_permissions"]) for r in rows),
        "rejected_records": sum(len(r["rejected_records"]) for r in rows),
        "coverage_notes": dict(Counter(r["coverage_note"] for r in rows)),
        "provider_calls_this_invocation": len(gateway.seen) if gateway else 0, "new_cost_micro_usd": gateway.cost if gateway else 0,
        "run_counter": counter, "funding_before": before, "full_batch_executed": False, "serving_records_added": 0,
        "funding_cap_changed": False, "finished_at": datetime.now(UTC).isoformat()}
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
    print(json.dumps({"status": "FROZEN_LOCAL_ONLY", "source_units": len(plan["selection"]),
        "full_inventory_unique_texts": plan["full_inventory_unique_texts"],
        "full_inventory_input_proxy_tokens": plan["full_inventory_input_proxy_tokens"],
        "forecast_micro_usd_by_output_tokens_per_unit": plan["forecast_micro_usd_by_output_tokens_per_unit"],
        "model_calls": 0, "aws_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
