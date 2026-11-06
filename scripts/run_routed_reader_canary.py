"""Bounded native routing/reading diagnostic; no answers, deploy or retries."""

import argparse
import json
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.graph_collection_migration import collection_records
from app.model_gateway import CallContext, safe_failure_code
from app.profiled_source_context import SourceContextProfile, render_profiled_context
from app.routed_source_reader import (
    READ_SYSTEM,
    ROUTE_SYSTEM,
    ReaderStageCache,
    build_directory,
    parse_ids,
    routed_catalog,
)
from scripts import run_scoped_flow_canary as parent
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable

ROOT = parent.ROOT
RUN = "routed-reader319-20261105-v1"
OUT = ROOT / "results/routed-reader319-2026-11-05-v1"
CACHE = ROOT / "data/cache/routed-reader319-2026-11-05-v1"


def freeze():
    collection, _, _ = load_migrated()
    docs, _, _ = collection_records(collection)
    tasks = parent.tasks()
    directories = [build_directory(collection, docs, question=t["question"], search=t["search"], language=t["language"])
        for t in tasks]
    scope = {"version": RUN, "collection_sha256": collection.manifest_sha256,
        "tasks": tasks, "directory_fingerprints": [d.fingerprint for d in directories],
        "route_recipe": sha256_bytes(ROUTE_SYSTEM.encode()), "read_recipe": sha256_bytes(READ_SYSTEM.encode()),
        "max_calls": 8, "run_cap_micro_usd": 500000, "published_usage_soft_stop_micro_usd": 80000,
        "funding_cap_micro_usd": 8000000, "exposure": "TWO_EXPOSED_INTENTS_ENGINEERING_DIAGNOSTIC_NOT_BENCHMARK",
        "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_bytes(p.read_bytes()) for p in
            [*(ROOT / "app").rglob("*.py"), Path(__file__)]},
        "price_sha256": sha256_bytes((ROOT / parent.PRICE).read_bytes()),
        "routes_sha256": sha256_bytes((ROOT / "configs/models/external-direct-v1.json").read_bytes()),
        "serving_changed": False, "source_annotations_received_by_runtime": False}
    immutable(OUT / "scope.json", scope)
    print(json.dumps({"status": "FROZEN_NO_DISPATCH", "tasks": len(tasks), "max_calls": 8,
        "directory_bytes": [len(d.message.encode()) for d in directories]}))


def checked_scope():
    scope = json.loads((OUT / "scope.json").read_bytes())
    if (any(sha256_bytes((ROOT / p).read_bytes()) != h for p, h in scope["code_sha256"].items())
            or scope["price_sha256"] != sha256_bytes((ROOT / parent.PRICE).read_bytes())
            or scope["routes_sha256"] != sha256_bytes((ROOT / "configs/models/external-direct-v1.json").read_bytes())):
        raise ValueError("ROUTED_READER_SCOPE_PIN_CHANGED")
    return scope


def managed(scope):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("ROUTED_READER_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    if funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 500000:
        raise ValueError("ROUTED_READER_FUNDING_HEADROOM")
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((ROOT / parent.PRICE).read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region,
        now=datetime.now(UTC), live=True)
    provider = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    metadata = deepseek_preflight(provider)
    if not metadata["available"] or metadata["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("ROUTED_READER_MODEL_METADATA_CHANGED")
    ledger.initialize_month("kg-r72", datetime.now(UTC).strftime("%Y-%m"))
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=scope["run_cap_micro_usd"])
    counter = ledger.store.get(ledger.run_key(RUN))
    if counter["halted"] or counter["reserved_micro_usd"]:
        raise ValueError("ROUTED_READER_RECONCILIATION_REQUIRED")
    immutable(OUT / "preflight.json", {"checked_at": datetime.now(UTC).isoformat(),
        "funding": funding.model_dump(mode="json"), "model_metadata_sha256": metadata["metadata_sha256"],
        "run_cap_micro_usd": 500000, "published_usage_soft_stop_micro_usd": 80000})
    return ExternalModelGateway(ledger, provider, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "FROZEN_ROUTED_READER319_V1"}), prices), ledger


class ControlledGateway:
    def __init__(self, base, scope):
        self.base, self.provider, self.scope = base, base.provider, scope
        self.allowed, self.calls, self.cost, self.halted = {}, 0, 0, False

    def allow(self, context, system, prompt, cap):
        if (context.run_id != RUN or context.attempt != 1 or (context.stage, system, cap) not in
                {("PLAN_QUERY", ROUTE_SYSTEM, 256), ("EXTRACT", READ_SYSTEM, 768)}
                or len(canonical_json({"model": "deepseek-flash", "messages": [
                    {"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "max_tokens": cap, "thinking": {"type": "disabled"}, "temperature": 0,
                    "stream": False, "response_format": {"type": "json_object"}})) > 100000):
            raise ValueError("ROUTED_READER_STAGE_OR_BODY_BOUND")
        self.allowed[context.request_id] = (context.stage, system, prompt, cap)

    def generate(self, context, *, system, prompt, max_output_tokens):
        if (self.halted or self.calls >= 8 or self.cost >= 80000 or context.run_id != RUN
                or self.allowed.pop(context.request_id, None) != (context.stage, system, prompt, max_output_tokens)):
            raise ValueError("ROUTED_READER_DISPATCH_NOT_ALLOWED")
        immutable(OUT / "dispatch-intents" / (context.request_id + ".json"),
            {"context": context.model_dump(mode="json"), "system_sha256": sha256_bytes(system.encode()),
                "prompt_sha256": sha256_bytes(prompt.encode()), "max_output_tokens": max_output_tokens})
        self.calls += 1
        try:
            result = self.base.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
            self.cost += result.actual_micro_usd
            immutable(OUT / "dispatch-receipts" / (result.operation_id + ".json"),
                {"operation_id": result.operation_id, "usage": result.usage.model_dump(mode="json"),
                    "published_cost_micro_usd": result.actual_micro_usd, "raw_output": result.text,
                    "stop_reason": result.stop_reason})
            return result
        except Exception:
            self.halted = True
            raise


def run(replay):
    scope = checked_scope()
    base, ledger = (None, None) if replay else managed(scope)
    gateway = None if replay else ControlledGateway(base, scope)
    collection, _, _ = load_migrated()
    docs, chunks, _ = collection_records(collection)
    route_cache, read_cache = [ReaderStageCache(CACHE, kind=kind) for kind in ("route", "read")]
    rows = []
    try:
        for i, task in enumerate(scope["tasks"]):
            directory = build_directory(collection, docs, question=task["question"], search=task["search"], language=task["language"])
            if directory.fingerprint != scope["directory_fingerprints"][i]:
                raise ValueError("ROUTED_READER_DIRECTORY_PIN")
            route_context = CallContext(tenant_id="demo", run_id=RUN,
                request_id=stable_id("reader-route-v1", directory=directory.fingerprint), stage="PLAN_QUERY")
            if gateway:
                gateway.allow(route_context, ROUTE_SYSTEM, directory.message, 256)
            route = route_cache.resolve_stage(route_context, system=ROUTE_SYSTEM, prompt=directory.message,
                max_output_tokens=256, binding_sha256=directory.fingerprint, recipe_sha256=scope["route_recipe"],
                parser=lambda raw, stop: parse_ids(raw, stop, "document_ids", "d", directory.document_ids, 6),
                gateway=gateway, dispatch=None if gateway is None else gateway.generate)
            catalog = routed_catalog(directory, route.value, docs, chunks)
            read_context = CallContext(tenant_id="demo", run_id=RUN,
                request_id=stable_id("reader-select-v1", catalog=catalog.fingerprint), stage="EXTRACT")
            if gateway:
                gateway.allow(read_context, READ_SYSTEM, catalog.message, 768)
            reading = read_cache.resolve_stage(read_context, system=READ_SYSTEM, prompt=catalog.message,
                max_output_tokens=768, binding_sha256=catalog.fingerprint, recipe_sha256=scope["read_recipe"],
                parser=lambda raw, stop: parse_ids(raw, stop, "source_ids", "r", catalog.source_ids, 12),
                gateway=gateway, dispatch=None if gateway is None else gateway.generate)
            context = render_profiled_context(docs, chunks, reading.value, question=task["question"],
                language=task["language"], tenant_id="demo", collection_sha256=collection.manifest_sha256,
                profile=SourceContextProfile(max_chunks=12), enforce_budget=False)
            row = {"task": task, "route_document_ids": route.value, "directory_sha256": directory.fingerprint,
                "catalog_sha256": catalog.fingerprint, "catalog_chunk_ids": catalog.source_ids,
                "selected_chunk_ids": reading.value, "catalog_message": catalog.message,
                "source_message": context.user_message, "context_fingerprint": context.fingerprint,
                "context_proxy_tokens": context.proxy_tokens, "fits_4096_proxy_budget": context.proxy_tokens <= 4096,
                "route_receipt": route.receipt.model_dump(mode="json"), "read_receipt": reading.receipt.model_dump(mode="json")}
            target = OUT / "cells" / (task["intent"] + "-" + task["language"] + ".json")
            if replay:
                if canonical_json(json.loads(target.read_bytes())) != canonical_json(row):
                    raise ValueError("ROUTED_READER_REPLAY_CHANGED")
            else:
                immutable(target, row)
            rows.append(row)
            print(json.dumps({"intent": task["intent"], "language": task["language"],
                "documents": len(route.value), "catalog_chunks": len(catalog.source_ids), "selected": len(reading.value),
                "fits_budget": row["fits_4096_proxy_budget"], "replay": replay}), flush=True)
        if ledger:
            receipts = [json.loads(p.read_bytes()) for p in (OUT / "dispatch-receipts").glob("*.json")]
            if any(ledger.operation(r["operation_id"]).status != "SETTLED" for r in receipts):
                raise ValueError("ROUTED_READER_LEDGER_NOT_SETTLED")
            if sum(r["published_cost_micro_usd"] for r in receipts) != gateway.cost:
                raise ValueError("ROUTED_READER_COST_MISMATCH")
        report = {"status": "NATIVE_CACHE_REPLAY_PASS" if replay else "NATIVE_ROUTING_READING_RECORDED",
            "tasks": len(rows), "new_calls": 0 if replay else gateway.calls,
            "new_published_cost_micro_usd": 0 if replay else gateway.cost,
            "new_answers": 0, "serving_changed": False, "human_reviewers": 0}
        immutable(OUT / ("replay.json" if replay else "report.json"), report)
        print(json.dumps(report), flush=True)
    except Exception as exc:
        record = {"status": "HALTED_NO_RETRY", "failure": safe_failure_code(exc), "completed_tasks": len(rows),
            "dispatch_intents": 0 if gateway is None else gateway.calls,
            "published_cost_micro_usd": 0 if gateway is None else gateway.cost}
        if not replay:
            immutable(OUT / "halt.json", record)
        print(json.dumps(record), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute", action="store_true")
    action.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.execute or args.replay:
        with ExitStack() as stack:
            if args.replay:
                for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
                    stack.enter_context(patch(name, side_effect=AssertionError("READER_REPLAY_NETWORK_FORBIDDEN")))
            run(args.replay)
    else:
        freeze()


if __name__ == "__main__":
    main()
