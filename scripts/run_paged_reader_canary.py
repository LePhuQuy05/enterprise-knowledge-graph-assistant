"""Native S3 EN/VI route -> pages -> consolidation; no model answers yet."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.graph_collection_migration import collection_records
from app.model_gateway import CallContext, safe_failure_code
from app.paged_source_reader_v3 import PAGE_SYSTEM, catalog_pages, consolidation_catalog
from app.profiled_source_context import SourceContextProfile, render_profiled_context
from app.routed_reader_json_v2 import READ_SYSTEM, ROUTE_SYSTEM
from app.routed_source_reader import ReaderStageCache, build_directory, parse_ids
from scripts import run_scoped_flow_canary as parent
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.reader_canary_runtime import managed_job

ROOT = parent.ROOT
RUN = "paged-reader319-s3-20261105-v3"
OUT = ROOT / "results/paged-reader319-s3-2026-11-05-v3"
CACHE = ROOT / "data/cache/paged-reader319-s3-2026-11-05-v3"


def protocols():
    return [(ROUTE_SYSTEM, "PLAN_QUERY", 256), (PAGE_SYSTEM, "EXTRACT", 768), (READ_SYSTEM, "EXTRACT", 768)]


def freeze():
    collection, _, _ = load_migrated()
    scope = {"version": RUN, "collection_sha256": collection.manifest_sha256,
        "tasks": [t for t in parent.tasks() if t["intent"] == "s3-permission-layers"],
        "allowed_protocols": protocols(), "max_calls": 20, "max_pages_per_task": 8,
        "page_bytes": 45000, "page_selection_cap": 4, "final_selection_cap": 12,
        "run_cap_micro_usd": 500000, "published_usage_soft_stop_micro_usd": 100000,
        "shared_cap_micro_usd": 8000000, "price_file": parent.PRICE,
        "price_sha256": sha256_bytes((ROOT / parent.PRICE).read_bytes()),
        "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_bytes(p.read_bytes()) for p in
            [*(ROOT / "app").rglob("*.py"), Path(__file__), ROOT / "scripts/reader_canary_runtime.py"]},
        "quality_scope": "ONE_EXPOSED_S3_INTENT_EN_VI_ENGINEERING_DIAGNOSTIC_NOT_BENCHMARK",
        "serving_changed": False, "new_model_answers": 0}
    immutable(OUT / "scope.json", scope)
    print(json.dumps({"status": "FROZEN_NO_DISPATCH", "tasks": 2, "max_calls": 20}))


def checked_scope():
    scope = json.loads((OUT / "scope.json").read_bytes())
    if any(sha256_bytes((ROOT / p).read_bytes()) != h for p, h in scope["code_sha256"].items()):
        raise ValueError("PAGED_READER_CODE_PIN")
    if scope["price_sha256"] != sha256_bytes((ROOT / scope["price_file"]).read_bytes()):
        raise ValueError("PAGED_READER_PRICE_PIN")
    scope["allowed_protocols"] = [tuple(p) for p in scope["allowed_protocols"]]
    return scope


def run(replay):
    scope = checked_scope()
    job = None if replay else managed_job(ROOT, OUT, RUN, scope)
    collection, _, _ = load_migrated()
    docs, chunks, _ = collection_records(collection)
    caches = {kind: ReaderStageCache(CACHE, kind=kind) for kind in ("route", "read")}
    completed = []

    def stage(kind, fingerprint, system, prompt, keys, prefix, cap, output_cap):
        context = CallContext(tenant_id="demo", run_id=RUN,
            request_id=stable_id("paged-reader-stage-v3", kind=kind, fingerprint=fingerprint, system=sha256_bytes(system.encode())),
            stage="PLAN_QUERY" if kind == "route" else "EXTRACT")
        if job:
            job.allow(context, system, prompt, output_cap)
        return caches[kind].resolve_stage(context, system=system, prompt=prompt, max_output_tokens=output_cap,
            binding_sha256=fingerprint, recipe_sha256=sha256_bytes(system.encode()),
            parser=lambda raw, stop: parse_ids(raw, stop, "document_ids" if kind == "route" else "source_ids", prefix, keys, cap),
            gateway=job, dispatch=None if job is None else job.generate)

    try:
        for t in scope["tasks"]:
            directory = build_directory(collection, docs, question=t["question"], search=t["search"], language=t["language"])
            route = stage("route", directory.fingerprint, ROUTE_SYSTEM, directory.message, directory.document_ids, "d", 6, 256)
            pages = catalog_pages(directory, route.value, docs, chunks)
            page_records, selected_by_page = [], []
            for page in pages:
                selected = stage("read", page.fingerprint, PAGE_SYSTEM, page.message, page.source_ids, "r", 4, 768)
                selected_by_page.append(selected.value)
                record = {"catalog_sha256": page.fingerprint, "chunk_ids": page.source_ids,
                    "selected_ids": selected.value, "receipt": selected.receipt.model_dump(mode="json")}
                page_records.append(record)
                immutable(OUT / "pages" / (t["language"] + "-" + str(len(page_records)) + ".json"), record)
            catalog = consolidation_catalog(directory, selected_by_page, pages, docs, chunks)
            final = stage("read", catalog.fingerprint, READ_SYSTEM, catalog.message, catalog.source_ids, "r", 12, 768)
            context = render_profiled_context(docs, chunks, final.value, question=t["question"], language=t["language"],
                tenant_id="demo", collection_sha256=collection.manifest_sha256,
                profile=SourceContextProfile(max_chunks=12), enforce_budget=False)
            row = {"task": t, "directory_sha256": directory.fingerprint, "route_document_ids": route.value,
                "route_receipt": route.receipt.model_dump(mode="json"), "pages": page_records,
                "catalog_sha256": catalog.fingerprint, "catalog_ids": catalog.source_ids,
                "selected_ids": final.value, "reading_receipt": final.receipt.model_dump(mode="json"),
                "source_message": context.user_message, "context_fingerprint": context.fingerprint,
                "context_proxy_tokens": context.proxy_tokens, "fits_budget": context.proxy_tokens <= 4096,
                "whole_chunk_truncations": 0}
            target = OUT / "cells" / (t["intent"] + "-" + t["language"] + ".json")
            if replay and canonical_json(json.loads(target.read_bytes())) != canonical_json(row):
                raise ValueError("PAGED_READER_EXACT_REPLAY_MISMATCH")
            if not replay:
                immutable(target, row)
            completed.append(row)
            print(json.dumps({"language": t["language"], "documents": len(route.value), "pages": len(pages),
                "catalog_chunks": sum(len(p.source_ids) for p in pages), "selected": len(final.value),
                "context_proxy_tokens": context.proxy_tokens, "fits_budget": row["fits_budget"], "replay": replay}), flush=True)
        report = {"status": "EXACT_PAGED_NATIVE_REPLAY_PASS" if replay else "NATIVE_PAGED_READING_RECORDED",
            "tasks": len(completed), "new_model_answers": 0, "serving_changed": False,
            "human_reviewers": 0} | ({"new_calls": 0, "published_cost_micro_usd": 0} if job is None else job.reconciled())
        immutable(OUT / ("replay.json" if replay else "report.json"), report)
        print(json.dumps(report), flush=True)
    except Exception as exc:
        report = {"status": "HALTED_NO_AUTOMATIC_RETRY", "failure": safe_failure_code(exc),
            "completed_tasks": len(completed), "dispatch_intents": 0 if job is None else job.calls,
            "published_cost_micro_usd": 0 if job is None else job.cost}
        if not replay:
            immutable(OUT / "halt.json", report)
        print(json.dumps(report), flush=True)
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.execute or args.replay:
        with ExitStack() as stack:
            if args.replay:
                for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
                    stack.enter_context(patch(name, side_effect=AssertionError("PAGED_READER_NETWORK_FORBIDDEN")))
            run(args.replay)
    else:
        freeze()
