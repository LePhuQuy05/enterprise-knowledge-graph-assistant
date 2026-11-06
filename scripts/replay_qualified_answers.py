"""Replay real V14 receipts through the common display gate, with no network."""

import argparse
import json
from contextlib import ExitStack
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes
from app.generation import AnswerBinding
from app.graph_collection_migration import collection_records
from app.model_gateway import CallContext
from app.qualified_answer_service import resolve_qualified_answer
from app.routed_source_reader import ReaderStageCache
from app.task_scoped_generation_v14 import TaskScopedRecipeV14
from scripts import run_task_scope_v14_canary as native
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable

OUT = native.ROOT / "results/qualified-answer-replay-2026-11-05-v2"


def run(replay):
    scope = native.checked_scope()
    collection, _, _ = load_migrated()
    docs, chunks, _ = collection_records(collection)
    cache = ReaderStageCache(native.CACHE, kind="answer-v13")
    cache_before = sha256_bytes(cache.path.read_bytes())
    recipe = TaskScopedRecipeV14.model_validate_json(canonical_json(scope["generation_recipe"]))
    code = ("app/answer_qualification.py", "app/qualified_answer_service.py", "app/citation_antecedents.py",
        "scripts/replay_qualified_answers.py")
    input_pins = {name: sha256_bytes((native.OUT / "cells" / name).read_bytes()) for name in native.INPUTS}
    manifest = {"version": "qualified-answer-replay-v2", "raw_native_scope": scope["version"],
        "source_sha256": collection.manifest_sha256, "inputs_sha256": input_pins,
        "code_sha256": {p: sha256_bytes((native.ROOT / p).read_bytes()) for p in code},
        "new_model_calls": 0, "new_aws_calls": 0, "serving_changed": False}
    immutable(OUT / "scope.json", manifest)
    rows = []
    for name, original in zip(native.INPUTS, native.inputs(), strict=True):
        saved = json.loads((native.OUT / "cells" / name).read_bytes())
        task = saved["task"]
        source = native.context_for(original, collection, docs, chunks)
        response = saved["response_v14"]
        call = CallContext(tenant_id="demo", run_id=native.RUN, request_id=response["request_id"], stage="GENERATE")
        binding = AnswerBinding(request_id=call.request_id, tenant_id="demo", owner_id="v14-local-review",
            mode_requested=saved["method"], mode_used=saved["method"], used_graph=False,
            warnings=("EXPERIMENTAL_TASK_SCOPE_V14_FIXED_CONTEXT_NOT_SERVING_APPROVED",))
        result = resolve_qualified_answer(call, source_context=source, documents=docs, chunks=chunks, binding=binding,
            recipe=recipe, search=task["search"], raw_cache=cache)
        if not result.cache_hit or result.new_cost_micro_usd or result.receipt.model_dump(mode="json") != saved["receipt"]:
            raise ValueError("QUALIFIED_REPLAY_NATIVE_RECEIPT_CHANGED_OR_NEW_CALL")
        if result.qualification.native_response.model_dump(mode="json") != response:
            raise ValueError("QUALIFIED_REPLAY_RAW_RESPONSE_REWRITTEN")
        row = {"input": name, "public_payload": result.public_payload(),
            "owner_bound_review": result.qualification.review_record(),
            "raw_operation_id": result.receipt.operation_id, "raw_cache_hit": True,
            "raw_output_sha256": sha256_bytes(result.receipt.raw_output.encode()),
            "new_model_calls": 0, "new_cost_micro_usd": 0}
        target = OUT / "cells" / name
        if replay and canonical_json(json.loads(target.read_bytes())) != canonical_json(row):
            raise ValueError("QUALIFIED_REPLAY_OUTPUT_CHANGED")
        if not replay:
            immutable(target, row)
        rows.append(row)
        print(json.dumps({"cell": name, "qualification": result.qualification.status,
            "flags": [f.rule_id for f in result.qualification.flags], "cache_hit": True}), flush=True)
    if sha256_bytes(cache.path.read_bytes()) != cache_before:
        raise ValueError("QUALIFIED_REPLAY_RAW_SQLITE_CHANGED")
    bad = next(r for r in rows if r["input"] == "s3-permission-layers-vi-hybrid_multi_query.json")
    if bad["public_payload"]["response"] is not None or bad["public_payload"]["qualification"] != "REVIEW_REQUIRED":
        raise ValueError("QUALIFIED_REPLAY_KNOWN_BAD_RESPONSE_NOT_WITHHELD")
    report = {"status": "QUALIFIED_NATIVE_CACHE_REPLAY_PASS", "cells": len(rows), "native_cache_hits": len(rows),
        "review_required": sum(r["public_payload"]["qualification"] == "REVIEW_REQUIRED" for r in rows),
        "no_mechanical_flags": sum(r["public_payload"]["qualification"] == "NO_MECHANICAL_FLAGS" for r in rows),
        "raw_sqlite_sha256_unchanged": cache_before, "new_model_calls": 0, "new_aws_calls": 0,
        "semantic_accuracy_measured": False, "serving_changed": False}
    immutable(OUT / ("replay.json" if replay else "report.json"), report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(name, side_effect=AssertionError("QUALIFIED_REPLAY_NETWORK_FORBIDDEN")))
        run(args.replay)
