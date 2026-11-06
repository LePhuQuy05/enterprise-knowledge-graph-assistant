"""Same frozen contexts, native V13 versus recorded V12; no retrieval retuning."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import AnswerBinding
from app.graph_collection_migration import collection_records
from app.model_gateway import CallContext, safe_failure_code
from app.profiled_generation import ProfiledGenerationRecipe, resolve_profiled_answer
from app.profiled_source_context import SourceContextProfile, render_profiled_context
from app.routed_source_reader import ReaderStageCache
from app.task_scoped_generation_v13 import (
    TaskScopedGateway,
    TaskScopedRecipeV13,
    parse_task_scoped_answer,
    task_scoped_request,
)
from scripts import run_scoped_flow_canary as parent
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.reader_canary_runtime import managed_job

ROOT = parent.ROOT
RUN = "task-scope-v13-fixedcontext-20261105-v1"
OUT = ROOT / "results/task-scope-v13-2026-11-05-v1"
CACHE = ROOT / "data/cache/task-scope-v13-2026-11-05-v1"
INPUTS = ("kms-cross-account-vi-hybrid.json", "kms-cross-account-vi-hybrid_multi_query.json",
    "s3-permission-layers-en-hybrid_multi_query.json", "s3-permission-layers-vi-hybrid_multi_query.json")


def inputs():
    return [json.loads((parent.OUT / "cells" / name).read_bytes()) for name in INPUTS]


def context_for(row, collection, docs, chunks):
    t = row["task"]
    context = render_profiled_context(docs, chunks, tuple(row["context_chunk_ids"]), question=t["question"],
        language=t["language"], tenant_id="demo", collection_sha256=collection.manifest_sha256,
        profile=SourceContextProfile(max_chunks=12))
    if context.fingerprint != row["context_fingerprint"]:
        raise ValueError("TASK_V13_FIXED_CONTEXT_CHANGED")
    return context


def freeze():
    c, _, _ = load_migrated()
    docs, chunks, _ = collection_records(c)
    metadata = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())["account_model_metadata_sha256"]
    recipe = TaskScopedRecipeV13(model_metadata_sha256=metadata)
    systems = [task_scoped_request(context_for(row, c, docs, chunks), docs, chunks, recipe, search=row["task"]["search"])[0]
        for row in inputs()]
    scope = {"version": RUN, "collection_sha256": c.manifest_sha256,
        "generation_recipe": recipe.model_dump(mode="json"), "max_calls": 4,
        "allowed_protocols": [(s, "GENERATE", 1600) for s in dict.fromkeys(systems)],
        "inputs_sha256": {name: sha256_bytes((parent.OUT / "cells" / name).read_bytes()) for name in INPUTS},
        "code_sha256": {str(p.relative_to(ROOT)).replace("\\", "/"): sha256_bytes(p.read_bytes()) for p in
            [*(ROOT / "app").rglob("*.py"), Path(__file__), ROOT / "scripts/reader_canary_runtime.py"]},
        "price_file": parent.PRICE, "price_sha256": sha256_bytes((ROOT / parent.PRICE).read_bytes()),
        "retrieval_and_context_changed": False, "route_or_reader_used": False,
        "quality_scope": "FOUR_EXPOSED_FIXED_CONTEXTS_NOT_INDEPENDENT_BENCHMARK", "serving_changed": False}
    immutable(OUT / "scope.json", scope)
    print(json.dumps({"status": "FROZEN_NO_DISPATCH", "max_calls": 4}))


def checked_scope():
    scope = json.loads((OUT / "scope.json").read_bytes())
    if (any(sha256_bytes((ROOT / p).read_bytes()) != h for p, h in scope["code_sha256"].items())
            or any(sha256_bytes((parent.OUT / "cells" / p).read_bytes()) != h for p, h in scope["inputs_sha256"].items())
            or sha256_bytes((ROOT / scope["price_file"]).read_bytes()) != scope["price_sha256"]):
        raise ValueError("TASK_V13_FROZEN_INPUT_OR_CODE_CHANGED")
    scope["allowed_protocols"] = [tuple(p) for p in scope["allowed_protocols"]]
    return scope


def run(replay):
    scope = checked_scope()
    job = None if replay else managed_job(ROOT, OUT, RUN, scope)
    collection, _, _ = load_migrated()
    docs, chunks, _ = collection_records(collection)
    recipe = TaskScopedRecipeV13.model_validate_json(canonical_json(scope["generation_recipe"]))
    cache = ReaderStageCache(CACHE, kind="answer-v13")
    cells = []
    try:
        for name, row in zip(INPUTS, inputs(), strict=True):
            task = row["task"]
            context = context_for(row, collection, docs, chunks)
            request = stable_id("task-scope-v13", parent=row["response"]["request_id"],
                context=context.fingerprint, recipe=recipe.fingerprint, owner="v13-local-review")
            call = CallContext(tenant_id="demo", run_id=RUN, request_id=request, stage="GENERATE")
            binding = AnswerBinding(request_id=request, tenant_id="demo", owner_id="v13-local-review",
                mode_requested=row["method"], mode_used=row["method"], used_graph=False,
                warnings=("EXPERIMENTAL_TASK_SCOPE_V13_FIXED_CONTEXT_NOT_SERVING_APPROVED",))
            system, prompt = task_scoped_request(context, docs, chunks, recipe, search=task["search"])

            def parse(raw, stop):
                output = parse_task_scoped_answer(raw, stop, question=task["question"], search=task["search"])
                return resolve_profiled_answer(output, context, docs, chunks, binding=binding,
                    recipe=ProfiledGenerationRecipe(model_metadata_sha256=recipe.model_metadata_sha256))

            if job:
                typed = TaskScopedGateway(job.base)
                job.allow(call, system, prompt, 1600, dispatcher=lambda: typed.generate_task_scoped(call,
                    source_context=context, documents=docs, chunks=chunks, recipe=recipe, search=task["search"]))
            answer = cache.resolve_stage(call, system=system, prompt=prompt, max_output_tokens=1600,
                binding_sha256=sha256_bytes(canonical_json(binding.model_dump(mode="json"))), recipe_sha256=recipe.fingerprint,
                parser=parse, gateway=job, dispatch=None if job is None else job.generate)
            cell = {"parent_cell": name, "task": task, "method": row["method"], "recipe_sha256": recipe.fingerprint,
                "context_fingerprint": context.fingerprint, "context_chunk_ids": context.chunk_ids,
                "baseline_v12": row["response"], "response_v13": answer.value.model_dump(mode="json"),
                "receipt": answer.receipt.model_dump(mode="json"), "context_unchanged": True,
                "full_user_message_proxy_budget_checked": True}
            target = OUT / "cells" / name
            if replay and canonical_json(json.loads(target.read_bytes())) != canonical_json(cell):
                raise ValueError("TASK_V13_EXACT_REPLAY_MISMATCH")
            if not replay:
                immutable(target, cell)
            cells.append(cell)
            print(json.dumps({"cell": name, "baseline_status": row["response"]["answer_status"],
                "v13_status": answer.value.answer_status, "limitations": [g.question_part for g in
                    parse_task_scoped_answer(answer.receipt.raw_output, answer.receipt.stop_reason,
                        question=task["question"], search=task["search"]).limitations], "replay": replay}), flush=True)
        report = {"status": "EXACT_V13_CACHE_REPLAY_PASS" if replay else "NATIVE_FIXED_CONTEXT_V13_RECORDED",
            "cells": len(cells), "source_contexts_unchanged": True, "human_reviewers": 0,
            "answer_accuracy_measured": False, "serving_changed": False} | (
                {"new_calls": 0, "published_cost_micro_usd": 0} if job is None else job.reconciled())
        immutable(OUT / ("replay.json" if replay else "report.json"), report)
        print(json.dumps(report), flush=True)
    except Exception as exc:
        record = {"status": "HALTED_NO_AUTOMATIC_RETRY", "failure": safe_failure_code(exc), "completed_cells": len(cells),
            "dispatch_intents": 0 if job is None else job.calls, "published_cost_micro_usd": 0 if job is None else job.cost}
        if not replay:
            immutable(OUT / "halt.json", record)
        print(json.dumps(record), flush=True)
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
                    stack.enter_context(patch(name, side_effect=AssertionError("TASK_V13_REPLAY_NETWORK_FORBIDDEN")))
            run(args.replay)
    else:
        freeze()
