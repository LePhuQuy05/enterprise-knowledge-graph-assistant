"""Freeze/replay source-message packing on real corpus316, without inference.

Two sections: 22 exposed graph controls, and 16 EN/VI method preparations using
existing native planner/rewrite receipts. Changed contexts have no new answer
receipts: cached answers from the old experiment are never relabelled here.
"""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from app.authorization_evidence import load_authorization_facts
from app.authorization_retrieval import combine_evidence_graph, prepare_authorization_answer
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.generation import generation_request
from app.ingestion.text import ProxyTokenizer
from app.reviewed_source_graph import load_reviewed_source_graph
from app.source_message_runtime import (
    prepare_source_message_graph,
    repack_source_answer,
    repack_strict_query_answer,
)
from scripts.check_authorization_evidence import CONTROLS
from scripts.pilot_reviewed_source_answers import (
    code_hashes as frozen_answer_code_hashes,
)
from scripts.pilot_reviewed_source_answers import prepare_cases
from scripts.prepare_authorization_evidence import ROOT, load_sources
from scripts.run_collection_benchmark import OfflineRuntime, load_inputs

CODE = ("app/source_message_packing.py", "app/source_message_runtime.py", "scripts/compare_source_message_packing.py",
    "app/authorization_retrieval.py", "app/authorization_evidence.py", "app/context.py", "app/generation.py")


def compare(output, *, replay=False):
    collection, chunks, documents = load_sources()
    _, _, _, _, pipeline = load_inputs()
    old_config = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    base_graph = load_reviewed_source_graph(ROOT / old_config["release_path"], collection,
        trusted_sha256=old_config["release_sha256"], allow_experimental=True)
    auth_config = json.loads((ROOT / "configs/graph/authorization-evidence-local-v1.json").read_bytes())
    auth = load_authorization_facts(ROOT / auth_config["release_path"], collection, chunks, documents,
        trusted_sha256=auth_config["release_sha256"], allow_experimental=True)
    graph = combine_evidence_graph(collection, base_graph, auth)
    old_scope = json.loads((ROOT / "results/reviewed-source-answers-v1/scope.json").read_bytes())
    if frozen_answer_code_hashes() != old_scope["code_sha256"]:
        raise ValueError("SOURCE_PACKING_FROZEN_PAID_CODE_CHANGED")
    plan = {"version": "source-message-packing-comparison-v1", "purpose": "DEVELOPMENT_CONTEXT_ONLY_NOT_ANSWER_QUALITY",
        "source_manifest_sha256": collection.manifest_sha256, "combined_graph_release_sha256": graph.release_sha256,
        "controls": CONTROLS, "native_preparation_scope_sha256": sha256_bytes(canonical_json(old_scope)),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "shared_limits": {"max_chunks": 8, "generation_user_message_proxy_tokens": 4096, "output_tokens": 800},
        "new_provider_calls": 0, "aws_calls": 0, "deploy": False}
    plan_raw = canonical_json(plan) + b"\n"
    if replay:
        if (output / "plan.json").read_bytes() != plan_raw:
            raise ValueError("SOURCE_PACKING_PLAN_CHANGED")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "plan.json").write_bytes(plan_raw)
    rows, timings = [], []
    t = ProxyTokenizer()
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))

        def vector(text):
            lookup = runtime.cache.resolve_local("demo", text=query_embedding_input(text), recipe=runtime.query_recipe, encoder=runtime.encoder)
            return np.asarray(lookup.record.vector, dtype=np.float32)

        def record(name, section, view, before, after, trace, language="en", before_trace=None):
            old_system, old_prompt = generation_request(before.question, view, before.packed, "demo", before.recipe, answer_language=language)
            new_view, new_prepared = after
            system, prompt = generation_request(new_prepared.question, new_view, new_prepared.packed, "demo", new_prepared.recipe, answer_language=language)
            if (old_system != system or new_prepared.packed.serialized != prompt
                    or new_prepared.packed.proxy_tokens != t.count(prompt) or t.count(prompt) > 4096
                    or new_prepared.binding.mode_used != before.binding.mode_used
                    or new_prepared.binding.request_id == before.binding.request_id):
                raise ValueError("SOURCE_PACKING_MESSAGE_OR_METHOD_BINDING")
            packing = trace["source_message_packing"] if "source_message_packing" in trace else trace
            if packing["policy"] != "GENERATION_USER_MESSAGE_BUDGET_ATOMIC_SOURCE_PATHS_V1":
                raise ValueError("SOURCE_PACKING_COMMON_POLICY_REQUIRED")
            rows.append({"case": name, "section": section, "language": language, "method": before.binding.mode_used,
                "before": {"chunk_ids": before.packed.chunk_ids, "assertion_ids": before.packed.assertion_ids,
                    "packer_proxy_tokens": before.packed.proxy_tokens, "actual_user_message_proxy_tokens": t.count(old_prompt),
                    "request_id": before.binding.request_id, "trace": before_trace},
                "after": {"chunk_ids": new_prepared.packed.chunk_ids, "assertion_ids": new_prepared.packed.assertion_ids,
                    "actual_user_message_proxy_tokens": t.count(prompt), "request_id": new_prepared.binding.request_id,
                    "user_message": json.loads(prompt), "system_sha256": sha256_bytes(system.encode()),
                    "user_message_sha256": sha256_bytes(prompt.encode()), "trace": trace},
                "source_provenance_retained": True, "answer_generation_executed": False,
                "native_preparation_reused_for_context_only": section == "NATIVE_PREPARATION_MATRIX"})

        for name, question in CONTROLS:
            v = vector(question)
            for hops in (1, 2):
                params = dict(tenant_id="demo", owner_id="source-message-comparison", run_id="source-message-v1",
                    recipe=runtime.recipe, hops=hops, allow_experimental=True)
                before = prepare_authorization_answer(collection, lexical, graph, question, v, **params)
                started = perf_counter()
                after = prepare_source_message_graph(collection, lexical, graph, question, v, **params)
                timings.append({"case": name, "method": f"graph_{hops}hop", "new_prepare_ms": round((perf_counter() - started) * 1000, 3)})
                if before.prepared is None or after.prepared is None:
                    raise ValueError("SOURCE_PACKING_GRAPH_MECHANISM_FAILED")
                record(name, "AUTHORIZATION_GRAPH_CONTROLS", before.view, before.prepared,
                    (after.view, after.prepared), after.trace, before_trace=before.trace)
            print(json.dumps({"case": name, "rows": len(rows)}), flush=True)
        # All three prep receipts already exist; gateway=None makes cache misses
        # explicit errors and cannot dispatch a new model call.
        cases, preparations = prepare_cases(collection, lexical, base_graph, runtime, gateway=None)
        for case in cases:
            method, language, p = case["method"], case["language"], case["prepared"]
            if method.startswith("graph_"):
                # Compare packing only: both before/after use the same NEW graph.
                params = dict(tenant_id="demo", owner_id="source-answer-local", run_id=p.context.run_id,
                    recipe=runtime.recipe, hops=int(method[6]), allow_experimental=True,
                    search_query=case["trace"]["search_query"], answer_language=language)
                v = vector(params["search_query"])
                before = prepare_authorization_answer(collection, lexical, graph, p.question, v, **params)
                after = prepare_source_message_graph(collection, lexical, graph, p.question, v, **params)
                record(case["id"], "NATIVE_PREPARATION_MATRIX", before.view, before.prepared,
                    (after.view, after.prepared), after.trace, language, before.trace)
            elif method.startswith("hybrid_multi_query"):
                # Rebuild the validated strict wrapper from the saved native plan;
                # no planner, subquery or query-balanced selection is synthesized.
                from app.method_execution import MethodExecutionV2, StrictPreparedMethod

                original = StrictPreparedMethod(MethodExecutionV2.model_validate_json(canonical_json(case["outcome"])),
                    case["view"], p, case["trace"], language)
                after = repack_strict_query_answer(original)
                record(case["id"], "NATIVE_PREPARATION_MATRIX", case["view"], p,
                    (after.view, after.prepared), after.trace, language, case["trace"])
            else:
                after, trace = repack_source_answer(case["view"], p, answer_language=language)
                record(case["id"], "NATIVE_PREPARATION_MATRIX", case["view"], p, (case["view"], after), trace, language)
            print(json.dumps({"case": case["id"], "rows": len(rows)}), flush=True)
    if plan["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
        raise ValueError("SOURCE_PACKING_CODE_CHANGED_DURING_RUN")
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    summary = {"status": "SOURCE_MESSAGE_CONTEXT_COMPARISON_COMPLETE_NOT_NEW_ANSWERS", "rows": len(rows),
        "sections": dict(Counter(r["section"] for r in rows)), "native_preparation_receipts_reused": len(preparations),
        "new_model_calls": 0, "aws_calls": 0, "new_cost_micro_usd": 0, "human_reviewed": 0,
        "local_onnx_query_calls": runtime.encoder.inference_calls, "retrieval_methods_changed": False,
        "source_count_increased": sum(len(r["after"]["chunk_ids"]) > len(r["before"]["chunk_ids"]) for r in rows),
        "source_count_decreased": sum(len(r["after"]["chunk_ids"]) < len(r["before"]["chunk_ids"]) for r in rows),
        "records": [{"case": r["case"], "section": r["section"], "method": r["method"],
            "before_sources": len(r["before"]["chunk_ids"]), "after_sources": len(r["after"]["chunk_ids"]),
            "before_packer_proxy_tokens": r["before"]["packer_proxy_tokens"],
            "before_user_message_proxy_tokens": r["before"]["actual_user_message_proxy_tokens"],
            "after_user_message_proxy_tokens": r["after"]["actual_user_message_proxy_tokens"]} for r in rows],
        "receipts_sha256": sha256_bytes(raw), "serving_published": False, "quality_accuracy_measured": False}
    if replay:
        if (output / "receipts.jsonl").read_bytes() != raw:
            raise ValueError("SOURCE_PACKING_REPLAY_MISMATCH")
        (output / "replay.json").write_bytes(canonical_json({"status": "EXACT_CONTEXT_AND_PROMPT_REPLAY_MATCH",
            "rows": len(rows), "receipts_sha256": sha256_bytes(raw), "new_model_calls": 0, "aws_calls": 0}) + b"\n")
    else:
        (output / "receipts.jsonl").write_bytes(raw)
        (output / "report.json").write_bytes(canonical_json(summary) + b"\n")
        (output / "timing.json").write_bytes(canonical_json({"backend": "LOCAL_WINDOWS_CPU", "rows": timings}) + b"\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    compare(args.output_dir, replay=args.replay)
