"""Second allocation experiment after V1 lost the exposed VI READ witness.

Compare both two/four source quotas with a document-diverse tail. Keep the V1
failure and all source swaps visible. Controls are exposed development inputs.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.generation import generation_request
from app.graph_document_allocation import reallocate_graph_documents
from app.ingestion.text import ProxyTokenizer
from app.method_execution import ReviewedGraphExecutionV3, StrictPreparedMethod
from app.source_message_runtime import prepare_source_message_graph
from scripts.check_authorization_evidence import CONTROLS
from scripts.compare_graph_source_allocation import CODE as PARENT_CODE
from scripts.compare_graph_source_allocation import DISTRACTORS, IMMUTABLE_TRACE, load_graph
from scripts.continue_source_message_answers import load_cases
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT, load_sources
from scripts.run_collection_benchmark import OfflineRuntime, load_inputs

OUTPUT = ROOT / "results/graph-document-allocation-v2"
PARENT = ROOT / "results/graph-source-allocation-v1"
CODE = (*PARENT_CODE, "app/graph_document_allocation.py", "scripts/compare_graph_document_allocation.py")
READ_QUOTE = "If your Lambda function code accesses other AWS resources, such as reading an object from an S3 bucket, the execution role needs permission to perform that action."


def source_witnesses():
    """Diagnostics only; these source IDs are never passed into either selector."""
    _, chunks, _ = load_sources()
    facts = json.loads((ROOT / "results/authorization-evidence-v1/facts.json").read_bytes())["records"]
    witnesses = {f["candidate"]["operation"]["quote"]: [f["candidate"]["chunk_id"]] for f in facts}
    witnesses["source_scoped_LambdaEdge_READ_example"] = sorted(k for k, c in chunks.items() if READ_QUOTE in c.text)
    if not witnesses["source_scoped_LambdaEdge_READ_example"]:
        raise ValueError("DOCUMENT_ALLOCATION_READ_CONTROL_MISSING")
    return witnesses


def compare(*, replay=False):
    parent_plan = json.loads((PARENT / "plan.json").read_bytes())
    if parent_plan["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in PARENT_CODE}:
        raise ValueError("DOCUMENT_ALLOCATION_PARENT_CODE_CHANGED")
    parent_rows = [json.loads(line) for line in (PARENT / "receipts.jsonl").read_text(encoding="utf-8").splitlines()]
    if sha256_bytes((PARENT / "receipts.jsonl").read_bytes()) != json.loads((PARENT / "report.json").read_bytes())["receipts_sha256"]:
        raise ValueError("DOCUMENT_ALLOCATION_PARENT_RECEIPTS_CHANGED")
    parent_by_id = {r["id"]: r for r in parent_rows if r["arm"] == "PREDECESSOR_TWO_CANDIDATES"}
    witnesses = source_witnesses()
    plan = {"version": "graph-priority-document-tail-comparison-v2", "controls": CONTROLS, "distractors": DISTRACTORS,
        "reason_for_new_experiment": "V1 four-source quota restored decrypt but removed the exposed VI READ example",
        "arms": ["PREDECESSOR_TWO_CANDIDATES", "DOCUMENT_TAIL_PRIORITY_2", "DOCUMENT_TAIL_PRIORITY_4"],
        "parent_plan_sha256": sha256_bytes((PARENT / "plan.json").read_bytes()),
        "parent_receipts_sha256": sha256_bytes((PARENT / "receipts.jsonl").read_bytes()),
        "source_witnesses_for_diagnostics_only": witnesses,
        "read_witness_scope": "Lambda@Edge page source-scoped example; not a newly extracted generic READ graph fact",
        "same_pool": True, "max_chunks": 8, "user_message_proxy_tokens": 4096,
        "new_model_calls": 0, "aws_calls": 0, "human_reviewed": 0, "independent_quality_measured": False,
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}, "deploy": False}
    immutable(OUTPUT / "plan.json", plan)
    collection, graph = load_graph()
    _, _, _, _, pipeline = load_inputs()
    rows, t = [], ProxyTokenizer()
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ())

        def record(case_id, before):
            recorded = parent_by_id[case_id]
            p = before.prepared
            _, base_prompt = generation_request(p.question, before.view, p.packed, "demo", p.recipe,
                answer_language=before.answer_language)
            if (sha256_bytes(base_prompt.encode()) != recorded["user_message_sha256"]
                    or tuple(recorded["context_chunk_ids"]) != p.packed.chunk_ids
                    or canonical_json(before.trace) != canonical_json(recorded["trace"])):
                raise ValueError("DOCUMENT_ALLOCATION_PARENT_CONTEXT_OR_TRACE_CHANGED")
            arms = [("PREDECESSOR_TWO_CANDIDATES", before)] + [(f"DOCUMENT_TAIL_PRIORITY_{cap}",
                reallocate_graph_documents(collection, lexical, graph, before, priority_source_cap=cap)) for cap in (2, 4)]
            for name, result in arms:
                after = result.prepared
                system, prompt = generation_request(after.question, result.view, after.packed, "demo", after.recipe,
                    answer_language=result.answer_language)
                if (set(after.retrieved_chunk_ids) != set(p.retrieved_chunk_ids)
                        or any(canonical_json(result.trace[k]) != canonical_json(before.trace[k]) for k in IMMUTABLE_TRACE)
                        or len(after.packed.chunk_ids) > 8 or t.count(prompt) != after.packed.proxy_tokens
                        or t.count(prompt) > 4096 or after.packed.serialized != prompt):
                    raise ValueError("DOCUMENT_ALLOCATION_SOURCE_OR_BUDGET_INVARIANT")
                if not before.trace["paths"] and (after.packed.chunk_ids != p.packed.chunk_ids
                        or result.outcome.mechanism_status != before.outcome.mechanism_status):
                    raise ValueError("DOCUMENT_ALLOCATION_NEGATIVE_CHANGED")
                rows.append({"id": case_id, "section": recorded["section"], "arm": name,
                    "question": after.question, "language": result.answer_language, "method": result.outcome.requested_method,
                    "outcome": result.outcome.model_dump(mode="json"), "trace": result.trace,
                    "request_id": after.binding.request_id, "source_count": len(after.packed.chunk_ids),
                    "context_chunk_ids": after.packed.chunk_ids, "user_message_proxy_tokens": after.packed.proxy_tokens,
                    "system_sha256": sha256_bytes(system.encode()), "user_message_sha256": sha256_bytes(prompt.encode()),
                    "sources_added": sorted(set(after.packed.chunk_ids) - set(p.packed.chunk_ids)),
                    "sources_removed": sorted(set(p.packed.chunk_ids) - set(after.packed.chunk_ids)),
                    "source_witness_presence_not_answer_accuracy": {label: bool(set(keys) & set(after.packed.chunk_ids))
                        for label, keys in witnesses.items()}, "same_pool": True, "answer_generated": False})
            print(json.dumps({"case": case_id, "arms": 3}), flush=True)

        for name, question in (*CONTROLS, *DISTRACTORS):
            lookup = runtime.cache.resolve_local("demo", text=query_embedding_input(question), recipe=runtime.query_recipe, encoder=runtime.encoder)
            vector = np.asarray(lookup.record.vector, dtype=np.float32)
            for hops in (1, 2):
                before = prepare_source_message_graph(collection, lexical, graph, question, vector,
                    tenant_id="demo", owner_id="graph-allocation-local", run_id="graph-source-allocation-v1",
                    recipe=runtime.recipe, hops=hops, allow_experimental=True)
                record(f"{name}-g{hops}", before)
        _, cases, _ = load_cases()
        for case in cases:
            if case["method"].startswith("graph_"):
                record(case["id"], StrictPreparedMethod(ReviewedGraphExecutionV3.model_validate_json(canonical_json(case["outcome"])),
                    case["view"], case["prepared"], case["trace"], case["language"]))
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    summary = {"status": "DOCUMENT_TAIL_CONTEXT_ONLY_COMPARISON_COMPLETE", "cases": len(rows) // 3, "arm_rows": len(rows),
        "new_model_calls": 0, "aws_calls": 0, "human_reviewed": 0, "serving_published": False,
        "receipts_sha256": sha256_bytes(raw), "all_same_pool": True,
        "arms": {arm: {"changed_contexts": sum(bool(r["sources_added"] or r["sources_removed"]) for r in rows if r["arm"] == arm),
            "added_sources": sum(len(r["sources_added"]) for r in rows if r["arm"] == arm),
            "removed_sources": sum(len(r["sources_removed"]) for r in rows if r["arm"] == arm),
            "zero_expansion": sum(r["outcome"]["mechanism_status"] == "GRAPH_ZERO_EXPANSION" for r in rows if r["arm"] == arm),
            "max_user_message_proxy_tokens": max(r["user_message_proxy_tokens"] for r in rows if r["arm"] == arm)}
            for arm in plan["arms"]},
        "native_graph_cell_source_presence": [{"id": r["id"], "arm": r["arm"],
            "presence": r["source_witness_presence_not_answer_accuracy"]} for r in rows if r["section"] == "NATIVE_EN_VI_GRAPH_CELLS"]}
    if replay:
        if (OUTPUT / "receipts.jsonl").read_bytes() != raw:
            raise ValueError("DOCUMENT_ALLOCATION_REPLAY_MISMATCH")
        immutable(OUTPUT / "replay.json", {"status": "ALL_126_CONTEXTS_AND_PATHS_EXACT_REPLAY", "network_blocked": True,
            "receipts_sha256": sha256_bytes(raw), "new_model_calls": 0, "aws_calls": 0})
    else:
        (OUTPUT / "receipts.jsonl").write_bytes(raw)
        immutable(OUTPUT / "report.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "native_graph_cell_source_presence"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=RuntimeError("DOCUMENT_ALLOCATION_NETWORK_BLOCKED")))
        compare(replay=args.replay)
