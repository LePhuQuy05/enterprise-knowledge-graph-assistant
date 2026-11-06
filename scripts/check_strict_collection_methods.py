"""Run strict Graph/Q/QB against pinned AWS sources without any provider calls.

Old planner text is revalidated explicitly as diagnostic input, not promoted to
new V2 inference. This checks mechanisms; it does not score answer quality.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.query_planning_v2 import make_validated_plan_v2, validate_search_queries_v2
from app.strict_collection_methods import (
    prepare_strict_graph_answer,
    prepare_strict_multi_query_answer,
)
from scripts.run_augmented_collection_benchmark import load_shared_queries
from scripts.run_collection_benchmark import ROOT, OfflineRuntime, load_inputs

CODE_PATHS = ("app/method_execution.py", "app/query_planning_v2.py", "app/query_preparation_v3.py",
    "app/query_rewrite_v2.py", "app/collection_graph_v2.py", "app/strict_collection_methods.py",
    "scripts/check_strict_collection_methods.py")


def code_hashes():
    return {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE_PATHS}


def check(output, case_ids=None):
    protocol_raw, dataset, collection, _, pipeline = load_inputs()
    bundle, bundle_sha, _ = load_shared_queries(dataset, protocol_raw)
    tasks = {(s.scenario_id, lang): getattr(s, "question_" + lang) for s in dataset.scenarios for lang in ("en", "vi")}
    if case_ids and not set(case_ids) <= {s.scenario_id for s in dataset.scenarios}:
        raise ValueError("STRICT_DIAGNOSTIC_UNKNOWN_CASE")
    selected = [r for r in bundle.rows if not case_ids or r.scenario_id in case_ids]
    before = code_hashes()
    output.mkdir(parents=True, exist_ok=False)
    outcomes, plan_checks = [], []
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("graph_1hop", "graph_2hop"))
        vectors = {}

        def vector(text):
            if text not in vectors:
                cached = runtime.cache.resolve_local(collection.manifest.tenant_id, text=query_embedding_input(text),
                    recipe=runtime.query_recipe, encoder=runtime.encoder)
                vectors[text] = np.asarray(cached.record.vector, dtype=np.float32)
            return vectors[text]

        with (output / "receipts.jsonl").open("xb") as handle:
            for row in selected:
                question = tasks[(row.scenario_id, row.language)]
                raw = row.planner_receipt.raw_output
                queries = validate_search_queries_v2(raw, row.search_query)
                plan = make_validated_plan_v2(question, row.search_query, row.language, queries,
                    operation_id=row.planner_receipt.operation_id,
                    receipt_sha256=sha256_bytes(canonical_json(row.planner_receipt.model_dump(mode="json"))),
                    origin="V1_REVALIDATED_DIAGNOSTIC")
                plan_checks.append({"scenario_id": row.scenario_id, "language": row.language,
                    "old_status": row.preparation_status, "new_validation": "VALID", "subquery_count": len(queries)})
                params = dict(question=question, query_vector=vector(row.search_query), search_query=row.search_query,
                    tenant_id=collection.manifest.tenant_id, owner_id="strict-local", run_id="method-correctness-v2",
                    recipe=runtime.recipe)
                for method in ("graph_1hop", "graph_2hop", "hybrid_multi_query", "hybrid_multi_query_balanced"):
                    if method.startswith("graph_"):
                        result = prepare_strict_graph_answer(collection, lexical, runtime.graph, **params,
                            hops=int(method[6]), allow_experimental=True, clock=lambda: 0.0)
                    else:
                        result = prepare_strict_multi_query_answer(collection, lexical, **params,
                            mode=method, plan=plan, subquery_vectors=tuple(vector(q) for q in queries),
                            language=row.language, allow_legacy_diagnostic=True)
                    record = {"scenario_id": row.scenario_id, "language": row.language,
                        "outcome": result.outcome.model_dump(mode="json"), "trace": result.trace,
                        "generation_executed": False}
                    outcomes.append(record)
                    handle.write(canonical_json(record) + b"\n")
                    handle.flush()
                print(json.dumps({"input": row.scenario_id + "." + row.language, "receipts": len(outcomes)}), flush=True)
    if before != code_hashes():
        raise ValueError("STRICT_DIAGNOSTIC_CODE_CHANGED_KEEP_RECEIPTS")
    for case, language in tasks:
        own = [r for r in outcomes if r["scenario_id"] == case and r["language"] == language
               and r["outcome"]["requested_method"].startswith("hybrid_multi_query")]
        if own:
            if len(own) != 2 or own[0]["trace"]["multi_query"] != own[1]["trace"]["multi_query"]:
                raise ValueError("STRICT_DIAGNOSTIC_Q_QB_POOL_MISMATCH")
    summaries = []
    for method in ("graph_1hop", "graph_2hop", "hybrid_multi_query", "hybrid_multi_query_balanced"):
        for lang in ("en", "vi"):
            own = [r for r in outcomes if r["language"] == lang and r["outcome"]["requested_method"] == method]
            summaries.append({"method": method, "language": lang, "inputs": len(own),
                "statuses": dict(Counter(r["outcome"]["mechanism_status"] for r in own))})
    result = {"status": "OFFLINE_MECHANISM_CHECK_COMPLETE_NOT_NEW_INFERENCE_OR_QUALITY_BENCHMARK",
        "code_sha256": before, "shared_bundle_sha256": bundle_sha, "source_manifest_sha256": collection.manifest_sha256,
        "inputs": len(selected), "receipts": len(outcomes), "summary": summaries, "plan_checks": plan_checks,
        "provider_calls": 0, "aws_calls": 0, "generation_calls": 0, "strict_hybrid_fallbacks": 0,
        "local_embedding_onnx_calls": runtime.encoder.inference_calls,
        "receipts_sha256": sha256_bytes((output / "receipts.jsonl").read_bytes()),
        "human_approved": 0, "graph_full_extraction": False, "planner_origin": "V1_REVALIDATED_DIAGNOSTIC"}
    (output / "report.json").write_bytes(canonical_json(result) + b"\n")
    print(json.dumps({k: result[k] for k in ("status", "inputs", "receipts", "summary", "provider_calls")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    check(args.output_dir, args.case_ids)


if __name__ == "__main__":
    main()
