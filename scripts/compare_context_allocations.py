"""Fixed same-pool context allocation experiment; no answer, planner or AWS calls."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np

from app.collection_answering import _evidence_view
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context import pack_context
from app.context_selection import ContextSelectionRecipe, select_context_order
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import EmbeddingCache, query_embedding_input
from app.graph_retrieval import GraphRetrievalResult
from app.local_bge import BgeEncoder
from app.multi_query import MultiQueryTraceV1, retrieve_multi_query
from scripts.answer_local import ROOT
from scripts.run_reranker_answer_pilot import write_once

POLICIES = ("RR_ID_ONLY", "RR_SCOPED_EXACT_DEDUP")
CONTROLS = ROOT / "configs/evaluation/context-allocation-controls-v1.json"


def read_controls(path):
    data = json.loads(path.read_bytes())
    required = {"schema_version", "version", "purpose", "query_policy", "cases"}
    if (type(data) is not dict or not required <= set(data) or set(data) - required - {"notice"}
            or type(data.get("schema_version")) is not int or data["schema_version"] != 1
            or data.get("version") != "context-allocation-controls-v1"
            or type(data.get("cases")) is not list
            or data.get("purpose") != "DEVELOPMENT_CONTEXT_ALLOCATION_INPUTS_NOT_GOLD_OR_HOLDOUT"
            or data.get("query_policy") != "MANUAL_DEV_FIXED_NOT_LLM" or len(data.get("cases", [])) != 4):
        raise ValueError("ALLOCATION_CONTROL_CONFIG_INVALID")
    seen = set()
    for row in data["cases"]:
        if (type(row) is not dict
                or set(row) != {"id", "language", "question", "search_query", "additional_search_queries"}
                or type(row["id"]) is not str or not row["id"].strip() or row["id"] in seen
                or row["language"] not in {"en", "vi"}
                or type(row["additional_search_queries"]) is not list or len(row["additional_search_queries"]) > 2
                or any(type(text) is not str or not text.strip() or len(text) > 2000
                       for text in (row["question"], row["search_query"]))
                or any(type(text) is not str or not text.strip() or len(text) > 240 for text in row["additional_search_queries"])):
            raise ValueError("ALLOCATION_CONTROL_RUNTIME_FIELDS_INVALID")
        queries = [row["search_query"], *row["additional_search_queries"]]
        if len({" ".join(text.split()).casefold() for text in queries}) != len(queries):
            raise ValueError("ALLOCATION_CONTROL_QUERY_DUPLICATE")
        seen.add(row["id"])
    return data


def describe(view, trace, ordered):
    ids = tuple(ordered)
    packed = pack_context(view, GraphRetrievalResult(ids, ids, (), (), "hybrid_multi_query", False, (), 0, 0))
    sources = []
    for key in packed.chunk_ids:
        chunk = view.chunks[key]
        doc, ref = view.documents[chunk.document_version_id], view.source_refs[key]
        if doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text:
            raise ValueError("ALLOCATION_SOURCE_SLICE_MISMATCH")
        sources.append({"chunk_id": key, "document_version_id": doc.document_version_id,
            "source_uri": doc.source_uri, "title": doc.title, "section_path": list(chunk.section_path),
            "start_char": chunk.start_char, "end_char": chunk.end_char, "text": chunk.text,
            "text_sha256": chunk.text_sha256, "shard_id": ref.shard_id, "shard_manifest_sha256": ref.manifest_sha256})
    return {"chunk_ids": list(packed.chunk_ids), "proxy_tokens": packed.proxy_tokens,
        "serialized_sha256": sha256_bytes(packed.serialized.encode()), "warnings": list(packed.warnings),
        "sources": sources, "packed_chunks_found_by_each_query": [
            sum(key in query.ranked_chunk_ids for key in packed.chunk_ids) for query in trace.queries],
        "query_exposure_is_not_fact_coverage": True}


def compare(view, trace):
    baseline = describe(view, trace, trace.ranked_chunk_ids)
    variants = {"Q_RRF": {"context": baseline, "ordered_chunk_ids": list(trace.ranked_chunk_ids)}}
    for policy in POLICIES:
        recipe, observations, first = ContextSelectionRecipe(policy=policy), [], None
        for _ in range(3):
            start = perf_counter()
            selected = select_context_order(trace, tuple(view.chunks.values()), tenant_id="demo", recipe=recipe)
            observations.append(round((perf_counter() - start) * 1000, 3))
            value = selected.model_dump(mode="json")
            if first is not None and value != first:
                raise ValueError("ALLOCATION_NONDETERMINISTIC_TRACE")
            first = value
        packed = describe(view, trace, selected.ordered_chunk_ids)
        variants[policy] = {"selection_trace": first, "ordered_chunk_ids": list(selected.ordered_chunk_ids),
            "context": packed, "selector_observations_ms": observations,
            "added_ids": [key for key in packed["chunk_ids"] if key not in baseline["chunk_ids"]],
            "removed_ids": [key for key in baseline["chunk_ids"] if key not in packed["chunk_ids"]]}
    return variants


def verify(directory):
    plan_raw, result_raw = (directory / "plan.json").read_bytes(), (directory / "experiment.json").read_bytes()
    plan, result = json.loads(plan_raw), json.loads(result_raw)
    if result["plan_sha256"] != sha256_bytes(plan_raw):
        raise ValueError("ALLOCATION_PLAN_CHECKSUM")
    if sha256_bytes((ROOT / "app/context_selection.py").read_bytes()) != plan["selector_module_sha256"]:
        raise ValueError("ALLOCATION_IMPLEMENTATION_CHANGED")
    if (plan["policies"] != [ContextSelectionRecipe(policy=policy).model_dump(mode="json") for policy in POLICIES]
            or plan["cases"] != 7 or plan["frozen_previously_inspected_cases"] != 3
            or plan["manual_fixed_query_new_controls"] != 4):
        raise ValueError("ALLOCATION_PREREGISTERED_RECIPE_OR_COUNTS_CHANGED")
    controls_path = Path(plan["controls_path"])
    if sha256_bytes(controls_path.read_bytes()) != plan["controls_sha256"]:
        raise ValueError("ALLOCATION_CONTROL_INPUT_CHANGED")
    controls = read_controls(controls_path)
    if len(plan["frozen_q_inputs"]) != 3:
        raise ValueError("ALLOCATION_FROZEN_INPUT_COUNT_CHANGED")
    frozen = []
    for pinned in plan["frozen_q_inputs"]:
        raw = Path(pinned["path"]).read_bytes()
        if sha256_bytes(raw) != pinned["sha256"]:
            raise ValueError("ALLOCATION_FROZEN_INPUT_CHANGED")
        frozen.append(json.loads(raw))
    expected = [{"id": old["pilot_case_id"], "phase": "PREVIOUSLY_INSPECTED_FROZEN_Q_DEVELOPMENT",
        "question": old["input"]["original_question"], "language": old["answer_language"],
        "trace": old["methods"][1]["multi_query_trace"],
        "context_chunk_ids": old["methods"][1]["context_chunk_ids"]} for old in frozen]
    expected.extend({"id": row["id"], "phase": "MANUAL_FIXED_QUERY_DEVELOPMENT_NOT_LLM_PIPELINE",
        "question": row["question"], "language": row["language"],
        "queries": [row["search_query"], *row["additional_search_queries"]]} for row in controls["cases"])
    if len(result["cases"]) != len(expected):
        raise ValueError("ALLOCATION_CASE_COUNT_CHANGED")
    collection = load_collection(Path(plan["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    if collection.manifest_sha256 != plan["collection_manifest_sha256"]:
        raise ValueError("ALLOCATION_COLLECTION_CHANGED")
    with load_global_lexical_index(Path(plan["lexical_manifest_path"]), collection) as lexical:
        for case, pinned in zip(result["cases"], expected, strict=True):
            if any(case[key] != pinned[key] for key in ("id", "phase", "question", "language")):
                raise ValueError("ALLOCATION_CASE_INPUT_BINDING_MISMATCH")
            trace = MultiQueryTraceV1.model_validate_json(canonical_json(case["multi_query_trace"]))
            if (("trace" in pinned and trace.model_dump(mode="json") != pinned["trace"])
                    or ("queries" in pinned and [query.query for query in trace.queries] != pinned["queries"])):
                raise ValueError("ALLOCATION_QUERY_TRACE_INPUT_BINDING_MISMATCH")
            if set(case["variants"]) != {"Q_RRF", *POLICIES}:
                raise ValueError("ALLOCATION_VARIANT_KEYS_MISMATCH")
            view = _evidence_view(collection, lexical, trace.ranked_chunk_ids)
            baseline = describe(view, trace, trace.ranked_chunk_ids)
            if (baseline != case["variants"]["Q_RRF"]["context"]
                    or case["variants"]["Q_RRF"]["ordered_chunk_ids"] != list(trace.ranked_chunk_ids)
                    or ("context_chunk_ids" in pinned and baseline["chunk_ids"] != pinned["context_chunk_ids"])):
                raise ValueError("ALLOCATION_BASELINE_RECONSTRUCTION_MISMATCH")
            for policy in POLICIES:
                variant = case["variants"][policy]
                selected = select_context_order(trace, tuple(view.chunks.values()), tenant_id="demo",
                                                recipe=ContextSelectionRecipe(policy=policy))
                if (selected.model_dump(mode="json") != variant["selection_trace"]
                        or list(selected.ordered_chunk_ids) != variant["ordered_chunk_ids"]
                        or describe(view, trace, selected.ordered_chunk_ids) != variant["context"]
                        or variant["added_ids"] != [key for key in variant["context"]["chunk_ids"] if key not in baseline["chunk_ids"]]
                        or variant["removed_ids"] != [key for key in baseline["chunk_ids"] if key not in variant["context"]["chunk_ids"]]):
                    raise ValueError("ALLOCATION_VARIANT_RECONSTRUCTION_MISMATCH")
    report = {"status": "PASS_SAVED_TRACE_SOURCE_CONTEXT_RECONSTRUCTION", "checked_at": datetime.now(UTC).isoformat(),
        "experiment_sha256": sha256_bytes(result_raw), "cases": len(result["cases"]), "variants_per_case": 3,
        "encoder_calls": 0, "model_calls": 0, "aws_calls": 0, "paid_calls": 0,
        "note": "Rebuilds selector/packer from saved retrieval traces; not retrieval/embedding replay or semantic assessment."}
    write_once(directory / "verification.json", report)
    print(json.dumps(report))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/context-allocation/development-v1")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args(argv)
    if args.verify:
        return verify(args.directory)
    if (args.directory / "plan.json").exists():
        raise ValueError("ALLOCATION_EXPERIMENT_ALREADY_PLANNED_NO_AUTOMATIC_RESUME")
    controls = read_controls(CONTROLS)
    frozen_paths = [ROOT / f"results/multi-query/pilot-v1/case-{index}.json" for index in range(1, 4)]
    frozen = [json.loads(path.read_bytes()) for path in frozen_paths]
    if any(row["status"] != "COMPLETED" or row["methods"][1]["effective_method"] != "hybrid_multi_query" for row in frozen):
        raise ValueError("ALLOCATION_FROZEN_INPUT_INVALID")
    pipeline = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    cache = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE")
    plan = {"schema_version": 1, "created_at": datetime.now(UTC).isoformat(),
        "kind": "PREREGISTERED_DEVELOPMENT_SAME_POOL_CONTEXT_ALLOCATION_NOT_GOLD_OR_HOLDOUT",
        "hypothesis": "RR can preserve source opportunities from distinct query angles within fixed context; it can lose consensus or admit query drift, so no universal improvement is predicted.",
        "policies": [ContextSelectionRecipe(policy=policy).model_dump(mode="json") for policy in POLICIES],
        "selector_module_sha256": sha256_bytes((ROOT / "app/context_selection.py").read_bytes()),
        "controls_path": str(CONTROLS), "controls_sha256": sha256_bytes(CONTROLS.read_bytes()),
        "frozen_q_inputs": [{"path": str(path), "sha256": sha256_bytes(path.read_bytes())} for path in frozen_paths],
        "collection_path": pipeline["collection_path"], "lexical_manifest_path": pipeline["lexical_manifest_path"],
        "collection_manifest_sha256": collection.manifest_sha256,
        "query_recipe_sha256": encoder.recipe("query").fingerprint,
        "source_control_draft_passed_to_runtime": False, "same_pool_max40": True,
        "packer": "UNCHANGED_8_CHUNKS_4096_PROXY", "selector_repeats": 3,
        "cases": 7, "frozen_previously_inspected_cases": 3, "manual_fixed_query_new_controls": 4,
        "human_reviewed": False, "gold": False, "holdout": False,
        "generation_calls": 0, "planner_calls": 0, "aws_calls": 0, "paid_calls": 0}
    write_once(args.directory / "plan.json", plan)
    output = []
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        for old in frozen:
            trace = MultiQueryTraceV1.model_validate_json(canonical_json(old["methods"][1]["multi_query_trace"]))
            view = _evidence_view(collection, lexical, trace.ranked_chunk_ids)
            variants = compare(view, trace)
            if variants["Q_RRF"]["context"]["chunk_ids"] != old["methods"][1]["context_chunk_ids"]:
                raise ValueError("ALLOCATION_ORIGINAL_Q_CONTEXT_MISMATCH")
            output.append({"id": old["pilot_case_id"], "phase": "PREVIOUSLY_INSPECTED_FROZEN_Q_DEVELOPMENT",
                "question": old["input"]["original_question"], "language": old["answer_language"],
                "multi_query_trace": trace.model_dump(mode="json"), "variants": variants})
        for row in controls["cases"]:
            queries = (row["search_query"], *row["additional_search_queries"])
            vectors = tuple(np.asarray(cache.resolve_local("demo", text=query_embedding_input(query),
                recipe=encoder.recipe("query"), encoder=encoder).record.vector, dtype=np.float32) for query in queries)
            trace = retrieve_multi_query(collection, lexical, queries[0], vectors[0],
                subqueries=tuple(queries[1:]), subquery_vectors=tuple(vectors[1:]))
            view = _evidence_view(collection, lexical, trace.ranked_chunk_ids)
            output.append({"id": row["id"], "phase": "MANUAL_FIXED_QUERY_DEVELOPMENT_NOT_LLM_PIPELINE",
                "question": row["question"], "language": row["language"],
                "multi_query_trace": trace.model_dump(mode="json"), "variants": compare(view, trace)})
    result = {"status": "COMPLETED_CONTEXT_EXPERIMENT_NOT_ANSWER_QUALITY", "created_at": datetime.now(UTC).isoformat(),
        "plan_sha256": sha256_bytes((args.directory / "plan.json").read_bytes()), "cases": output,
        "local_query_encoder_calls": encoder.inference_calls, "generation_calls": 0, "planner_calls": 0,
        "reranker_calls": 0, "aws_calls": 0, "paid_calls": 0, "new_api_cost_micro_usd": 0,
        "quality_metrics": None, "human_reviewed": False, "cloud_deployed": False}
    write_once(args.directory / "experiment.json", result)
    print(json.dumps({key: result[key] for key in ("status", "local_query_encoder_calls", "paid_calls", "aws_calls")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
