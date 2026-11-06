"""Frozen local section-field ablation; no AWS, native inference or generation.

Default freezes scope. --prepare-index encodes metadata locally. --run records
96 cells; --replay rebuilds them from cached vectors with network blocked.
"""

import argparse
import io
import json
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.faceted_query_v5 import FacetPlanV5
from app.graph_collection_migration import collection_records
from app.ingestion.text import ProxyTokenizer
from app.section_recall import (
    SectionRecipe,
    build_section_pool,
    literal_probes,
    make_index,
    metadata_rows,
    pack_section_pool,
)
from app.source_dependency_audit import SourceDependency, audit_source_dependencies
from scripts.audit_collection319_methods import source_reachability
from scripts.audit_facet_recall_v5 import rendered_sources
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT
from scripts.run_collection_benchmark import OfflineRuntime

OUTPUT = ROOT / "results/section-recall-v1"
ASSETS = ROOT / "data/section-recall-v1"
PARENT = ROOT / "results/facet-recall-v5-canary-v1"
RUN = "literal-section-recall-local-v1"
CODE = ("app/section_recall.py", "scripts/run_section_recall.py", "app/document_recall_v2.py",
    "app/faceted_query_v5.py", "app/collection_answering.py", "app/source_dependency_audit.py",
    "app/generation.py", "app/local_bge.py", "app/collections.py", "app/retrieval.py",
    "app/strict_collection_methods.py", "app/ingestion/text.py")


def load():
    collection, paths, _ = load_migrated()
    raw = (PARENT / "receipts.json").read_bytes()
    report = json.loads((PARENT / "report.json").read_bytes())
    parent_scope = json.loads((PARENT / "scope.json").read_bytes())
    if (sha256_bytes(raw) != report["receipts_sha256"] or any(sha256_bytes((ROOT / p).read_bytes()) != h
            for p, h in parent_scope["code_sha256"].items())):
        raise ValueError("SECTION_PARENT_PIN")
    return collection, paths, json.loads(raw)


def recipes():
    return tuple(SectionRecipe(field_mode=mode, admission=admission, radius=radius)
        for mode in ("BM25", "DENSE", "RRF") for admission in ("RANKED", "DOCUMENT_DIVERSE") for radius in (1, 2))


def freeze(collection, payload):
    docs, chunks, _ = collection_records(collection)
    rows = metadata_rows(docs, chunks, collection.manifest.tenant_id)
    tasks = [{k: prep[k] for k in ("intent", "language", "question", "search", "plan")}
        for prep in payload["native_preparations"]]
    scope = {"version": RUN, "purpose": "EXPOSED_LOCAL_DEVELOPMENT_ABLATION_NOT_INDEPENDENT_BENCHMARK",
        "collection_manifest_sha256": collection.manifest_sha256, "documents": len(docs), "chunks": len(chunks),
        "metadata_projection": "DEDUPLICATED_TITLE_AND_SECTION_PATH_ONLY_V1", "metadata_units": len(rows),
        "metadata_rows_sha256": sha256_bytes(canonical_json(rows)),
        "empty_section_paths": sum(not c.section_path for c in chunks.values()),
        "parent_receipts_sha256": sha256_bytes((PARENT / "receipts.json").read_bytes()), "tasks": tasks,
        "recipes": [r.model_dump(mode="json") for r in recipes()], "packers": ["Q", "QB"],
        "new_cells": len(tasks)*len(recipes())*2, "recorded_v5_baseline_cells": 24,
        "metadata_top_sections_per_channel": 20, "metadata_channels_max": 4,
        "metadata_rank_occurrences_max_per_field": 80,
        "body_vector_lexical_baseline_reused_not_recomputed": True,
        "additional_dense_metadata_dot_products_per_recipe_input_max": len(rows)*4,
        "source_review_annotations_sha256": sha256_bytes((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes()),
        "source_annotations_used_for_ranking": False, "queries_are_derived_literal_probes_not_native_rewrites": True,
        "native_planner_calls": 0, "generation_calls": 0, "extraction_calls": 0, "aws_calls": 0,
        "new_graph_records": 0, "human_reviewed": 0, "serving_published": False,
        "model_manifest_sha256": sha256_bytes((ROOT / "configs/models/bge-small-en-v1.5-onnx.json").read_bytes()),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}}
    immutable(OUTPUT / "scope.json", scope)
    return scope, rows


def prepare_index(collection, paths, rows, scope):
    manifest_path = ASSETS / "manifest.json"
    if manifest_path.exists():
        return load_index(collection, rows, scope)
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ())
        texts = list(dict.fromkeys(row["metadata_text"] for row in rows))
        counts = [runtime.encoder.count(text, role="document") for text in texts]
        if max(counts) > 512:
            raise ValueError("SECTION_METADATA_TOO_LONG_NO_TRUNCATION")
        start = perf_counter()
        unique = runtime.encoder.encode(texts, role="document", batch_size=8)
        lookup = {text: vector for text, vector in zip(texts, unique, strict=True)}
        matrix = np.asarray([lookup[row["metadata_text"]] for row in rows], dtype=np.float32)
        index = make_index(collection, rows, matrix)
        ASSETS.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        np.save(buffer, matrix, allow_pickle=False)
        raw = buffer.getvalue()
        with (ASSETS / "vectors.npy").open("xb") as handle:
            handle.write(raw)
        manifest = {"version": RUN, "collection_manifest_sha256": collection.manifest_sha256,
            "metadata_rows_sha256": scope["metadata_rows_sha256"], "index_fingerprint": index.fingerprint,
            "projection": scope["metadata_projection"], "encoder_backend_recipe": runtime.encoder.recipe("document").model_dump(mode="json"),
            "projection_is_separate_from_body_embedding_recipe": True,
            "vectors_npy_sha256": sha256_bytes(raw), "shape": list(matrix.shape), "dtype": "float32",
            "unique_texts_encoded": len(texts), "max_wordpiece_tokens": max(counts),
            "bge_local_batches": runtime.encoder.inference_calls, "model_api_calls": 0,
            "encoding_seconds": round(perf_counter()-start, 3), "created_at": datetime.now(UTC).isoformat()}
        immutable(manifest_path, manifest)
        immutable(OUTPUT / "metadata-inventory.json", manifest | {"rows": rows})
        print(json.dumps({k: manifest[k] for k in ("shape", "unique_texts_encoded", "max_wordpiece_tokens", "encoding_seconds", "model_api_calls")}), flush=True)
        return index


def load_index(collection, rows, scope):
    manifest = json.loads((ASSETS / "manifest.json").read_bytes())
    raw = (ASSETS / "vectors.npy").read_bytes()
    if (manifest["collection_manifest_sha256"] != collection.manifest_sha256
            or manifest["metadata_rows_sha256"] != scope["metadata_rows_sha256"]
            or manifest["vectors_npy_sha256"] != sha256_bytes(raw)
            or manifest["encoder_backend_recipe"]["artifact_manifest_sha256"] != scope["model_manifest_sha256"]):
        raise ValueError("SECTION_INDEX_ARTIFACT_PIN")
    index = make_index(collection, rows, np.load(io.BytesIO(raw), allow_pickle=False))
    if index.fingerprint != manifest["index_fingerprint"]:
        raise ValueError("SECTION_INDEX_FINGERPRINT")
    return index


def run(replay=False):
    collection, paths, payload = load()
    scope, metadata = freeze(collection, payload)
    index = load_index(collection, metadata, scope)
    docs, chunks, _ = collection_records(collection)
    control = json.loads((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())
    deps = [SourceDependency.model_validate_json(canonical_json(r["dependency"])) for r in control["source_controls"]]
    deps.append(SourceDependency.model_validate_json(canonical_json(control["parent_kms_dependency"])))
    cells, pairs = [], []
    sources, spans = 0, 0
    tokenizer = ProxyTokenizer()
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ())
        for task in payload["native_preparations"]:
            plan = FacetPlanV5.model_validate_json(canonical_json(task["plan"]))
            queries = literal_probes(collection, task["question"], task["search"], task["language"], plan)
            vectors = tuple(np.asarray(runtime.cache.resolve_local("demo", text=query_embedding_input(q),
                recipe=runtime.query_recipe, encoder=runtime.encoder).record.vector, dtype=np.float32) for q in queries)
            base = next(r for r in payload["recall_cells"] if r["intent"] == task["intent"]
                and r["language"] == task["language"] and r["arm"] == "V5_CHUNK_Q")["pool_ids"]
            for recipe in recipes():
                pool = build_section_pool(collection, lexical, index, question=task["question"], search=task["search"],
                    language=task["language"], plan=plan, vectors=vectors, base_ids=base, recipe=recipe)
                own = []
                for balanced in (False, True):
                    answer, audit = pack_section_pool(collection, lexical, pool, balanced=balanced,
                        generation_recipe=runtime.recipe, run_id=RUN, owner_id="section-recall-local")
                    arm = f"{recipe.field_mode}_{recipe.admission}_R{recipe.radius}_" + ("QB" if balanced else "Q")
                    baseline = next(r for r in payload["recall_cells"] if r["intent"] == task["intent"]
                        and r["language"] == task["language"] and r["arm"] == f"V5_WINDOW_R{recipe.radius}_" + ("QB" if balanced else "Q"))
                    row = {k: task[k] for k in ("intent", "language", "question", "search")}
                    row.update(arm=arm, recipe=recipe.model_dump(mode="json"), pool_ids=pool.pool_ids,
                        pool_fingerprint=pool.fingerprint, pool_trace=json.loads(pool.trace_json), bundles=json.loads(pool.bundles_json),
                        context_chunk_ids=answer.packed.chunk_ids, source_message=answer.packed.serialized,
                        user_message_proxy_tokens=answer.packed.proxy_tokens, binding=answer.binding.model_dump(mode="json"), packing_audit=audit,
                        context_audit=audit_source_dependencies(docs, chunks, answer.packed.chunk_ids, deps, tenant_id="demo"),
                        source_reachability=[source_reachability(dep, pool.pool_ids, answer.packed.chunk_ids, chunks) for dep in deps],
                        context_added_vs_v5=sorted(set(answer.packed.chunk_ids)-set(baseline["context_chunk_ids"])),
                        context_removed_vs_v5=sorted(set(baseline["context_chunk_ids"])-set(answer.packed.chunk_ids)),
                        generation_executed=False, graph_records_used=0)
                    n, s = rendered_sources(row, docs, chunks, tokenizer)
                    sources += n
                    spans += s
                    own.append(row)
                    cells.append(row)
                if own[0]["pool_fingerprint"] != own[1]["pool_fingerprint"] or own[0]["pool_trace"] != own[1]["pool_trace"]:
                    raise ValueError("SECTION_PAIR_POOL")
                pairs.append({"intent": task["intent"], "language": task["language"], "recipe_sha256": recipe.fingerprint,
                    "same_pool_plan_bundles": True, "context_changed": own[0]["context_chunk_ids"] != own[1]["context_chunk_ids"]})
            print(json.dumps({"intent": task["intent"], "language": task["language"], "cells": len(cells)}), flush=True)
        local_calls = runtime.encoder.inference_calls
    if replay and local_calls:
        raise ValueError("SECTION_REPLAY_MUST_REUSE_QUERY_VECTORS")
    if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
        raise ValueError("SECTION_CODE_CHANGED_DURING_RUN")
    immutable(OUTPUT / "receipts.json", {"cells": cells, "recorded_v5_baseline": payload["recall_cells"]})
    summary = []
    for arm in sorted({r["arm"] for r in cells}):
        own = [r for r in cells if r["arm"] == arm]
        summary.append({"arm": arm, "inputs": len(own),
            "annotation_statuses": dict(Counter(c["status"] for r in own for c in r["context_audit"]["checks"])),
            "reachability": dict(Counter(c["stage"] for r in own for c in r["source_reachability"])),
            "context_chunks": [len(r["context_chunk_ids"]) for r in own],
            "proxy_tokens": [r["user_message_proxy_tokens"] for r in own],
            "sources_removed_vs_v5": [len(r["context_removed_vs_v5"]) for r in own]})
    result = {"status": "EXACT_OFFLINE_REPLAY_MATCH" if replay else "LOCAL_ABLATION_RECORDED",
        "cells": len(cells), "q_qb_pairs": len(pairs), "rendered_source_appearances": sources, "exact_source_spans": spans,
        "receipts_sha256": sha256_bytes((OUTPUT / "receipts.json").read_bytes()), "arms": summary,
        "metadata_index_fingerprint": index.fingerprint, "new_model_api_calls": 0, "aws_calls": 0, "generation_calls": 0,
        "extraction_calls": 0, "quality_accuracy_measured": False, "human_reviewed": 0, "serving_published": False}
    immutable(OUTPUT / ("replay.json" if replay else "report.json"), result)
    immutable(OUTPUT / ("replay-pairs.json" if replay else "pairs.json"), pairs)
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--prepare-index", action="store_true")
    choice.add_argument("--run", action="store_true")
    choice.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=AssertionError("SECTION_EXPERIMENT_NETWORK_BLOCKED")))
        if args.run or args.replay:
            run(replay=args.replay)
        else:
            collection, paths, payload = load()
            frozen, rows = freeze(collection, payload)
            if args.prepare_index:
                prepare_index(collection, paths, rows, frozen)
            else:
                print(json.dumps({"status": "SCOPE_FROZEN_NO_RANKING_OR_CALLS", "metadata_units": len(rows), "planned_cells": frozen["new_cells"]}))
