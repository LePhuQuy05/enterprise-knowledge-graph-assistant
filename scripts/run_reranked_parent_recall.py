"""Local reranker diagnostic on RRF parent anchors; no native or cloud calls."""

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.faceted_query_v5 import FacetPlanV5
from app.graph_collection_migration import collection_records
from app.ingestion.text import ProxyTokenizer
from app.reranked_parent_recall import RerankedParentRecipe, build_reranked_parent_pool
from app.section_recall import literal_probes, metadata_rows, pack_section_pool
from app.source_dependency_audit import SourceDependency, audit_source_dependencies
from scripts.audit_collection319_methods import source_reachability
from scripts.audit_facet_recall_v5 import rendered_sources
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT
from scripts.run_collection_benchmark import OfflineRuntime
from scripts.run_section_recall import ASSETS, load, load_index
from scripts.run_section_recall import CODE as SECTION_CODE

OUTPUT = ROOT / "results/reranked-parent-recall-v1"
RUN = "reranked-parent-recall-local-v1"
CODE = (*SECTION_CODE, "app/parent_section_recall.py", "scripts/run_parent_section_recall.py", "app/reranked_parent_recall.py", "scripts/run_reranked_parent_recall.py", "app/reranking.py")


def recipes():
    # Parent proposals are already unique per document. DIVERSE and RANKED
    # would coincide, so do not manufacture a duplicate admission ablation.
    return tuple(RerankedParentRecipe(field_mode=mode, admission="RANKED", radius=radius)
        for mode in ("RRF",) for radius in (1, 2))


def freeze(collection):
    predecessor = ROOT / "results/parent-section-recall-v1"
    original = json.loads((predecessor / "scope.json").read_bytes())
    if any(sha256_bytes((ROOT / p).read_bytes()) != h for p, h in original["code_sha256"].items()):
        raise ValueError("PARENT_SECTION_PREDECESSOR_CODE_CHANGED")
    scope = original | {"version": RUN,
        "predecessor_scope_sha256": sha256_bytes((predecessor / "scope.json").read_bytes()),
        "predecessor_receipts_sha256": sha256_bytes((predecessor / "receipts.json").read_bytes()),
        "metadata_asset_sha256": sha256_bytes((ASSETS / "manifest.json").read_bytes()),
        "candidate_policy": "TOP20_DISTINCT_DOCUMENTS_BEFORE_FIELD_AND_QUERY_RRF",
        "recipes": [r.model_dump(mode="json") for r in recipes()], "new_cells": 16,
        "local_reranker_anchor_cap": 40, "local_reranker_passes_max": 8,
        "reranker_manifest_sha256": sha256_bytes((ROOT / "configs/models/ms-marco-MiniLM-L2-v2-onnx.json").read_bytes()),
        "metadata_top_sections_per_channel": None, "metadata_top_documents_per_channel": 20,
        "one_proposal_per_parent_document": True,
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}}
    if scope["collection_manifest_sha256"] != collection.manifest_sha256:
        raise ValueError("PARENT_SECTION_COLLECTION_CHANGED")
    immutable(OUTPUT / "scope.json", scope)
    return scope


def run(replay=False):
    collection, paths, payload = load()
    scope = freeze(collection)
    docs, chunks, _ = collection_records(collection)
    metadata = metadata_rows(docs, chunks, collection.manifest.tenant_id)
    index = load_index(collection, metadata, scope)
    annotations = json.loads((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())
    deps = [SourceDependency.model_validate_json(canonical_json(r["dependency"])) for r in annotations["source_controls"]]
    deps.append(SourceDependency.model_validate_json(canonical_json(annotations["parent_kms_dependency"])))
    cells, pairs = [], []
    count, spans = 0, 0
    tokenizer = ProxyTokenizer()
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
        for task in payload["native_preparations"]:
            plan = FacetPlanV5.model_validate_json(canonical_json(task["plan"]))
            queries = literal_probes(collection, task["question"], task["search"], task["language"], plan)
            vectors = tuple(np.asarray(runtime.cache.resolve_local("demo", text=query_embedding_input(q),
                recipe=runtime.query_recipe, encoder=runtime.encoder).record.vector, dtype=np.float32) for q in queries)
            base = next(r for r in payload["recall_cells"] if r["intent"] == task["intent"] and r["language"] == task["language"]
                and r["arm"] == "V5_CHUNK_Q")["pool_ids"]
            for recipe in recipes():
                pool = build_reranked_parent_pool(collection, lexical, index, question=task["question"], search=task["search"],
                    language=task["language"], plan=plan, vectors=vectors, base_ids=base, recipe=recipe, reranker=runtime.reranker)
                pair = []
                for balanced in (False, True):
                    answer, audit = pack_section_pool(collection, lexical, pool, balanced=balanced,
                        generation_recipe=runtime.recipe, run_id=RUN, owner_id="parent-section-local")
                    arm = f"CE_PARENT_{recipe.field_mode}_R{recipe.radius}_" + ("QB" if balanced else "Q")
                    baseline = next(r for r in payload["recall_cells"] if r["intent"] == task["intent"] and r["language"] == task["language"]
                        and r["arm"] == f"V5_WINDOW_R{recipe.radius}_" + ("QB" if balanced else "Q"))
                    row = {k: task[k] for k in ("intent", "language", "question", "search")}
                    row.update(arm=arm, pool_ids=pool.pool_ids, pool_fingerprint=pool.fingerprint, pool_trace=json.loads(pool.trace_json),
                        bundles=json.loads(pool.bundles_json), context_chunk_ids=answer.packed.chunk_ids,
                        source_message=answer.packed.serialized, user_message_proxy_tokens=answer.packed.proxy_tokens,
                        binding=answer.binding.model_dump(mode="json"), packing_audit=audit,
                        context_audit=audit_source_dependencies(docs, chunks, answer.packed.chunk_ids, deps, tenant_id="demo"),
                        source_reachability=[source_reachability(dep, pool.pool_ids, answer.packed.chunk_ids, chunks) for dep in deps],
                        context_added_vs_v5=sorted(set(answer.packed.chunk_ids)-set(baseline["context_chunk_ids"])),
                        context_removed_vs_v5=sorted(set(baseline["context_chunk_ids"])-set(answer.packed.chunk_ids)),
                        generation_executed=False, graph_records_used=0)
                    n, s = rendered_sources(row, docs, chunks, tokenizer)
                    count += n
                    spans += s
                    pair.append(row)
                    cells.append(row)
                if pair[0]["pool_fingerprint"] != pair[1]["pool_fingerprint"] or pair[0]["pool_trace"] != pair[1]["pool_trace"]:
                    raise ValueError("PARENT_SECTION_PAIR_POOL")
                pairs.append({"intent": task["intent"], "language": task["language"], "recipe_sha256": recipe.fingerprint,
                    "same_plan_pool_bundles": True, "context_changed": pair[0]["context_chunk_ids"] != pair[1]["context_chunk_ids"]})
            print(json.dumps({"intent": task["intent"], "language": task["language"], "cells": len(cells)}), flush=True)
        if runtime.encoder.inference_calls:
            raise ValueError("PARENT_SECTION_REUSES_QUERY_VECTORS_ONLY")
    immutable(OUTPUT / "receipts.json", {"cells": cells, "recorded_v5_baseline": payload["recall_cells"]})
    arms = []
    for arm in sorted({r["arm"] for r in cells}):
        own = [r for r in cells if r["arm"] == arm]
        arms.append({"arm": arm, "annotation_statuses": dict(Counter(c["status"] for r in own for c in r["context_audit"]["checks"])),
            "reachability": dict(Counter(c["stage"] for r in own for c in r["source_reachability"])),
            "context_chunks": [len(r["context_chunk_ids"]) for r in own], "proxy_tokens": [r["user_message_proxy_tokens"] for r in own]})
    if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
        raise ValueError("PARENT_SECTION_CODE_CHANGED")
    result = {"status": "EXACT_OFFLINE_REPLAY_MATCH" if replay else "RERANKED_PARENT_DIAGNOSTIC_RECORDED", "cells": len(cells),
        "q_qb_pairs": len(pairs), "local_reranker_batches_this_invocation": runtime.reranker.inference_calls, "rendered_source_appearances": count, "exact_source_spans": spans,
        "receipts_sha256": sha256_bytes((OUTPUT / "receipts.json").read_bytes()), "arms": arms,
        "new_model_api_calls": 0, "new_local_encoding_calls": 0, "aws_calls": 0, "generation_calls": 0,
        "extraction_calls": 0, "quality_accuracy_measured": False, "human_reviewed": 0, "serving_published": False}
    immutable(OUTPUT / ("replay.json" if replay else "report.json"), result)
    immutable(OUTPUT / ("replay-pairs.json" if replay else "pairs.json"), pairs)
    print(json.dumps({k: v for k, v in result.items() if k != "arms"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--run", action="store_true")
    choice.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=AssertionError("PARENT_SECTION_NETWORK_BLOCKED")))
        if args.run or args.replay:
            run(replay=args.replay)
        else:
            collection, _, _ = load()
            frozen = freeze(collection)
            print(json.dumps({"status": "PARENT_SCOPE_FROZEN_NO_RANKING", "planned_cells": frozen["new_cells"]}))
