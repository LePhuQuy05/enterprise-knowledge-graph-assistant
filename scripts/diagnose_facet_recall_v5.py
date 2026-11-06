"""Post-hoc focus-only probes, never a replacement native plan or benchmark."""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np
from rank_bm25 import BM25Okapi

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.graph_collection_migration import collection_records
from app.retrieval import lexical_tokens, reciprocal_rank_fusion
from app.source_dependency_audit import SourceDependency
from app.strict_collection_methods import STOPWORDS
from scripts.audit_collection319_methods import source_reachability
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.run_collection_benchmark import OfflineRuntime
from scripts.run_facet_recall_v5 import OUTPUT, ROOT


def diagnose():
    collection, paths, _ = load_migrated()
    docs, chunks, _ = collection_records(collection)
    p = json.loads((OUTPUT / "receipts.json").read_bytes())
    annotations = json.loads((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())
    deps = [SourceDependency.model_validate_json(canonical_json(r["dependency"])) for r in annotations["source_controls"]]
    deps.append(SourceDependency.model_validate_json(canonical_json(annotations["parent_kms_dependency"])))
    ids = sorted(docs)
    tokens = [tuple(t for t in lexical_tokens(docs[k].title) if t not in STOPWORDS) for k in ids]
    bm25 = BM25Okapi(tokens)
    sets = list(map(set, tokens))
    result = []
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ())
        for prep in p["native_preparations"]:
            # Algorithm gets literal user task spans only; source annotations enter scoring nowhere.
            queries = (prep["search"], *(f["focus_quote"] for f in prep["plan"]["facets"]))
            rankings = []
            for query in queries:
                vector = np.asarray(runtime.cache.resolve_local("demo", text=query_embedding_input(query),
                    recipe=runtime.query_recipe, encoder=runtime.encoder).record.vector, dtype=np.float32)
                v = [hit.chunk_id for hit in collection.search_vector(vector, k=20)]
                b = [key for key, _ in lexical.search(query, k=20)]
                rankings.append(tuple(reciprocal_rank_fusion([v, b])))
            focus_pool = tuple(reciprocal_rank_fusion(rankings))[:40]
            actual = next(r for r in p["recall_cells"] if r["intent"] == prep["intent"]
                and r["language"] == prep["language"] and r["arm"] == "V5_WINDOW_R2_Q")
            actual_rankings = actual["pool_trace"]["title_rankings"]
            controls = []
            for dep in deps:
                per_query = []
                for ranking in actual_rankings:
                    terms = tuple(t for t in lexical_tokens(ranking["query"]) if t not in STOPWORDS)
                    scores = bm25.get_scores(terms)
                    eligible = [i for i in range(len(ids)) if sets[i] & set(terms)]
                    order = [ids[i] for i in sorted(eligible, key=lambda i: (-float(scores[i]), ids[i]))]
                    windows = [b for b in actual["bundles"] if b["kind"] == "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT"
                        and b["query_index"] == ranking["query_index"] and b["document_version_id"] == dep.document_version_id]
                    per_query.append({"query_index": ranking["query_index"], "query": ranking["query"],
                        "target_title_rank": order.index(dep.document_version_id)+1 if dep.document_version_id in order else None,
                        "top4_title_candidate": dep.document_version_id in ranking["document_ranking"],
                        "admitted_windows": windows})
                controls.append({"source_uri": dep.source_uri, "dependency_id": dep.dependency_id,
                    "native_window_reachability": source_reachability(dep, actual["pool_ids"], actual["context_chunk_ids"], chunks),
                    "focus_only_pool_reachability": source_reachability(dep, focus_pool, (), chunks),
                    "native_title_ranking": per_query})
            result.append({"intent": prep["intent"], "language": prep["language"],
                "derived_queries_not_native_search_queries": queries,
                "derived_focus_pool_ids": focus_pool, "controls": controls})
    artifact = {"status": "POST_HOC_DIAGNOSTIC_ONLY_NO_PROMOTION", "inputs": result,
        "source_annotations_used_for_ranking": False, "derived_from_exposed_native_focus_quotes": True,
        "receipts_sha256": sha256_bytes((OUTPUT / "receipts.json").read_bytes()),
        "model_calls": 0, "aws_calls": 0, "generation_calls": 0, "quality_accuracy_measured": False,
        "code_sha256": sha256_bytes(Path(__file__).read_bytes())}
    immutable(OUTPUT / "focus-diagnostic.json", artifact)
    print(json.dumps({"status": artifact["status"], "inputs": len(result), "model_calls": 0,
        "native_stages": [r["native_window_reachability"]["stage"] for x in result for r in x["controls"]],
        "focus_pool_stages_context_not_packed": [r["focus_only_pool_reachability"]["stage"] for x in result for r in x["controls"]]}))


if __name__ == "__main__":
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=AssertionError("FACET_DIAGNOSTIC_NETWORK_BLOCKED")))
        diagnose()
