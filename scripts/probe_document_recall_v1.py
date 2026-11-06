"""Local, nonblind diagnostic of document breadth and adjacent whole chunks.

No serving changes or generation. Retrieval functions accept source data and
query text only; reviewed witnesses enter solely after pools are frozen.
"""

import json
from collections import defaultdict
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from rank_bm25 import BM25Okapi

from app.contracts import embedding_input, sha256_bytes
from app.graph_collection_migration import collection_records
from app.profiled_source_context import SourceContextProfile, pack_profiled_context
from app.retrieval import lexical_tokens, reciprocal_rank_fusion
from scripts import run_scoped_flow_canary as parent
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable

OUT = parent.ROOT / "results/document-recall-probe-2026-11-05-v1"


class DocumentRecallProbe:
    def __init__(self, documents, chunks):
        self.documents, self.chunks = documents, chunks
        self.doc_ids, self.ids = sorted(documents), sorted(chunks)
        self.doc_terms = [lexical_tokens(documents[k].title + "\n" + documents[k].normalized_text) for k in self.doc_ids]
        self.chunk_terms = [lexical_tokens(embedding_input(documents[chunks[k].document_version_id], chunks[k])) for k in self.ids]
        self.doc_index, self.chunk_index = BM25Okapi(self.doc_terms), BM25Okapi(self.chunk_terms)
        self.by_doc, self.positions = defaultdict(list), {}
        for key in self.ids:
            self.by_doc[chunks[key].document_version_id].append(key)
        for keys in self.by_doc.values():
            keys.sort(key=lambda k: (chunks[k].start_char, k))
            self.positions.update({key: i for i, key in enumerate(keys)})

    def query(self, query):
        tokens = lexical_tokens(query)
        own = set(tokens)
        ds, cs = self.doc_index.get_scores(tokens), self.chunk_index.get_scores(tokens)
        eligible = [i for i, terms in enumerate(self.doc_terms) if own.intersection(terms)]
        docs = sorted(eligible, key=lambda i: (-float(ds[i]), self.doc_ids[i]))[:8]
        indices = {key: i for i, key in enumerate(self.ids)}
        heads, tails = [], []
        for i in docs:
            keys = self.by_doc[self.doc_ids[i]]
            ranked = sorted(keys, key=lambda k: (-float(cs[indices[k]]), k))[:2]
            heads.extend(ranked)
            for key in ranked:
                j = self.positions[key]
                tails.extend(keys[max(0, j-1):j+2])
        return tuple(dict.fromkeys((*heads, *tails)))

    def pool(self, queries, original, *, graph_prefix=()):
        rows = [self.query(q) for q in queries]
        # First two from each actual query, then RRF. Graph support remains whole.
        prefix = [key for depth in range(2) for row in rows for key in row[depth:depth+1]]
        fused = reciprocal_rank_fusion([original, *rows])
        return tuple(dict.fromkeys((*graph_prefix, *prefix, *fused)))[:40]


def diversified_bundles(bundles, chunks, per_doc):
    """Prefer bounded page concentration, retaining every atomic graph bundle."""
    chosen, remaining, counts = [], [], defaultdict(int)
    for bundle in bundles:
        docs = [chunks[key].document_version_id for key in bundle["chunk_ids"]]
        additions = {d: docs.count(d) for d in set(docs)}
        if bundle.get("assertion_ids") or all(counts[d] + n <= per_doc for d, n in additions.items()):
            chosen.append(bundle)
            for d, n in additions.items():
                counts[d] += n
        else:
            remaining.append(bundle)
    return chosen + remaining


def run():
    parent.checked_scope()
    collection, _, graph = load_migrated()
    docs, chunks, _ = collection_records(collection)
    sources = sorted((parent.OUT / "cells").glob("*.json"))
    if len(sources) != 20:
        raise ValueError("DOCUMENT_PROBE_CANARY_INCOMPLETE")
    scope = {"version": "document-recall-probe-v1", "collection": collection.manifest_sha256,
        "code_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "input_sha256": {p.name: sha256_bytes(p.read_bytes()) for p in sources},
        "policies": [2, 4], "max_document_candidates_per_query": 8, "per_document_heads": 2,
        "adjacent_chunk_radius": 1, "max_pool": 40, "max_sources": 12,
        "retrieval_receives_no_review_or_witness_ids": True, "exposure": "NONBLIND_DIAGNOSTIC_NOT_BENCHMARK",
        "new_api_calls": 0, "serving_changed": False}
    immutable(OUT / "scope.json", scope)
    retriever = DocumentRecallProbe(docs, chunks)
    rows = []
    for path in sources:
        source = json.loads(path.read_bytes())
        task, trace = source["task"], source["trace"]
        graph_bundles = [b for b in trace["bundles"] if b.get("assertion_ids")][:2]
        graph_prefix = tuple(dict.fromkeys(k for b in graph_bundles for k in b["chunk_ids"]))
        pool = retriever.pool(trace["queries"], source["pool_ids"], graph_prefix=graph_prefix)
        eligible_graph = [b for b in graph_bundles if set(b["chunk_ids"]) <= set(pool)]
        bundled = {k for b in eligible_graph for k in b["chunk_ids"]}
        bundles = eligible_graph + [{"bundle_id": "doc-recall-" + k, "chunk_ids": [k], "query_index": 0}
            for k in pool if k not in bundled]
        for cap in (2, 4):
            ordered = diversified_bundles(bundles, chunks, cap)
            context, packing = pack_profiled_context(docs, chunks, pool, ordered,
                question=task["question"], language=task["language"], tenant_id=collection.manifest.tenant_id,
                collection_sha256=collection.manifest_sha256, profile=SourceContextProfile(max_chunks=12),
                graph_records={f.assertion_id: json.loads(f.context_json) for f in graph.facts})
            rows.append({"parent_cell": path.name, "task": task, "method": source["method"], "per_document_cap": cap,
                "pool_ids": pool, "context_ids": context.chunk_ids, "context_fingerprint": context.fingerprint,
                "packing": packing, "native_qb_allocator_reexecuted": False,
                "policy_scope": "DOCUMENT_ADMISSION_ABLATION_NOT_METHOD_PROMOTION"})
    immutable(OUT / "pools.json", {"rows": rows})
    # Evaluation is deliberately after immutable runtime receipts.
    from scripts.audit_scoped_flow_canary import pool_missing
    from scripts.run_context_profiles import load_review

    rubric, witnesses, _ = load_review()
    reviews = []
    for row in rows:
        intent = row["task"]["intent"]
        review = {"cell": row["parent_cell"], "per_document_cap": row["per_document_cap"],
            "missing_pool": pool_missing(rubric, witnesses, chunks, row["pool_ids"], intent),
            "missing_context": pool_missing(rubric, witnesses, chunks, row["context_ids"], intent)}
        reviews.append(review)
        print(json.dumps(review))
    immutable(OUT / "review.json", {"reviews": reviews, "new_model_calls": 0,
        "human_reviewers": 0, "answer_quality_measured": False})


if __name__ == "__main__":
    with ExitStack() as stack:
        for call in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(call, side_effect=AssertionError("DOCUMENT_PROBE_NETWORK_FORBIDDEN")))
        run()
