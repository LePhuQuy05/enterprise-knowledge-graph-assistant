"""Build and exercise a sparse graph on all 316 documents. No API calls.

Existing extraction outputs may migrate only through unique source-quote
matches. Coverage is reported explicitly; this is not full relation extraction.
"""

import dataclasses
import json
from pathlib import Path

import numpy as np

from app.collection_answering import _evidence_view
from app.collection_graph import (
    expand_graph,
    graph_fingerprint,
    literal_mentions,
    reanchor_assertions,
    reanchor_mentions,
    service_entities,
)
from app.collection_lexical import load_global_lexical_index, retrieve_collection_hybrid
from app.collections import load_collection
from app.context import pack_context
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.graph_retrieval import GraphCandidate, GraphRetrievalResult
from app.local_bge import BgeEncoder
from app.retrieval import reciprocal_rank_fusion
from app.snapshot import SnapshotContent, load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    receipt = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(receipt["collection_path"]), tenant_id="demo", embedding_model_id="BAAI/bge-small-en-v1.5")
    index = load_global_lexical_index(Path(receipt["lexical_manifest_path"]), collection)
    documents, chunks = {}, {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        documents.update(shard.documents)
        chunks.update(shard.chunks)
    old = json.loads((ROOT / "results/pilot_graph_snapshot.json").read_bytes())
    previous = load_snapshot(Path(old["snapshot_directory"]), tenant_id="demo", embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=old["manifest_sha256"])
    entities = service_entities(receipt["service_counts"], previous.content.entities)
    assertions, rejected, provenance = reanchor_assertions(previous.documents, previous.content.assertions, tuple(documents.values()), tuple(chunks.values()))
    # Generic extraction candidates ("Resource", "key policy", etc.) must not
    # become globally merged identities merely because their spelling repeats.
    scoped_spans = {}
    for a in assertions:
        for key in (a.subject_id, a.object_id):
            scoped_spans.setdefault((key, a.chunk_id), []).append(a.evidence)
    global_entities = {e.entity_id for e in entities if e.canonical_key.startswith("aws/") and e.entity_type == "AWS_SERVICE"}
    all_mentions = literal_mentions(entities, tuple(chunks.values()), canonicalization_version="collection-literal-entities-v1")
    scoped_mentions = reanchor_mentions(previous.documents, previous.content.mentions, tuple(documents.values()), tuple(chunks.values()))
    chosen = {m.mention_id: m for m in all_mentions if m.entity_id in global_entities or any(
        span.start_char <= m.start_char < m.end_char <= span.end_char for span in scoped_spans.get((m.entity_id, m.chunk_id), ()))}
    chosen.update((m.mention_id, m) for m in scoped_mentions)
    mentions = tuple(sorted(chosen.values(), key=lambda m: m.mention_id))
    fingerprint = graph_fingerprint(collection.manifest_sha256, entities, mentions, assertions)
    output = {"schema_version": 1, "kind": "SPARSE_SOURCE_REANCHORED_COLLECTION_GRAPH", "collection_version": collection.manifest.collection_version,
        "collection_manifest_sha256": collection.manifest_sha256, "graph_sha256": fingerprint,
        "documents": len(documents), "chunks": len(chunks), "service_groups": len(receipt["service_counts"]),
        "entities": [e.model_dump(mode="json") for e in entities], "mentions": [m.model_dump(mode="json") for m in mentions],
        "assertions": [a.model_dump(mode="json") for a in assertions], "rejected_migrations": rejected,
        "migration_provenance": provenance, "review": "PENDING", "full_relation_extraction": False,
        "relation_source_chunks": len({a.chunk_id for a in assertions}), "relation_source_documents": len({a.document_version_id for a in assertions}),
        "mention_covered_chunks": len({m.chunk_id for m in mentions}), "inspected_chunks": len(chunks),
        "model_calls": 0, "aws_calls": 0, "cloud_published": False, "traces": []}
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    if encoder.recipe("document") != collection.manifest.embedding_recipe:
        raise ValueError("COLLECTION_GRAPH_ENCODER_MISMATCH")
    question = "Which policies must permit cross-account use of an AWS KMS key?"
    vector = np.asarray(encoder.encode([query_embedding_input(question)], role="query")[0], dtype=np.float32)
    baseline = tuple(retrieve_collection_hybrid(collection, index, question, vector)[:40])
    scores = {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        scores.update((key, float(score)) for key, score in zip(shard.chunk_ids, shard.vectors @ vector, strict=True))
    for hops in (1, 2):
        trace = expand_graph(entities, mentions, assertions, baseline, scores, hops=hops)
        ranked = tuple(reciprocal_rank_fusion([baseline, [c["chunk_id"] for c in trace["graph_candidates"]]])[:40])
        candidates = tuple(GraphCandidate(**c) for c in trace["graph_candidates"] if c["chunk_id"] in ranked and set(c["supporting_chunk_ids"]) <= set(ranked))
        view = _evidence_view(collection, index, ranked)
        view = dataclasses.replace(view, entities={e.entity_id: e for e in entities}, content=SnapshotContent((), (), entities, mentions, assertions))
        result = GraphRetrievalResult(ranked, baseline, candidates, tuple(trace["seed_entity_ids"]), f"graph_{hops}hop", bool(candidates), ("SPARSE_GRAPH_SEMANTIC_REVIEW_PENDING",), trace["assertion_visits"], trace["visited_entities"])
        packed = pack_context(view, result)
        output["traces"].append({"question": question, **trace, "final_ranking": ranked,
            "baseline_ranking": baseline, "context_chunk_ids": packed.chunk_ids,
            "context_assertion_ids": packed.assertion_ids, "context_proxy_tokens": packed.proxy_tokens,
            "warnings": packed.warnings, "used_graph": bool(candidates), "mode_used": result.mode_used})
    directory = ROOT / "data/collection-graph" / fingerprint
    directory.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(output) + b"\n"
    path = directory / "walkthrough-v2.json"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("IMMUTABLE_COLLECTION_GRAPH_CHANGED")
    path.write_bytes(raw)
    report = {k: output[k] for k in ("kind", "collection_version", "collection_manifest_sha256", "graph_sha256", "documents", "chunks", "service_groups", "relation_source_chunks", "relation_source_documents", "mention_covered_chunks", "inspected_chunks", "review", "full_relation_extraction", "model_calls", "aws_calls")}
    report.update(graph_path=str(path), graph_file_sha256=sha256_bytes(raw), entities=len(entities), mentions=len(mentions), assertions=len(assertions), rejected_migrations=len(rejected))
    target = ROOT / "results/collection-graph/walkthrough-v1.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
