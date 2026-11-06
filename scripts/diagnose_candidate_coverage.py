"""Read-only candidate diagnosis from a saved query vector; never encode or call APIs."""

import argparse
import json
import re
import sqlite3
from collections import Counter, defaultdict
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from app.collection_answering import validate_collection_lexical_binding
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import LocalBgeRecipe, canonical_json, embedding_cache_key, sha256_bytes
from app.embeddings import MAX_CACHE_RECORD_BYTES, CachedEmbedding, query_embedding_input
from app.rag_methods import MethodResultV1, RunInputV1
from app.retrieval import lexical_tokens, reciprocal_rank_fusion

ROOT = Path(__file__).resolve().parents[1]
TARGET = "chk_e32449e2e56a676a9878b4bac5dfe196b8ac516c21a0aadc6c356faaf5eac18e"
MAX_CHUNKS = 20_000
MAX_DOCUMENTS = 1_000
MAX_SHARDS = 10
SIMILARITY_THRESHOLD = 0.75


def bounded_json(path, limit):
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= limit:
        raise ValueError("CANDIDATE_INPUT_PATH_OR_SIZE")
    raw = path.read_bytes()
    if not 1 <= len(raw) <= limit:
        raise ValueError("CANDIDATE_INPUT_SIZE_CHANGED")
    return json.loads(raw), sha256_bytes(raw)


def cached_query(database, *, tenant_id, query, recipe, namespace="LIVE"):
    """Read exactly one existing READY row without creating a DB, claim or encoder."""
    if namespace not in {"LIVE", "FIXTURE"} or database.is_symlink() or not database.is_file():
        raise ValueError("CANDIDATE_CACHE_PATH_OR_NAMESPACE")
    query = query_embedding_input(query)
    key = embedding_cache_key(tenant_id, query, recipe)
    db = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=5)
    try:
        db.execute("PRAGMA trusted_schema=OFF")
        db.execute("PRAGMA query_only=ON")
        row = db.execute(
            "SELECT state,length(payload),payload_sha256 FROM embeddings "
            "WHERE namespace=? AND cache_key=?", (namespace, key),
        ).fetchone()
        if row is None or row[0] != "READY":
            raise ValueError("CANDIDATE_READY_QUERY_VECTOR_REQUIRED")
        if row[1] is None or not 1 <= row[1] <= MAX_CACHE_RECORD_BYTES:
            raise ValueError("CANDIDATE_CACHE_PAYLOAD_SIZE")
        raw = db.execute("SELECT payload FROM embeddings WHERE namespace=? AND cache_key=?",
                         (namespace, key)).fetchone()[0]
    finally:
        db.close()
    if not isinstance(raw, bytes) or sha256_bytes(raw) != row[2]:
        raise ValueError("CANDIDATE_CACHE_CHECKSUM")
    record = CachedEmbedding.model_validate_json(raw)
    if (record.namespace != namespace or record.tenant_id != tenant_id or record.cache_key != key
            or record.input_text != query or record.recipe != recipe):
        raise ValueError("CANDIDATE_CACHE_BINDING")
    return record, row[2]


def complete_rankings(collection, lexical, query, vector):
    """Target-blind rankings over a bounded, pinned local collection."""
    counts = collection.manifest.counts
    if (not 1 <= counts.chunks <= MAX_CHUNKS or not 1 <= counts.documents <= MAX_DOCUMENTS
            or not 1 <= len(collection.manifest.shards) <= MAX_SHARDS):
        raise ValueError("CANDIDATE_COLLECTION_DIAGNOSTIC_BOUND")
    if (not isinstance(vector, np.ndarray) or vector.dtype != np.dtype("float32")
            or vector.shape != (collection.manifest.embedding_recipe.dimension,)
            or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5, rtol=0)):
        raise ValueError("CANDIDATE_QUERY_VECTOR_INVALID")
    dense, records = [], {}
    for ref in sorted(collection.manifest.shards, key=lambda item: item.shard_id):
        snapshot = collection.get_shard(ref.shard_id)
        if snapshot.manifest_sha256 != ref.manifest_sha256:
            raise ValueError("CANDIDATE_SHARD_BINDING")
        scores = snapshot.vectors @ vector
        for key, score in zip(snapshot.chunk_ids, scores, strict=True):
            chunk = snapshot.chunks[key]
            document = snapshot.documents[chunk.document_version_id]
            if (key in records or chunk.tenant_id != collection.manifest.tenant_id
                    or document.tenant_id != collection.manifest.tenant_id
                    or document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text
                    or not np.isfinite(score)):
                raise ValueError("CANDIDATE_SOURCE_INTEGRITY")
            records[key] = (chunk, document, ref)
            dense.append((key, float(score)))
        del snapshot
    if len(records) != counts.chunks:
        raise ValueError("CANDIDATE_SOURCE_COUNT")
    dense.sort(key=lambda row: (-row[1], row[0]))
    # This extends the existing BM25 scoring definition for diagnostic ranks
    # beyond top 20. It is not a new runtime retriever or a changed query.
    scores = {}
    for term, query_tf in Counter(lexical_tokens(query)).items():
        row = lexical._connection.execute("SELECT idf FROM terms WHERE term=?", (term,)).fetchone()
        if row is None:
            continue
        for key, tf, length in lexical._connection.execute(
            "SELECT p.chunk_id,p.tf,c.document_length FROM postings p "
            "JOIN chunks c ON c.chunk_id=p.chunk_id WHERE p.term=?", (term,),
        ):
            if key not in records:
                raise ValueError("CANDIDATE_LEXICAL_FOREIGN_SOURCE")
            denominator = tf + 1.5 * (1 - 0.75 + 0.75 * length / lexical.manifest.average_document_length)
            contribution = row[0] * (tf * 2.5 / denominator)
            scores[key] = scores.get(key, 0.0) + query_tf * contribution
    bm25 = sorted(scores.items(), key=lambda row: (-row[1], row[0]))
    if bm25[:20] != lexical.search(query, k=20):
        raise ValueError("CANDIDATE_BM25_PREFIX_MISMATCH")
    return dense, bm25, records


def source_description(key, records):
    chunk, document, ref = records[key]
    parts = urlsplit(document.source_uri).path.strip("/").split("/")
    return {
        "chunk_id": key, "document_version_id": document.document_version_id,
        "source_uri": document.source_uri, "title": document.title,
        "source_family_path_prefix": parts[0] if parts else "",
        "section_path": list(chunk.section_path), "start_char": chunk.start_char,
        "end_char": chunk.end_char, "text_sha256": chunk.text_sha256,
        "document_normalized_sha256": document.normalized_sha256,
        "document_raw_sha256": document.raw_sha256, "shard_id": ref.shard_id,
        "shard_manifest_sha256": ref.manifest_sha256,
    }


def pool_redundancy(ids, records):
    exact = defaultdict(list)
    overlaps, similar = [], []
    token_sets = {}
    for key in ids:
        chunk = records[key][0]
        exact[chunk.text_sha256].append(key)
        token_sets[key] = set(lexical_tokens(chunk.text))
    for left, right in combinations(ids, 2):
        a, b = records[left][0], records[right][0]
        start, end = max(a.start_char, b.start_char), min(a.end_char, b.end_char)
        if a.document_version_id == b.document_version_id and start < end:
            overlap = records[left][1].normalized_text[start:end]
            overlaps.append({"left_chunk_id": left, "right_chunk_id": right,
                "document_version_id": a.document_version_id, "start_char": start,
                "end_char": end, "overlap_characters": end - start,
                "overlap_text_sha256": sha256_bytes(overlap.encode()),
                "overlap_fraction_of_shorter_chunk": (end - start) / min(len(a.text), len(b.text))})
        union = token_sets[left] | token_sets[right]
        score = len(token_sets[left] & token_sets[right]) / len(union) if union else 0
        if score >= SIMILARITY_THRESHOLD:
            similar.append({"left_chunk_id": left, "right_chunk_id": right,
                            "token_set_jaccard": score,
                            "exact_text_equal": a.text_sha256 == b.text_sha256})
    return {
        "exact_text_duplicate_groups": [{"text_sha256": digest, "chunk_ids": keys}
            for digest, keys in sorted(exact.items()) if len(keys) > 1],
        "same_document_span_overlap_pairs": overlaps,
        "lexical_similarity_pairs": similar,
        "lexical_similarity_definition": "Jaccard of sets from ASCII-alnum-lower tokens of chunk text only",
        "lexical_similarity_threshold": SIMILARITY_THRESHOLD,
        "limitations": ["Same-document overlap means shared source characters, not independent evidence.",
            "Token similarity is diagnostic only, not semantic duplication or proof that facts are interchangeable.",
            "No source, chunk, query, candidate or packer was deduplicated or changed."],
    }


def diagnose(collection, lexical, receipt, record, target):
    if re.fullmatch(r"chk_[0-9a-f]{64}", target) is None:
        raise ValueError("CANDIDATE_TARGET_ID_INVALID")
    validate_collection_lexical_binding(collection, lexical)
    request = RunInputV1.model_validate_json(canonical_json(receipt["input"]))
    if (receipt.get("status") != "COMPLETED" or request.source_scope != "local_collection"
            or receipt["collection_manifest_sha256"] != collection.manifest_sha256
            or receipt["collection_version"] != collection.manifest.collection_version):
        raise ValueError("CANDIDATE_RECEIPT_COLLECTION_BINDING")
    methods = [MethodResultV1.model_validate_json(canonical_json(row)) for row in receipt["methods"]]
    baseline = next((row for row in methods if row.effective_method == "hybrid"), None)
    if baseline is None or baseline.status not in {"COMPLETED", "RETRIEVED"}:
        raise ValueError("CANDIDATE_COMPLETED_HYBRID_REQUIRED")
    document_recipe = collection.manifest.embedding_recipe
    expected_query = document_recipe.model_copy(update={
        "input_role": "query", "input_recipe_version": "question-text-v1"})
    if (not isinstance(expected_query, LocalBgeRecipe) or record.recipe != expected_query
            or record.tenant_id != collection.manifest.tenant_id
            or record.input_text != query_embedding_input(receipt["search_query"])
            or baseline.config.query_recipe_sha256 != record.recipe.fingerprint
            or baseline.config.document_recipe_sha256 != document_recipe.fingerprint
            or baseline.config.collection_manifest_sha256 != collection.manifest_sha256
            or baseline.config.lexical_index_version != lexical.manifest.index_version):
        raise ValueError("CANDIDATE_QUERY_OR_INDEX_BINDING")
    query = record.input_text
    dense, bm25, records = complete_rankings(
        collection, lexical, query, np.asarray(record.vector, dtype=np.float32))
    # Target has not participated in ranking, candidate construction or query transformation.
    if target not in records:
        raise ValueError("CANDIDATE_TARGET_NOT_IN_CORPUS")
    dense_ids, bm25_ids = [key for key, _ in dense], [key for key, _ in bm25]
    pool = reciprocal_rank_fusion([dense_ids[:20], bm25_ids[:20]])
    if tuple(pool) != baseline.ranked_chunk_ids:
        raise ValueError("CANDIDATE_ORIGINAL_HYBRID_REPLAY_MISMATCH")
    for row in methods:
        if row.effective_method == "hybrid_reranked" and row.rerank_trace is not None:
            if row.rerank_trace.baseline_candidate_ids != tuple(pool):
                raise ValueError("CANDIDATE_RERANK_BASELINE_POOL_MISMATCH")
    families = defaultdict(list)
    for key in pool:
        families[source_description(key, records)["source_family_path_prefix"]].append(key)
    def branch(rows, ids):
        return {"target_rank": ids.index(target) + 1 if target in ids else None,
                "target_score": dict(rows).get(target), "target_in_top20": target in ids[:20],
                "ranked_eligible_chunks": len(rows),
                "top20": [{**source_description(key, records), "rank": rank, "score": score}
                          for rank, (key, score) in enumerate(rows[:20], 1)]}
    ch = records[target][0]
    return {
        "schema_version": 1, "kind": "SAVED_QUERY_CANDIDATE_COVERAGE_DIAGNOSTIC",
        "status": "PASS_EXACT_ORIGINAL_HYBRID_RECONSTRUCTION", "checked_at": datetime.now(UTC).isoformat(),
        "original_question": request.original_question, "search_query": query,
        "query_transform": receipt.get("query_transform"), "query_recipe_sha256": record.recipe.fingerprint,
        "query_embedding_cache_key": record.cache_key,
        "query_vector_float32_sha256": sha256_bytes(np.asarray(record.vector, dtype=np.float32).tobytes()),
        "collection_version": collection.manifest.collection_version,
        "collection_manifest_sha256": collection.manifest_sha256,
        "lexical_index_version": lexical.manifest.index_version,
        "documents": collection.manifest.counts.documents, "chunks": collection.manifest.counts.chunks,
        "target": {**source_description(target, records), "text": ch.text,
            "target_used_for": "POST_RETRIEVAL_DIAGNOSTIC_ONLY_NOT_INPUT_TO_RANKING",
            "in_original_hybrid_pool": target in pool,
            "original_hybrid_pool_rank": pool.index(target) + 1 if target in pool else None,
            "context_membership_by_method": {row.effective_method: target in row.context_chunk_ids for row in methods}},
        "dense": branch(dense, dense_ids), "bm25": branch(bm25, bm25_ids),
        "original_hybrid_pool": {"candidate_count": len(pool), "candidate_chunk_ids": pool,
            "candidate_pool_sha256": sha256_bytes(canonical_json(pool)),
            "receipt_rank_ids_equal": True,
            "source_family_definition": "First path component of the source URI, not a semantic service label",
            "source_families": [{"path_prefix": family, "candidate_count": len(keys),
                "document_count": len({records[key][0].document_version_id for key in keys}),
                "chunk_ids": keys} for family, keys in sorted(families.items())],
            "sources": [source_description(key, records) for key in pool]},
        "source_redundancy": pool_redundancy(pool, records),
        "actions": {"new_embeddings": 0, "model_encoder_calls": 0, "reranker_calls": 0,
                    "paid_api_calls": 0, "aws_calls": 0, "network_calls": 0,
                    "queries_changed": 0, "candidates_injected": 0},
        "quality_metrics": None, "human_reviewed": False, "gold": False, "holdout": False,
        "limitations": ["One previously inspected development query; target ranks are not gold recall or answer accuracy.",
            "Reranking cannot select a source excluded from both top-20 branches.",
            "The full BM25 diagnostic preserves eligible zero/negative scores and validates its top 20 against the runtime index.",
            "No new answer was generated; semantic alternatives require separate source review.",
            "The saved actual rewrite is reused unchanged; this does not evaluate rewrite quality."],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, default=ROOT / "results/reranker-answering/pilot-v1/case-2.json")
    parser.add_argument("--pipeline", type=Path, default=ROOT / "results/corpus-expansion-v2/pipeline.json")
    parser.add_argument("--cache-db", type=Path, default=ROOT / "data/cache/local-bge/embeddings.sqlite3")
    parser.add_argument("--target-chunk-id", default=TARGET)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise ValueError("CANDIDATE_DIAGNOSTIC_OUTPUT_IMMUTABLE")
    if args.output.resolve() in {p.resolve() for p in (args.receipt, args.pipeline, args.cache_db)}:
        raise ValueError("CANDIDATE_DIAGNOSTIC_CANNOT_OVERWRITE_INPUT")
    if re.fullmatch(r"chk_[0-9a-f]{64}", args.target_chunk_id) is None:
        raise ValueError("CANDIDATE_TARGET_ID_INVALID")
    receipt, receipt_sha = bounded_json(args.receipt, 4 * 1024 * 1024)
    pipeline, pipeline_sha = bounded_json(args.pipeline, 64 * 1024)
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=0)
    document = collection.manifest.embedding_recipe
    if not isinstance(document, LocalBgeRecipe):
        raise ValueError("CANDIDATE_LOCAL_BGE_REQUIRED")
    recipe = document.model_copy(update={"input_role": "query", "input_recipe_version": "question-text-v1"})
    record, payload_sha = cached_query(args.cache_db, tenant_id="demo", query=receipt["search_query"], recipe=recipe)
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        report = diagnose(collection, lexical, receipt, record, args.target_chunk_id)
    report["input_bindings"] = {"receipt_path": str(args.receipt), "receipt_sha256": receipt_sha,
        "pipeline_path": str(args.pipeline), "pipeline_sha256": pipeline_sha,
        "cache_database_path": str(args.cache_db), "cache_payload_sha256": payload_sha,
        "cache_read_mode": "SQLITE_MODE_RO_QUERY_ONLY_NO_CLAIM_OR_MUTATION"}
    raw = canonical_json(report) + b"\n"
    if len(raw) > 2 * 1024 * 1024:
        raise ValueError("CANDIDATE_DIAGNOSTIC_OUTPUT_SIZE")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"status": report["status"], "output": str(args.output),
        "target_dense_rank": report["dense"]["target_rank"],
        "target_bm25_rank": report["bm25"]["target_rank"],
        "target_in_hybrid_pool": report["target"]["in_original_hybrid_pool"],
        "new_model_calls": 0, "aws_calls": 0}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
