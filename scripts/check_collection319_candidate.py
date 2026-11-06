"""Reload source/vector parity and run sixteen local source-gap retrieval cells.

Exposed English development questions, not a benchmark or generated answers.
Graph/Q/QB migration and cloud selectors are deliberately separate gates.
"""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.collection_answering import prepare_collection_answer
from app.collection_lexical import build_global_lexical_index, load_global_lexical_index
from app.collections import load_collection
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.source_dependency_audit import SourceDependency, audit_source_dependencies
from app.source_message_runtime import repack_source_answer
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT, load_sources
from scripts.qualify_permission_source_context import covering_chunks
from scripts.run_collection_benchmark import OfflineRuntime

OUTPUT = ROOT / "results/collection319-local"
QUESTIONS = (
    ("two-permission-groups", "How are permissions for Amazon S3 to invoke Lambda different from execution role permissions to access S3?"),
    ("tutorial-read-role", "Which permission does the S3 trigger tutorial put in the Lambda execution role to get an object from an S3 bucket?"),
    ("async-event-202", "With asynchronous Lambda invocation type Event, what does a successful StatusCode 202 response establish?"),
    ("acceptance-vs-processing", "Does successful asynchronous Lambda invocation show that the function code has completed reading or decrypting an S3 object?"),
)
METHODS = ("bm25", "vector", "hybrid", "hybrid_reranked")
CODE = ("scripts/check_collection319_candidate.py", "app/collection_indexing.py", "app/collection_lexical.py",
    "app/source_dependency_audit.py", "app/source_message_runtime.py", "app/source_message_packing.py")


def check(*, replay=False):
    preparation = json.loads((OUTPUT / "preparation.json").read_bytes())
    index = json.loads((OUTPUT / "index.json").read_bytes())
    collection = load_collection(Path(index["collection_path"]), tenant_id="demo", embedding_model_id="BAAI/bge-small-en-v1.5")
    collection.validate_all()
    parent, old_chunks, old_docs = load_sources()
    if parent.manifest_sha256 != preparation["parent_manifest_sha256"]:
        raise ValueError("COLLECTION319_PARENT_MANIFEST_CHANGED")
    documents, chunks, vectors = {}, {}, {}
    for ref in collection.manifest.shards:
        snap = collection.get_shard(ref.shard_id)
        documents.update(snap.documents)
        chunks.update(snap.chunks)
        vectors.update({c.chunk_id: snap.vectors[i].copy() for i, c in enumerate(snap.content.chunks)})
    if (len(documents) != 319 or len(chunks) != 3024 or len({c.text_sha256 for c in chunks.values()}) != 2942
            or any(documents.get(k) != d for k, d in old_docs.items())
            or any(chunks.get(k) != c for k, c in old_chunks.items())):
        raise ValueError("COLLECTION319_SOURCE_OR_COUNT_PARITY")
    verified = 0
    for ref in parent.manifest.shards:
        snap = parent.get_shard(ref.shard_id)
        for i, chunk in enumerate(snap.content.chunks):
            if not np.array_equal(snap.vectors[i], vectors[chunk.chunk_id]):
                raise ValueError("COLLECTION319_OLD_VECTOR_CHANGED")
            verified += 1
    if verified != 2978 or index["prior_input_vectors_reused"] != 2978 or index["new_vectors"] != 46:
        raise ValueError("COLLECTION319_VECTOR_REUSE_ACCOUNTING")
    qualification = json.loads((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())
    dependencies = []
    for row in qualification["source_controls"]:
        dependency = SourceDependency.model_validate_json(canonical_json(row["dependency"]))
        doc = documents[dependency.document_version_id]
        selected = covering_chunks(doc, (dependency.anchor, *dependency.required_spans), chunks)
        audit = audit_source_dependencies(documents, chunks, selected, (dependency,), tenant_id="demo")
        if audit["incomplete_count"]:
            raise ValueError("COLLECTION319_RECHUNKED_SOURCE_DEPENDENCY_LOST")
        dependencies.append({"control_id": row["control_id"], "audit": audit})
    lexical_path = (Path(json.loads((OUTPUT / "runtime.json").read_bytes())["lexical_manifest_path"]) if replay
        else build_global_lexical_index(collection, ROOT / "data/collection319-local/lexical"))
    immutable(OUTPUT / "runtime.json", {"collection_path": str(collection.path), "collection_manifest_sha256": collection.manifest_sha256,
        "lexical_manifest_path": str(lexical_path), "documents": 319, "chunks": 3024, "serving_published": False,
        "local_methods_checked": METHODS, "graph_migrated": False, "q_qb_native_preparation_done": False})
    plan = {"version": "collection319-local-source-gap-retrieval-v1", "questions": QUESTIONS, "methods": METHODS,
        "collection_manifest_sha256": collection.manifest_sha256, "index_report_sha256": sha256_bytes((OUTPUT / "index.json").read_bytes()),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}, "max_chunks": 8, "user_message_proxy_tokens": 4096,
        "new_paid_model_calls": 0, "aws_calls": 0, "purpose": "EXPOSED_ENGLISH_SOURCE_DIAGNOSTICS_NOT_QUALITY_BENCHMARK"}
    immutable(OUTPUT / "retrieval-plan.json", plan)
    rows = []
    with load_global_lexical_index(lexical_path, collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
        for name, question in QUESTIONS:
            lookup = runtime.cache.resolve_local("demo", text=query_embedding_input(question), recipe=runtime.query_recipe, encoder=runtime.encoder)
            vector = np.asarray(lookup.record.vector, dtype=np.float32)
            for method in METHODS:
                view, before = prepare_collection_answer(collection, lexical, question, None if method == "bm25" else vector,
                    tenant_id="demo", owner_id="collection319-local", run_id="collection319-source-gap-v1", recipe=runtime.recipe,
                    mode=method, reranker=runtime.reranker)
                prepared, trace = repack_source_answer(view, before)
                rows.append({"case": name, "method": method, "question": question,
                    "ranked_chunk_ids": prepared.retrieved_chunk_ids, "context_chunk_ids": prepared.packed.chunk_ids,
                    "user_message_proxy_tokens": prepared.packed.proxy_tokens,
                    "user_message_sha256": trace["user_message_sha256"], "sources": trace["audit"]["sources"],
                    "supplement_document_ids_in_context": sorted(set(c.document_version_id for c in (chunks[k] for k in prepared.packed.chunk_ids)) - set(old_docs)),
                    "answer_generated": False, "human_reviewed": 0})
            print(json.dumps({"case": name, "local_cells_completed": len(rows)}), flush=True)
    raw = canonical_json({"retrieval_cells": rows, "rechunked_dependency_audits": dependencies}) + b"\n"
    report = {"status": "CANDIDATE319_SOURCE_VECTOR_DEPENDENCY_AND_RETRIEVAL_CHECKED", "documents": 319, "chunks": 3024,
        "unique_body_texts": 2942, "unchanged_parent_documents": len(old_docs), "unchanged_parent_chunks": len(old_chunks),
        "bit_exact_parent_vectors_verified": verified, "new_local_vectors": 46, "collection_manifest_sha256": collection.manifest_sha256,
        "source_dependencies_revalidated_after_rechunk": len(dependencies), "english_retrieval_cells": len(rows),
        "retrieval_cells_with_some_supplement_source": sum(bool(r["supplement_document_ids_in_context"]) for r in rows),
        "max_actual_user_message_proxy_tokens": max(r["user_message_proxy_tokens"] for r in rows),
        "receipts_sha256": sha256_bytes(raw), "new_paid_model_calls": 0, "aws_calls": 0,
        "generation_answers_tested": False, "graph_migrated": False, "q_qb_native_preparation_done": False,
        "quality_accuracy_measured": False, "human_reviewed": 0, "serving_published": False}
    if replay:
        if (OUTPUT / "source-retrieval-receipts.json").read_bytes() != raw:
            raise ValueError("COLLECTION319_RETRIEVAL_SOURCE_REPLAY_CHANGED")
        immutable(OUTPUT / "replay.json", report | {"status": "CANDIDATE319_PARITY_AND_RETRIEVAL_EXACT_REPLAY"})
    else:
        (OUTPUT / "source-retrieval-receipts.json").write_bytes(raw)
        immutable(OUTPUT / "validation.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=AssertionError("COLLECTION319_NETWORK_BLOCKED")))
        check(replay=parser.parse_args().replay)
