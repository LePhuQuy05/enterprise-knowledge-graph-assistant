"""Offline BGE indexing of prepared corpus shards with verified reusable checkpoints."""

import io
import json
from pathlib import Path

import numpy as np

from app.collections import CollectionError, build_collection, load_collection
from app.contracts import ExtractionRecipe, canonical_json, embedding_input, sha256_bytes, stable_id
from app.corpus import load_corpus
from app.ingestion.pipeline import _write
from app.snapshot import SnapshotContent, build_snapshot


def no_extraction_recipe() -> ExtractionRecipe:
    return ExtractionRecipe(
        version="not-extracted-coverage-v1", model_id="not-invoked", invocation_id="not-invoked",
        source_region="local", parameters={"performed": False}, prompt_version="none",
        output_schema_version="none", ontology_version="none", canonicalization_version="none",
    )


def _checkpoint(root, key, matrix, metadata):
    output = io.BytesIO()
    np.lib.format.write_array(output, matrix, allow_pickle=False, version=(1, 0))
    raw = output.getvalue()
    _write(root, f"vector-cache/{key}.npy", raw)
    _write(root, f"vector-cache/{key}.json", canonical_json({
        **metadata, "sha256": sha256_bytes(raw), "bytes": len(raw),
    }))


def _cached_matrix(root, key, metadata):
    path = root / f"vector-cache/{key}.json"
    array = root / f"vector-cache/{key}.npy"
    if not path.exists() and not array.exists():
        return None
    if not path.exists():
        # A crash after array publication is safe to recompute locally; immutable
        # publication subsequently checks identical bytes instead of overwriting.
        return None
    if (not path.resolve().is_relative_to(root) or not array.resolve().is_relative_to(root)
            or path.is_symlink() or array.is_symlink() or path.stat().st_size > 65536):
        raise ValueError("VECTOR_CHECKPOINT_UNSAFE")
    saved = json.loads(path.read_bytes())
    if not isinstance(saved, dict) or {k: saved.get(k) for k in metadata} != metadata:
        raise ValueError("VECTOR_CHECKPOINT_RECIPE_MISMATCH")
    expected = metadata["rows"] * metadata["dimension"] * 4
    if array.stat().st_size > expected + 4096 or saved.get("bytes") != array.stat().st_size:
        raise ValueError("VECTOR_CHECKPOINT_SIZE_INVALID")
    raw = array.read_bytes()
    if sha256_bytes(raw) != saved.get("sha256"):
        raise ValueError("VECTOR_CHECKPOINT_CHECKSUM_INVALID")
    with io.BytesIO(raw) as stream:
        version = np.lib.format.read_magic(stream)
        if version != (1, 0):
            raise ValueError("VECTOR_CHECKPOINT_FORMAT_INVALID")
        shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream, max_header_size=4096)
        if shape != (metadata["rows"], metadata["dimension"]) or fortran or dtype != np.dtype("float32"):
            raise ValueError("VECTOR_CHECKPOINT_SHAPE_INVALID")
        if stream.tell() + expected != len(raw):
            raise ValueError("VECTOR_CHECKPOINT_LENGTH_INVALID")
    matrix = np.load(io.BytesIO(raw), allow_pickle=False)
    if (not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1, atol=1e-5, rtol=0)):
        raise ValueError("VECTOR_CHECKPOINT_NORMALIZATION_INVALID")
    return matrix


def _reuse_prior_inputs(prior, texts, matrix):
    """Copy only needed rows; never retain views into evicted prior shards."""
    needed = {}
    for row, text in enumerate(texts):
        digest = sha256_bytes(text.encode("utf-8"))
        needed.setdefault(digest, []).append(row)
    reused = 0
    if prior is not None:
        for ref in prior.manifest.shards:
            snapshot = prior.get_shard(ref.shard_id)
            for row, chunk in enumerate(snapshot.content.chunks):
                text = embedding_input(snapshot.documents[chunk.document_version_id], chunk)
                target_rows = needed.pop(sha256_bytes(text.encode("utf-8")), None)
                if target_rows is not None:
                    # Fancy-index assignment copies values into the target
                    # matrix instead of preserving a whole prior array via a view.
                    matrix[target_rows] = snapshot.vectors[row]
                    reused += len(target_rows)
            del snapshot
            if not needed:
                break
    missing = sorted(row for rows in needed.values() for row in rows)
    return reused, missing


def index_corpus_directories(corpus_directories, output_root: Path, encoder, *, tenant_id="demo",
                             progress=None, batch_size=8,
                             reuse_collection_path: Path | None = None) -> dict:
    if not corpus_directories or not 1 <= len(corpus_directories) <= 128:
        raise ValueError("INDEX_SHARD_COUNT_INVALID")
    if type(batch_size) is not int or not 1 <= batch_size <= 16:
        raise ValueError("INDEX_BATCH_SIZE_INVALID")
    # Validate the complete input set before loading/encoding the first shard.
    # Preparation can legitimately partition different normalizer versions;
    # one controlled collection must not silently merge those recipes.
    expected, versions, document_ids = None, set(), set()
    for directory in map(Path, corpus_directories):
        corpus = load_corpus(directory, tenant_id=tenant_id)
        if corpus.version in versions:
            raise ValueError("INDEX_DUPLICATE_CORPUS")
        versions.add(corpus.version)
        current = (corpus.recipe["normalizer_version"], corpus.chunks[0].chunking_version,
                   corpus.chunks[0].tokenizer_version)
        if expected is not None and current != expected:
            raise ValueError("INDEX_MIXED_RECIPES_CREATE_SEPARATE_COLLECTIONS")
        expected = current
        if (any(doc.normalizer_version != current[0] for doc in corpus.documents)
                or any((chunk.chunking_version, chunk.tokenizer_version) != current[1:]
                       for chunk in corpus.chunks)):
            raise ValueError("INDEX_INCONSISTENT_SHARD_RECIPES")
        for doc in corpus.documents:
            if doc.document_id in document_ids:
                raise ValueError("INDEX_DUPLICATE_DOCUMENT")
            document_ids.add(doc.document_id)
    recipe = encoder.recipe("document")
    purpose = "FIXTURE" if encoder.fixture_only else "DEMO"
    prior = None
    if reuse_collection_path is not None:
        try:
            prior = load_collection(
                Path(reuse_collection_path), tenant_id=tenant_id, embedding_model_id=recipe.model_id,
                allow_fixture=purpose == "FIXTURE", cache_shards=1,
            )
            if prior.manifest.embedding_recipe != recipe:
                raise ValueError("INDEX_REUSE_EMBEDDING_RECIPE_MISMATCH")
            if prior.manifest.purpose != purpose:
                raise ValueError("INDEX_REUSE_PURPOSE_MISMATCH")
            # Fail before encoding even when a later prior shard is corrupt or
            # every target row could otherwise be found in the first shard.
            prior.validate_all()
        except CollectionError as exc:
            raise ValueError("INDEX_REUSE_COLLECTION_INVALID") from exc
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    shards, summaries, seen = [], [], set()
    extraction = no_extraction_recipe()
    for directory in map(Path, corpus_directories):
        corpus = load_corpus(directory, tenant_id=tenant_id)
        if corpus.version in seen:
            raise ValueError("INDEX_DUPLICATE_CORPUS")
        seen.add(corpus.version)
        docs = {doc.document_version_id: doc for doc in corpus.documents}
        texts = [embedding_input(docs[c.document_version_id], c) for c in corpus.chunks]
        if any(encoder.count(text, role="document") > 512 for text in texts):
            raise ValueError("INDEX_INPUT_EXCEEDS_MODEL_CONTEXT")
        # Vector values depend on ordered input texts and encoder recipe, not
        # the surrounding release/selection metadata. Keep corpus provenance in
        # snapshot manifests without invalidating unchanged input checkpoints.
        metadata = {"schema_version": 1,
                    "recipe_sha256": recipe.fingerprint, "rows": len(texts),
                    "input_sha256": sha256_bytes(canonical_json(texts)),
                    "dimension": recipe.dimension, "purpose": purpose}
        key = stable_id("vectors", **metadata)
        matrix = _cached_matrix(root, key, metadata)
        cache_hit = matrix is not None
        reused = new_vectors = 0
        if matrix is None:
            matrix = np.empty((len(texts), recipe.dimension), dtype=np.float32)
            reused, missing = _reuse_prior_inputs(prior, texts, matrix)
            if prior is not None and progress:
                progress({"stage": "PRIOR_COLLECTION_REUSE", "corpus_version": corpus.version,
                          "completed": reused, "total": len(texts),
                          "prior_input_vectors_reused": reused, "new_vectors": 0})
            for offset in range(0, len(missing), batch_size):
                rows = missing[offset:offset + batch_size]
                part = encoder.encode([texts[row] for row in rows], role="document",
                                      batch_size=batch_size)
                if part.dtype != np.dtype("float32") or part.shape != (
                    len(rows), recipe.dimension,
                ) or not np.isfinite(part).all() or not np.allclose(
                    np.linalg.norm(part, axis=1), 1, atol=1e-5, rtol=0,
                ):
                    raise ValueError("INDEX_ENCODER_VECTOR_INVALID")
                matrix[rows] = part
                new_vectors += len(rows)
                if progress:
                    progress({"stage": "LOCAL_EMBED", "corpus_version": corpus.version,
                              "completed": reused + new_vectors, "total": len(texts),
                              "prior_input_vectors_reused": reused, "new_vectors": new_vectors})
            _checkpoint(root, key, matrix, metadata)
        elif progress:
            progress({"stage": "VECTOR_CHECKPOINT_HIT", "corpus_version": corpus.version,
                      "completed": len(texts), "total": len(texts),
                      "prior_input_vectors_reused": 0, "new_vectors": 0})
        path = build_snapshot(
            root, SnapshotContent(corpus.documents, corpus.chunks),
            {c.chunk_id: matrix[i] for i, c in enumerate(corpus.chunks)},
            tenant_id=tenant_id, purpose=purpose, extraction_recipe=extraction,
            embedding_recipe=recipe, corpus_version=corpus.version,
            normalizer_version=corpus.recipe["normalizer_version"],
            chunking_version=corpus.chunks[0].chunking_version,
            tokenizer_version=corpus.chunks[0].tokenizer_version,
        )
        shards.append(path)
        summaries.append({"snapshot_directory": str(path), "corpus_version": corpus.version,
                          "documents": len(corpus.documents), "chunks": len(corpus.chunks),
                          "vector_checkpoint_hit": cache_hit,
                          "new_vectors": new_vectors,
                          "prior_input_vectors_reused": reused,
                          "block_checkpoint_vectors_reused": len(texts) if cache_hit else 0})
        del matrix
    collection_path = build_collection(
        root, shards, tenant_id=tenant_id, embedding_model_id=recipe.model_id,
        allow_fixture=purpose == "FIXTURE",
    )
    collection = load_collection(
        collection_path, tenant_id=tenant_id, embedding_model_id=recipe.model_id,
        allow_fixture=purpose == "FIXTURE",
    )
    collection.validate_all()
    return {"status": "INDEXED_LOCAL_QUALITY_NOT_EVALUATED", "collection_path": str(collection_path),
            "purpose": purpose, "embedding_recipe": recipe.model_dump(mode="json"),
            "shards": summaries, "documents": sum(s["documents"] for s in summaries),
            "chunks": sum(s["chunks"] for s in summaries),
            "new_vectors": sum(s["new_vectors"] for s in summaries), "paid_model_calls": 0,
            "prior_input_vectors_reused": sum(s["prior_input_vectors_reused"] for s in summaries),
            "block_checkpoint_vectors_reused": sum(s["block_checkpoint_vectors_reused"] for s in summaries),
            "reuse_collection_path": str(prior.path) if prior is not None else None,
            "graph_status": "NOT_BUILT", "semantic_review": "NOT_REVIEWED",
            "cloud_deployments": 0, "serving_status": "NOT_CONNECTED"}
