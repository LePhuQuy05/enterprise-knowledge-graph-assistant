"""Collection spike fixtures are synthetic and measure integrity, not AWS QA quality."""

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.collections import (
    CollectionError,
    CollectionManifest,
    build_collection,
    load_collection,
)
from app.contracts import Chunk, DocumentVersion, SnapshotManifest, canonical_json, text_hash
from app.fixtures import (
    FIXTURE_EMBEDDING_MODEL,
    FIXTURE_TENANT,
    fixture_recipes,
    fixture_vector,
)
from app.snapshot import SnapshotContent, build_snapshot


def make_shard(root: Path, start: int, count: int, *, source_offset: int = 0, **overrides) -> Path:
    documents, chunks = [], []
    for index in range(start, start + count):
        text = f"Synthetic document {index}: component {index % 7} supports feature {index % 11}."
        document = DocumentVersion.create(
            tenant_id=FIXTURE_TENANT,
            source_uri=f"https://example.invalid/collection/{index + source_offset}",
            title=f"Synthetic {index}", fetched_at=datetime(2026, 10, 23, tzinfo=UTC),
            raw_sha256=text_hash(text), normalizer_version="normalize-v1",
            normalized_text_path=f"sources/{index}.txt", normalized_text=text, source_kind="TEXT",
        )
        documents.append(document)
        chunks.append(Chunk.create(
            tenant_id=FIXTURE_TENANT, document_version_id=document.document_version_id,
            chunking_version="chunk-v1", section_path=("Synthetic",), position=0,
            start_char=0, end_char=len(text), text=text, token_count=len(text.split()),
            tokenizer_version="fixture-words-v1",
        ))
    extraction, embedding = fixture_recipes()
    params = dict(
        tenant_id=FIXTURE_TENANT, purpose="FIXTURE", extraction_recipe=extraction,
        embedding_recipe=embedding, corpus_version="collection-synthetic-v1",
        normalizer_version="normalize-v1", chunking_version="chunk-v1",
        tokenizer_version="fixture-words-v1",
    ) | overrides
    return build_snapshot(
        root, SnapshotContent(tuple(documents), tuple(chunks)),
        {chunk.chunk_id: fixture_vector(chunk.text) for chunk in chunks}, **params,
    )


def publish(root, shards):
    return build_collection(root, shards, tenant_id=FIXTURE_TENANT,
                            embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)


def read(path, **overrides):
    return load_collection(path, **(dict(
        tenant_id=FIXTURE_TENANT, embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True,
    ) | overrides))


def rewrite_collection(path, change):
    payload = json.loads(path.read_bytes())
    change(payload)
    payload["collection_version"] = CollectionManifest.version_for(payload)
    path.write_bytes(canonical_json(payload))


@pytest.fixture
def collection_files(tmp_path):
    shards = [make_shard(tmp_path, 0, 60), make_shard(tmp_path, 60, 60)]
    return publish(tmp_path, shards), shards


def test_more_than_one_hundred_documents_preserves_v2_per_shard_limits(collection_files):
    path, shards = collection_files
    collection = read(path, cache_shards=1)
    assert collection.manifest.schema_version == 1
    assert collection.manifest.counts.documents == 120
    assert collection.manifest.counts.chunks == 120
    assert collection.validate_all().documents == 120
    assert collection.cache_info.resident_shards == 1
    for shard in shards:
        header = SnapshotManifest.model_validate_json((shard / "manifest.json").read_bytes())
        assert header.schema_version == 2
        assert header.counts.documents == 60
        assert header.counts.chunks <= 5000
        assert sum(file.uncompressed_size for file in header.files) <= 100 * 1024 * 1024


def test_open_is_lazy_and_lru_cache_is_bounded(collection_files, monkeypatch):
    path, _ = collection_files
    import app.collections as module
    original, calls = module.load_snapshot, []

    def observed(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "load_snapshot", observed)
    collection = read(path, cache_shards=1)
    assert calls == []
    first, second = [ref.shard_id for ref in collection.manifest.shards]
    snapshot = collection.get_shard(first)
    assert collection.get_shard(first) is snapshot
    assert len(calls) == 1
    assert collection.cache_info.hits == 1
    collection.get_shard(second)
    assert len(calls) == 2
    assert collection.cache_info.resident_shards == 1
    assert collection.cache_info.evictions == 1
    collection.get_shard(first)
    assert len(calls) == 3
    assert collection.cache_info.evictions == 2


def test_byte_budget_can_skip_cache_without_skipping_validation(collection_files):
    collection = read(collection_files[0], cache_bytes=1)
    first = collection.manifest.shards[0].shard_id
    collection.get_shard(first)
    collection.get_shard(first)
    assert collection.cache_info.resident_shards == 0
    assert collection.cache_info.accounted_bytes == 0
    assert collection.cache_info.loads == 2


def test_exact_vector_topk_matches_monolithic_ranking(collection_files):
    collection = read(collection_files[0], cache_shards=1)
    query = fixture_vector("component 3 feature 9")
    # Dot products can differ in their final float32 ulp for batched BLAS, so
    # use the exact same matrix multiplication as the serving baseline.
    expected = []
    for ref in collection.manifest.shards:
        snapshot = collection.get_shard(ref.shard_id)
        expected.extend(zip(snapshot.chunk_ids, map(float, snapshot.vectors @ query), strict=True))
    expected.sort(key=lambda item: (-item[1], item[0]))
    result = collection.search_vector(query, k=20)
    assert [(hit.chunk_id, hit.score) for hit in result] == expected[:20]
    assert all(hit.snapshot_version == hit.shard_id for hit in result)
    assert collection.cache_info.resident_shards == 1


def test_shard_selection_is_explicit_and_hybrid_fails_honestly(collection_files):
    collection = read(collection_files[0])
    first = collection.manifest.shards[0].shard_id
    result = collection.search(fixture_vector("supports feature"), k=5, shard_ids=(first,))
    assert len(result) == 5
    assert {hit.shard_id for hit in result} == {first}
    with pytest.raises(CollectionError, match="global BM25"):
        collection.search(fixture_vector("feature"), mode="hybrid")


@pytest.mark.parametrize("selected", [(), ("unknown",), ("duplicate", "duplicate")])
def test_invalid_shard_selection_rejected(collection_files, selected):
    with pytest.raises(CollectionError, match="Selected shards"):
        read(collection_files[0]).search_vector(fixture_vector("x"), shard_ids=selected)


@pytest.mark.parametrize("kind", ["dimension", "dtype", "zero", "nan", "shape"])
def test_invalid_query_vectors_rejected(collection_files, kind):
    vector = fixture_vector("test")
    if kind == "dimension":
        vector = vector[:384]
    elif kind == "dtype":
        vector = vector.astype(np.float64)
    elif kind == "zero":
        vector[:] = 0
    elif kind == "nan":
        vector[0] = np.nan
    else:
        vector = vector.reshape(1, -1)
    with pytest.raises(CollectionError, match="Query vector"):
        read(collection_files[0]).search_vector(vector)


@pytest.mark.parametrize("override", [
    {"tenant_id": "other"}, {"embedding_model_id": "other"}, {"allow_fixture": False},
    {"trusted_manifest_sha256": "0" * 64}, {"cache_shards": -1}, {"cache_bytes": 0},
])
def test_approved_context_and_cache_bounds_required(collection_files, override):
    with pytest.raises(CollectionError):
        read(collection_files[0], **override)


def test_deterministic_order_and_idempotent_publication(collection_files):
    path, shards = collection_files
    original = path.read_bytes()
    assert publish(path.parent, list(reversed(shards))) == path
    assert path.read_bytes() == original
    assert not list(path.parent.glob(".collection-*.tmp"))
    read(path, trusted_manifest_sha256=text_hash(original.decode("utf-8")))


@pytest.mark.parametrize("unsafe", [
    "../outside", "/absolute", "C:/outside", "snap/../escape", "snap//other", "snap\\other",
    "snap/./other", "CON", "snap.", "https://example.invalid/shard",
])
def test_untrusted_paths_rejected_even_when_root_identity_is_resealed(collection_files, unsafe):
    path, _ = collection_files
    rewrite_collection(path, lambda payload: payload["shards"][0].update(relative_path=unsafe))
    with pytest.raises(CollectionError):
        read(path)


def test_missing_shard_manifest_fails_at_open(collection_files):
    path, shards = collection_files
    (shards[0] / "manifest.json").unlink()
    with pytest.raises(CollectionError, match="artifact"):
        read(path)


def test_tampered_shard_manifest_fails_pin_check(collection_files):
    path, shards = collection_files
    manifest_path = shards[0] / "manifest.json"
    payload = json.loads(manifest_path.read_bytes())
    payload["git_dirty"] = not payload["git_dirty"]
    payload["snapshot_version"] = SnapshotManifest.version_for(payload)
    manifest_path.write_bytes(canonical_json(payload))
    with pytest.raises(CollectionError, match="Pinned"):
        read(path)


def test_corrupt_unloaded_data_is_rejected_on_access_and_full_validation(collection_files):
    path, shards = collection_files
    collection = read(path)
    target = next(ref for ref in collection.manifest.shards if ref.relative_path == shards[1].name)
    (shards[1] / "chunks.jsonl").write_bytes(b"corrupt")
    assert collection.cache_info.loads == 0
    with pytest.raises(CollectionError, match="content validation"):
        collection.get_shard(target.shard_id)
    with pytest.raises(CollectionError):
        collection.validate_all()


def test_full_validation_rereads_previously_cached_artifacts(collection_files):
    path, _ = collection_files
    collection = read(path)
    ref = collection.manifest.shards[0]
    collection.get_shard(ref.shard_id)
    (path.parent / ref.relative_path / "chunks.jsonl").write_bytes(b"corrupt after first read")
    with pytest.raises(CollectionError, match="content validation"):
        collection.validate_all()


def test_collection_inventory_is_checked_against_loaded_records(collection_files):
    path, _ = collection_files

    def forge(payload):
        payload["shards"][0]["document_version_ids"][0] = "forged-doc-version"
        payload["shards"][0]["document_version_ids"].sort()

    rewrite_collection(path, forge)
    collection = read(path)
    with pytest.raises(CollectionError, match="inventory"):
        collection.get_shard(collection.manifest.shards[0].shard_id)


def test_duplicate_source_versions_and_incompatible_recipes_fail_build(tmp_path):
    first = make_shard(tmp_path, 0, 2)
    duplicate_version = make_shard(tmp_path, 0, 1)
    with pytest.raises(CollectionError, match="duplicate"):
        publish(tmp_path, [first, duplicate_version])
    # Same canonical source but a different immutable version is still forbidden.
    duplicate_source = make_shard(tmp_path, 2, 1, source_offset=-2)
    with pytest.raises(CollectionError, match="duplicate"):
        publish(tmp_path, [first, duplicate_source])
    incompatible = make_shard(tmp_path, 10, 1, extraction_recipe=fixture_recipes()[0].model_copy(
        update={"prompt_version": "different-extraction-recipe"},
    ))
    with pytest.raises(CollectionError, match="recipes"):
        publish(tmp_path, [first, incompatible])


def test_distinct_shard_corpus_versions_are_pinned_in_aggregate_collection(tmp_path):
    first = make_shard(tmp_path, 0, 2, corpus_version="content-hash-a")
    second = make_shard(tmp_path, 2, 2, corpus_version="content-hash-b")
    path = publish(tmp_path, [first, second])
    collection = read(path)
    assert {ref.corpus_version for ref in collection.manifest.shards} == {
        "content-hash-a", "content-hash-b",
    }
    assert collection.manifest.corpus_version.startswith("corpus_collection_")
    assert collection.validate_all().documents == 4


def test_root_count_recipe_and_dimension_tampering_fail_open(collection_files):
    path, _ = collection_files
    original = path.read_bytes()
    for update in (
        lambda payload: payload["counts"].update(documents=121),
        lambda payload: payload.update(tokenizer_version="changed"),
        lambda payload: payload["embedding_recipe"].update(dimension=384),
    ):
        path.write_bytes(original)
        rewrite_collection(path, update)
        with pytest.raises(CollectionError):
            read(path)


def test_symlink_shards_are_rejected_when_platform_permits(tmp_path):
    shard = make_shard(tmp_path, 0, 1)
    link = tmp_path / "linked-shard"
    try:
        link.symlink_to(shard, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Symlink creation is unavailable on this Windows account")
    with pytest.raises(CollectionError):
        publish(tmp_path, [link])


def test_root_duplicate_json_keys_are_rejected(collection_files):
    path, _ = collection_files
    raw = path.read_bytes()
    path.write_bytes(b'{"schema_version":1,' + raw[1:])
    with pytest.raises(CollectionError):
        read(path)


def test_benchmark_distinguishes_cache_capacity_from_process_memory(collection_files):
    from scripts.benchmark_collection import benchmark_search

    profiles = benchmark_search(
        collection_files[0], tenant_id=FIXTURE_TENANT,
        embedding_model_id=FIXTURE_EMBEDDING_MODEL, query_vectors=[fixture_vector("feature")],
        repeats=3, cache_shards=1, allow_fixture=True,
    )
    all_shards, selected = profiles
    assert all_shards["selected_documents"] == 120
    assert all_shards["selected_chunks"] == 120
    assert all(row["loads"] == 2 for row in all_shards["observations"])
    assert all_shards["hit_ids_consistent_across_repetitions"]
    assert selected["selected_documents"] == 60
    assert selected["observations"][0]["loads"] == 1
    assert selected["observations"][1]["loads"] == 0
    assert selected["cache_final"]["resident_shards"] == 1
    assert selected["process_memory_after_profile"]["peak_rss_bytes"] > 0
    assert selected["repeated_search_latency"]["observations"] == 2


def test_benchmark_questions_are_bounded_and_normalized(tmp_path):
    from scripts.benchmark_collection import _questions

    path = tmp_path / "questions.json"
    path.write_text(json.dumps(["  S3\r\nLambda  "]), encoding="utf-8")
    assert _questions(path) == ("S3\nLambda",)
    path.write_text(json.dumps(["x"] * 6), encoding="utf-8")
    with pytest.raises(ValueError, match="1..5"):
        _questions(path)
