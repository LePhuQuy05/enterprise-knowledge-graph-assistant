import json
import math
import sqlite3
from datetime import UTC, datetime

import pytest

from app.collection_lexical import (
    CollectionLexicalError,
    GlobalLexicalManifest,
    build_global_lexical_index,
    load_global_lexical_index,
    retrieve_collection_hybrid,
)
from app.collections import build_collection, load_collection
from app.contracts import Chunk, DocumentVersion, canonical_json, text_hash
from app.fixtures import FIXTURE_EMBEDDING_MODEL, FIXTURE_TENANT, fixture_recipes, fixture_vector
from app.retrieval import LexicalIndex, reciprocal_rank_fusion
from app.snapshot import SnapshotContent, build_snapshot


def make_collection(root, texts, *, corpus_label="part"):
    extraction, embedding = fixture_recipes()
    shards = []
    for start in range(0, len(texts), 2):
        documents, chunks = [], []
        for offset, text in enumerate(texts[start:start + 2], start):
            document = DocumentVersion.create(
                tenant_id=FIXTURE_TENANT,
                source_uri=f"https://example.invalid/lexical/{corpus_label}/{offset}",
                title="資料", fetched_at=datetime(2026, 10, 23, tzinfo=UTC),
                raw_sha256=text_hash(text), normalizer_version="normalize-v1",
                normalized_text_path=f"sources/{offset}.txt", normalized_text=text, source_kind="TEXT",
            )
            chunk = Chunk.create(
                tenant_id=FIXTURE_TENANT, document_version_id=document.document_version_id,
                chunking_version="chunk-v1", section_path=("節",), position=0,
                start_char=0, end_char=len(text), text=text, token_count=max(1, len(text.split())),
                tokenizer_version="fixture-words-v1",
            )
            documents.append(document)
            chunks.append(chunk)
        shards.append(build_snapshot(
            root, SnapshotContent(tuple(documents), tuple(chunks)),
            {chunk.chunk_id: fixture_vector(chunk.text) for chunk in chunks},
            tenant_id=FIXTURE_TENANT, purpose="FIXTURE", extraction_recipe=extraction,
            embedding_recipe=embedding, corpus_version=f"{corpus_label}-{start}",
            normalizer_version="normalize-v1", chunking_version="chunk-v1",
            tokenizer_version="fixture-words-v1",
        ))
    path = build_collection(root, shards, tenant_id=FIXTURE_TENANT,
                            embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    return load_collection(path, tenant_id=FIXTURE_TENANT,
                           embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True, cache_shards=1)


def baseline(collection):
    documents, chunks = [], []
    for ref in collection.manifest.shards:
        snapshot = collection.get_shard(ref.shard_id)
        documents.extend(snapshot.content.documents)
        chunks.extend(snapshot.content.chunks)
    return LexicalIndex(documents, chunks, tenant_id=FIXTURE_TENANT)


@pytest.fixture
def corpus(tmp_path):
    collection = make_collection(tmp_path / "shards", (
        "common common alpha AWS S3", "common beta lambda", "common gamma kms",
        "common delta encryption", "common epsilon", "rare zeta",
    ))
    path = build_global_lexical_index(collection, tmp_path / "lexical")
    return collection, path


@pytest.mark.parametrize("question", [
    "common", "rare", "alpha", "ALPHA aws", "common rare common", "lambda kms",
    "zero unknown nonexistent", "", "   ", "日本語", "common zeta",
])
def test_global_scores_match_combined_monolithic_baseline(corpus, question):
    collection, path = corpus
    expected = baseline(collection).search(question)
    with load_global_lexical_index(path, collection) as index:
        actual = index.search(question)
        assert [key for key, _ in actual] == [key for key, _ in expected]
        assert [score for _, score in actual] == pytest.approx(
            [score for _, score in expected], rel=1e-12, abs=1e-12,
        )
        for key, _ in actual:
            shard = collection.get_shard(index.shard_for_chunk(key))
            assert key in shard.chunks


def test_negative_global_average_idf_and_tied_scores_are_preserved(tmp_path):
    collection = make_collection(tmp_path / "shards", ["same same"] * 6)
    path = build_global_lexical_index(collection, tmp_path / "lexical")
    with load_global_lexical_index(path, collection) as index:
        assert index.manifest.average_idf == pytest.approx(math.log(0.5) - math.log(6.5))
        result = index.search("same same")
        assert len(result) == 6
        assert all(score < 0 for _, score in result)
        assert [key for key, _ in result] == sorted(key for key, _ in result)
        expected = baseline(collection).search("same same")
        assert [score for _, score in result] == pytest.approx([score for _, score in expected])


def test_zero_idf_matching_chunks_remain_eligible(tmp_path):
    collection = make_collection(tmp_path / "shards", ["alpha", "alpha", "beta", "gamma"])
    path = build_global_lexical_index(collection, tmp_path / "lexical")
    with load_global_lexical_index(path, collection) as index:
        result = index.search("alpha")
        assert len(result) == 2
        assert [score for _, score in result] == [0.0, 0.0]
        assert result == baseline(collection).search("alpha")


def test_empty_ascii_vocabulary_is_safe_and_honest(tmp_path):
    collection = make_collection(tmp_path / "shards", ["日本語", "中文"])
    path = build_global_lexical_index(collection, tmp_path / "lexical")
    with load_global_lexical_index(path, collection) as index:
        assert index.manifest.terms == 0
        assert index.manifest.total_tokens == 0
        assert index.search("日本語 alpha") == []


def test_hybrid_uses_global_lexical_rrf_over_entire_collection(corpus):
    collection, path = corpus
    question, vector = "common rare", fixture_vector("rare")
    with load_global_lexical_index(path, collection) as index:
        expected = reciprocal_rank_fusion([
            [hit.chunk_id for hit in collection.search_vector(vector, k=4)],
            [key for key, _ in baseline(collection).search(question, k=3)],
        ])
        actual = retrieve_collection_hybrid(collection, index, question, vector,
                                             vector_k=4, lexical_k=3)
        assert actual == expected


def test_idempotent_build_does_not_change_published_artifacts(corpus):
    collection, path = corpus
    original = path.read_bytes(), (path.parent / "index.sqlite3").read_bytes()
    assert build_global_lexical_index(collection, path.parent.parent) == path
    assert original == (path.read_bytes(), (path.parent / "index.sqlite3").read_bytes())
    assert not list(path.parent.parent.glob(".lexical-*"))


def test_wrong_collection_is_rejected_by_loader_and_hybrid(corpus, tmp_path):
    collection, path = corpus
    other = make_collection(tmp_path / "other", ["other corpus", "different evidence"])
    with pytest.raises(CollectionLexicalError, match="collection"):
        load_global_lexical_index(path, other)
    with load_global_lexical_index(path, collection) as index:
        with pytest.raises(CollectionLexicalError, match="collection"):
            retrieve_collection_hybrid(other, index, "other", fixture_vector("other"))


@pytest.mark.parametrize("change", ["missing", "database", "manifest", "trusted"])
def test_missing_or_modified_artifacts_fail_closed(corpus, change):
    collection, path = corpus
    options = {}
    if change == "missing":
        (path.parent / "index.sqlite3").unlink()
    elif change == "database":
        with (path.parent / "index.sqlite3").open("ab") as stream:
            stream.write(b"tamper")
    elif change == "manifest":
        payload = json.loads(path.read_bytes())
        payload["average_idf"] += 1
        path.write_bytes(canonical_json(payload))
    else:
        options["trusted_manifest_sha256"] = "0" * 64
    with pytest.raises(CollectionLexicalError):
        load_global_lexical_index(path, collection, **options)


def test_resealed_wrong_schema_is_rejected_and_connection_is_read_only(corpus):
    collection, path = corpus
    with load_global_lexical_index(path, collection) as index:
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            index._connection.execute("DELETE FROM chunks")
    database = path.parent / "index.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("PRAGMA user_version=99")
    payload = json.loads(path.read_bytes())
    from app.contracts import sha256_bytes

    payload["sqlite_sha256"] = sha256_bytes(database.read_bytes())
    payload["index_version"] = GlobalLexicalManifest.version_for(payload)
    path.write_bytes(canonical_json(payload))
    with pytest.raises(CollectionLexicalError, match="schema"):
        load_global_lexical_index(path, collection)


def test_query_caps_and_unknown_chunk_are_explicit(corpus):
    collection, path = corpus
    with load_global_lexical_index(path, collection) as index:
        for k in (0, 21, True):
            with pytest.raises(CollectionLexicalError):
                index.search("common", k=k)
        with pytest.raises(CollectionLexicalError):
            index.search("x" * 2001)
        with pytest.raises(CollectionLexicalError):
            index.shard_for_chunk("missing")


def test_hybrid_benchmark_runs_actual_global_index_and_vector_search(corpus):
    from scripts.benchmark_collection import benchmark_hybrid

    collection, path = corpus
    questions = ("common rare", "alpha")
    result = benchmark_hybrid(
        collection.path, path, tenant_id=FIXTURE_TENANT,
        embedding_model_id=FIXTURE_EMBEDDING_MODEL, questions=questions,
        query_vectors=[fixture_vector(question) for question in questions],
        repeats=3, cache_shards=1, allow_fixture=True,
    )
    assert result["global_chunks"] == 6
    assert result["scope"] == "all_collection_shards"
    assert result["lexical_repeated_latency"]["observations"] == 4
    assert result["hybrid_repeated_latency"]["observations"] == 4
    assert result["hit_ids_consistent_across_repetitions"]
    assert all(row["loads"] == 3 for row in result["observations"])
    assert result["process_memory_after_profile"]["peak_rss_bytes"] > 0
