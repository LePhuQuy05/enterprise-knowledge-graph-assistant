"""Local fixtures exercise candidate diagnosis; no encoders, network or paid APIs."""

import copy
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from app.collection_answering import prepare_collection_answer
from app.contracts import canonical_json, embedding_cache_key, sha256_bytes
from app.cost import TokenUsage
from app.embeddings import CachedEmbedding
from app.generation import GenerationRecipeV9
from app.rag_methods import MethodConfigV1, MethodResultV1, RunInputV1
from scripts import diagnose_candidate_coverage as diagnostic
from tests.test_collection_review import review_context as review_context


@pytest.fixture
def diagnostic_inputs(review_context):
    collection, lexical, _, vectors, params = review_context
    query, recipe = "Service B", params["query_recipe"]
    vector = vectors["dev-01:en"]
    generation = GenerationRecipeV9(model_metadata_sha256="0" * 64)
    _, prepared = prepare_collection_answer(
        collection, lexical, query, vector, tenant_id=collection.manifest.tenant_id,
        owner_id="fixture-owner", run_id="fixture-diagnostic", recipe=generation,
        search_query=query, mode="hybrid",
    )
    config = MethodConfigV1(
        method_id="hybrid", fusion="RRF60",
        collection_manifest_sha256=collection.manifest_sha256,
        lexical_index_version=lexical.manifest.index_version,
        document_recipe_sha256=params["document_recipe"].fingerprint,
        query_recipe_sha256=recipe.fingerprint, generation_recipe_sha256=generation.fingerprint,
    )
    method = MethodResultV1(
        requested_method="hybrid", effective_method="hybrid", config=config,
        config_sha256=config.fingerprint, collection_version=collection.manifest.collection_version,
        status="RETRIEVED", ranked_chunk_ids=prepared.retrieved_chunk_ids,
        context_chunk_ids=prepared.packed.chunk_ids, context_proxy_tokens=prepared.packed.proxy_tokens,
    )
    receipt = {
        "status": "COMPLETED", "input": RunInputV1(
            original_question=query, method_ids=("hybrid",), idempotency_key="diagnostic-fixture",
        ).model_dump(mode="json"), "search_query": query, "query_transform": "NONE",
        "collection_version": collection.manifest.collection_version,
        "collection_manifest_sha256": collection.manifest_sha256,
        "methods": [method.model_dump(mode="json")],
    }
    record = CachedEmbedding(
        namespace="FIXTURE", tenant_id=collection.manifest.tenant_id,
        cache_key=embedding_cache_key(collection.manifest.tenant_id, query, recipe),
        input_text=query, input_sha256=sha256_bytes(query.encode()), recipe=recipe,
        operation_id="localembed_diagnostic_fixture", usage=TokenUsage(input_tokens=10, output_tokens=0),
        original_cost_micro_usd=0, vector=tuple(float(x) for x in vector),
    )
    return collection, lexical, receipt, record


def write_cache(path, record, *, state="READY", payload=None, checksum=None):
    raw = payload if payload is not None else canonical_json(record.model_dump(mode="json"))
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE embeddings(namespace TEXT,cache_key TEXT,state TEXT,"
                   "payload BLOB,payload_sha256 TEXT,PRIMARY KEY(namespace,cache_key))")
        db.execute("INSERT INTO embeddings VALUES(?,?,?,?,?)", (
            "FIXTURE", record.cache_key, state, raw, checksum or sha256_bytes(raw),
        ))


def test_cached_query_is_read_only_and_pins_exact_input_recipe_and_vector(tmp_path, diagnostic_inputs):
    _, _, _, record = diagnostic_inputs
    database = tmp_path / "embeddings.sqlite3"
    write_cache(database, record)
    before = database.read_bytes()
    restored, checksum = diagnostic.cached_query(
        database, tenant_id=record.tenant_id, query=record.input_text,
        recipe=record.recipe, namespace="FIXTURE",
    )
    assert restored == record
    assert checksum == sha256_bytes(canonical_json(record.model_dump(mode="json")))
    assert database.read_bytes() == before
    assert not database.with_name(database.name + "-journal").exists()


@pytest.mark.parametrize("state", ["IN_PROGRESS", "FAILED"])
def test_unsettled_cache_record_cannot_encode_retry_or_write(tmp_path, diagnostic_inputs, state):
    _, _, _, record = diagnostic_inputs
    database = tmp_path / "embeddings.sqlite3"
    write_cache(database, record, state=state)
    before = database.read_bytes()
    with pytest.raises(ValueError, match="READY_QUERY_VECTOR_REQUIRED"):
        diagnostic.cached_query(database, tenant_id=record.tenant_id, query=record.input_text,
                                recipe=record.recipe, namespace="FIXTURE")
    assert database.read_bytes() == before


def test_cache_miss_does_not_create_a_database_or_new_query_record(tmp_path, diagnostic_inputs):
    _, _, _, record = diagnostic_inputs
    database = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError, match="CACHE_PATH_OR_NAMESPACE"):
        diagnostic.cached_query(database, tenant_id=record.tenant_id, query=record.input_text,
                                recipe=record.recipe, namespace="FIXTURE")
    assert not database.exists()
    write_cache(database, record)
    before = database.read_bytes()
    with pytest.raises(ValueError, match="READY_QUERY_VECTOR_REQUIRED"):
        diagnostic.cached_query(database, tenant_id=record.tenant_id, query="Different query",
                                recipe=record.recipe, namespace="FIXTURE")
    assert database.read_bytes() == before


@pytest.mark.parametrize("damage", ["checksum", "namespace"])
def test_cache_corruption_or_namespace_relabel_is_rejected(tmp_path, diagnostic_inputs, damage):
    _, _, _, record = diagnostic_inputs
    database = tmp_path / "embeddings.sqlite3"
    value = record.model_copy(update={"namespace": "LIVE"}) if damage == "namespace" else record
    write_cache(database, value, checksum="0" * 64 if damage == "checksum" else None)
    with pytest.raises(ValueError, match="CACHE_CHECKSUM|CACHE_BINDING"):
        diagnostic.cached_query(database, tenant_id=record.tenant_id, query=record.input_text,
                                recipe=record.recipe, namespace="FIXTURE")


def test_target_only_changes_diagnostic_fields_not_query_candidates_or_provenance(diagnostic_inputs):
    collection, lexical, receipt, record = diagnostic_inputs
    ids = receipt["methods"][0]["ranked_chunk_ids"]
    assert len(ids) >= 2
    before = copy.deepcopy(receipt)
    first = diagnostic.diagnose(collection, lexical, receipt, record, ids[0])
    second = diagnostic.diagnose(collection, lexical, receipt, record, ids[-1])
    assert first["original_hybrid_pool"] == second["original_hybrid_pool"]
    assert first["search_query"] == second["search_query"] == record.input_text
    assert first["target"]["chunk_id"] != second["target"]["chunk_id"]
    assert first["actions"] == {"new_embeddings": 0, "model_encoder_calls": 0,
        "reranker_calls": 0, "paid_api_calls": 0, "aws_calls": 0,
        "network_calls": 0, "queries_changed": 0, "candidates_injected": 0}
    assert receipt == before
    for source in first["original_hybrid_pool"]["sources"]:
        shard = collection.get_shard(source["shard_id"])
        chunk = shard.chunks[source["chunk_id"]]
        doc = shard.documents[source["document_version_id"]]
        assert doc.normalized_text[source["start_char"]:source["end_char"]] == chunk.text
        assert source["text_sha256"] == sha256_bytes(chunk.text.encode())
        assert source["shard_manifest_sha256"] == shard.manifest_sha256


def test_complete_diagnostic_ranks_match_runtime_branch_prefixes(diagnostic_inputs):
    collection, lexical, receipt, record = diagnostic_inputs
    report = diagnostic.diagnose(
        collection, lexical, receipt, record, receipt["methods"][0]["ranked_chunk_ids"][0],
    )
    vector = np.asarray(record.vector, dtype=np.float32)
    assert [r["chunk_id"] for r in report["dense"]["top20"]] == [
        hit.chunk_id for hit in collection.search_vector(vector, k=20)]
    assert [(r["chunk_id"], r["score"]) for r in report["bm25"]["top20"]] == lexical.search(
        record.input_text, k=20)
    assert report["original_hybrid_pool"]["candidate_chunk_ids"] == receipt["methods"][0]["ranked_chunk_ids"]
    assert report["quality_metrics"] is None and not report["human_reviewed"]


def test_changed_query_is_rejected_before_ranking(diagnostic_inputs, monkeypatch):
    collection, lexical, receipt, record = diagnostic_inputs
    damaged = copy.deepcopy(receipt)
    damaged["search_query"] = "An altered search query"
    monkeypatch.setattr(diagnostic, "complete_rankings", lambda *a: pytest.fail("No ranking"))
    with pytest.raises(ValueError, match="QUERY_OR_INDEX_BINDING"):
        diagnostic.diagnose(collection, lexical, damaged, record,
                            receipt["methods"][0]["ranked_chunk_ids"][0])


def test_persisted_pool_changes_cannot_be_hidden_by_reconstructing_a_different_pool(diagnostic_inputs):
    collection, lexical, receipt, record = diagnostic_inputs
    damaged = copy.deepcopy(receipt)
    damaged["methods"][0]["ranked_chunk_ids"] = damaged["methods"][0]["ranked_chunk_ids"][1:]
    with pytest.raises(ValueError, match="ORIGINAL_HYBRID_REPLAY_MISMATCH"):
        diagnostic.diagnose(collection, lexical, damaged, record,
                            receipt["methods"][0]["ranked_chunk_ids"][0])


def test_unknown_target_is_not_injected(diagnostic_inputs):
    collection, lexical, receipt, record = diagnostic_inputs
    with pytest.raises(ValueError, match="TARGET_NOT_IN_CORPUS"):
        diagnostic.diagnose(collection, lexical, receipt, record, "chk_" + "f" * 64)


def test_large_collection_is_rejected_before_loading_any_shard():
    collection = SimpleNamespace(manifest=SimpleNamespace(
        counts=SimpleNamespace(chunks=20_001, documents=316), shards=(object(),)))
    with pytest.raises(ValueError, match="COLLECTION_DIAGNOSTIC_BOUND"):
        diagnostic.complete_rankings(collection, None, "Fixture", None)


def test_duplicate_exact_text_and_document_overlap_are_distinct_diagnostics():
    full = "alpha beta gamma"
    def row(text, start, end, version):
        chunk = SimpleNamespace(text=text, start_char=start, end_char=end,
            document_version_id=version, text_sha256=sha256_bytes(text.encode()))
        return chunk, SimpleNamespace(normalized_text=full), None

    records = {"a": row(full[:10], 0, 10, "doc-a"),
               "b": row(full[6:], 6, 16, "doc-a"),
               "c": row(full[:10], 0, 10, "doc-b")}
    report = diagnostic.pool_redundancy(["a", "b", "c"], records)
    assert report["exact_text_duplicate_groups"][0]["chunk_ids"] == ["a", "c"]
    overlap, = report["same_document_span_overlap_pairs"]
    assert (overlap["left_chunk_id"], overlap["right_chunk_id"]) == ("a", "b")
    assert overlap["overlap_characters"] == 4 and overlap["overlap_fraction_of_shorter_chunk"] == 0.4
    assert overlap["overlap_text_sha256"] == sha256_bytes(b"beta")
    assert report["lexical_similarity_pairs"] == [{
        "left_chunk_id": "a", "right_chunk_id": "c", "token_set_jaccard": 1.0,
        "exact_text_equal": True,
    }]


@pytest.mark.parametrize("guard", ["existing_output", "input_alias", "bad_target"])
def test_cli_guards_run_before_loading_collection_or_cache(tmp_path, monkeypatch, guard):
    def forbidden(*args, **kwargs):
        pytest.fail("Early CLI guard must precede collection/cache reads")

    monkeypatch.setattr(diagnostic, "load_collection", forbidden)
    monkeypatch.setattr(diagnostic, "cached_query", forbidden)
    receipt, output = tmp_path / "receipt.json", tmp_path / "out.json"
    argv = ["--receipt", str(receipt), "--output", str(output)]
    if guard == "existing_output":
        output.write_bytes(b"original")
        error = "OUTPUT_IMMUTABLE"
    elif guard == "input_alias":
        argv = ["--receipt", str(receipt), "--output", str(tmp_path / "x/../receipt.json")]
        error = "CANNOT_OVERWRITE_INPUT"
    else:
        argv += ["--target-chunk-id", "not-a-chunk"]
        error = "TARGET_ID_INVALID"
    with pytest.raises(ValueError, match=error):
        diagnostic.main(argv)
    if guard == "existing_output":
        assert output.read_bytes() == b"original"
    else:
        assert not output.exists()


def test_bounded_json_rejects_empty_and_oversized_files(tmp_path):
    path = tmp_path / "record.json"
    for raw in (b"", b"x" * 33):
        path.write_bytes(raw)
        with pytest.raises(ValueError, match="INPUT_PATH_OR_SIZE"):
            diagnostic.bounded_json(path, 32)
    path.write_bytes(b'{"fixture":true}')
    value, checksum = diagnostic.bounded_json(path, 32)
    assert value == {"fixture": True} and checksum == sha256_bytes(path.read_bytes())
