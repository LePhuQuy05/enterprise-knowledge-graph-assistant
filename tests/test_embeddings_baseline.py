import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.cli import main
from app.contracts import Chunk, canonical_json, embedding_input, sha256_bytes, stable_id
from app.corpus import load_corpus
from app.cost import PriceSheet
from app.embeddings import (
    CachedEmbedding,
    EmbeddingCache,
    EmbeddingCacheBlocked,
    EmbeddingCacheError,
    plan_embeddings,
    query_embedding_input,
    titan_recipe,
)
from app.fixtures import FIXTURE_TENANT, fixture_content
from app.ledger import Ledger, MemoryLedgerStore
from app.model_gateway import CallContext, FixtureProvider, GatewayError, ModelGateway, ModelRoutes
from app.retrieval import LexicalIndex, lexical_tokens, retrieve_baseline


class CountingProvider(FixtureProvider):
    def __init__(self, fault=None):
        self.calls = 0
        self.fault = fault

    def invoke(self, route, body):
        self.calls += 1
        if self.fault == "timeout":
            raise TimeoutError("secret text should not be persisted")
        result = super().invoke(route, body)
        if self.fault == "rounding":
            result["embedding"][0] = 0.9995
        if self.fault == "zero":
            result["embedding"] = [0.0] * 1024
        return result


@pytest.fixture
def configured():
    now = datetime(2026, 10, 16, 17, tzinfo=UTC)
    ledger = Ledger(MemoryLedgerStore(), clock=lambda: now)
    ledger.initialize_month("kg-r72", "2026-10")
    ledger.initialize_run("kg-r72", "demo", "cache-test", cap_micro_usd=100_000)
    routes = ModelRoutes.model_validate_json(
        Path("configs/models/bedrock-us-east-1-v2.json").read_bytes())
    prices = PriceSheet.model_validate_json(
        Path("configs/prices/aws-us-east-1-2026-10-16.json").read_bytes())
    provider = CountingProvider()
    gateway = ModelGateway(ledger, provider, routes, prices.model_copy(update={"purpose": "FIXTURE"}))
    context = CallContext(tenant_id="demo", run_id="cache-test", request_id="r1", stage="EMBED")
    return gateway, context, provider


@pytest.fixture
def corpus_dir(tmp_path):
    content = fixture_content()
    payloads = {
        "documents.jsonl": b"".join(d.model_dump_json().encode() + b"\n" for d in
                                    sorted(content.documents, key=lambda d: d.document_version_id)),
        "chunks.jsonl": b"".join(c.model_dump_json().encode() + b"\n" for c in
                                 sorted(content.chunks, key=lambda c: c.chunk_id)),
        "sources.jsonl": b"[]\n", "recipe.json": canonical_json({"version": "synthetic-test"}),
    }
    version = stable_id("corpus", files={n: sha256_bytes(b) for n, b in payloads.items()})
    directory = tmp_path / version
    directory.mkdir()
    for name, raw in payloads.items():
        (directory / name).write_bytes(raw)
    return directory


def test_cache_hit_survives_reopen_without_gateway_or_new_ledger_charge(tmp_path, configured):
    gateway, context, provider = configured
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")
    first = cache.resolve(context, text="Synthetic doc", recipe=titan_recipe(), gateway=gateway)
    operation = gateway.ledger.operation(first.record.operation_id)
    assert not first.cache_hit and operation.status == "SETTLED"
    reopened = EmbeddingCache(tmp_path, namespace="FIXTURE")
    second = reopened.resolve(context.model_copy(update={"request_id": "r2"}),
                              text="Synthetic doc", recipe=titan_recipe())
    assert second.cache_hit and second.record == first.record and provider.calls == 1
    assert second.new_cost_micro_usd == 0 and first.new_cost_micro_usd == 1
    assert gateway.ledger.operation(first.record.operation_id) == operation


def test_cache_scope_tenant_and_full_recipe_prevent_reuse(tmp_path, configured):
    gateway, context, provider = configured
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")
    cache.resolve(context, text="Same text", recipe=titan_recipe(), gateway=gateway)
    assert EmbeddingCache(tmp_path).get("demo", "Same text", titan_recipe()) is None
    assert cache.get("other", "Same text", titan_recipe()) is None
    changed = titan_recipe().model_copy(update={"version": "next-recipe"})
    assert cache.get("demo", "Same text", changed) is None
    with pytest.raises(EmbeddingCacheError, match="GATEWAY_RECIPE"):
        EmbeddingCache(tmp_path).resolve(context, text="Same text", recipe=titan_recipe(), gateway=gateway)
    assert provider.calls == 1


def test_input_prefix_and_query_preprocessing_are_explicit(corpus_dir):
    corpus = load_corpus(corpus_dir, tenant_id=FIXTURE_TENANT)
    chunk = corpus.chunks[0]
    doc = next(d for d in corpus.documents if d.document_version_id == chunk.document_version_id)
    assert embedding_input(doc, chunk) == doc.title + "\n" + " > ".join(chunk.section_path) + "\n\n" + chunk.text
    assert query_embedding_input("  IAM\r\nKMS\rpolicy  ") == "IAM\nKMS\npolicy"
    assert query_embedding_input("Q?") == "Q?"
    for text in ["   ", "q" * 2001]:
        with pytest.raises(EmbeddingCacheError):
            query_embedding_input(text)


def test_failed_uncertain_call_never_retries_even_with_new_request_id(tmp_path, configured):
    gateway, context, provider = configured
    provider.fault = "timeout"
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")
    with pytest.raises(GatewayError):
        cache.resolve(context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
    with pytest.raises(EmbeddingCacheBlocked):
        cache.resolve(context.model_copy(update={"request_id": "r2"}), text="Synthetic",
                      recipe=titan_recipe(), gateway=gateway)
    assert provider.calls == 1
    with sqlite3.connect(cache.path) as db:
        state, claim, error = db.execute("SELECT state,claim,error_category FROM embeddings").fetchone()
    details = json.loads(claim)
    assert state == "FAILED" and error == "UncertainModelCall"
    assert gateway.ledger.operation(details["operation_id"]).status == "UNKNOWN_SPEND"
    assert details["context"]["request_id"] == "r1" and "secret" not in claim.decode()


def test_cache_write_failure_after_settlement_does_not_dispatch_again(tmp_path, configured, monkeypatch):
    gateway, context, provider = configured
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")

    def fail(*args, **kwargs):
        raise OSError("simulated persistence failure")

    monkeypatch.setattr(CachedEmbedding, "model_dump_json", fail)
    with pytest.raises(OSError):
        cache.resolve(context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
    with pytest.raises(EmbeddingCacheBlocked):
        cache.resolve(context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
    with sqlite3.connect(cache.path) as db:
        claim = json.loads(db.execute("SELECT claim FROM embeddings").fetchone()[0])
    assert gateway.ledger.operation(claim["operation_id"]).status == "SETTLED"
    assert provider.calls == 1


def test_process_crash_leaves_durable_claim_for_new_process(tmp_path, configured):
    gateway, context, provider = configured

    class SimulatedProcessCrash(BaseException):
        pass

    def crash(*args, **kwargs):
        provider.calls += 1
        raise SimulatedProcessCrash()

    provider.invoke = crash
    with pytest.raises(SimulatedProcessCrash):
        EmbeddingCache(tmp_path, namespace="FIXTURE").resolve(
            context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
    with pytest.raises(EmbeddingCacheBlocked):
        EmbeddingCache(tmp_path, namespace="FIXTURE").resolve(
            context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
    assert provider.calls == 1


def test_competing_reader_cannot_take_over_in_progress_owner(tmp_path, configured):
    gateway, context, provider = configured
    entered, release = threading.Event(), threading.Event()
    original = provider.invoke

    def waiting(*args):
        entered.set()
        assert release.wait(5)
        return original(*args)

    provider.invoke = waiting
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(cache.resolve, context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
        try:
            assert entered.wait(5)
            with pytest.raises(EmbeddingCacheBlocked):
                EmbeddingCache(tmp_path, namespace="FIXTURE").resolve(
                    context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
        finally:
            release.set()
        assert not future.result().cache_hit
    assert provider.calls == 1


def test_output_is_tightened_to_snapshot_float32_norm(tmp_path, configured):
    gateway, context, provider = configured
    provider.fault = "rounding"
    record = EmbeddingCache(tmp_path, namespace="FIXTURE").resolve(
        context, text="Synthetic", recipe=titan_recipe(), gateway=gateway).record
    assert np.linalg.norm(np.asarray(record.vector, dtype=np.float32)) == 1


def test_bad_output_never_becomes_ready_or_zero_vector(tmp_path, configured):
    gateway, context, provider = configured
    provider.fault = "zero"
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")
    with pytest.raises(GatewayError):
        cache.resolve(context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
    with pytest.raises(EmbeddingCacheBlocked):
        cache.get("demo", "Synthetic", titan_recipe())
    assert provider.calls == 1


@pytest.mark.parametrize("recompute_checksum", [False, True])
def test_corrupt_cache_fails_closed_without_new_call(tmp_path, configured, recompute_checksum):
    gateway, context, provider = configured
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")
    cache.resolve(context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
    with sqlite3.connect(cache.path) as db:
        row = db.execute("SELECT payload FROM embeddings").fetchone()[0]
        data = json.loads(row)
        data["vector"] = [0.0] * 512
        raw = canonical_json(data)
        if recompute_checksum:
            db.execute("UPDATE embeddings SET payload=?,payload_sha256=?", (raw, sha256_bytes(raw)))
        else:
            db.execute("UPDATE embeddings SET payload=?", (raw,))
    with pytest.raises(ValueError):
        cache.resolve(context, text="Synthetic", recipe=titan_recipe(), gateway=gateway)
    assert provider.calls == 1


def test_cache_miss_without_gateway_or_wrong_stage_does_no_work(tmp_path, configured):
    _, context, provider = configured
    cache = EmbeddingCache(tmp_path)
    with pytest.raises(EmbeddingCacheError, match="NO_GATEWAY"):
        cache.resolve(context, text="Synthetic", recipe=titan_recipe())
    with pytest.raises(EmbeddingCacheError, match="EMBED_STAGE"):
        cache.resolve(context.model_copy(update={"stage": "GENERATE"}), text="Synthetic", recipe=titan_recipe())
    with sqlite3.connect(cache.path) as db:
        assert db.execute("SELECT count(*) FROM embeddings").fetchone()[0] == 0
    assert provider.calls == 0


def test_plan_has_no_vectors_or_implicit_calls(corpus_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("boto3.Session", lambda **kw: pytest.fail("Must stay offline"))
    corpus = load_corpus(corpus_dir, tenant_id=FIXTURE_TENANT)
    report = plan_embeddings(corpus, EmbeddingCache(tmp_path / "cache"), titan_recipe())
    assert report["cache_hits"] == report["inference_calls"] == report["blocked_inputs"] == 0
    assert report["cache_misses"] == report["unique_inputs"] == len(corpus.chunks)


def test_corpus_hash_and_tenant_guards(corpus_dir):
    with pytest.raises(ValueError, match="TENANT"):
        load_corpus(corpus_dir, tenant_id="other")
    path = corpus_dir / "chunks.jsonl"
    path.write_bytes(path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="VERSION"):
        load_corpus(corpus_dir, tenant_id=FIXTURE_TENANT)


def _reseal(directory):
    payloads = {p.name: p.read_bytes() for p in directory.iterdir()}
    version = stable_id("corpus", files={n: sha256_bytes(b) for n, b in payloads.items()})
    destination = directory.parent / version
    destination.mkdir()
    for name, raw in payloads.items():
        (destination / name).write_bytes(raw)
    return destination


@pytest.mark.parametrize("fault", ["duplicate", "gap", "missing_document"])
def test_corpus_rejects_rehashed_cross_record_errors(corpus_dir, fault):
    path = corpus_dir / "chunks.jsonl"
    records = path.read_bytes().splitlines()
    if fault == "duplicate":
        records.insert(0, records[0])
    elif fault == "gap":
        record = json.loads(records[0])
        record["position"] += 100
        records[0] = canonical_json(record)
    else:
        record = Chunk.model_validate_json(records[0]).model_dump()
        record["document_version_id"] = "missing-document"
        record.pop("chunk_id")
        records[0] = Chunk.create(**record).model_dump_json().encode()
        records.sort(key=lambda raw: json.loads(raw)["chunk_id"])
    path.write_bytes(b"\n".join(records) + b"\n")
    directory = _reseal(corpus_dir)
    with pytest.raises(ValueError):
        load_corpus(directory, tenant_id=FIXTURE_TENANT)


def test_lexical_keeps_actual_matches_when_idf_is_negative(loaded):
    doc = loaded.content.documents[0]
    chunk = next(c for c in loaded.content.chunks if c.document_version_id == doc.document_version_id)
    index = LexicalIndex((doc,), (chunk,), tenant_id=FIXTURE_TENANT)
    result = index.search(lexical_tokens(embedding_input(doc, chunk))[0])
    assert len(result) == 1 and result[0][0] == chunk.chunk_id and result[0][1] < 0


def test_bm25_no_match_is_empty_instead_of_arbitrary_zero_score_chunks(loaded):
    index = LexicalIndex(loaded.content.documents, loaded.content.chunks, tenant_id=FIXTURE_TENANT)
    assert index.search("zzzzunmatchedidentifier") == []
    assert index.search("?!") == []
    assert lexical_tokens("AWS_Lambda SSE-KMS café") == ["aws", "lambda", "sse", "kms", "caf"]


def test_lexical_terms_have_deterministic_ranks_and_no_unmatched_candidates(loaded):
    index = LexicalIndex(loaded.content.documents, loaded.content.chunks, tenant_id=FIXTURE_TENANT)
    first = index.search("Permission", k=20)
    assert first and first == index.search("Permission", k=20)
    for chunk_id, _ in first:
        chunk = loaded.chunks[chunk_id]
        assert "permission" in lexical_tokens(embedding_input(loaded.documents[chunk.document_version_id], chunk))


def test_lexical_rejects_other_tenant_before_scoring(loaded):
    with pytest.raises(ValueError, match="TENANT"):
        LexicalIndex(loaded.content.documents, loaded.content.chunks, tenant_id="other")


def test_hybrid_no_match_preserves_vector_order(loaded):
    vector = np.zeros(1024, dtype=np.float32)
    vector[0] = 1
    vector_only = retrieve_baseline(loaded, "zzzzunmatchedidentifier", vector, mode="vector")
    hybrid = retrieve_baseline(loaded, "zzzzunmatchedidentifier", vector, mode="hybrid")
    assert hybrid == vector_only


@pytest.mark.parametrize("command", ["embedding-plan", "lexical-search"])
def test_real_corpus_cli_stays_offline(corpus_dir, tmp_path, monkeypatch, capsys, command):
    monkeypatch.setattr("boto3.Session", lambda **kw: pytest.fail("Must stay offline"))
    args = [command, "--corpus-dir", str(corpus_dir), "--tenant", FIXTURE_TENANT,
            "--report", str(tmp_path / "report.json")]
    args += (["--cache-dir", str(tmp_path / "cache")] if command == "embedding-plan" else
             ["--question", "Permission"])
    assert main(args) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["inference_calls"] == 0
    if command == "lexical-search":
        assert report["hits"] and "answer" not in report


def test_cli_rejects_wrong_tenant_without_echoing_private_question(corpus_dir, tmp_path, capsys):
    secret = "private question text"
    assert main(["lexical-search", "--corpus-dir", str(corpus_dir), "--tenant", "other",
                 "--question", secret, "--report", str(tmp_path / "report.json")]) == 1
    assert secret not in capsys.readouterr().out
