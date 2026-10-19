"""Local encoder/recipe boundaries, distinct from paid gateway and model-quality tests."""

import io
import json

import numpy as np
import pytest
from pydantic import ValidationError
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.processors import TemplateProcessing

from app.bge_model_store import ensure_bge_bundle
from app.contracts import EmbeddingRecipe, LocalBgeRecipe, embedding_cache_key, sha256_bytes
from app.embeddings import EmbeddingCache, EmbeddingCacheBlocked, EmbeddingCacheError
from app.fixtures import fixture_content, fixture_recipes
from app.local_assets import verify_bundle
from app.local_bge import BgeEncoder, BgeError
from app.retrieval import retrieve_baseline
from app.snapshot import SnapshotError, build_snapshot, load_snapshot


class Session:
    def __init__(self):
        self.calls = []

    def run(self, output_names, feed):
        self.calls.append(feed)
        batch, width = feed["input_ids"].shape
        states = np.zeros((batch, width, 384), dtype=np.float32)
        states[:, 0, 0] = 3
        states[:, 1:, 1] = 7  # Mean pooling would be observably wrong.
        return [states]


def encoder():
    tokenizer = Tokenizer(WordLevel({"[UNK]": 0, "[CLS]": 1, "[SEP]": 2, "a": 3, "b": 4}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.post_processor = TemplateProcessing(single="[CLS] $A [SEP]", special_tokens=[("[CLS]", 1), ("[SEP]", 2)])
    instance = BgeEncoder.__new__(BgeEncoder)
    instance.config = {"query_prefix": "Represent this sentence for searching relevant passages: "}
    instance.manifest_sha256 = "0" * 64
    instance.tokenizer, instance.session = tokenizer, Session()
    instance.inference_calls, instance.fixture_only = 0, True
    return instance


def test_cls_pooling_padding_role_prompt_and_float32_l2():
    e = encoder()
    result = e.encode(["a", "a b"], role="document")
    assert result.shape == (2, 384) and result.dtype == np.float32
    assert np.all(result[:, 0] == 1) and np.all(result[:, 1:] == 0)
    assert e.session.calls[0]["attention_mask"].tolist() == [[1, 1, 1, 0], [1, 1, 1, 1]]
    assert e.count("a", role="query") > e.count("a", role="document")
    assert e.inference_calls == 1


@pytest.mark.parametrize("text", ["a " * 511, "a " * 512])
def test_overlength_including_specials_rejected_before_forward(text):
    e = encoder()
    with pytest.raises(BgeError, match="EXCEEDS_512"):
        e.encode([text], role="document")
    assert e.inference_calls == 0
    assert e.encode(["a " * 510], role="document").shape == (1, 384)


@pytest.mark.parametrize("batch_size", [0, 17, True])
def test_bounded_batch_and_invalid_roles_fail(batch_size):
    e = encoder()
    with pytest.raises(BgeError, match="BATCH_LIMIT"):
        e.encode(["a"], role="document", batch_size=batch_size)
    with pytest.raises(BgeError, match="INPUT_INVALID"):
        e.encode(["a"], role="unknown")


def test_recipe_version_roles_legacy_dimension_and_runtime_identity():
    e = encoder()
    doc, query = e.recipe("document"), e.recipe("query")
    assert doc.dimension == 384 and doc.schema_version == 3
    assert embedding_cache_key("demo", "a", doc) != embedding_cache_key("demo", "a", query)
    changed = doc.model_copy(update={"artifact_manifest_sha256": "1" * 64})
    assert changed.fingerprint != doc.fingerprint
    with pytest.raises(ValidationError):
        EmbeddingRecipe(version="legacy", model_id="other", invocation_id="other", source_region="fixture", dimension=384)
    with pytest.raises(ValidationError, match="role/input"):
        LocalBgeRecipe.model_validate_json(json.dumps(doc.model_dump(mode="json") | {"input_role": "query"}))


def test_local_cache_reopen_has_no_forward_and_keeps_free_receipt(tmp_path):
    e = encoder()
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")
    first = cache.resolve_local("demo", text="a", recipe=e.recipe("document"), encoder=e)
    assert first.record.operation_id.startswith("localembed_") and first.new_cost_micro_usd == 0
    assert type(first.record.recipe) is LocalBgeRecipe and len(first.record.vector) == 384
    replay = EmbeddingCache(tmp_path, namespace="FIXTURE").resolve_local("demo", text="a", recipe=e.recipe("document"))
    assert replay.cache_hit and replay.record == first.record and e.inference_calls == 1
    with pytest.raises(EmbeddingCacheError, match="LOCAL_ENCODER_RECIPE_OR_NAMESPACE"):
        EmbeddingCache(tmp_path / "live", namespace="LIVE").resolve_local("demo", text="a", recipe=e.recipe("document"), encoder=e)


def test_failed_local_encoding_is_not_hidden_retry(tmp_path):
    e = encoder()
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")
    with pytest.raises(BgeError):
        cache.resolve_local("demo", text="a " * 511, recipe=e.recipe("document"), encoder=e)
    with pytest.raises(EmbeddingCacheBlocked):
        cache.resolve_local("demo", text="a " * 511, recipe=e.recipe("document"), encoder=e)
    assert e.inference_calls == 0


def test_snapshot384_roundtrip_baseline_and_reject_wrong_dimension(tmp_path):
    e = encoder()
    content = fixture_content()
    extraction, _ = fixture_recipes()
    vectors = {c.chunk_id: np.eye(1, 384, dtype=np.float32)[0] for c in content.chunks}
    kwargs = dict(tenant_id="tenant_fixture", purpose="FIXTURE", extraction_recipe=extraction,
                  corpus_version="synthetic", normalizer_version="normalize-v1", chunking_version="chunk-v1",
                  tokenizer_version="fixture-tokenizer-v1")
    # Use the source's exact metadata; this test changes the encoder, not provenance.
    kwargs.update(normalizer_version=content.documents[0].normalizer_version,
                  chunking_version=content.chunks[0].chunking_version,
                  tokenizer_version=content.chunks[0].tokenizer_version)
    path = build_snapshot(tmp_path, content, vectors, embedding_recipe=e.recipe("document"), **kwargs)
    loaded = load_snapshot(path, tenant_id="tenant_fixture", embedding_model_id=e.recipe("document").model_id, allow_fixture=True)
    assert loaded.vectors.shape == (5, 384)
    assert len(retrieve_baseline(loaded, "a", vectors[content.chunks[0].chunk_id], mode="vector")) == 5
    with pytest.raises(ValueError, match="snapshot dimension"):
        retrieve_baseline(loaded, "a", np.eye(1, 1024, dtype=np.float32)[0])
    with pytest.raises((ValueError, SnapshotError), match="document embedding recipe"):
        build_snapshot(tmp_path, content, vectors, embedding_recipe=e.recipe("query"), **kwargs)
    with pytest.raises(SnapshotError, match="dimension"):
        build_snapshot(tmp_path, content, {k: np.zeros(1024, dtype=np.float32) for k in vectors},
                       embedding_recipe=e.recipe("document"), **kwargs)


def test_model_manifest_rejects_unsafe_paths_before_loading():
    with pytest.raises(ValueError, match="MANIFEST_INVALID"):
        verify_bundle(None, {"files": [{"path": "../elsewhere"}], "model_id": "BAAI/bge-small-en-v1.5",
                             "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"})


@pytest.mark.parametrize("failure", [None, "size", "checksum", "overrun"])
def test_model_store_validates_before_use_and_reuses_immutable_tmp(tmp_path, failure):
    from pathlib import Path

    config = json.loads((Path(__file__).resolve().parents[1] / "configs/models/bge-small-en-v1.5-onnx.json").read_bytes())
    for row in config["files"]:
        row.update(bytes=3, sha256=sha256_bytes(b"abc"))
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps(config))

    class S3:
        calls = 0

        def get_object(self, **kwargs):
            self.calls += 1
            assert kwargs["Key"].startswith("models/bge-small-en-v1.5/")
            return {"ContentLength": 4 if failure == "size" else 3,
                    "Body": io.BytesIO(b"abcd" if failure == "overrun" else b"bad" if failure == "checksum" else b"abc")}

    client = S3()
    kwargs = dict(bucket="fixture-private-bucket", manifest_path=manifest, cache_root=tmp_path / "cache")
    if failure:
        with pytest.raises(ValueError, match="BGE_STORE_"):
            ensure_bge_bundle(client, **kwargs)
        assert not list((tmp_path / "cache").rglob("*.partial"))
    else:
        path = ensure_bge_bundle(client, **kwargs)
        assert client.calls == 8
        assert ensure_bge_bundle(client, **kwargs) == path and client.calls == 8
        (path / "tokenizer.json").write_bytes(b"bad")
        with pytest.raises(ValueError, match="CHECKSUM_INVALID"):
            ensure_bge_bundle(client, **kwargs)
