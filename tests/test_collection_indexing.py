from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from app.collection_indexing import index_corpus_directories
from app.collections import load_collection
from app.contracts import (
    Chunk,
    DocumentVersion,
    LocalBgeRecipe,
    canonical_json,
    embedding_input,
    sha256_bytes,
    stable_id,
)
from app.corpus import load_corpus


class FixtureEncoder:
    fixture_only = True

    def __init__(self, *, invalid=False):
        self.calls = 0
        self.invalid = invalid

    def recipe(self, role):
        return LocalBgeRecipe(input_role=role, input_recipe_version="title-section-text-v1",
                              artifact_manifest_sha256="a" * 64,
                              onnxruntime_version="fixture", tokenizers_version="fixture")

    def count(self, text, *, role):
        return len(text.split()) + 2

    def encode(self, texts, *, role, batch_size):
        self.calls += 1
        result = np.zeros((len(texts), 384), dtype=np.float32)
        if not self.invalid:
            result[:, 0] = 1
        return result


def corpus(tmp_path, name, *, text="Source evidence has a condition.",
           normalizer="fixture-normalizer", recipe_extra=None, tenant="demo", title="Fixture"):
    doc = DocumentVersion.create(
        tenant_id=tenant, source_uri=f"https://docs.aws.amazon.com/fixture/{name}.html",
        title=title, fetched_at=datetime(2026, 10, 23, tzinfo=UTC),
        raw_sha256=sha256_bytes(text.encode()), normalizer_version=normalizer,
        normalized_text_path=f"fixture/{name}.txt", normalized_text=text, source_kind="HTML",
    )
    chunk = Chunk.create(
        tenant_id=tenant, document_version_id=doc.document_version_id,
        chunking_version="fixture-chunk", section_path=("Fixture",), position=0,
        start_char=0, end_char=len(text), text=text, token_count=len(text.split()),
        tokenizer_version="fixture-tokens",
    )
    payloads = {
        "documents.jsonl": canonical_json(doc.model_dump(mode="json")) + b"\n",
        "chunks.jsonl": canonical_json(chunk.model_dump(mode="json")) + b"\n",
        "sources.jsonl": canonical_json({"source_uri": doc.source_uri}) + b"\n",
        "recipe.json": canonical_json({"normalizer_version": doc.normalizer_version,
                                       **(recipe_extra or {})}),
    }
    version = stable_id("corpus", files={key: sha256_bytes(raw) for key, raw in payloads.items()})
    directory = tmp_path / version
    directory.mkdir()
    for name, raw in payloads.items():
        (directory / name).write_bytes(raw)
    return directory


def test_indexes_distinct_corpus_versions_and_reuses_verified_checkpoints(tmp_path):
    directories = [corpus(tmp_path, "a"), corpus(tmp_path, "b")]
    encoder = FixtureEncoder()
    output = tmp_path / "output"
    report = index_corpus_directories(directories, output, encoder)
    assert report["documents"] == 2 and report["chunks"] == 2
    # Different source IDs with identical embedding inputs legitimately share a checkpoint.
    assert report["new_vectors"] == 1 and encoder.calls == 1
    assert report["purpose"] == "FIXTURE"
    assert report["serving_status"] == "NOT_CONNECTED"
    collection = load_collection(
        Path(report["collection_path"]), tenant_id="demo", embedding_model_id=encoder.recipe("document").model_id,
        allow_fixture=True,
    )
    query = np.zeros(384, dtype=np.float32)
    query[0] = 1
    assert len(collection.search_vector(query)) == 2
    replay = index_corpus_directories(directories, output, encoder)
    assert replay["new_vectors"] == 0 and encoder.calls == 1
    assert replay["collection_path"] == report["collection_path"]
    assert all(row["vector_checkpoint_hit"] for row in replay["shards"])


def test_tampered_checkpoint_is_rejected_without_reencoding(tmp_path):
    directory, encoder = corpus(tmp_path, "a"), FixtureEncoder()
    output = tmp_path / "output"
    index_corpus_directories([directory], output, encoder)
    checkpoint = next((output / "vector-cache").glob("*.npy"))
    raw = bytearray(checkpoint.read_bytes())
    raw[-1] ^= 1
    checkpoint.write_bytes(raw)
    with pytest.raises(ValueError, match="CHECKSUM_INVALID"):
        index_corpus_directories([directory], output, encoder)
    assert encoder.calls == 1


def test_invalid_vector_fails_before_snapshot_publication(tmp_path):
    directory = corpus(tmp_path, "a")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="ENCODER_VECTOR_INVALID"):
        index_corpus_directories([directory], output, FixtureEncoder(invalid=True))
    assert not list(output.glob("snap_*"))


def test_oversized_complete_model_input_not_silently_truncated(tmp_path):
    directory = corpus(tmp_path, "a", text="word " * 600)
    encoder = FixtureEncoder()
    with pytest.raises(ValueError, match="EXCEEDS_MODEL_CONTEXT"):
        index_corpus_directories([directory], tmp_path / "output", encoder)
    assert encoder.calls == 0


def test_mixed_normalizers_rejected_before_any_encoding(tmp_path):
    a = corpus(tmp_path, "a")
    b = corpus(tmp_path, "b", normalizer="other-normalizer")
    encoder = FixtureEncoder()
    with pytest.raises(ValueError, match="MIXED_RECIPES_CREATE_SEPARATE_COLLECTIONS"):
        index_corpus_directories([a, b], tmp_path / "output", encoder)
    assert encoder.calls == 0


def test_changed_release_metadata_reuses_identical_ordered_embedding_inputs(tmp_path):
    a = corpus(tmp_path, "a", recipe_extra={"selection_hash": "before"})
    b = corpus(tmp_path, "a", recipe_extra={"selection_hash": "after"})
    encoder = FixtureEncoder()
    first = index_corpus_directories([a], tmp_path / "output", encoder)
    second = index_corpus_directories([b], tmp_path / "output", encoder)
    assert first["collection_path"] != second["collection_path"]
    assert second["new_vectors"] == 0 and encoder.calls == 1


@pytest.mark.parametrize("directories,batch_size", [([], 8), (["unused"], 0), (["unused"], 17)])
def test_invalid_index_bounds(directories, batch_size, tmp_path):
    with pytest.raises(ValueError, match="INVALID"):
        index_corpus_directories(directories, tmp_path / "output", FixtureEncoder(), batch_size=batch_size)


class InputEncoder(FixtureEncoder):
    """Distinct toy vectors make accidental row reordering visible."""

    def __init__(self):
        super().__init__()
        self.encoded_texts = []

    @staticmethod
    def vector(text):
        row = np.zeros(384, dtype=np.float32)
        row[int(sha256_bytes(text.encode())[:8], 16) % 384] = 1
        return row

    def encode(self, texts, *, role, batch_size):
        self.calls += 1
        self.encoded_texts.extend(texts)
        return np.stack([self.vector(text) for text in texts])


def combined_corpus(root, directories):
    loaded = [load_corpus(directory, tenant_id="demo") for directory in directories]
    docs = sorted((doc for corpus in loaded for doc in corpus.documents),
                  key=lambda doc: doc.document_version_id)
    chunks = sorted((chunk for corpus in loaded for chunk in corpus.chunks),
                    key=lambda chunk: chunk.chunk_id)
    payloads = {
        "documents.jsonl": b"".join(canonical_json(doc.model_dump(mode="json")) + b"\n" for doc in docs),
        "chunks.jsonl": b"".join(canonical_json(chunk.model_dump(mode="json")) + b"\n" for chunk in chunks),
        "sources.jsonl": b"".join(canonical_json({"source_uri": doc.source_uri}) + b"\n" for doc in docs),
        "recipe.json": canonical_json(loaded[0].recipe),
    }
    version = stable_id("corpus", files={key: sha256_bytes(raw) for key, raw in payloads.items()})
    directory = root / version
    directory.mkdir()
    for name, raw in payloads.items():
        (directory / name).write_bytes(raw)
    return directory


def test_prior_collection_reuses_exact_inputs_across_changed_blocks_and_rebinds_sources(tmp_path):
    prior_inputs = [corpus(tmp_path, "old-a", text="Alpha evidence has a condition."),
                    corpus(tmp_path, "old-b", text="Beta evidence needs permission.")]
    encoder = InputEncoder()
    prior = index_corpus_directories(prior_inputs, tmp_path / "old-index", encoder)
    previous = load_collection(Path(prior["collection_path"]), tenant_id="demo",
                               embedding_model_id=encoder.recipe("document").model_id,
                               allow_fixture=True)
    prior_document_ids = {
        doc.document_id for ref in previous.manifest.shards
        for doc in previous.get_shard(ref.shard_id).content.documents
    }
    # New source identities and a different chunk/normalizer release still have
    # the same complete title/section/text input for two of the three rows.
    target_inputs = [corpus(tmp_path, "new-a", text="Alpha evidence has a condition.",
                            normalizer="changed-normalizer"),
                     corpus(tmp_path, "new-b", text="Beta evidence needs permission.",
                            normalizer="changed-normalizer"),
                     corpus(tmp_path, "new-c", text="Gamma is genuinely new input.",
                            normalizer="changed-normalizer")]
    target = combined_corpus(tmp_path, target_inputs)
    current = load_corpus(target, tenant_id="demo")
    events = []
    encoder.encoded_texts.clear()
    report = index_corpus_directories(
        [target], tmp_path / "new-index", encoder, reuse_collection_path=Path(prior["collection_path"]),
        progress=events.append, batch_size=1,
    )
    assert report["new_vectors"] == 1
    assert report["prior_input_vectors_reused"] == 2
    assert report["block_checkpoint_vectors_reused"] == 0
    assert report["shards"][0]["vector_checkpoint_hit"] is False
    assert len(encoder.encoded_texts) == 1 and "Gamma is genuinely new input." in encoder.encoded_texts[0]
    assert events[0]["stage"] == "PRIOR_COLLECTION_REUSE" and events[0]["completed"] == 2
    assert events[-1]["completed"] == events[-1]["total"] == 3
    indexed = load_collection(Path(report["collection_path"]), tenant_id="demo",
                              embedding_model_id=encoder.recipe("document").model_id, allow_fixture=True)
    snapshot = indexed.get_shard(indexed.manifest.shards[0].shard_id)
    assert set(snapshot.documents) == {doc.document_version_id for doc in current.documents}
    assert not prior_document_ids.intersection(doc.document_id for doc in snapshot.content.documents)
    for row, chunk in enumerate(snapshot.content.chunks):
        text = embedding_input(snapshot.documents[chunk.document_version_id], chunk)
        np.testing.assert_array_equal(snapshot.vectors[row], InputEncoder.vector(text))
    calls = encoder.calls
    replay = index_corpus_directories(
        [target], tmp_path / "new-index", encoder, reuse_collection_path=Path(prior["collection_path"]),
    )
    assert encoder.calls == calls
    assert replay["new_vectors"] == replay["prior_input_vectors_reused"] == 0
    assert replay["block_checkpoint_vectors_reused"] == 3
    assert replay["shards"][0]["vector_checkpoint_hit"] is True


@pytest.mark.parametrize("mismatch", ["recipe", "purpose", "tenant", "fixture_to_live"])
def test_incompatible_prior_context_is_rejected_before_target_encoding(tmp_path, mismatch):
    prior_encoder = InputEncoder()
    prior_tenant = "other" if mismatch == "tenant" else "demo"
    if mismatch == "purpose":
        prior_encoder.fixture_only = False
    old = corpus(tmp_path, "old", tenant=prior_tenant)
    prior = index_corpus_directories([old], tmp_path / "old-index", prior_encoder,
                                     tenant_id=prior_tenant)
    target = corpus(tmp_path, "new")
    encoder = InputEncoder()
    if mismatch == "recipe":
        original = encoder.recipe
        encoder.recipe = lambda role: original(role).model_copy(update={"onnxruntime_version": "different"})
    elif mismatch == "fixture_to_live":
        encoder.fixture_only = False
    with pytest.raises(ValueError, match="INDEX_REUSE"):
        index_corpus_directories([target], tmp_path / "target-index", encoder,
                                 reuse_collection_path=Path(prior["collection_path"]))
    assert encoder.calls == 0 and not list((tmp_path / "target-index").glob("snap_*"))


def test_prior_tampering_blocks_before_encoding_even_with_a_whole_block_checkpoint(tmp_path):
    inputs = [corpus(tmp_path, "old-a", text="Alpha evidence."),
              corpus(tmp_path, "old-b", text="Beta evidence.")]
    encoder = InputEncoder()
    output = tmp_path / "index"
    report = index_corpus_directories(inputs, output, encoder)
    prior_path = Path(report["collection_path"])
    prior = load_collection(prior_path, tenant_id="demo",
                            embedding_model_id=encoder.recipe("document").model_id, allow_fixture=True)
    last = prior.manifest.shards[-1]
    (prior_path.parent / last.relative_path / "chunks.jsonl").write_bytes(b"corrupt prior source")
    calls = encoder.calls
    with pytest.raises(ValueError, match="INDEX_REUSE_COLLECTION_INVALID"):
        index_corpus_directories(inputs, output, encoder, reuse_collection_path=prior_path)
    assert encoder.calls == calls


def test_prior_reuse_hashes_complete_embedding_input_including_title(tmp_path):
    old = corpus(tmp_path, "old", title="Original title")
    encoder = InputEncoder()
    prior = index_corpus_directories([old], tmp_path / "old-index", encoder)
    target = corpus(tmp_path, "new", title="Changed title")
    encoder.encoded_texts.clear()
    report = index_corpus_directories([target], tmp_path / "new-index", encoder,
                                      reuse_collection_path=Path(prior["collection_path"]))
    assert report["prior_input_vectors_reused"] == 0
    assert report["new_vectors"] == 1
    assert len(encoder.encoded_texts) == 1 and encoder.encoded_texts[0].startswith("Changed title\n")
