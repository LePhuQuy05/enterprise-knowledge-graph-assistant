import dataclasses
import json
from types import SimpleNamespace

import numpy as np
import pytest
from tokenizers import Tokenizer, models, normalizers, pre_tokenizers, processors

from app.contracts import Chunk, canonical_json, sha256_bytes
from app.reranking import (
    MANIFEST_VERSION,
    MODEL_ID,
    MODEL_REVISION,
    OnnxCrossEncoder,
    RerankingError,
    verify_reranker_bundle,
)


class SyntheticSession:
    """Only synthetic rank invariants, never model-quality or ONNX parity evidence."""

    def __init__(self):
        self.feeds = []
        self.behavior = "normal"

    def get_inputs(self):
        return [SimpleNamespace(name=name, type="tensor(int64)", shape=["batch", "tokens"])
                for name in ("input_ids", "attention_mask", "token_type_ids")]

    def get_outputs(self):
        return [SimpleNamespace(name="logits", type="tensor(float)", shape=["batch", 1])]

    def run(self, _, feed):
        self.feeds.append({key: value.copy() for key, value in feed.items()})
        if self.behavior == "failure":
            raise RuntimeError("SYNTHETIC_RUNTIME_FAILURE")
        selected = (feed["input_ids"] == 5) & (feed["token_type_ids"] == 1) & (feed["attention_mask"] == 1)
        logits = selected.sum(axis=1).astype(np.float32).reshape(-1, 1) - np.float32(3)
        if self.behavior == "nan":
            logits[0, 0] = np.nan
        elif self.behavior == "inf":
            logits[0, 0] = np.inf
        elif self.behavior == "wrong_shape":
            return [logits[:, 0]]
        elif self.behavior == "wrong_dtype":
            return [logits.astype(np.float64)]
        elif self.behavior == "extra_output":
            return [logits, logits]
        return [logits]


@pytest.fixture
def bundle(tmp_path):
    directory = tmp_path / "bundle"
    (directory / "onnx").mkdir(parents=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "x": 4, "signal": 5,
             "q": 6, "a": 7, "b": 8, "tiếng": 9, "Việt": 10, "🚀": 11}
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer.enable_truncation(max_length=8)
    tokenizer.enable_padding(length=8)
    tokenizer.save(str(directory / "tokenizer.json"))
    files = {"onnx/model.onnx": b"SYNTHETIC_ONNX_BYTES_NOT_A_REAL_MODEL", "config.json": b"{}",
             "tokenizer_config.json": b"{}", "special_tokens_map.json": b"{}", "README.md": b"Synthetic fixture"}
    for name, content in files.items():
        (directory / name).write_bytes(content)
    names = [*files, "tokenizer.json"]
    config = {"version": MANIFEST_VERSION, "model_id": MODEL_ID, "revision": MODEL_REVISION, "max_tokens": 512,
        "files": [{"path": name, "bytes": (directory / name).stat().st_size,
                   "sha256": sha256_bytes((directory / name).read_bytes())} for name in names]}
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(canonical_json(config))
    return directory, manifest, config


@pytest.fixture
def reranker(bundle, monkeypatch):
    import onnxruntime as ort

    session = SyntheticSession()
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: session)
    return OnnxCrossEncoder(bundle[0], bundle[1]), session


def chunk(text, *, position=0, tenant="demo"):
    return Chunk.create(tenant_id=tenant, document_version_id="docv_synthetic", chunking_version="fixture-v1",
        section_path=("Fixture",), position=position, start_char=0, end_char=len(text), text=text,
        token_count=max(1, len(text.split())), tokenizer_version="fixture-v1")


def test_offline_rank_order_raw_logits_and_batch_padding_use_real_pair_tokenizer(reranker):
    encoder, session = reranker
    rows = [chunk("x", position=0), chunk("signal signal", position=1), chunk("signal", position=2)]
    result = encoder.rerank("q", rows, tenant_id="demo", batch_size=2)
    assert result.ranked_chunk_ids == (rows[1].chunk_id, rows[2].chunk_id, rows[0].chunk_id)
    assert tuple(score.chunk_id for score in result.scores) == result.ranked_chunk_ids
    assert [score.raw_logit for score in result.scores] == [-1.0, -2.0, -3.0]
    assert [score.original_rank for score in result.scores] == [2, 3, 1]
    assert encoder.inference_calls == len(session.feeds) == 2
    assert result.recipe_sha256 == encoder.recipe.fingerprint
    assert encoder.recipe.max_tokens == 512 and encoder.recipe.window_aggregation == "max-raw-logit"
    for feed in session.feeds:
        assert all(value.dtype == np.int64 for value in feed.values())
        assert feed["input_ids"].shape[1] <= 512
        assert not feed["attention_mask"][feed["input_ids"] == 0].any()
        assert ((feed["token_type_ids"] == 1) & (feed["attention_mask"] == 1)).any()


def test_empty_returns_no_inference_and_ties_keep_original_candidate_order(reranker):
    encoder, session = reranker
    empty = encoder.rerank("q", [], "demo")
    assert not empty.ranked_chunk_ids and not empty.scores and not empty.windows and not session.feeds
    rows = sorted([chunk("a"), chunk("b", position=1)], key=lambda value: value.chunk_id, reverse=True)
    result = encoder.rerank("q", rows, "demo")
    assert result.ranked_chunk_ids == tuple(row.chunk_id for row in rows)
    assert [score.original_rank for score in result.scores] == [1, 2]


def test_long_passage_tail_and_unicode_whitespace_fully_covered_without_truncation(reranker):
    encoder, _ = reranker
    text = "  \n" + "x " * 620 + "tiếng Việt 🚀 signal \n  "
    row = chunk(text)
    result = encoder.rerank("q", [row, chunk("x", position=1)], "demo")
    windows = [window for window in result.windows if window.chunk_id == row.chunk_id]
    assert len(windows) == 2 and windows[0].start_char == 0 and windows[-1].end_char == len(text)
    assert windows[0].end_char == windows[1].start_char
    assert "".join(text[window.start_char:window.end_char] for window in windows) == text
    assert all(window.offset_basis == "CHUNK_LOCAL_UNICODE" and window.input_tokens <= 512 for window in windows)
    assert windows[-1].raw_logit > windows[0].raw_logit
    assert result.ranked_chunk_ids[0] == row.chunk_id and result.scores[0].raw_logit == windows[-1].raw_logit
    # The saved tokenizer has truncation at8: successful >8-token windows prove
    # request-time loading disables silent tokenizer truncation.
    assert windows[0].input_tokens == 512


def test_wordpiece_cut_reencoding_and_accent_normalization_keep_original_char_coverage(bundle, monkeypatch):
    import onnxruntime as ort

    directory, manifest, config = bundle
    vocabulary = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3, "q": 4, "signal": 5,
                  "un": 6, "##bel": 7, "##iev": 8, "##able": 9}
    tokenizer = Tokenizer(models.WordPiece(vocabulary, unk_token="[UNK]"))
    tokenizer.normalizer = normalizers.BertNormalizer(lowercase=True, strip_accents=True)
    tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
    tokenizer.post_processor = processors.TemplateProcessing(single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1", special_tokens=[("[CLS]", 2), ("[SEP]", 3)])
    tokenizer.save(str(directory / "tokenizer.json"))
    for entry in config["files"]:
        if entry["path"] == "tokenizer.json":
            content = (directory / "tokenizer.json").read_bytes()
            entry.update(bytes=len(content), sha256=sha256_bytes(content))
    manifest.write_bytes(canonical_json(config))
    session = SyntheticSession()
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: session)
    encoder = OnnxCrossEncoder(directory, manifest)
    query = "q " * 506  # 3 passage tokens fit, cutting inside a 4-piece word.
    text = " \nunbelievable unbelievable tie\u0302ng Việt 🚀 signal  \n"
    result = encoder.rerank(query, [chunk(text)], "demo")
    assert 2 <= len(result.windows) <= 8
    assert "".join(text[window.start_char:window.end_char] for window in result.windows) == text
    assert all(window.input_tokens <= 512 for window in result.windows)
    assert result.windows[-1].raw_logit == -2.0
    for window in result.windows:
        actual = tokenizer.encode(query, text[window.start_char:window.end_char])
        assert len(actual.ids) == window.input_tokens


@pytest.mark.parametrize("kind", ["cross_tenant", "hash", "identity", "duplicate", "too_many"])
def test_all_candidates_are_validated_before_first_inference(reranker, kind):
    encoder, session = reranker
    valid = chunk("signal")
    rows = [valid, chunk("x", position=1)]
    if kind == "cross_tenant":
        rows[-1] = chunk("x", tenant="other")
    elif kind == "hash":
        rows[-1] = rows[-1].model_copy(update={"text": "forged"})
    elif kind == "identity":
        rows[-1] = rows[-1].model_copy(update={"chunk_id": "chk_forged"})
    elif kind == "duplicate":
        rows[-1] = valid
    else:
        rows = [chunk("x", position=i) for i in range(41)]
    with pytest.raises(RerankingError):
        encoder.rerank("q", rows, "demo", batch_size=1)
    assert not session.feeds and encoder.inference_calls == 0


@pytest.mark.parametrize("kind", ["query", "chunk_windows", "total_windows", "text_length"])
def test_length_and_work_limits_fail_closed_before_scoring(reranker, kind):
    encoder, session = reranker
    query, rows = "q", [chunk("x")]
    if kind == "query":
        query = "q " * 509  # query509 + actualpair3 leaves no passage room.
    elif kind == "chunk_windows":
        query, rows = "q " * 508, [chunk("x " * 9)]
    elif kind == "total_windows":
        query, rows = "q " * 508, [chunk("x " * 5, position=i) for i in range(33)]
    else:
        rows = [chunk("x" * 50001)]
    with pytest.raises(RerankingError):
        encoder.rerank(query, rows, "demo")
    assert not session.feeds and encoder.inference_calls == 0


def test_exact_query_pair_boundary_is_allowed_and_query_is_never_truncated(reranker):
    encoder, session = reranker
    result = encoder.rerank("q " * 508, [chunk("signal")], "demo")
    assert result.windows[0].input_tokens == 512
    assert int((session.feeds[0]["input_ids"] == 6).sum()) == 508
    assert result.scores[0].raw_logit == -2.0


@pytest.mark.parametrize("behavior", ["nan", "inf", "wrong_shape", "wrong_dtype", "extra_output", "failure"])
def test_invalid_logits_or_runtime_failure_never_return_partial_or_fallback_ranking(reranker, behavior):
    encoder, session = reranker
    session.behavior = behavior
    expected = RuntimeError if behavior == "failure" else RerankingError
    with pytest.raises(expected):
        encoder.rerank("q", [chunk("signal"), chunk("x", position=1)], "demo", batch_size=1)
    assert encoder.inference_calls == len(session.feeds) == 1


@pytest.mark.parametrize("kind", ["revision", "model", "max_tokens", "duplicate", "traversal", "bad_size", "checksum", "missing"])
def test_specialized_asset_verifier_rejects_unpinned_unsafe_or_corrupt_bundles(bundle, kind):
    directory, _, original = bundle
    config = json.loads(json.dumps(original))
    if kind == "revision":
        config["revision"] = "f" * 40
    elif kind == "model":
        config["model_id"] = "other/model"
    elif kind == "max_tokens":
        config["max_tokens"] = True
    elif kind == "duplicate":
        config["files"].append(config["files"][0])
    elif kind == "traversal":
        config["files"][0]["path"] = "../model.onnx"
    elif kind == "bad_size":
        config["files"][0]["bytes"] = True
    elif kind == "checksum":
        path = directory / config["files"][0]["path"]
        path.write_bytes(b"X" * path.stat().st_size)
    else:
        (directory / config["files"][0]["path"]).unlink()
    with pytest.raises(RerankingError):
        verify_reranker_bundle(directory, config)


@pytest.mark.parametrize("bad", ["input_type", "missing_input", "output_width", "output_type"])
def test_constructor_checks_model_input_and_output_signatures(bundle, monkeypatch, bad):
    import onnxruntime as ort

    session = SyntheticSession()
    inputs, outputs = session.get_inputs(), session.get_outputs()
    if bad == "input_type":
        inputs[0].type = "tensor(float)"
    elif bad == "missing_input":
        inputs.pop()
    elif bad == "output_width":
        outputs[0].shape[-1] = 2
    else:
        outputs[0].type = "tensor(double)"
    session.get_inputs = lambda: inputs
    session.get_outputs = lambda: outputs
    monkeypatch.setattr(ort, "InferenceSession", lambda *args, **kwargs: session)
    with pytest.raises(RerankingError, match="SIGNATURE_INVALID"):
        OnnxCrossEncoder(bundle[0], bundle[1])
    assert not session.feeds


def test_recipe_and_results_are_immutable_and_pin_runtime_asset_window_policy(reranker):
    encoder, _ = reranker
    result = encoder.rerank("q", [chunk("signal")], "demo")
    assert len(encoder.recipe.artifact_manifest_sha256) == 64
    assert encoder.recipe.model_revision == MODEL_REVISION
    assert encoder.recipe.onnxruntime_version and encoder.recipe.tokenizers_version
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.ranked_chunk_ids = ()
    with pytest.raises(ValueError):
        encoder.recipe.max_tokens = 256
