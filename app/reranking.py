"""Offline bounded CrossEncoder scoring; logits are ranks, not factual confidence."""

import hashlib
import importlib.metadata
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from app.contracts import Chunk, Record, Sha256, canonical_json, sha256_bytes

MODEL_ID = "cross-encoder/ms-marco-MiniLM-L2-v2"
MODEL_REVISION = "1b5cd67b15209f24824c50370e0397743aa9b787"
MANIFEST_VERSION = "ms-marco-minilm-l2-v2-onnx-fp32-v1"
MAX_SEQUENCE = 512
MAX_CANDIDATES = 40
MAX_WINDOWS_PER_CHUNK = 8
MAX_TOTAL_WINDOWS = 160


class RerankingError(ValueError):
    """Fail closed before ranking or publishing a partial reranker result."""


class RerankerRecipe(Record):
    version: Literal["cross-encoder-windowed-onnx-v1"] = "cross-encoder-windowed-onnx-v1"
    model_id: Literal["cross-encoder/ms-marco-MiniLM-L2-v2"] = MODEL_ID
    model_revision: Literal["1b5cd67b15209f24824c50370e0397743aa9b787"] = MODEL_REVISION
    artifact_manifest_sha256: Sha256
    onnxruntime_version: str
    tokenizers_version: str
    max_tokens: Literal[512] = MAX_SEQUENCE
    max_candidates: Literal[40] = MAX_CANDIDATES
    max_windows_per_chunk: Literal[8] = MAX_WINDOWS_PER_CHUNK
    max_total_windows: Literal[160] = MAX_TOTAL_WINDOWS
    input_recipe: Literal["query-original-passage-no-prefix-v1"] = "query-original-passage-no-prefix-v1"
    window_policy: Literal["contiguous-offset-covered-no-truncation-v1"] = "contiguous-offset-covered-no-truncation-v1"
    window_aggregation: Literal["max-raw-logit"] = "max-raw-logit"
    tie_break: Literal["original-rank-then-chunk-id"] = "original-rank-then-chunk-id"
    offset_basis: Literal["CHUNK_LOCAL_UNICODE"] = "CHUNK_LOCAL_UNICODE"

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


@dataclass(frozen=True)
class WindowScore:
    chunk_id: str
    original_rank: int
    window_index: int
    start_char: int
    end_char: int
    input_tokens: int
    raw_logit: float
    offset_basis: Literal["CHUNK_LOCAL_UNICODE"] = "CHUNK_LOCAL_UNICODE"


@dataclass(frozen=True)
class ChunkScore:
    chunk_id: str
    original_rank: int
    raw_logit: float
    window_count: int


@dataclass(frozen=True)
class RerankResult:
    ranked_chunk_ids: tuple[str, ...]
    scores: tuple[ChunkScore, ...]
    windows: tuple[WindowScore, ...]
    recipe_sha256: str


def verify_reranker_bundle(directory: Path, config: dict):
    """Separate policy from BGE verification; fixed revision, bounded files and hashes."""
    required = {"onnx/model.onnx", "tokenizer.json", "config.json", "tokenizer_config.json",
                "special_tokens_map.json", "README.md"}
    allowed = required | {"vocab.txt", "LICENSE", "LICENSE.txt"}
    rows = config.get("files")
    if (config.get("version") != MANIFEST_VERSION or config.get("model_id") != MODEL_ID
            or config.get("revision") != MODEL_REVISION or type(config.get("max_tokens")) is not int
            or config["max_tokens"] != MAX_SEQUENCE or type(rows) is not list or not rows
            or any(type(row) is not dict or set(row) != {"path", "bytes", "sha256"} for row in rows)):
        raise RerankingError("RERANKER_MANIFEST_INVALID")
    paths = [row["path"] for row in rows]
    if (any(type(path) is not str for path in paths) or len(paths) != len(set(paths))
            or not required.issubset(paths) or set(paths) - allowed or directory.is_symlink()):
        raise RerankingError("RERANKER_MANIFEST_INVALID")
    for row in rows:
        path = directory / row["path"]
        if (type(row["bytes"]) is not int or not 1 <= row["bytes"] <= 80 * 1024 * 1024
                or type(row["sha256"]) is not str or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None
                or not path.is_file() or path.is_symlink()
                or not path.resolve().is_relative_to(directory.resolve()) or path.stat().st_size != row["bytes"]):
            raise RerankingError("RERANKER_ASSET_MISSING_OR_SIZE_INVALID")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != row["sha256"]:
            raise RerankingError("RERANKER_ASSET_CHECKSUM_INVALID")


class OnnxCrossEncoder:
    fixture_only = False

    def __init__(self, directory: Path, manifest_path: Path, threads=2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if type(threads) is not int or not 1 <= threads <= 8:
            raise RerankingError("RERANKER_THREAD_LIMIT")
        if manifest_path.is_symlink() or not manifest_path.is_file() or not 1 <= manifest_path.stat().st_size <= 64 * 1024:
            raise RerankingError("RERANKER_MANIFEST_SIZE_OR_PATH")
        raw = manifest_path.read_bytes()
        try:
            config = json.loads(raw)
        except (ValueError, UnicodeError):
            raise RerankingError("RERANKER_MANIFEST_INVALID") from None
        if type(config) is not dict:
            raise RerankingError("RERANKER_MANIFEST_INVALID")
        verify_reranker_bundle(directory, config)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.pad_id = self.tokenizer.token_to_id("[PAD]")
        if type(self.pad_id) is not int or self.pad_id < 0:
            raise RerankingError("RERANKER_PAD_TOKEN_INVALID")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(directory / "onnx/model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if ({item.name for item in inputs} != {"input_ids", "attention_mask", "token_type_ids"}
                or len(inputs) != 3 or any(item.type != "tensor(int64)" or len(item.shape) != 2 for item in inputs)
                or len(outputs) != 1 or outputs[0].type != "tensor(float)" or len(outputs[0].shape) != 2
                or outputs[0].shape[-1] != 1):
            raise RerankingError("RERANKER_ONNX_SIGNATURE_INVALID")
        self.recipe = RerankerRecipe(artifact_manifest_sha256=sha256_bytes(raw),
            onnxruntime_version=importlib.metadata.version("onnxruntime"),
            tokenizers_version=importlib.metadata.version("tokenizers"))
        self.inference_calls = 0

    def _encode_pair(self, query, passage):
        encoded = self.tokenizer.encode(query, passage, add_special_tokens=True)
        if (not encoded.ids or len(encoded.ids) != len(encoded.type_ids) or len(encoded.ids) != len(encoded.attention_mask)
                or any(type(value) is not int or value < 0 for value in encoded.ids)
                or any(value not in (0, 1) for value in encoded.type_ids)
                or any(value != 1 for value in encoded.attention_mask)):
            raise RerankingError("RERANKER_TOKENIZER_PAIR_INVALID")
        return encoded

    def _windows(self, query, text, passage_capacity):
        whole = self._encode_pair(query, text)
        if len(whole.ids) <= MAX_SEQUENCE:
            return [(0, len(text), whole)]
        passage = self.tokenizer.encode(text, add_special_tokens=False)
        offsets = passage.offsets
        if (len(offsets) != len(passage.ids) or not offsets
                or any(not 0 <= start < end <= len(text) for start, end in offsets)
                or any(start < previous[0] or end < previous[1]
                       for previous, (start, end) in zip(offsets, offsets[1:]))):
            raise RerankingError("RERANKER_PASSAGE_OFFSETS_INVALID")
        windows, cursor = [], 0
        while cursor < len(text):
            if len(windows) >= MAX_WINDOWS_PER_CHUNK:
                raise RerankingError("RERANKER_CHUNK_WINDOW_LIMIT")
            first = next((i for i, (_, end) in enumerate(offsets) if end > cursor), len(offsets))
            last = min(first + passage_capacity, len(offsets)) - 1
            end = len(text) if last >= len(offsets) - 1 else offsets[last][1]
            if end <= cursor:
                end = cursor + 1
            encoded = self._encode_pair(query, text[cursor:end])
            # WordPiece re-tokenization at a cut may change token count. Check
            # actual pairs and shrink to preceding real-token character ends.
            while len(encoded.ids) > MAX_SEQUENCE:
                preceding = [stop for _, stop in offsets if cursor < stop < end]
                if preceding:
                    end = preceding[-1]
                elif end > cursor + 1:
                    end = cursor + 1
                else:
                    raise RerankingError("RERANKER_SINGLE_CHARACTER_PAIR_OVERFLOW")
                encoded = self._encode_pair(query, text[cursor:end])
            windows.append((cursor, end, encoded))
            cursor = end
        if (windows[0][0] != 0 or windows[-1][1] != len(text)
                or any(left[1] != right[0] for left, right in zip(windows, windows[1:]))):
            raise RerankingError("RERANKER_WINDOW_COVERAGE_INVALID")
        return windows

    def rerank(self, query: str, chunks: Sequence[Chunk], tenant_id: str, batch_size=4) -> RerankResult:
        if (type(query) is not str or not query.strip() or len(query) > 2000 or type(tenant_id) is not str
                or not tenant_id or type(batch_size) is not int or not 1 <= batch_size <= 16
                or not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes))
                or len(chunks) > MAX_CANDIDATES):
            raise RerankingError("RERANKER_INPUT_LIMIT")
        # Validate every candidate before any inference, even a later bad item.
        ids = []
        for chunk in chunks:
            if not isinstance(chunk, Chunk) or chunk.tenant_id != tenant_id:
                raise RerankingError("RERANKER_CHUNK_TYPE_OR_TENANT")
            if len(chunk.text) > 50000 or sha256_bytes(chunk.text.encode()) != chunk.text_sha256:
                raise RerankingError("RERANKER_SOURCE_TEXT_OR_HASH_INVALID")
            try:
                Chunk.model_validate_json(chunk.model_dump_json())
            except ValueError:
                raise RerankingError("RERANKER_CHUNK_IDENTITY_INVALID") from None
            ids.append(chunk.chunk_id)
        if len(ids) != len(set(ids)):
            raise RerankingError("RERANKER_DUPLICATE_CHUNKS")
        if not chunks:
            return RerankResult((), (), (), self.recipe.fingerprint)
        query_tokens = self.tokenizer.encode(query, add_special_tokens=False)
        special = self.tokenizer.num_special_tokens_to_add(is_pair=True)
        capacity = MAX_SEQUENCE - len(query_tokens.ids) - special
        if capacity < 1 or len(self._encode_pair(query, "").ids) >= MAX_SEQUENCE:
            raise RerankingError("RERANKER_QUERY_TOO_LONG_NO_TRUNCATION")
        pending = []
        for rank, chunk in enumerate(chunks, 1):
            windows = self._windows(query, chunk.text, capacity)
            for index, (start, end, encoded) in enumerate(windows, 1):
                pending.append((chunk.chunk_id, rank, index, start, end, encoded))
            if len(pending) > MAX_TOTAL_WINDOWS:
                raise RerankingError("RERANKER_TOTAL_WINDOW_LIMIT")
        windows = []
        for start in range(0, len(pending), batch_size):
            batch = pending[start:start + batch_size]
            width = max(len(row[5].ids) for row in batch)
            input_ids = np.full((len(batch), width), self.pad_id, dtype=np.int64)
            attention_mask = np.zeros_like(input_ids)
            token_type_ids = np.zeros_like(input_ids)
            for index, row in enumerate(batch):
                encoded = row[5]
                length = len(encoded.ids)
                input_ids[index, :length] = encoded.ids
                attention_mask[index, :length] = encoded.attention_mask
                token_type_ids[index, :length] = encoded.type_ids
            self.inference_calls += 1  # Counts attempted calls, including a failing runtime.
            outputs = self.session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask,
                                               "token_type_ids": token_type_ids})
            if (not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], np.ndarray)
                    or outputs[0].dtype != np.dtype("float32") or outputs[0].shape != (len(batch), 1)
                    or not np.isfinite(outputs[0]).all()):
                raise RerankingError("RERANKER_LOGITS_INVALID")
            for row, logit in zip(batch, outputs[0][:, 0], strict=True):
                chunk_id, rank, index, begin, end, encoded = row
                windows.append(WindowScore(chunk_id, rank, index, begin, end, len(encoded.ids), float(logit)))
        scores = []
        for rank, chunk in enumerate(chunks, 1):
            selected = [window.raw_logit for window in windows if window.chunk_id == chunk.chunk_id]
            scores.append(ChunkScore(chunk.chunk_id, rank, max(selected), len(selected)))
        scores.sort(key=lambda score: (-score.raw_logit, score.original_rank, score.chunk_id))
        return RerankResult(tuple(score.chunk_id for score in scores), tuple(scores), tuple(windows), self.recipe.fingerprint)
