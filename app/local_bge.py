"""Pinned BGE ONNX CPU inference. Loading and encoding perform no network operations."""

import importlib.metadata
import json
from pathlib import Path

import numpy as np

from app.contracts import LocalBgeRecipe, sha256_bytes
from app.local_assets import verify_bundle


class BgeError(ValueError):
    pass


class BgeEncoder:
    fixture_only = False

    def __init__(self, directory: Path, manifest_path: Path, *, threads: int = 2):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if type(threads) is not int or not 1 <= threads <= 8:
            raise BgeError("BGE_THREAD_LIMIT")
        manifest_raw = manifest_path.read_bytes()
        self.config = json.loads(manifest_raw)
        verify_bundle(directory, self.config)
        self.manifest_sha256 = sha256_bytes(manifest_raw)
        self.tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(str(directory / "onnx/model.onnx"), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        if self.input_names != {"input_ids", "attention_mask", "token_type_ids"}:
            raise BgeError("BGE_ONNX_INPUT_MISMATCH")
        if self.session.get_outputs()[0].shape[-1] != 384:
            raise BgeError("BGE_ONNX_OUTPUT_MISMATCH")
        self.inference_calls = 0

    def recipe(self, role: str) -> LocalBgeRecipe:
        return LocalBgeRecipe(input_role=role,
            input_recipe_version="title-section-text-v1" if role == "document" else "question-text-v1",
            artifact_manifest_sha256=self.manifest_sha256,
            onnxruntime_version=importlib.metadata.version("onnxruntime"),
            tokenizers_version=importlib.metadata.version("tokenizers"))

    def tokenize(self, text: str, *, role: str):
        if role not in {"document", "query"} or not isinstance(text, str) or not text.strip() or len(text) > 50000:
            raise BgeError("BGE_INPUT_INVALID")
        prefix = self.config["query_prefix"] if role == "query" else ""
        # Includes CLS/SEP and query instructions; no truncation or implicit model fetch.
        return self.tokenizer.encode(prefix + text, add_special_tokens=True)

    def count(self, text: str, *, role: str = "document") -> int:
        return len(self.tokenize(text, role=role).ids)

    def encode(self, texts: list[str], *, role: str, batch_size: int = 8) -> np.ndarray:
        if not texts or type(batch_size) is not int or not 1 <= batch_size <= 16 or len(texts) > 5000:
            raise BgeError("BGE_BATCH_LIMIT")
        encoded = [self.tokenize(text, role=role) for text in texts]
        if any(not 2 <= len(item.ids) <= 512 for item in encoded):
            raise BgeError("BGE_INPUT_EXCEEDS_512_NO_TRUNCATION")
        rows = []
        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            width = max(len(e.ids) for e in batch)
            ids = np.zeros((len(batch), width), dtype=np.int64)
            masks = np.zeros_like(ids)
            types = np.zeros_like(ids)
            for index, item in enumerate(batch):
                n = len(item.ids)
                ids[index, :n], masks[index, :n], types[index, :n] = item.ids, item.attention_mask, item.type_ids
            outputs = self.session.run(None, {"input_ids": ids, "attention_mask": masks, "token_type_ids": types})
            self.inference_calls += 1
            states = outputs[0]
            if states.shape != (len(batch), width, 384) or not np.isfinite(states).all():
                raise BgeError("BGE_OUTPUT_INVALID")
            vectors = states[:, 0, :].astype(np.float32)  # BGE uses CLS, not mean pooling.
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(norms).all() or np.any(norms <= 0):
                raise BgeError("BGE_ZERO_OR_INVALID_NORM")
            rows.append(vectors / norms)
        return np.concatenate(rows).astype(np.float32)
