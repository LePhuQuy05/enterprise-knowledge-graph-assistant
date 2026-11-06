"""Offline CPU numerical parity of the real reranker against local safetensors.

Run in the isolated reference venv. This does not evaluate retrieval/answer quality.
All fixtures are synthetic or the pinned model-card example, never experiment labels.
"""

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import sys
from datetime import UTC, datetime
from pathlib import Path

EXPECTED_ONNX_SHA256 = "97ae94c94faf17d97398a7af38e44ee0ef326eff99ece940423b187e885eae6a"
EXPECTED_SAFETENSORS_SHA256 = "88f11fa671e11c53b5cfe88bb6594139ec4991eaf8cf6a10bd61c9abbc4f691a"


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fixtures():
    return [
        {
            "id": "model-card-berlin",
            "source": "Pinned README SentenceTransformers example; not a task benchmark",
            "query": "How many people live in Berlin?",
            "passages": [
                ("population", "Berlin had a population of 3,520,031 registered inhabitants "
                 "in an area of 891.82 square kilometers."),
                ("museums", "Berlin is well known for its museums."),
            ],
        },
        {
            "id": "synthetic-exact-negation-unrelated",
            "source": "Independent synthetic inputs; no relevance labels or task-quality claims",
            "query": "The Indigo service permits anonymous file uploads.",
            "passages": [
                ("exact", "The Indigo service permits anonymous file uploads."),
                ("negation", "The Indigo service does not permit anonymous file uploads. "
                 "A registered account is required."),
                ("unrelated", "A gardener grows rosemary in a clay pot beside a wooden fence."),
            ],
        },
        {
            "id": "synthetic-unicode",
            "source": "Independent Unicode inputs; this is not multilingual model evaluation",
            "query": "What is the name of the café in Zürich?",
            "passages": [
                ("accented", "The café in Zürich is called Hikari. Its sign says Café Hikari."),
                ("japanese", "喫茶店の名前はひかりです。Hikari café serves tea in Zürich. ☕"),
                ("unrelated", "The mountain railway closes during heavy snowfall."),
            ],
        },
        {
            "id": "synthetic-long-tail",
            "source": "Independent long fixture for contiguous window coverage only",
            "query": "What is the lighthouse height?",
            "passages": [
                ("long", "  \n" + "A quiet library stores books and opens every morning. " * 90
                 + "The final record lists the lighthouse height as 42 meters. \n  "),
            ],
            "must_window": True,
        },
        {
            "id": "synthetic-long-unicode",
            "source": "Independent long Unicode fixture for window offsets only",
            "query": "Which café is listed in the final atlas record?",
            "passages": [
                ("long", " \n" + "The atlas keeps a café entry for Zürich. " * 90
                 + "最後の記録: Hikari is the listed café. cafe\u0301 ☕ \n  "),
            ],
            "must_window": True,
        },
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("data/models/ms-marco-MiniLM-L2-v2"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("configs/models/ms-marco-MiniLM-L2-v2-onnx.json"))
    parser.add_argument("--output", type=Path, default=Path("results/reranker/reference-parity.json"))
    args = parser.parse_args()
    # Set before importing HuggingFace/PyTorch. Loading model/tokenizer cannot fetch.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
                      CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false")
    network_attempts = []

    def reject_network(*args, **kwargs):
        network_attempts.append("blocked socket connection attempt")
        raise RuntimeError("REFERENCE_VALIDATION_NETWORK_FORBIDDEN")

    socket.create_connection = reject_network
    socket.socket.connect = reject_network
    socket.socket.connect_ex = reject_network
    import numpy as np
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    from app.contracts import Chunk
    from app.reranking import MODEL_ID, MODEL_REVISION, OnnxCrossEncoder

    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)
    directory = args.model_dir.resolve()
    assert digest(directory / "onnx/model.onnx") == EXPECTED_ONNX_SHA256, "Unexpected pinned ONNX asset"
    assert digest(directory / "model.safetensors") == EXPECTED_SAFETENSORS_SHA256, "Unexpected pinned reference weights"
    source_sha256 = digest(Path("app/reranking.py"))
    encoder = OnnxCrossEncoder(directory, args.manifest, threads=2)
    reference_tokenizer = AutoTokenizer.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False, use_fast=True)
    reference = AutoModelForSequenceClassification.from_pretrained(
        str(directory), local_files_only=True, trust_remote_code=False,
        use_safetensors=True, attn_implementation="eager")
    reference.to(device="cpu", dtype=torch.float32)
    reference.eval()
    assert not reference.training
    assert {parameter.device.type for parameter in reference.parameters()} == {"cpu"}
    assert {parameter.dtype for parameter in reference.parameters()} == {torch.float32}
    assert torch.version.cuda is None, "Reference must use an actual CPU-only PyTorch build"

    class CaptureSession:
        def __init__(self, session):
            self.session, self.calls = session, []

        def run(self, output_names, feeds):
            output = self.session.run(output_names, feeds)
            self.calls.append(({key: value.copy() for key, value in feeds.items()}, output[0].copy()))
            return output

    encoder.session = capture = CaptureSession(encoder.session)
    tolerance = 1e-4
    report_groups = []
    all_differences, all_tokenizer_matches, all_rank_matches = [], [], []
    for fixture in fixtures():
        chunks = [Chunk.create(
            tenant_id="reference", document_version_id=f"docv_reference_{fixture['id']}",
            chunking_version="reference-fixture-v1", section_path=("Reference fixture",),
            position=position, start_char=0, end_char=len(text), text=text,
            token_count=max(1, len(text.split())), tokenizer_version="reference-fixture-v1")
            for position, (_, text) in enumerate(fixture["passages"])]
        before = len(capture.calls)
        result = encoder.rerank(fixture["query"], chunks, tenant_id="reference", batch_size=4)
        calls = capture.calls[before:]
        by_id = {chunk.chunk_id: chunk for chunk in chunks}
        reference_windows, batches, flat = [], [], 0
        for batch_index, (feed, onnx_logits) in enumerate(calls, 1):
            assert all(array.dtype == np.int64 for array in feed.values())
            assert feed["input_ids"].ndim == 2 and feed["input_ids"].shape[1] <= 512
            with torch.no_grad():
                logits = reference(**{key: torch.from_numpy(value.copy())
                                      for key, value in feed.items()}).logits.cpu().numpy()
            assert logits.shape == onnx_logits.shape and np.isfinite(logits).all()
            differences = np.abs(logits.astype(np.float64) - onnx_logits.astype(np.float64))
            all_differences.extend(differences[:, 0].tolist())
            rows = []
            for row_index in range(logits.shape[0]):
                window = result.windows[flat]
                flat += 1
                chunk = by_id[window.chunk_id]
                passage = chunk.text[window.start_char:window.end_char]
                independently_encoded = reference_tokenizer(
                    fixture["query"], passage, add_special_tokens=True,
                    padding=False, truncation=False,
                    return_attention_mask=True, return_token_type_ids=True)
                count = window.input_tokens
                matches = all(independently_encoded[key] == feed[key][row_index, :count].tolist()
                              for key in ("input_ids", "attention_mask", "token_type_ids"))
                all_tokenizer_matches.append(matches)
                assert int(feed["attention_mask"][row_index].sum()) == count
                assert not feed["attention_mask"][row_index, count:].any()
                assert (feed["input_ids"][row_index, count:] == encoder.pad_id).all()
                reference_windows.append((window.chunk_id, float(logits[row_index, 0])))
                rows.append({
                    **dataclasses.asdict(window), "passage": passage,
                    "reference_raw_logit": float(logits[row_index, 0]),
                    "absolute_difference": float(differences[row_index, 0]),
                    "independent_tokenizer_input_match": matches,
                })
            batches.append({"batch_index": batch_index, "input_dtype": "int64",
                            "output_dtype": "float32", "shape": list(feed["input_ids"].shape),
                            "inputs": {key: value.tolist() for key, value in feed.items()}, "rows": rows})
        assert flat == len(result.windows)
        reference_scores = {chunk.chunk_id: max(score for chunk_id, score in reference_windows
                                               if chunk_id == chunk.chunk_id) for chunk in chunks}
        ranks = tuple(chunk.chunk_id for _, chunk in sorted(
            enumerate(chunks, 1), key=lambda row: (-reference_scores[row[1].chunk_id], row[0], row[1].chunk_id)))
        ranks_match = ranks == result.ranked_chunk_ids
        all_rank_matches.append(ranks_match)
        coverage = []
        for name, chunk in zip((row[0] for row in fixture["passages"]), chunks, strict=True):
            windows = [window for window in result.windows if window.chunk_id == chunk.chunk_id]
            contiguous = (windows[0].start_char == 0 and windows[-1].end_char == len(chunk.text)
                          and all(left.end_char == right.start_char for left, right in zip(windows, windows[1:])))
            reconstructs = "".join(chunk.text[window.start_char:window.end_char] for window in windows) == chunk.text
            whole_tokens = len(encoder.tokenizer.encode(fixture["query"], chunk.text, add_special_tokens=True).ids)
            assert contiguous and reconstructs
            if fixture.get("must_window"):
                assert whole_tokens > 512 and len(windows) >= 2
            coverage.append({"fixture_name": name, "chunk_id": chunk.chunk_id,
                             "text_characters_unicode": len(chunk.text), "whole_untruncated_pair_tokens": whole_tokens,
                             "window_count": len(windows), "contiguous_coverage": contiguous,
                             "exact_text_reconstruction": reconstructs})
        report_groups.append({"id": fixture["id"], "source": fixture["source"], "query": fixture["query"],
                              "coverage": coverage, "batches": batches,
                              "onnx_chunk_scores": [dataclasses.asdict(score) for score in result.scores],
                              "reference_max_window_scores": reference_scores,
                              "onnx_ranking": list(result.ranked_chunk_ids), "reference_ranking": list(ranks),
                              "ranks_match": ranks_match})
    maximum = max(all_differences)
    assert digest(Path("app/reranking.py")) == source_sha256, "Reranker source changed during validation"
    passed = maximum <= tolerance and all(all_tokenizer_matches) and all(all_rank_matches)
    dependencies = {name: importlib.metadata.version(name) for name in (
        "torch", "transformers", "onnxruntime", "tokenizers", "safetensors", "numpy", "pydantic")}
    artifact_files = ["onnx/model.onnx", "model.safetensors", "config.json", "tokenizer.json",
                      "tokenizer_config.json", "special_tokens_map.json", "vocab.txt", "README.md"]
    report = {
        "schema_version": 1, "status": "NUMERICAL_PARITY_PASSED" if passed else "NUMERICAL_PARITY_FAILED",
        "recorded_at": datetime.now(UTC).isoformat(), "model_id": MODEL_ID, "model_revision": MODEL_REVISION,
        "validation_scope": "Independent CPU PyTorch reference on exact real application ONNX input tensors",
        "not_evaluated": ["retrieval quality", "answer factuality", "citation semantic support", "production latency",
                          "multilingual task quality", "control or holdout outcomes"],
        "quality_metrics": None, "paid_model_calls": 0, "aws_calls": 0,
        "model_loading_network_calls": 0, "blocked_network_connection_attempts": len(network_attempts),
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "dependencies": dependencies, "isolated_venv": sys.prefix,
                        "cpu_only_torch_build": torch.version.cuda is None, "device": "cpu", "dtype": "float32",
                        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
                        "torch_attention_implementation": "eager", "onnx_provider": "CPUExecutionProvider",
                        "torch_install_index": "https://download.pytorch.org/whl/cpu",
                        "other_dependencies_index": "https://pypi.org/simple"},
        "safety": {"local_files_only": True, "trust_remote_code": False, "use_safetensors": True,
                   "eval_mode": True, "no_grad": True, "implicit_truncation": False,
                   "socket_connections_blocked": True,
                   "source": "Synthetic fixtures and pinned model-card example only; no experiment target labels read"},
        "model_artifacts": [{"path": name, "bytes": (directory / name).stat().st_size,
                             "sha256": digest(directory / name)} for name in artifact_files],
        "runtime_manifest_sha256": digest(args.manifest),
        "runtime_recipe": encoder.recipe.model_dump(mode="json"), "runtime_recipe_sha256": encoder.recipe.fingerprint,
        "script_sha256": digest(Path(__file__)), "reranking_module_sha256": source_sha256,
        "comparison": {"activation": "none; raw classifier logits", "window_aggregation": "max raw logit",
                       "absolute_tolerance": tolerance, "relative_tolerance": 0.0,
                       "maximum_absolute_difference": maximum, "compared_pairs_or_windows": len(all_differences),
                       "fixture_chunks": sum(len(group["passages"]) for group in fixtures()),
                       "long_chunk_fixtures": 2, "all_tokenizer_inputs_match": all(all_tokenizer_matches),
                       "all_rankings_match": all(all_rank_matches), "application_onnx_batches": encoder.inference_calls},
        "fixtures": report_groups,
        "limitations": ["Numerical parity is limited to these recorded fixtures, CPU dependency versions and artifact hashes.",
                        "Matching ranks and logits does not establish relevance, factual confidence or general AWS quality.",
                        "The MS MARCO model is English-focused; Unicode input acceptance is not multilingual quality validation."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
                           encoding="utf-8")
    print(json.dumps({"status": report["status"], **report["comparison"], "output": str(args.output)}, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
