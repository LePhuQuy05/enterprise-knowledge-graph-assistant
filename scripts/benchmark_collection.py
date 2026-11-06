"""Measure local BGE collection loading, retrieval scans and process memory.

No downloads, provider APIs, answer generation or quality labels are involved.
The benchmark opens fresh application caches, not fresh OS filesystem caches.
Repeated scans are not called fully warm when the collection exceeds its LRU.
"""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.collection_lexical import load_global_lexical_index, retrieve_collection_hybrid
from app.collections import CollectionError, load_collection
from app.contracts import text_hash
from app.embeddings import query_embedding_input
from app.local_bge import BgeEncoder

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = (
    "Why can S3 invoke Lambda but the function cannot read an SSE-KMS object?",
    "Which policies must permit cross-account use of an AWS KMS key?",
    "Can a standby in an RDS Multi-AZ DB instance deployment serve read traffic?",
)


def process_memory() -> dict[str, Any]:
    """Process-lifetime peak, including ONNX/native arrays, unlike tracemalloc."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        current_process = ctypes.windll.kernel32.GetCurrentProcess
        current_process.restype = wintypes.HANDLE
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessMemoryCounters), wintypes.DWORD]
        get_memory.restype = wintypes.BOOL
        if not get_memory(current_process(), ctypes.byref(counters), counters.cb):
            raise OSError("Windows process memory measurement failed")
        return {
            "current_rss_bytes": counters.WorkingSetSize,
            "peak_rss_bytes": counters.PeakWorkingSetSize,
            "method": "Windows GetProcessMemoryInfo working set / lifetime peak",
        }
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    current = None
    if sys.platform.startswith("linux"):
        resident_pages = int(Path("/proc/self/statm").read_text().split()[1])
        current = resident_pages * os.sysconf("SC_PAGE_SIZE")
    return {
        "current_rss_bytes": current,
        "peak_rss_bytes": int(peak if sys.platform == "darwin" else peak * 1024),
        "method": "resource ru_maxrss lifetime peak; /proc current RSS when available",
    }


def _latency_summary(values: list[float]) -> dict[str, int | float]:
    return {
        "observations": len(values), "min_ms": round(min(values), 3),
        "median_ms": round(float(np.median(values)), 3),
        "p95_ms": round(float(np.percentile(values, 95)), 3),
        "max_ms": round(max(values), 3),
    }


def benchmark_search(
    path: Path, *, tenant_id: str, embedding_model_id: str,
    query_vectors: list[np.ndarray], repeats: int = 3, cache_shards: int = 2,
    cache_bytes: int = 128 * 1024 * 1024, selected_shards: tuple[str, ...] | None = None,
    allow_fixture: bool = False,
) -> list[dict[str, Any]]:
    if type(repeats) is not int or not 3 <= repeats <= 5:
        raise CollectionError("Benchmark repetitions must be 3..5")
    if not 1 <= len(query_vectors) <= 5:
        raise CollectionError("Benchmark requires 1..5 query vectors")
    profiles = []
    for name in ("all_shards", "selected_shards"):
        started = time.perf_counter()
        collection = load_collection(
            path, tenant_id=tenant_id, embedding_model_id=embedding_model_id,
            cache_shards=cache_shards, cache_bytes=cache_bytes,
            allow_fixture=allow_fixture,
        )
        open_ms = (time.perf_counter() - started) * 1000
        selected = None if name == "all_shards" else (
            selected_shards or (collection.manifest.shards[0].shard_id,)
        )
        chosen = tuple(ref.shard_id for ref in collection.manifest.shards) if selected is None else selected
        # Search validates unknown/duplicate shard IDs before loading any data.
        rows, initial_results, repeated_latencies = [], {}, []
        consistent = True
        for repetition in range(repeats):
            for index, vector in enumerate(query_vectors):
                before = collection.cache_info
                started = time.perf_counter()
                hits = collection.search_vector(vector, k=5, shard_ids=selected)
                elapsed = (time.perf_counter() - started) * 1000
                after = collection.cache_info
                identities = [(hit.shard_id, hit.chunk_id) for hit in hits]
                if repetition == 0:
                    initial_results[index] = identities
                else:
                    consistent = consistent and identities == initial_results[index]
                    repeated_latencies.append(elapsed)
                rows.append({
                    "repetition": repetition + 1, "question_index": index,
                    "application_cache_empty_at_start": before.resident_shards == 0,
                    "elapsed_ms": round(elapsed, 3),
                    "loads": after.loads - before.loads, "cache_hits": after.hits - before.hits,
                    "evictions": after.evictions - before.evictions,
                    "cache_after": asdict(after),
                    "top_5": [asdict(hit) for hit in hits],
                })
        profiles.append({
            "scope": name, "selected_shard_ids": list(chosen),
            "selected_documents": sum(ref.counts.documents for ref in collection.manifest.shards
                                      if ref.shard_id in chosen),
            "selected_chunks": sum(ref.counts.chunks for ref in collection.manifest.shards
                                   if ref.shard_id in chosen),
            "collection_open_ms": round(open_ms, 3),
            "first_search_empty_application_cache_ms": rows[0]["elapsed_ms"],
            "repeated_search_latency": _latency_summary(repeated_latencies),
            "hit_ids_consistent_across_repetitions": consistent,
            "cache_final": asdict(collection.cache_info),
            "process_memory_after_profile": process_memory(), "observations": rows,
        })
        del collection
    return profiles


def _questions(path: Path | None) -> tuple[str, ...]:
    if path is None:
        return DEFAULT_QUESTIONS
    if not path.is_file() or path.stat().st_size > 64 * 1024:
        raise ValueError("Questions JSON missing or larger than 64 KiB")
    questions = json.loads(path.read_bytes())
    if (not isinstance(questions, list) or not 1 <= len(questions) <= 5
            or any(not isinstance(question, str) or not question.strip() or len(question) > 2000
                   for question in questions)):
        raise ValueError("Questions JSON must contain 1..5 nonempty strings, each <=2000 chars")
    return tuple(query_embedding_input(question) for question in questions)


def benchmark_hybrid(
    path: Path, lexical_manifest: Path, *, tenant_id: str, embedding_model_id: str,
    questions: tuple[str, ...], query_vectors: list[np.ndarray], repeats: int = 3,
    cache_shards: int = 2, cache_bytes: int = 128 * 1024 * 1024, allow_fixture: bool = False,
) -> dict[str, Any]:
    """Measure actual global lexical and hybrid executions over all shards."""
    if (type(repeats) is not int or not 3 <= repeats <= 5 or not 1 <= len(questions) <= 5
            or len(questions) != len(query_vectors)):
        raise CollectionError("Hybrid benchmark requires 1..5 paired queries and 3..5 repetitions")
    collection = load_collection(
        path, tenant_id=tenant_id, embedding_model_id=embedding_model_id,
        cache_shards=cache_shards, cache_bytes=cache_bytes, allow_fixture=allow_fixture,
    )
    started = time.perf_counter()
    with load_global_lexical_index(lexical_manifest, collection) as lexical:
        open_ms = (time.perf_counter() - started) * 1000
        rows, lexical_times, hybrid_times, initial = [], [], [], {}
        consistent = True
        for repetition in range(repeats):
            for index, (question, vector) in enumerate(zip(questions, query_vectors, strict=True)):
                started = time.perf_counter()
                lexical_hits = lexical.search(question, k=20)
                lexical_ms = (time.perf_counter() - started) * 1000
                before = collection.cache_info
                started = time.perf_counter()
                hybrid_hits = retrieve_collection_hybrid(collection, lexical, question, vector)
                hybrid_ms = (time.perf_counter() - started) * 1000
                after = collection.cache_info
                if repetition == 0:
                    initial[index] = hybrid_hits
                else:
                    consistent = consistent and hybrid_hits == initial[index]
                    lexical_times.append(lexical_ms)
                    hybrid_times.append(hybrid_ms)
                rows.append({
                    "repetition": repetition + 1, "question_index": index,
                    "lexical_ms": round(lexical_ms, 3), "hybrid_ms": round(hybrid_ms, 3),
                    "loads": after.loads - before.loads,
                    "cache_hits": after.hits - before.hits,
                    "application_cache_empty_at_start": before.resident_shards == 0,
                    "lexical_top_5": [{"chunk_id": key, "score": score}
                                      for key, score in lexical_hits[:5]],
                    "hybrid_top_5": hybrid_hits[:5], "cache_after": asdict(after),
                })
        return {
            "scope": "all_collection_shards", "method": "global_BM25_plus_vector_RRF",
            "lexical_index_version": lexical.manifest.index_version,
            "lexical_index_sqlite_bytes": lexical.manifest.sqlite_bytes,
            "global_chunks": lexical.manifest.chunks,
            "global_terms": lexical.manifest.terms,
            "global_average_document_length": lexical.manifest.average_document_length,
            "global_average_idf": lexical.manifest.average_idf,
            "index_open_with_checksum_validation_ms": round(open_ms, 3),
            "lexical_repeated_latency": _latency_summary(lexical_times),
            "hybrid_repeated_latency": _latency_summary(hybrid_times),
            "first_hybrid_empty_application_cache_ms": rows[0]["hybrid_ms"],
            "hit_ids_consistent_across_repetitions": consistent,
            "cache_final": asdict(collection.cache_info), "observations": rows,
            "process_memory_after_profile": process_memory(),
            "timing_note": "Hybrid executes vector search, global BM25 and RRF; lexical-only is timed separately.",
        }


def _cache_mib(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Cache MiB must be an integer in 1..512") from exc
    if not 1 <= parsed <= 512:
        raise argparse.ArgumentTypeError("Cache MiB must be an integer in 1..512")
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("collection", type=Path)
    parser.add_argument("--tenant-id", default="demo")
    parser.add_argument("--model-directory", type=Path, default=ROOT / "data/models/bge-small-en-v1.5")
    parser.add_argument("--model-manifest", type=Path,
                        default=ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    parser.add_argument("--questions", type=Path, help="Optional JSON array, at most five questions")
    parser.add_argument("--lexical-manifest", "--lexical", dest="lexical", type=Path,
                        help="Optional prebuilt global BM25 manifest for hybrid benchmark")
    parser.add_argument("--repeats", type=int, choices=range(3, 6), default=3)
    parser.add_argument("--threads", type=int, choices=range(1, 9), default=2)
    parser.add_argument("--cache-shards", type=int, default=2)
    parser.add_argument("--cache-mib", type=_cache_mib, metavar="1..512", default=128)
    parser.add_argument("--selected-shards", nargs="+", help="Default: first shard in sorted catalog")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    questions = _questions(args.questions)
    memory_before = process_memory()
    started = time.perf_counter()
    encoder = BgeEncoder(args.model_directory, args.model_manifest, threads=args.threads)
    model_load_ms = (time.perf_counter() - started) * 1000
    memory_after_model = process_memory()
    collection = load_collection(
        args.collection, tenant_id=args.tenant_id,
        embedding_model_id=encoder.recipe("document").model_id,
    )
    if collection.manifest.embedding_recipe != encoder.recipe("document"):
        raise CollectionError("Query BGE assets/runtime do not match the collection document recipe")
    vectors, encodings = [], []
    for question in questions:
        started = time.perf_counter()
        vectors.append(encoder.encode([question], role="query", batch_size=1)[0])
        encodings.append({
            "question_sha256": text_hash(question), "characters": len(question),
            "complete_query_tokens": encoder.count(question, role="query"),
            "encode_ms": round((time.perf_counter() - started) * 1000, 3),
        })
    profiles = benchmark_search(
        args.collection, tenant_id=args.tenant_id,
        embedding_model_id=encoder.recipe("document").model_id, query_vectors=vectors,
        repeats=args.repeats, cache_shards=args.cache_shards,
        cache_bytes=args.cache_mib * 1024 * 1024,
        selected_shards=tuple(args.selected_shards) if args.selected_shards else None,
    )
    hybrid_profile = None
    if args.lexical:
        hybrid_profile = benchmark_hybrid(
            args.collection, args.lexical, tenant_id=args.tenant_id,
            embedding_model_id=encoder.recipe("document").model_id,
            questions=questions, query_vectors=vectors, repeats=args.repeats,
            cache_shards=args.cache_shards, cache_bytes=args.cache_mib * 1024 * 1024,
        )
    report = {
        "schema_version": 1, "kind": "LOCAL_COLLECTION_RETRIEVAL_RUNTIME_SPIKE",
        "created_at": datetime.now(UTC).isoformat(),
        "collection_version": collection.manifest.collection_version,
        "collection_manifest_sha256": collection.manifest_sha256,
        "corpus_version": collection.manifest.corpus_version,
        "counts": collection.manifest.counts.model_dump(),
        "shards": len(collection.manifest.shards),
        "embedding_recipe": encoder.recipe("document").model_dump(mode="json"),
        "platform": platform.platform(), "python_version": platform.python_version(),
        "cpu_threads": args.threads, "logical_cpu_count": os.cpu_count(),
        "model_load_ms": round(model_load_ms, 3), "query_encodings": encodings,
        "query_encoder_inference_calls": encoder.inference_calls,
        "query_vectors_reused_for_all_comparisons": True,
        "cache_limits": {"shards": args.cache_shards, "serialized_bytes": args.cache_mib * 1024 * 1024},
        "memory_before": memory_before, "memory_after_model_load": memory_after_model,
        "memory_after_benchmark": process_memory(), "profiles": profiles,
        "global_lexical_and_hybrid_profile": hybrid_profile,
        "paid_calls": 0, "answer_quality_measured": False,
        "limitations": [
            "Runtime measurements only; there is no gold relevance or answer accuracy measurement.",
            "Application caches start empty; OS filesystem caches are not flushed.",
            "Repeated scans may reload shards when they exceed LRU capacity.",
            "Cache serialized-byte accounting is not process RSS or a memory guarantee.",
            "Process peak RSS is lifetime peak including the pinned BGE model/native arrays.",
            "Small-sample p95 is exploratory for this machine, not a deployment SLA.",
            "Selected-shard results cover only explicitly selected data; routing recall is not evaluated.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(args.output), "collection_version": report["collection_version"],
                      "documents": report["counts"]["documents"], "paid_calls": 0,
                      "answer_quality_measured": False}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
