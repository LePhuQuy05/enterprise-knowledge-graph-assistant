"""Index prepared local corpus shards using pinned BGE CPU; no paid APIs or network."""

import argparse
import json
import time
from pathlib import Path

from app.collection_indexing import index_corpus_directories
from app.contracts import canonical_json
from app.local_bge import BgeEncoder

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preparation", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/corpus-expansion/index")
    parser.add_argument("--output", type=Path, default=ROOT / "results/corpus-expansion/index.json")
    parser.add_argument("--limit-shards", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--reuse-collection", type=Path,
                        help="Optional prior collection for verified exact-input vector reuse")
    args = parser.parse_args()
    if args.preparation.stat().st_size > 10 * 1024 * 1024:
        parser.error("preparation report too large")
    preparation = json.loads(args.preparation.read_bytes())
    directories = preparation["corpus_directories"]
    if args.limit_shards is not None:
        if not 1 <= args.limit_shards <= len(directories):
            parser.error("limit-shards outside available range")
        directories = directories[:args.limit_shards]
    started = time.monotonic()
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5",
                         ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    report = index_corpus_directories(
        directories, args.output_root, encoder, batch_size=args.batch_size,
        reuse_collection_path=args.reuse_collection,
        progress=lambda row: print(json.dumps(row), flush=True),
    )
    report.update(seconds=round(time.monotonic() - started, 3),
                  preparation_report=str(args.preparation),
                  all_prepared_shards_indexed=len(directories) == len(preparation["corpus_directories"]),
                  onnx_forward_calls=encoder.inference_calls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
