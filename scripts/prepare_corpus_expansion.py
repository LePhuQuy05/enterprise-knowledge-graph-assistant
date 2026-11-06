"""Prepare balanced official guide corpus shards using archived bytes and local tokenizer."""

import argparse
import json
from pathlib import Path

from app.archive import ArchivePolicy
from app.corpus_expansion import prepare_expansion


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/aws-official-archive"))
    parser.add_argument("--policy", type=Path, default=Path("configs/corpus/aws-official-archive-v1.json"))
    parser.add_argument("--candidates", type=Path, default=Path("results/corpus-coverage/candidates.jsonl"))
    parser.add_argument("--guides", type=Path, default=Path("results/corpus-expansion/targeted-guide-sources.json"))
    parser.add_argument("--output-root", type=Path, default=Path("data/corpus-expansion"))
    parser.add_argument("--tenant-id", default="demo")
    parser.add_argument("--per-service", type=int, default=20)
    parser.add_argument("--total-limit", type=int, default=500)
    parser.add_argument("--shard-documents", type=int, default=100)
    args = parser.parse_args(argv)
    result = prepare_expansion(args.root, ArchivePolicy.model_validate_json(args.policy.read_bytes()),
        args.candidates, args.guides, args.output_root, tenant_id=args.tenant_id,
        per_service=args.per_service, total_limit=args.total_limit, shard_documents=args.shard_documents)
    print(json.dumps({key: result[key] for key in ("status", "selected_documents", "selected_chunks",
        "total_shortage", "service_counts", "corpus_directories", "max_embedding_input_tokens",
        "source_http_requests", "inference_calls")}, ensure_ascii=False, indent=2))
    return 0 if result["selected_documents"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
