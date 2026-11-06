"""Audit local archive coverage and emit candidates; no network, models, or AWS calls."""

import argparse
import json
from pathlib import Path

from app.archive import ArchivePolicy
from app.corpus_inventory import audit_archive


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/aws-official-archive"))
    parser.add_argument("--policy", type=Path, default=Path("configs/corpus/aws-official-archive-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/corpus-coverage/report.json"))
    parser.add_argument("--candidates", type=Path, default=Path("results/corpus-coverage/candidates.jsonl"))
    parser.add_argument("--selection", type=Path,
                        help="Optional bounded JSON object with sources[].url; audit only these URLs.")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.output.resolve().is_relative_to(root) or args.candidates.resolve().is_relative_to(root):
        parser.error("outputs must be outside the read-only archive")
    if args.output.resolve() == args.candidates.resolve():
        parser.error("report and candidate manifest must be different files")
    policy = ArchivePolicy.model_validate_json(args.policy.read_bytes())
    selected_urls = None
    if args.selection is not None:
        if not args.selection.is_file() or args.selection.stat().st_size > 16 * 1024 * 1024:
            parser.error("selection file missing or larger than 16 MiB")
        selection = json.loads(args.selection.read_bytes())
        if (not isinstance(selection, dict) or not isinstance(selection.get("sources"), list)
                or len(selection["sources"]) > 10_000
                or any(not isinstance(row, dict) or not isinstance(row.get("url"), str)
                       for row in selection["sources"])):
            parser.error("selection must contain at most 10,000 sources with URL strings")
        selected_urls = tuple(row["url"] for row in selection["sources"])
    result = audit_archive(root, policy, candidate_output=args.candidates, selected_urls=selected_urls)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "audit_scope": result["audit_scope"],
                      "selected_url_count": result["selected_url_count"], "counts": result["counts"],
                      "output": str(args.output), "candidates": str(args.candidates),
                      "source_http_requests": 0, "inference_calls": 0}, indent=2))
    return 0 if result["counts"]["invalid_done_records"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
