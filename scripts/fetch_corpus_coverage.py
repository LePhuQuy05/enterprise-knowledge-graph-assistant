"""Fetch an explicit bounded guide selection through the existing archive safeguards."""

import argparse
import json
from pathlib import Path

from app.archive import Archive, ArchivePolicy, archive_url
from app.contracts import canonical_json, sha256_bytes

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--archive", type=Path, default=ROOT / "data/aws-official-archive")
    parser.add_argument("--policy", type=Path, default=ROOT / "configs/corpus/aws-official-archive-v1.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results/corpus-expansion/fetch.json")
    parser.add_argument("--max-pages", type=int, default=120)
    parser.add_argument("--seconds", type=int, default=1200)
    parser.add_argument("--apply", action="store_true", help="Otherwise report selection only")
    args = parser.parse_args()
    if not 1 <= args.max_pages <= 500 or not 1 <= args.seconds <= 3600:
        parser.error("max-pages must be1..500 and seconds1..3600")
    if args.selection.stat().st_size > 2 * 1024 * 1024:
        parser.error("selection too large")
    data = json.loads(args.selection.read_bytes())
    rows = data["sources"]
    urls = tuple(row["url"] for row in rows)
    if not 1 <= len(urls) <= 1000 or len(set(urls)) != len(urls):
        parser.error("selection must contain1..1000unique sources")
    for url in urls:
        if archive_url(url) != url or not url.endswith(".html"):
            parser.error("selection must contain canonical official HTML pages")
    policy = ArchivePolicy.model_validate_json(args.policy.read_bytes())
    report = {"selection_sha256": sha256_bytes(canonical_json(urls)),
              "selected": len(urls), "paid_model_calls": 0, "cloud_deployments": 0}
    if args.apply:
        with Archive(args.archive, policy) as archive:
            report.update(archive.run(
                "PAGE", selected_urls=urls, max_resources=args.max_pages,
                max_seconds=args.seconds, progress=lambda row: print(json.dumps(row), flush=True),
            ))
    else:
        report["status"] = "PLANNED_ONLY_NO_FETCH"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical_json(report) + b"\n")
    summary_keys = ("status", "selected", "processed_this_run", "http_requests_this_run",
                    "duration_seconds", "stop_reason", "paid_model_calls", "cloud_deployments")
    print(json.dumps({key: report[key] for key in summary_keys if key in report}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
