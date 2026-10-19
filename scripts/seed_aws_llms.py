"""Seed official Markdown catalogs into the existing paced, resumable archive."""

import argparse
import json
from pathlib import Path

from app.archive import ROOT_LLMS, Archive, ArchivePolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps({"status": "PLANNED", "seed": ROOT_LLMS, "new_model_calls": 0}))
        return
    policy = ArchivePolicy.model_validate_json(Path("configs/corpus/aws-official-archive-v1.json").read_bytes())
    with Archive(Path("data/aws-official-archive"), policy) as archive:
        archive._add(ROOT_LLMS, "PAGE", "EXPLICIT_LLMS_DISCOVERY_EXTENSION", None)
        archive.db.commit()
        result = archive.status()
        result["extension"] = "Official llms.txt directories; not article text and not a full dump."
        Path("results/aws-knowledge/llms-seed.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": "SEEDED_RESUME_WORKER", "new_model_calls": 0}))


if __name__ == "__main__":
    main()
