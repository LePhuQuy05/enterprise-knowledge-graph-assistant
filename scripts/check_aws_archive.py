"""Independent offline archive hash/text/offset audit; never claims semantic coverage."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.archive import XML_LIMIT, Archive, ArchivePolicy, archive_url
from app.contracts import canonical_json, text_hash
from app.ingestion.models import Section


def check(root: Path, policy: ArchivePolicy) -> dict:
    checked_raw, documents, warnings = set(), 0, 0
    with Archive(root, policy, read_only=True) as archive:
        for row in archive.db.execute("SELECT * FROM resources WHERE raw_sha256 IS NOT NULL ORDER BY url"):
            if row["raw_sha256"] not in checked_raw:
                archive.read_blob(row["raw_sha256"], max(XML_LIMIT, policy.max_page_bytes, policy.max_pdf_bytes))
                checked_raw.add(row["raw_sha256"])
            if row["state"] == "DONE" and row["kind"] == "PAGE":
                doc = json.loads(archive.read_blob(row["document_sha256"]))
                if doc["source_uri"] != archive_url(row["url"]) or doc["raw_sha256"] != row["raw_sha256"]:
                    raise ValueError("ARCHIVE_AUDIT_SOURCE_MISMATCH")
                if text_hash(doc["text"]) != doc["normalized_sha256"]:
                    raise ValueError("ARCHIVE_AUDIT_TEXT_HASH_MISMATCH")
                if datetime.fromisoformat(doc["fetched_at"]).utcoffset().total_seconds() != 0:
                    raise ValueError("ARCHIVE_AUDIT_TIMEZONE")
                for value in doc["sections"]:
                    section = Section.model_validate_json(canonical_json(value))
                    if section.end_char > len(doc["text"]):
                        raise ValueError("ARCHIVE_AUDIT_SECTION_RANGE")
                documents += 1
                warnings += bool(doc["warnings"])
        status = archive.status()
    return {"status": "PASS_STRUCTURAL_AUDIT", "checked_at": datetime.now(UTC).isoformat(),
            "raw_payloads_checked": len(checked_raw), "normalized_documents_checked": documents,
            "documents_with_warnings": warnings, "archive_counts": status["counts"],
            "inference_calls": 0, "source_http_requests": 0,
            "semantic_review": "NOT_REVIEWED", "absolute_aws_completeness_claimed": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/aws-official-archive"))
    parser.add_argument("--policy", type=Path, default=Path("configs/corpus/aws-official-archive-v1.json"))
    parser.add_argument("--output", type=Path, default=Path("results/aws_archive_integrity.json"))
    args = parser.parse_args()
    result = check(args.root, ArchivePolicy.model_validate_json(args.policy.read_bytes()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
