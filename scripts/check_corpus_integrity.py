"""Re-read real ingestion artifacts; verify hashes, spans, coverage and sample layout.

This is an offline audit of stored artifacts, not a semantic gold-label review.
"""

import argparse
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from bs4 import BeautifulSoup

from app.contracts import Chunk, DocumentVersion, sha256_bytes
from app.ingestion.pipeline import _inside, _read
from app.ingestion.text import ProxyTokenizer


def check(report_path: Path, root: Path) -> dict:
    root = root.resolve()
    report = json.loads(report_path.read_bytes())
    corpus = Path(report["corpus_directory"]).resolve()
    if not corpus.is_relative_to(root):
        raise ValueError("CORPUS_OUTSIDE_ROOT")
    for name, digest in report["artifact_sha256"].items():
        raw = _read(root, (corpus / name).relative_to(root).as_posix(), 100 * 1024 * 1024)
        if sha256_bytes(raw) != digest:
            raise ValueError("CORPUS_ARTIFACT_CHECKSUM_FAILED")
    docs = [DocumentVersion.model_validate_json(line) for line in
            (corpus / "documents.jsonl").read_bytes().split(b"\n") if line]
    chunks = [Chunk.model_validate_json(line) for line in
              (corpus / "chunks.jsonl").read_bytes().split(b"\n") if line]
    by_id = {d.document_version_id: d for d in docs}
    if len(by_id) != len(docs) or len({c.chunk_id for c in chunks}) != len(chunks):
        raise ValueError("DUPLICATE_CORPUS_IDENTITIES")
    if len(docs) != report["unique_documents"] or len(chunks) != report["chunks"]:
        raise ValueError("CORPUS_COUNT_MISMATCH")
    tokenizer = ProxyTokenizer()
    for doc in docs:
        cursor = 0
        own = sorted((c for c in chunks if c.document_version_id == doc.document_version_id),
                     key=lambda c: c.position)
        if not own:
            raise ValueError("DOCUMENT_WITHOUT_CHUNKS")
        for position, chunk in enumerate(own):
            if chunk.position != position or chunk.tenant_id != doc.tenant_id:
                raise ValueError("CORPUS_CHUNK_OWNERSHIP_FAILED")
            if chunk.text != doc.normalized_text[chunk.start_char:chunk.end_char]:
                raise ValueError("CORPUS_SOURCE_SLICE_FAILED")
            if tokenizer.count(chunk.text) != chunk.token_count or chunk.token_count > 768:
                raise ValueError("CORPUS_TOKEN_BOUND_FAILED")
            if doc.normalized_text[cursor:chunk.start_char].strip():
                raise ValueError("CORPUS_NONWHITESPACE_GAP")
            cursor = max(cursor, chunk.end_char)
        if doc.normalized_text[cursor:].strip():
            raise ValueError("CORPUS_TAIL_MISSING")
    if any(c.document_version_id not in by_id for c in chunks):
        raise ValueError("DANGLING_CORPUS_CHUNK")
    selected = {"lambda-s3-events", "s3-destination-permissions", "kms-cross-account",
                "lambda-resource-policy", "s3-object-ownership"}
    samples = []
    for source in report["sources"]:
        if source["source_id"] not in selected:
            continue
        doc = by_id[source["document_version_id"]]
        raw = _read(root, source["raw_path"])
        if sha256_bytes(raw) != doc.raw_sha256:
            raise ValueError("AUDIT_RAW_HASH_FAILED")
        soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
        article = soup.find(id="main-col-body")
        codes = [p.get_text("", strip=False).replace("\r\n", "\n").replace("\r", "\n").strip("\n")
                 for p in article.find_all("pre")]
        cells = [re.sub(r"\s+", " ", c.get_text("", strip=False)).strip()
                 for c in article.select("table th, table td")]
        if any(code not in doc.normalized_text for code in codes):
            raise ValueError("AUDIT_CODE_WHITESPACE_LOST")
        if any(cell and cell not in doc.normalized_text for cell in cells):
            raise ValueError("AUDIT_TABLE_CELL_LOST")
        samples.append({"source_id": source["source_id"], "code_blocks_checked": len(codes),
                        "table_cells_checked": len(cells), "status": "PASS"})
        _inside(root, source["document_path"])  # Check provenance path containment too.
    return {"status": "PASS", "checked_at": datetime.now(UTC).isoformat(),
            "corpus_version": report["corpus_version"], "documents_checked": len(docs),
            "chunks_checked": len(chunks), "sample_layout_checks": samples,
            "inference_calls": 0, "source_network_calls": 0,
            "semantic_source_review": "NOT_REVIEWED",
            "notice": "Automated integrity/layout audit; not a labeled evidence benchmark."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/ingestion_pilot20_v2.json"))
    parser.add_argument("--root", type=Path, default=Path("data/ingestion"))
    parser.add_argument("--output", type=Path, default=Path("results/corpus_integrity.json"))
    args = parser.parse_args()
    result = check(args.report, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
