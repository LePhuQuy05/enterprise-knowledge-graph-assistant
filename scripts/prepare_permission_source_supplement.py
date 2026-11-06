"""Fetch/replay three targeted official sources without changing collection316.

Uses the existing allowlist/robots-aware ingestion path. No embeddings, graph
extraction, model calls, AWS account calls, collection rebuild or deployment.
"""

import argparse
import json
from pathlib import Path

from app.contracts import Chunk, DocumentVersion, sha256_bytes
from app.ingestion.pipeline import ingest
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT, load_sources

CONFIG = ROOT / "configs/corpus/lambda-permission-source-supplement-v1.json"
DATA = ROOT / "data/permission-source-supplement-v1"
OUTPUT = ROOT / "results/permission-source-supplement-v1"
CODE = ("scripts/prepare_permission_source_supplement.py", "app/ingestion/collector.py",
    "app/ingestion/pipeline.py", "app/ingestion/text.py", "app/ingestion/models.py")


def run(*, fetch=False, replay=False):
    collection, _, original_docs = load_sources()
    config = json.loads(CONFIG.read_bytes())
    historical = OUTPUT / "fetch-plan.json"
    if historical.exists():
        original = json.loads(historical.read_bytes())
        if sha256_bytes((OUTPUT / "fetch-implementation.py.txt").read_bytes()) != original["code_sha256"]["scripts/prepare_permission_source_supplement.py"]:
            raise ValueError("PERMISSION_SUPPLEMENT_ORIGINAL_FETCH_IMPLEMENTATION_CHANGED")
    plan = {"version": "permission-source-supplement-replay-plan-v2", "config_sha256": sha256_bytes(CONFIG.read_bytes()),
        "original_fetch_plan_sha256": sha256_bytes(historical.read_bytes()) if historical.exists() else None,
        "entrypoint_amendment": "Remove unused import and preserve original fetch plan/implementation; ingestion recipes and source bytes unchanged",
        "parent_collection_manifest_sha256": collection.manifest_sha256,
        "parent_documents": collection.manifest.counts.documents,
        "source_urls": [s["url"] for s in config["sources"]],
        "source_urls_absent_from_parent": [s["url"] for s in config["sources"] if s["url"] not in {d.source_uri for d in original_docs.values()}],
        "purpose": "EXPOSED_SOURCE_GAP_QUALIFICATION_NOT_RETRIEVAL_OR_ANSWER_BENCHMARK",
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "new_model_calls": 0, "aws_calls": 0, "changes_parent_collection": False, "serving_published": False}
    immutable(OUTPUT / "plan-v2.json", plan)
    if not (fetch or replay):
        print(json.dumps({"status": "THREE_SOURCE_FETCH_PLAN_FROZEN_NO_NETWORK"}))
        return
    if fetch and (OUTPUT / "report.json").exists():
        raise ValueError("PERMISSION_SUPPLEMENT_ALREADY_FETCHED_USE_REPLAY")
    result = ingest(CONFIG, DATA, tenant_id="demo", offline=replay)
    if result["status"] != "INGESTED" or result["sources_validated"] != 3 or result["failures"]:
        immutable(OUTPUT / "incomplete-ingestion.json", result)
        raise ValueError("PERMISSION_SUPPLEMENT_INGESTION_INCOMPLETE")
    corpus = Path(result["corpus_directory"])
    documents = [DocumentVersion.model_validate_json(line) for line in (corpus / "documents.jsonl").read_bytes().splitlines()]
    chunks = [Chunk.model_validate_json(line) for line in (corpus / "chunks.jsonl").read_bytes().splitlines()]
    by_id = {d.document_version_id: d for d in documents}
    if any(by_id[c.document_version_id].normalized_text[c.start_char:c.end_char] != c.text for c in chunks):
        raise ValueError("PERMISSION_SUPPLEMENT_CHUNK_SOURCE_MISMATCH")
    inventory = {"version": "permission-source-supplement-inventory-v1", "parent_collection_unchanged": True,
        "corpus_version": result["corpus_version"], "artifact_sha256": result["artifact_sha256"],
        "documents": [{"document_version_id": d.document_version_id, "source_uri": d.source_uri,
            "title": d.title, "raw_sha256": d.raw_sha256, "normalized_sha256": d.normalized_sha256,
            "fetched_at": d.fetched_at.isoformat(), "chunks": [c.chunk_id for c in chunks if c.document_version_id == d.document_version_id]}
            for d in documents], "chunks": len(chunks), "native_extraction_calls": 0, "human_reviewed": 0,
        "serving_published": False, "embedded": False, "not_in_frozen_collection316": True}
    immutable(OUTPUT / "inventory.json", inventory)
    report = {"status": "OFFLINE_SOURCE_SUPPLEMENT_REPLAY_MATCH" if replay else "THREE_OFFICIAL_SOURCES_INGESTED_FOR_QUALIFICATION",
        "documents": len(documents), "chunks": len(chunks), "source_cache_hits": result["source_cache_hits"],
        "http_requests_this_run": result["http_requests_this_run"], "new_model_calls": 0, "aws_calls": 0,
        "artifact_sha256": result["artifact_sha256"], "corpus_directory": str(corpus),
        "inventory_sha256": sha256_bytes((OUTPUT / "inventory.json").read_bytes()), "changes_parent_collection": False,
        "serving_published": False, "semantic_source_review": "PENDING"}
    if replay and result["http_requests_this_run"]:
        raise ValueError("PERMISSION_SUPPLEMENT_REPLAY_USED_NETWORK")
    immutable(OUTPUT / ("replay.json" if replay else "report.json"), report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--fetch", action="store_true")
    choice.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    run(fetch=args.fetch, replay=args.replay)
