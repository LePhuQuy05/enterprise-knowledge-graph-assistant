"""Nonblind, author-selected literal-probe diagnosis; no native-plan claims."""

import json
from pathlib import Path
from unittest.mock import patch

from app.collection_lexical import load_global_lexical_index
from app.contracts import sha256_bytes
from app.graph_collection_migration import collection_records
from app.scoped_method_flow import _ranking
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.run_collection_benchmark import OfflineRuntime

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/atomic-recall-diagnosis-2026-11-05-v1"
PROBES = ("S3 invocation permissions", "Lambda execution role", "s3:GetObject", "kms:Decrypt", "across accounts")


def main():
    c, paths, _ = load_migrated()
    immutable(OUT / "scope.json", {"kind": "AUTHOR_SELECTED_LITERAL_QUERY_DIAGNOSIS_NOT_NATIVE_NOT_BENCHMARK",
        "probes": PROBES, "methods": ["bm25", "vector", "hybrid"], "collection": c.manifest_sha256,
        "code_sha256": sha256_bytes(Path(__file__).read_bytes()), "new_api_calls": 0})
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), c) as lexical:
        runtime = OfflineRuntime(c, lexical, ())
        rows = [{"query": q, "method": m, "pool": _ranking(c, lexical, runtime, q, q, m)[0]}
            for q in PROBES for m in ("bm25", "vector", "hybrid")]
    immutable(OUT / "pools.json", {"rows": rows})
    from scripts.audit_scoped_flow_canary import pool_missing
    from scripts.run_context_profiles import load_review

    rubric, witnesses, _ = load_review()
    docs, chunks, _ = collection_records(c)
    report = []
    for row in rows:
        missing = pool_missing(rubric, witnesses, chunks, row["pool"], "s3-permission-layers")
        r = {"query": row["query"], "method": row["method"], "missing_reviewed_witnesses": missing,
            "top5_source_uris": [docs[chunks[k].document_version_id].source_uri for k in row["pool"][:5]]}
        report.append(r)
        print(json.dumps(r), flush=True)
    immutable(OUT / "diagnosis.json", {"rows": report, "new_model_calls": 0, "human_reviewers": 0})


if __name__ == "__main__":
    with patch("socket.socket.connect", side_effect=AssertionError("DIAGNOSIS_NETWORK_FORBIDDEN")):
        main()
