"""Freeze actual catalog receipts before applying the independent source rubric."""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.collection_lexical import load_global_lexical_index
from app.contracts import sha256_bytes
from app.literal_source_catalog import build_source_catalog
from app.scoped_query_v6 import ScopedPlanV6
from scripts import run_scoped_flow_canary as parent
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.run_collection_benchmark import OfflineRuntime

OUT = parent.ROOT / "results/literal-source-catalog-2026-11-05-v1"


def main():
    collection, paths, _ = load_migrated()
    inputs = sorted((parent.OUT / "preparations").glob("*.json"))
    immutable(OUT / "scope.json", {"version": "literal-source-catalog-diagnostic-v1",
        "collection": collection.manifest_sha256,
        "code_sha256": {p: sha256_bytes((parent.ROOT / p).read_bytes()) for p in
            ("app/literal_source_catalog.py", "scripts/probe_literal_catalog.py")},
        "input_sha256": {p.name: sha256_bytes(p.read_bytes()) for p in inputs},
        "exposure": "TWO_PREVIOUSLY_EXPOSED_INTENTS_DIAGNOSTIC_NOT_BENCHMARK",
        "new_api_calls": 0})
    rows = []
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ())
        for p in inputs:
            old = json.loads(p.read_bytes())
            t = old["task"]
            cat = build_source_catalog(collection, lexical, runtime, question=t["question"], search=t["search"],
                language=t["language"], plan=ScopedPlanV6.model_validate_json(json.dumps(old["plan"])))
            rows.append({"task": t, "catalog_fingerprint": cat.fingerprint, "source_ids": cat.source_ids,
                "message": cat.message, "trace": cat.trace})
    immutable(OUT / "catalogs.json", {"rows": rows})
    from app.graph_collection_migration import collection_records
    from scripts.audit_scoped_flow_canary import pool_missing
    from scripts.run_context_profiles import load_review

    rubric, witnesses, _ = load_review()
    _, chunks, _ = collection_records(collection)
    reviews = [{"intent": r["task"]["intent"], "language": r["task"]["language"],
        "chunks": len(r["source_ids"]), "bytes": len(r["message"].encode()),
        "missing_reviewed_witnesses": pool_missing(rubric, witnesses, chunks, r["source_ids"], r["task"]["intent"])}
        for r in rows]
    immutable(OUT / "review.json", {"rows": reviews, "new_api_calls": 0, "human_reviewers": 0})
    for r in reviews:
        print(json.dumps(r), flush=True)


if __name__ == "__main__":
    with ExitStack() as stack:
        for call in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(call, side_effect=AssertionError("CATALOG_NETWORK_FORBIDDEN")))
        main()
