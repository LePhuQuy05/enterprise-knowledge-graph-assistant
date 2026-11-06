"""Post-exposure authored search probes; never native plans or benchmark gold."""

import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.graph_collection_migration import collection_records
from app.retrieval import reciprocal_rank_fusion
from app.source_dependency_audit import SourceDependency
from scripts.audit_collection319_methods import source_reachability
from scripts.check_collection319_methods import QUESTIONS
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT
from scripts.run_collection_benchmark import OfflineRuntime

OUTPUT = ROOT / "results/collection319-recall-probes-v1"
QUERIES = (
    ("broad-user-question", QUESTIONS["s3-permission-layers"]["en"]),
    ("focused-invoke", "S3 permissions to invoke Lambda"),
    ("focused-read-role", "Lambda execution role permissions s3:GetObject reading an S3 object"),
    ("focused-decrypt", "kms:Decrypt SSE-KMS object access across accounts"),
    ("two-groups-diagnostic", "How are permissions for Amazon S3 to invoke Lambda different from execution role permissions to access S3?"),
    ("tutorial-read-diagnostic", "Which permission does the S3 trigger tutorial put in the Lambda execution role to get an object from an S3 bucket?"),
)


def probe():
    collection, paths, _ = load_migrated()
    _, chunks, _ = collection_records(collection)
    controls = json.loads((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())["source_controls"]
    dependencies = tuple(SourceDependency.model_validate_json(canonical_json(row["dependency"])) for row in controls)
    plan = {"version": "collection319-postexposure-source-recall-probes-v1", "queries": QUERIES,
        "purpose": "AI_AUTHORED_POSTHOC_SOURCE_REACHABILITY_NOT_NATIVE_PLANS_OR_QUALITY_EVALUATION",
        "source_manifest_sha256": collection.manifest_sha256, "retrieval_depth_per_channel": 20,
        "union_not_serving_context": True, "new_model_calls": 0, "aws_calls": 0,
        "implementation_sha256": sha256_bytes(Path(__file__).read_bytes()),
        "reachability_auditor_sha256": sha256_bytes((ROOT / "scripts/audit_collection319_methods.py").read_bytes())}
    immutable(OUTPUT / "plan.json", plan)
    pools, rows = {}, []
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ())
        for name, query in QUERIES:
            cached = runtime.cache.resolve_local("demo", text=query_embedding_input(query), recipe=runtime.query_recipe, encoder=runtime.encoder)
            vector = np.asarray(cached.record.vector, dtype=np.float32)
            lexical_ids = tuple(key for key, _ in lexical.search(query, k=20))
            vector_ids = tuple(hit.chunk_id for hit in collection.search_vector(vector, k=20))
            pools[name] = {"bm25": lexical_ids, "vector": vector_ids,
                "hybrid": tuple(reciprocal_rank_fusion([vector_ids, lexical_ids]))}
            for method, pool in pools[name].items():
                rows.append({"query_name": name, "query": query, "method": method, "pool_chunk_ids": pool,
                    "controls": [source_reachability(dep, pool, (), chunks) for dep in dependencies]})
        unions = []
        for method in ("bm25", "vector", "hybrid"):
            combined = tuple(dict.fromkeys(key for name in ("focused-invoke", "focused-read-role", "focused-decrypt") for key in pools[name][method]))
            unions.append({"method": method, "pool_size": len(combined), "serving_context": False,
                "controls": [source_reachability(dep, combined, (), chunks) for dep in dependencies]})
    receipt = {"probes": rows, "three_facet_unions": unions, "human_reviewed": 0, "new_model_calls": 0,
        "new_native_plans": 0, "serving_published": False, "quality_accuracy_measured": False,
        "warning": "Union may exceed 40 candidates/8 context chunks. Authored probes are not deployable Q/QB or evidence gold."}
    immutable(OUTPUT / "receipts.json", receipt)
    print(json.dumps({"status": "POSTHOC_SOURCE_REACHABILITY_PROBES_RECORDED", "retrieval_cells": len(rows),
        "union_controls": [{"method": u["method"], "pool_size": u["pool_size"],
            "stages": [c["stage"] for c in u["controls"]]} for u in unions], "new_model_calls": 0}))


if __name__ == "__main__":
    with ExitStack() as stack:
        for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(name, side_effect=AssertionError("SOURCE_RECALL_PROBES_NETWORK_BLOCKED")))
        probe()
