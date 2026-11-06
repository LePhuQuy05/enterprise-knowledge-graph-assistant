"""Create/replay a source-proven local graph inheritance release; never publish."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

from app.authorization_evidence import load_authorization_facts
from app.authorization_retrieval import combine_evidence_graph
from app.collections import load_collection
from app.contracts import sha256_bytes
from app.graph_collection_migration import collection_records, migrate_graph_payloads
from app.reviewed_source_graph import RELEASE_VERSION, load_reviewed_source_graph
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT, load_sources

OUTPUT = ROOT / "results/collection319-graph-migration-v1"
CODE = ("app/graph_collection_migration.py", "scripts/migrate_collection319_graph.py",
    "app/authorization_evidence.py", "app/authorization_retrieval.py", "app/reviewed_source_graph.py",
    "app/collection_graph_v2.py", "app/collections.py")


def load_target():
    runtime = json.loads((ROOT / "results/collection319-local/runtime.json").read_bytes())
    target = load_collection(Path(runtime["collection_path"]), tenant_id="demo", embedding_model_id="BAAI/bge-small-en-v1.5")
    if target.manifest_sha256 != runtime["collection_manifest_sha256"]:
        raise ValueError("MIGRATION_TARGET_RUNTIME_PIN")
    return target, runtime


def load_migrated():
    target, runtime = load_target()
    selection = json.loads((OUTPUT / "selection.json").read_bytes())
    docs, chunks, _ = collection_records(target)
    reviewed = load_reviewed_source_graph(OUTPUT / "reviewed/release.json", target,
        trusted_sha256=selection["reviewed_release_sha256"], allow_experimental=True)
    auth = load_authorization_facts(OUTPUT / "authorization/release.json", target, chunks, docs,
        trusted_sha256=selection["authorization_release_sha256"], allow_experimental=True)
    graph = combine_evidence_graph(target, reviewed, auth)
    if graph.release_sha256 != selection["combined_release_sha256"]:
        raise ValueError("MIGRATION_COMBINED_RELEASE_PIN")
    return target, runtime, graph


def migrate(*, replay=False):
    parent, chunks, docs = load_sources()
    target, _ = load_target()
    config = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    old_path = ROOT / config["release_path"]
    reviewed = load_reviewed_source_graph(old_path, parent, trusted_sha256=config["release_sha256"], allow_experimental=True)
    auth_config = json.loads((ROOT / "configs/graph/authorization-evidence-local-v1.json").read_bytes())
    auth_path = ROOT / auth_config["release_path"]
    auth = load_authorization_facts(auth_path, parent, chunks, docs,
        trusted_sha256=auth_config["release_sha256"], allow_experimental=True)
    pins = {str(p.relative_to(ROOT)): sha256_bytes(p.read_bytes()) for p in
        (old_path, old_path.parent / "relations-v2.json", old_path.parent / "actions.json",
         old_path.parent / "sidecar.json", auth_path, auth_path.parent / "facts.json")}
    plan = {"version": "collection319-local-graph-migration-v1", "source_artifact_sha256": pins,
        "parent_manifest_sha256": parent.manifest_sha256, "target_manifest_sha256": target.manifest_sha256,
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "new_model_calls": 0, "aws_calls": 0, "serving_published": False}
    immutable(OUTPUT / "plan.json", plan)
    migrated = migrate_graph_payloads(parent, target, reviewed, auth,
        action_envelope=json.loads((old_path.parent / "actions.json").read_bytes()),
        sidecar=json.loads((old_path.parent / "sidecar.json").read_bytes()))
    immutable(OUTPUT / "source-equivalence.json", migrated["proof"])
    immutable(OUTPUT / "lineage.json", migrated["lineage"])
    for name, payload in migrated["reviewed_payloads"].items():
        immutable(OUTPUT / "reviewed" / name, payload)
    release = {"version": RELEASE_VERSION, "source_manifest_sha256": target.manifest_sha256,
        "collection_version": target.manifest.collection_version,
        "artifacts": {n: sha256_bytes((OUTPUT / "reviewed" / n).read_bytes()) for n in migrated["reviewed_payloads"]},
        "human_reviewed": 0, "serving_published": False}
    immutable(OUTPUT / "reviewed/release.json", release)
    immutable(OUTPUT / "authorization/facts.json", migrated["authorization_payload"])
    immutable(OUTPUT / "authorization/release.json", migrated["authorization_release_base"] |
        {"records_sha256": sha256_bytes((OUTPUT / "authorization/facts.json").read_bytes())})
    new_docs, new_chunks, _ = collection_records(target)
    new_reviewed = load_reviewed_source_graph(OUTPUT / "reviewed/release.json", target,
        trusted_sha256=sha256_bytes((OUTPUT / "reviewed/release.json").read_bytes()), allow_experimental=True)
    new_auth = load_authorization_facts(OUTPUT / "authorization/release.json", target, new_chunks, new_docs,
        trusted_sha256=sha256_bytes((OUTPUT / "authorization/release.json").read_bytes()), allow_experimental=True)
    graph = combine_evidence_graph(target, new_reviewed, new_auth)
    immutable(OUTPUT / "selection.json", {"reviewed_release_sha256": new_reviewed.release_sha256,
        "authorization_release_sha256": new_auth.release_sha256, "combined_release_sha256": graph.release_sha256,
        "collection_manifest_sha256": target.manifest_sha256, "serving_published": False})
    if any(sha256_bytes((ROOT / p).read_bytes()) != digest for p, digest in pins.items()):
        raise ValueError("MIGRATION_PARENT_MUTATED")
    stored_docs = {new_chunks[e.chunk_id].document_version_id for e in new_reviewed.edges}
    stored_docs.update(row["candidate"]["document_version_id"] for row in new_auth.records())
    report = {"status": "LOCAL_SOURCE_PROVEN_GRAPH_MIGRATION_RELOADED", "documents": len(new_docs),
        "chunks": len(new_chunks), "stored_records": len(new_reviewed.edges) + len(new_auth.records_json),
        "traversable_records": len(graph.facts), "source_documents_with_stored_records": len(stored_docs),
        "new_document_graph_records": 0, "inherited_reviews_not_new_semantic_reviews": True,
        "source_equivalence_sha256": sha256_bytes((OUTPUT / "source-equivalence.json").read_bytes()),
        "lineage_sha256": sha256_bytes((OUTPUT / "lineage.json").read_bytes()),
        "combined_release_sha256": graph.release_sha256,
        "new_model_calls": 0, "aws_calls": 0, "human_reviewed": 0, "serving_published": False}
    immutable(OUTPUT / ("replay.json" if replay else "report.json"), report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    with ExitStack() as stack:
        for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(name, side_effect=AssertionError("MIGRATION_NETWORK_BLOCKED")))
        migrate(replay=parser.parse_args().replay)
