"""Real local Neo4j import/replay/export, corruption checks and snapshot parity."""

import json
import os
import secrets
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from neo4j import GraphDatabase

from app.graph_store import GraphStoreError, Neo4jGraphStore, graph_records
from app.snapshot import build_snapshot, load_snapshot

ROOT = Path(__file__).resolve().parents[1]


def main():
    receipt = json.loads((ROOT / "results/pilot_graph_snapshot.json").read_bytes())
    snapshot = load_snapshot(Path(receipt["snapshot_directory"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=receipt["manifest_sha256"])
    image = json.loads((ROOT / "configs/runtime_images.json").read_bytes())["neo4j"]
    name = "kg-r72-graph-roundtrip-" + uuid.uuid4().hex[:12]
    password = secrets.token_urlsafe(32)
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "FAIL", "image": image,
              "driver_version": __import__("neo4j").__version__, "paid_model_calls": 0, "aws_calls": 0,
              "bind_scope": "127.0.0.1", "semantic_review": "PENDING"}
    driver, started = None, False

    def docker(*args, **kwargs):
        result = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=40, **kwargs)
        if result.returncode:
            raise RuntimeError("LOCAL_DOCKER_COMMAND_FAILED")
        return result.stdout.strip()

    def save():
        (ROOT / "results/graph_roundtrip.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    try:
        docker("run", "--detach", "--rm", "--name", name, "--memory", "1536m", "--cpus", "1",
            "-p", "127.0.0.1::7687", "-e", "NEO4J_AUTH",
            "-e", "NEO4J_server_memory_heap_initial__size=256m",
            "-e", "NEO4J_server_memory_heap_max__size=512m",
            "-e", "NEO4J_server_memory_pagecache_size=128m", image,
            env=dict(os.environ, NEO4J_AUTH="neo4j/" + password))
        started = True
        bound = docker("port", name, "7687/tcp")
        if not bound.startswith("127.0.0.1:") or not bound.split(":")[-1].isdigit():
            raise RuntimeError("LOCAL_BIND_SCOPE_REQUIRED")
        driver = GraphDatabase.driver("bolt://" + bound, auth=("neo4j", password),
            max_transaction_retry_time=0, connection_timeout=3, connection_acquisition_timeout=5)
        deadline = time.monotonic() + 120
        while True:
            try:
                driver.verify_connectivity()
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise RuntimeError("LOCAL_NEO4J_READINESS_TIMEOUT") from None
                time.sleep(2)
        store = Neo4jGraphStore(driver)

        def checkpoint(batches):
            report["committed_batches"] = list(batches)
            save()

        first = store.import_snapshot(snapshot, checkpoint=checkpoint)
        second = store.import_snapshot(snapshot)
        exported = store.export_snapshot_content(snapshot)
        if graph_records(exported) != graph_records(snapshot.content) or first["counts"] != second["counts"]:
            raise RuntimeError("LOCAL_GRAPH_REPLAY_OR_EXPORT_FAILED")
        out = build_snapshot(ROOT / "data/snapshots", exported,
            {key: snapshot.vectors[i] for i, key in enumerate(snapshot.chunk_ids)}, tenant_id="demo", purpose="DEMO",
            extraction_recipe=snapshot.manifest.extraction_recipe, embedding_recipe=snapshot.manifest.embedding_recipe,
            corpus_version=snapshot.manifest.corpus_version, normalizer_version=snapshot.manifest.normalizer_version,
            chunking_version=snapshot.manifest.chunking_version, tokenizer_version=snapshot.manifest.tokenizer_version)
        if out.name != snapshot.manifest.snapshot_version:
            raise RuntimeError("LOCAL_GRAPH_SNAPSHOT_IDENTITY_CHANGED")
        scope = store.scope(snapshot)
        target = snapshot.content.assertions[0].assertion_id
        key = store._key(scope, target)
        with driver.session(database="neo4j") as session:
            saved = session.run("MATCH (n:KGRecord {key:$key}) RETURN n.payload_sha256 AS sha", key=key).single()["sha"]
            session.run("MATCH (n:KGRecord {key:$key}) SET n.payload_sha256=$bad", key=key, bad="0" * 64).consume()
        try:
            store.import_snapshot(snapshot)
        except GraphStoreError as exc:
            if str(exc) != "GRAPH_IMMUTABLE_RECORD_CONFLICT":
                raise
            report["immutable_conflict_rejected"] = True
        else:
            raise RuntimeError("LOCAL_GRAPH_CONFLICT_NOT_REJECTED")
        with driver.session(database="neo4j") as session:
            session.run("MATCH (n:KGRecord {key:$key}) SET n.payload_sha256=$sha", key=key, sha=saved).consume()
            session.run("MATCH (n:KGRelease {key:$key}) SET n.status='BUILDING'", key=scope["release_key"]).consume()
        try:
            store.export_snapshot_content(snapshot)
        except GraphStoreError:
            report["incomplete_release_export_rejected"] = True
        else:
            raise RuntimeError("LOCAL_GRAPH_INCOMPLETE_RELEASE_EXPORTED")
        store.import_snapshot(snapshot)
        with driver.session(database="neo4j") as session:
            session.run("MATCH (n:KGRecord {key:$key}) "
                "CREATE (f:KGRecord {key:$foreign,tenant:'foreign-fixture',release:'foreign-fixture'}) "
                "CREATE (f)-[:OBJECT]->(n)", key=key, foreign=name + "-foreign").consume()
        try:
            store.export_snapshot_content(snapshot)
        except GraphStoreError as exc:
            if str(exc) != "GRAPH_CROSS_SCOPE_EDGE":
                raise
            report["foreign_incoming_edge_rejected"] = True
        else:
            raise RuntimeError("LOCAL_GRAPH_CROSS_SCOPE_EDGE_EXPORTED")
        with driver.session(database="neo4j") as session:
            session.run("MATCH (f:KGRecord {key:$key}) DETACH DELETE f", key=name + "-foreign").consume()
        store.export_snapshot_content(snapshot)
        report.update(status="PASS", counts=first["counts"], structural_edges=first["structural_edges"],
                      idempotent_import_runs=3, snapshot_version=out.name, record_and_edge_parity="PASS",
                      source_quotes_and_polarity_preserved=True, corpus_coverage=receipt["status"])
    except Exception as exc:
        report["error_category"] = type(exc).__name__
        # No driver/Docker raw messages, URI auth, model output or credentials in reports.
        if isinstance(exc, GraphStoreError):
            report["safe_error"] = str(exc)
    finally:
        if driver:
            driver.close()
        if started:
            try:
                docker("stop", "--time", "10", name)
                report["temporary_container_stopped"] = True
            except Exception:
                report["temporary_container_stopped"] = False
        save()
    print(json.dumps({k: v for k, v in report.items() if k != "committed_batches"}, indent=2))
    return 0 if report["status"] == "PASS" and report.get("temporary_container_stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
