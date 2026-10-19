"""Local Docker readiness/provenance smoke; no host ports or AWS calls."""

import json
import secrets
import subprocess
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    output = ROOT / "results" / "neo4j_smoke"
    output.mkdir(parents=True, exist_ok=True)
    image = json.loads((ROOT / "configs/runtime_images.json").read_text())["neo4j"]
    name = "kg-r72-neo4j-smoke-" + uuid.uuid4().hex[:12]
    envfile = output / (name + ".env")
    envfile.write_text("NEO4J_AUTH=neo4j/" + secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    report = dict(status="FAIL", checked_at=datetime.now(UTC).isoformat(), image=image,
                  aws_calls=0, host_ports_published=False)
    started = False

    def run(args, **kwargs):
        return subprocess.run(["docker", *args], capture_output=True, text=True,
                              timeout=kwargs.pop("timeout", 30), **kwargs)

    def cypher(query):
        # Password stays in the temporary container environment, not the command line/report.
        return run(["exec", "-i", name, "sh", "-c",
                    'export NEO4J_PASSWORD="${NEO4J_AUTH#*/}"; '
                    'exec cypher-shell -u neo4j --format plain'], input=query + "\n", timeout=20)

    try:
        result = run(["run", "--detach", "--rm", "--name", name, "--memory", "1536m",
                      "--cpus", "1", "--env-file", str(envfile),
                      "-e", "NEO4J_server_memory_heap_initial__size=256m",
                      "-e", "NEO4J_server_memory_heap_max__size=512m",
                      "-e", "NEO4J_server_memory_pagecache_size=128m", image])
        if result.returncode:
            raise RuntimeError("container_start_failed")
        started = True
        deadline = time.monotonic() + 120
        while True:
            result = cypher("RETURN 1 AS ready;")
            if result.returncode == 0:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError("neo4j_readiness_timeout")
            time.sleep(2)
        version = cypher("CALL dbms.components() YIELD versions RETURN versions[0] AS version;")
        query = """
CREATE CONSTRAINT smoke_entity_id IF NOT EXISTS FOR (n:SmokeEntity) REQUIRE n.id IS UNIQUE;
CREATE CONSTRAINT smoke_assertion_id IF NOT EXISTS FOR (n:SmokeAssertion) REQUIRE n.id IS UNIQUE;
MERGE (s:SmokeEntity {id:'a'})
MERGE (o:SmokeEntity {id:'b'})
MERGE (r:SmokeAssertion {id:'r1'}) SET r.quote='Service A uses Service B.'
MERGE (s)-[:SUBJECT_OF]->(r) MERGE (r)-[:OBJECT]->(o);
MATCH (:SmokeEntity {id:'a'})-[:SUBJECT_OF]->(r:SmokeAssertion)-[:OBJECT]->(:SmokeEntity {id:'b'})
RETURN count(r) AS assertions, collect(r.quote)[0] AS quote;
"""
        for _ in range(2):
            result = cypher(query)
            if result.returncode or '1, "Service A uses Service B."' not in result.stdout:
                raise RuntimeError("provenance_or_merge_check_failed")
        report.update(status="PASS", database_version_output=version.stdout.strip(),
                      merge_runs=2, assertions_after_replay=1, quote_preserved=True)
    except (OSError, subprocess.TimeoutExpired, RuntimeError) as exc:
        report["error_category"] = type(exc).__name__
        # Never include Docker environment, raw authentication errors or secrets.
    finally:
        if started:
            stopped = run(["stop", "--time", "10", name], timeout=30)
            report["temporary_container_stopped"] = stopped.returncode == 0
        envfile.unlink(missing_ok=True)
        report["temporary_envfile_removed"] = not envfile.exists()
        (output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" and report.get("temporary_container_stopped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
