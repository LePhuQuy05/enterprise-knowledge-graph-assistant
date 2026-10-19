"""Local Neo4j staging: immutable records, explicit transactions and verified export.

No model calls. A graph release is visible to export only after full parity succeeds.
The serial writer scopes every key/query by tenant and immutable snapshot version.
"""

import json
from collections import defaultdict

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.snapshot import RECORD_TYPES, LoadedSnapshot, SnapshotContent

LABELS = {"documents": "KGDocumentVersion", "chunks": "KGChunk", "entities": "KGEntity",
          "mentions": "KGMention", "assertions": "KGAssertion"}
EDGE_TYPES = frozenset({"FROM_DOCUMENT", "IN_CHUNK", "MENTIONS", "SUBJECT_OF", "OBJECT", "EVIDENCED_BY"})


class GraphStoreError(ValueError):
    pass


def graph_records(content: SnapshotContent) -> dict[str, list[dict]]:
    result = {}
    for _, key, category in RECORD_TYPES.values():
        rows = []
        for record in sorted(getattr(content, category), key=lambda r: getattr(r, key)):
            payload = canonical_json(record.model_dump(mode="json")).decode()
            rows.append({"record_id": getattr(record, key), "payload": payload,
                         "payload_sha256": sha256_bytes(payload.encode())})
        result[category] = rows
    return result


def graph_edges(content: SnapshotContent) -> dict[str, list[tuple[str, str]]]:
    edges = defaultdict(set)
    for chunk in content.chunks:
        edges["FROM_DOCUMENT"].add((chunk.chunk_id, chunk.document_version_id))
    for mention in content.mentions:
        edges["IN_CHUNK"].add((mention.mention_id, mention.chunk_id))
        edges["MENTIONS"].add((mention.mention_id, mention.entity_id))
    for assertion in content.assertions:
        edges["SUBJECT_OF"].add((assertion.subject_id, assertion.assertion_id))
        edges["OBJECT"].add((assertion.assertion_id, assertion.object_id))
        edges["EVIDENCED_BY"].add((assertion.assertion_id, assertion.chunk_id))
    return {kind: sorted(rows) for kind, rows in sorted(edges.items())}


class Neo4jGraphStore:
    def __init__(self, driver):
        self.driver = driver

    def _transaction(self, work):
        # Explicit transactions never replay application/model work automatically.
        with self.driver.session(database="neo4j") as session, session.begin_transaction(timeout=15) as tx:
            result = work(tx)
            tx.commit()
            return result

    def initialize(self):
        for label in ("KGRecord", "KGRelease"):
            self._transaction(lambda tx, label=label: tx.run(
                f"CREATE CONSTRAINT {label.lower()}_key IF NOT EXISTS FOR (n:{label}) REQUIRE n.key IS UNIQUE"
            ).consume())

    @staticmethod
    def scope(snapshot: LoadedSnapshot):
        return {"tenant": snapshot.manifest.tenant_id, "release": snapshot.manifest.snapshot_version,
                "release_key": stable_id("graphrelease", tenant=snapshot.manifest.tenant_id,
                                         snapshot=snapshot.manifest.snapshot_version)}

    @staticmethod
    def _key(scope, record_id):
        return scope["release_key"] + ":" + record_id

    def import_snapshot(self, snapshot: LoadedSnapshot, *, checkpoint=None):
        self.initialize()
        scope = self.scope(snapshot)
        records = graph_records(snapshot.content)

        def begin(tx):
            row = tx.run("MERGE (r:KGRelease {key:$release_key}) "
                "ON CREATE SET r.tenant=$tenant,r.release=$release,r.manifest_sha=$sha,r.status='BUILDING' "
                "RETURN r.manifest_sha AS sha,r.tenant AS tenant,r.release AS release", **scope,
                sha=snapshot.manifest_sha256).single(strict=True)
            if (row["sha"], row["tenant"], row["release"]) != (snapshot.manifest_sha256, scope["tenant"], scope["release"]):
                raise GraphStoreError("GRAPH_IMMUTABLE_RELEASE_CONFLICT")
        self._transaction(begin)
        batches = []
        for category, rows in records.items():
            label = LABELS[category]  # Static whitelist, never source-supplied Cypher.
            for offset in range(0, len(rows), 100):
                batch = [row | {"key": self._key(scope, row["record_id"])} for row in rows[offset:offset + 100]]

                def write(tx):
                    found = list(tx.run(f"UNWIND $rows AS row MERGE (n:KGRecord:{label} {{key:row.key}}) "
                        "ON CREATE SET n.tenant=$tenant,n.release=$release,n.category=$category,"
                        "n.record_id=row.record_id,n.payload=row.payload,n.payload_sha256=row.payload_sha256 "
                        "RETURN n.key AS key,n.payload_sha256 AS sha,n.payload AS payload,"
                        "n.tenant AS tenant,n.release AS release", rows=batch, category=category, **scope))
                    expected = {r["key"]: r for r in batch}
                    if len(found) != len(batch) or any(
                        (r["sha"], r["payload"], r["tenant"], r["release"]) != (
                            expected[r["key"]]["payload_sha256"], expected[r["key"]]["payload"],
                            scope["tenant"], scope["release"]) for r in found
                    ):
                        raise GraphStoreError("GRAPH_IMMUTABLE_RECORD_CONFLICT")
                self._transaction(write)
                batches.append({"category": category, "offset": offset, "rows": len(batch),
                                "sha256": sha256_bytes(canonical_json(batch))})
                if checkpoint:
                    checkpoint(tuple(batches))
        for kind, rows in graph_edges(snapshot.content).items():
            if kind not in EDGE_TYPES:
                raise GraphStoreError("GRAPH_UNKNOWN_EDGE")
            for offset in range(0, len(rows), 100):
                batch = [{"source": self._key(scope, a), "target": self._key(scope, b)}
                         for a, b in rows[offset:offset + 100]]

                def write_edges(tx):
                    count = tx.run("UNWIND $rows AS row "
                        "MATCH (a:KGRecord {key:row.source,tenant:$tenant,release:$release}) "
                        "MATCH (b:KGRecord {key:row.target,tenant:$tenant,release:$release}) "
                        f"MERGE (a)-[r:{kind}]->(b) RETURN count(r) AS count", rows=batch, **scope).single(strict=True)["count"]
                    if count != len(batch):
                        raise GraphStoreError("GRAPH_MISSING_OR_DUPLICATE_ENDPOINT")
                self._transaction(write_edges)
        exported = self._read(snapshot, require_complete=False)
        if graph_records(exported) != records:
            raise GraphStoreError("GRAPH_RECORD_PARITY_FAILED")
        self._transaction(lambda tx: tx.run("MATCH (r:KGRelease {key:$release_key,manifest_sha:$sha}) "
            "SET r.status='COMPLETE' RETURN r.key", sha=snapshot.manifest_sha256, **scope).consume())
        return {"batches": batches, "counts": {k: len(v) for k, v in records.items()},
                "structural_edges": sum(len(v) for v in graph_edges(snapshot.content).values()),
                "release_status": "COMPLETE", "semantic_review": "PENDING"}

    def _read(self, snapshot, *, require_complete):
        scope = self.scope(snapshot)

        def read(tx):
            release = tx.run("MATCH (r:KGRelease {key:$release_key,tenant:$tenant,release:$release}) "
                             "RETURN r.status AS status,r.manifest_sha AS sha", **scope).single(strict=True)
            if release["sha"] != snapshot.manifest_sha256 or (require_complete and release["status"] != "COMPLETE"):
                raise GraphStoreError("GRAPH_RELEASE_NOT_COMPLETE_OR_UNTRUSTED")
            count = tx.run("MATCH (n:KGRecord {tenant:$tenant,release:$release}) RETURN count(n) AS count", **scope).single(strict=True)["count"]
            if count != sum(getattr(snapshot.manifest.counts, category) for category in LABELS):
                raise GraphStoreError("GRAPH_EXPORT_RECORD_COUNT")
            cross = tx.run("MATCH (a:KGRecord)-[r]->(b:KGRecord) "
                "WHERE ((a.tenant=$tenant AND a.release=$release) OR (b.tenant=$tenant AND b.release=$release)) "
                "AND (a.tenant<>$tenant OR a.release<>$release OR b.tenant<>$tenant OR b.release<>$release) "
                "RETURN count(r) AS count", **scope).single(strict=True)["count"]
            if cross:
                raise GraphStoreError("GRAPH_CROSS_SCOPE_EDGE")
            rows = list(tx.run("MATCH (n:KGRecord {tenant:$tenant,release:$release}) "
                "RETURN n.category AS category,n.record_id AS id,n.payload AS payload,n.payload_sha256 AS sha "
                "ORDER BY category,id", **scope))
            values = {category: [] for category in LABELS}
            for row in rows:
                if row["category"] not in values or sha256_bytes(row["payload"].encode()) != row["sha"]:
                    raise GraphStoreError("GRAPH_EXPORT_RECORD_HASH_OR_CATEGORY")
                model, key, _ = next(value for value in RECORD_TYPES.values() if value[2] == row["category"])
                record = model.model_validate_json(row["payload"])
                if record.tenant_id != scope["tenant"] or getattr(record, key) != row["id"]:
                    raise GraphStoreError("GRAPH_EXPORT_IDENTITY")
                values[row["category"]].append(record)
            content = SnapshotContent(**{k: tuple(v) for k, v in values.items()})
            actual = list(tx.run("MATCH (a:KGRecord {tenant:$tenant,release:$release})-[r]->(b) "
                "RETURN a.record_id AS source,b.record_id AS target,type(r) AS kind,"
                "b.tenant AS tenant,b.release AS release ORDER BY kind,source,target", **scope))
            expected = sorted((kind, a, b) for kind, pairs in graph_edges(snapshot.content).items() for a, b in pairs)
            if any((r["tenant"], r["release"]) != (scope["tenant"], scope["release"]) for r in actual) or [
                (r["kind"], r["source"], r["target"]) for r in actual
            ] != expected:
                raise GraphStoreError("GRAPH_EDGE_PARITY_OR_ISOLATION_FAILED")
            if graph_records(content) != graph_records(snapshot.content):
                raise GraphStoreError("GRAPH_EXPORT_CONTENT_PARITY_FAILED")
            return content
        return self._transaction(read)

    def export_snapshot_content(self, snapshot: LoadedSnapshot) -> SnapshotContent:
        return self._read(snapshot, require_complete=True)


def merge_canonical_rows(corpus_content: SnapshotContent, rows: list[dict]) -> SnapshotContent:
    """Union reviewed-identity aliases before immutable writes; never merge unknown entities by name."""
    records = {category: {} for category in ("entities", "mentions", "assertions")}
    for row in rows:
        for _, key, category in RECORD_TYPES.values():
            if category not in records:
                continue
            model = next(v[0] for v in RECORD_TYPES.values() if v[2] == category)
            for data in row[category]:
                record = model.model_validate_json(json.dumps(data))
                previous = records[category].get(getattr(record, key))
                if previous is not None:
                    if category == "entities":
                        if previous.model_dump(exclude={"aliases"}) != record.model_dump(exclude={"aliases"}):
                            raise GraphStoreError("GRAPH_ENTITY_IDENTITY_CONFLICT")
                        record = model(**(record.model_dump() | {"aliases": tuple(sorted(set(previous.aliases + record.aliases)))}))
                    elif previous != record:
                        raise GraphStoreError("GRAPH_RECORD_IDENTITY_CONFLICT")
                records[category][getattr(record, key)] = record
    return SnapshotContent(corpus_content.documents, corpus_content.chunks,
        **{category: tuple(values[k] for k in sorted(values)) for category, values in records.items()})
