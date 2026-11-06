"""Durable local state and bounded Query adapters; no live resources are created."""

import json
import sqlite3
from pathlib import Path
from uuid import uuid4

from app.contracts import canonical_json
from app.ledger import ConcurrentWrite, DynamoLedgerStore, MemoryLedgerStore, _validate_changes


class MemoryRunStore(MemoryLedgerStore):
    def run_history_keys(self, partition, *, limit, after=None):
        with self._lock:
            keys = sorted((key for key in self._items if key[0] == partition and key[1].startswith("RUNHIST#")
                and (after is None or key[1] < after)), reverse=True)
            selected = keys[:limit + 1]
            return [{"PK": key[0], "SK": key[1]} for key in selected[:limit]], (
                selected[limit - 1][1] if len(selected) > limit else None)


class SQLiteRunStore:
    """Local workbench persistence, never a replacement for live model budget admission."""

    backend = "sqlite_run_state_not_model_ledger"

    def __init__(self, path, *, read_only=False):
        self.path, self.read_only = Path(path).resolve(), read_only
        if self.read_only:
            if not self.path.is_file():
                raise ValueError("RUN_READ_ONLY_STATE_MISSING")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS run_items (pk TEXT, sk TEXT, value BLOB, PRIMARY KEY(pk,sk))")

    def connect(self):
        return sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True, timeout=5) if self.read_only else (
            sqlite3.connect(self.path, timeout=5))

    def get(self, key):
        with self.connect() as db:
            row = db.execute("SELECT value FROM run_items WHERE pk=? AND sk=?", key).fetchone()
            return json.loads(row[0]) if row else None

    def commit(self, changes):
        if self.read_only:
            raise ValueError("RUN_STATE_READ_ONLY")
        _validate_changes(changes)
        if any(len(canonical_json(c.item)) > 192 * 1024 for c in changes):
            raise ValueError("RUN_STATE_ROW_LIMIT")
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            for change in changes:
                row = db.execute("SELECT value FROM run_items WHERE pk=? AND sk=?", change.key).fetchone()
                actual = json.loads(row[0])["version"] if row else None
                if actual != change.expected_version:
                    raise ConcurrentWrite()
            for change in changes:
                db.execute("INSERT OR REPLACE INTO run_items VALUES (?,?,?)",
                    (*change.key, canonical_json(change.item)))

    def run_history_keys(self, partition, *, limit, after=None):
        with self.connect() as db:
            rows = db.execute("SELECT sk FROM run_items WHERE pk=? AND sk>=? AND sk<? AND sk<? "
                "ORDER BY sk DESC LIMIT ?", (partition, "RUNHIST#", "RUNHIST$", after or "RUNHIST$", limit + 1)).fetchall()
        return [{"PK": partition, "SK": row[0]} for row in rows[:limit]], (
            rows[limit - 1][0] if len(rows) > limit else None)


class DynamoRunStore(DynamoLedgerStore):
    def run_history_keys(self, partition, *, limit, after=None):
        args = {"TableName": self.table_name, "ConsistentRead": True, "Limit": limit, "ScanIndexForward": False,
            "KeyConditionExpression": "PK = :pk AND begins_with(SK, :prefix)",
            "ExpressionAttributeValues": {":pk": {"S": partition}, ":prefix": {"S": "RUNHIST#"}},
            "ProjectionExpression": "PK, SK"}
        if after is not None:
            args["ExclusiveStartKey"] = {"PK": {"S": partition}, "SK": {"S": after}}
        response = self.client.query(**args)
        return [{"PK": r["PK"]["S"], "SK": r["SK"]["S"]} for r in response["Items"]], (
            response.get("LastEvaluatedKey", {}).get("SK", {}).get("S"))


class SQLiteRunQueue:
    """Durable local at-least-once queue, no claim of real SQS integration."""

    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS run_queue (id TEXT PRIMARY KEY, body BLOB, acknowledged INTEGER)")

    def send(self, message):
        with sqlite3.connect(self.path) as db:
            db.execute("INSERT INTO run_queue VALUES (?,?,0)", (uuid4().hex, canonical_json(message)))

    def receive(self, *, limit=20):
        with sqlite3.connect(self.path) as db:
            rows = db.execute("SELECT id,body FROM run_queue WHERE acknowledged=0 ORDER BY rowid LIMIT ?", (limit,)).fetchall()
        return [(identifier, json.loads(body)) for identifier, body in rows]

    def ack(self, identifier):
        with sqlite3.connect(self.path) as db:
            db.execute("UPDATE run_queue SET acknowledged=1 WHERE id=?", (identifier,))
