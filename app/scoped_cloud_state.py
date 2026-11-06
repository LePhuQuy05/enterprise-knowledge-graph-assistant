"""V2 discovery slots and bounded READY-receipt retention on the existing table.

Quota/active counters remain shared with V1. Discovery has a separate partition
so the legacy watchdog cannot consume a V2 Graph message by accident.
"""

from datetime import UTC, datetime

from app.answer_cache import AnswerCacheBlocked
from app.ledger import Change, ConcurrentWrite
from app.owned_runs import TERMINAL
from app.run_storage import DynamoRunStore

WORK_PARTITION = "RUNWORK#kg-r72#scoped-v2"
MAX_WORK_SLOTS = 6
READY_RETENTION_SECONDS = 31*86400


def slot_key(number):
    if type(number) is not int or not 0 <= number < MAX_WORK_SLOTS:
        raise ValueError("SCOPED_WORK_SLOT_INVALID")
    return WORK_PARTITION, f"SLOT#{number}"


def binding(row):
    return {key: row[key] for key in ("run_id", "method_id", "admission_token")} | {"owner_partition": row["PK"]}


class ScopedWorkIndexMixin:
    def _epoch(self):
        return int(datetime.now(UTC).timestamp())

    def get(self, key):
        row = super().get(key)
        if row and row.get("record_type") == "NATIVE_ANSWER_RECEIPT_V1" and row.get("state") == "READY" and (
                type(row.get("expires_at")) is not int or row["expires_at"] <= self._epoch()):
            # Never turn an expired receipt into a cache miss and another call.
            raise AnswerCacheBlocked("SCOPED_RAW_RECEIPT_EXPIRED_RECONCILE")
        return row

    def work_slots(self):
        return [row for i in range(MAX_WORK_SLOTS) if (row := self.get(slot_key(i))) is not None]

    def commit(self, changes):
        changes, extra = list(changes), []
        parents = {(c.item["PK"], c.item.get("run_id")) for c in changes
            if c.expected_version is None and c.item.get("record_type") == "OWNED_SCOPED_RUN_V2"
            and c.item.get("schema_version") == 2}
        needs_slots = any(c.expected_version is None and c.item.get("record_type") == "OWNED_RUN_METHOD" for c in changes)
        slots = {i: self.get(slot_key(i)) for i in range(MAX_WORK_SLOTS)} if needs_slots else {}
        used = set()
        for index, change in enumerate(changes):
            row = change.item
            if row.get("record_type") == "NATIVE_ANSWER_RECEIPT_V1":
                if row.get("state") == "READY":
                    old = super().get(change.key)
                    expiry = old.get("expires_at") if old else None
                    if expiry is not None and (type(expiry) is not int or expiry <= self._epoch()):
                        raise AnswerCacheBlocked("SCOPED_RAW_RECEIPT_EXPIRED_RECONCILE")
                    row = row | {"expires_at": expiry or self._epoch()+READY_RETENTION_SECONDS}
                else:
                    # Incomplete/FAILED receipts and financial obligations are
                    # retained for explicit reconciliation, never TTL-refunded.
                    row = {key: value for key, value in row.items() if key != "expires_at"}
                changes[index] = Change(change.key, change.expected_version, row)
            if row.get("record_type") != "OWNED_RUN_METHOD":
                continue
            if change.expected_version is None:
                if (row["PK"], row.get("run_id")) not in parents or row["status"] != "QUEUED" or "work_slot" in row:
                    raise ValueError("SCOPED_WORK_PARENT_ADMISSION_BINDING")
                for i in range(MAX_WORK_SLOTS):
                    old = slots[i]
                    if i not in used and (old is None or old.get("status") == "FREE"):
                        used.add(i)
                        key = slot_key(i)
                        extra.append(Change(key, old["version"] if old else None,
                            {"PK": key[0], "SK": key[1], "version": (old["version"] if old else 0)+1,
                                "record_type": "SCOPED_RUN_WORK_SLOT_V2", "schema_version": 2, "status": "ACTIVE",
                                **binding(row)}))
                        changes[index] = Change(change.key, None, row | {"work_slot": i, "work_protocol": "SCOPED_V2"})
                        break
                else:
                    raise ConcurrentWrite()
            elif row["status"] in TERMINAL-{"UNKNOWN"}:
                if row.get("work_protocol") != "SCOPED_V2":
                    raise ValueError("SCOPED_WORK_PROTOCOL_BINDING")
                key = slot_key(row.get("work_slot"))
                old = self.get(key)
                if not old or old.get("status") != "ACTIVE" or any(old.get(k) != v for k, v in binding(row).items()):
                    raise ValueError("SCOPED_WORK_SLOT_BINDING_LOST")
                extra.append(Change(key, old["version"], {"PK": key[0], "SK": key[1], "version": old["version"]+1,
                    "record_type": "SCOPED_RUN_WORK_SLOT_V2", "schema_version": 2, "status": "FREE"}))
        return super().commit(changes+extra)


class ScopedCloudDynamoRunStore(ScopedWorkIndexMixin, DynamoRunStore):
    def work_slots(self):
        response = self.client.query(TableName=self.table_name, ConsistentRead=True, Limit=MAX_WORK_SLOTS+1,
            KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
            ExpressionAttributeValues={":pk": {"S": WORK_PARTITION}, ":prefix": {"S": "SLOT#"}})
        if response.get("LastEvaluatedKey") or len(response["Items"]) > MAX_WORK_SLOTS:
            raise ValueError("SCOPED_WORK_INDEX_BOUND")
        rows = []
        keys = {slot_key(i) for i in range(MAX_WORK_SLOTS)}
        for item in response["Items"]:
            key = item["PK"]["S"], item["SK"]["S"]
            if key not in keys:
                raise ValueError("SCOPED_WORK_SLOT_INVALID")
            row = self.get(key)
            if row is not None:
                rows.append(row)
        return rows
