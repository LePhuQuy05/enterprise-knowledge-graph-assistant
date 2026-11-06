"""Six durable work slots, acquired/released inside the existing admission transaction.

Discovery uses a bounded key Query, never a table Scan, stream, or new GSI.
UNKNOWN keeps its slot and financial obligation until explicit reconciliation.
"""

from app.ledger import Change, ConcurrentWrite
from app.owned_runs import TERMINAL
from app.run_storage import DynamoRunStore

WORK_PARTITION = "RUNWORK#kg-r72"
MAX_WORK_SLOTS = 6


def slot_key(number):
    if type(number) is not int or not 0 <= number < MAX_WORK_SLOTS:
        raise ValueError("RUN_WORK_SLOT_INVALID")
    return WORK_PARTITION, f"SLOT#{number}"


def binding(row):
    return {k: row[k] for k in ("run_id", "method_id", "admission_token")} | {"owner_partition": row["PK"]}


class WorkIndexMixin:
    """Opt-in store composition: default local/pilot stores remain unchanged."""

    def work_slots(self):
        return [row for number in range(MAX_WORK_SLOTS) if (row := self.get(slot_key(number))) is not None]

    def commit(self, changes):
        changes, extra = list(changes), []
        used = set()
        slots = {number: self.get(slot_key(number)) for number in range(MAX_WORK_SLOTS)} if any(
            c.expected_version is None and c.item.get("record_type") == "OWNED_RUN_METHOD" for c in changes) else {}
        for index, change in enumerate(changes):
            row = change.item
            if row.get("record_type") != "OWNED_RUN_METHOD":
                continue
            if change.expected_version is None:
                if row["status"] != "QUEUED" or "work_slot" in row:
                    raise ValueError("RUN_WORK_ADMISSION_INVALID")
                for number in range(MAX_WORK_SLOTS):
                    slot = slots[number]
                    if number not in used and (slot is None or slot.get("status") == "FREE"):
                        used.add(number)
                        key = slot_key(number)
                        extra.append(Change(key, slot["version"] if slot else None,
                            {"PK": key[0], "SK": key[1], "version": (slot["version"] if slot else 0) + 1,
                             "record_type": "OWNED_RUN_WORK_SLOT", "schema_version": 1, "status": "ACTIVE",
                             **binding(row)}))
                        changes[index] = Change(change.key, None, row | {"work_slot": number})
                        break
                else:
                    raise ConcurrentWrite()
            elif row["status"] in TERMINAL - {"UNKNOWN"}:
                key = slot_key(row.get("work_slot"))
                slot = self.get(key)
                if not slot or slot.get("status") != "ACTIVE" or any(slot.get(k) != v for k, v in binding(row).items()):
                    raise ValueError("RUN_WORK_SLOT_BINDING_LOST")
                # FREE slots contain no past owner/question/source data and have no TTL.
                extra.append(Change(key, slot["version"], {"PK": key[0], "SK": key[1],
                    "version": slot["version"] + 1, "record_type": "OWNED_RUN_WORK_SLOT",
                    "schema_version": 1, "status": "FREE"}))
        return super().commit(changes + extra)


class CloudDynamoRunStore(WorkIndexMixin, DynamoRunStore):
    def work_slots(self):
        result = self.client.query(TableName=self.table_name, ConsistentRead=True, Limit=MAX_WORK_SLOTS + 1,
            KeyConditionExpression="PK = :pk AND begins_with(SK, :prefix)",
            ExpressionAttributeValues={":pk": {"S": WORK_PARTITION}, ":prefix": {"S": "SLOT#"}})
        if result.get("LastEvaluatedKey") or len(result["Items"]) > MAX_WORK_SLOTS:
            raise ValueError("RUN_WORK_INDEX_BOUND_EXCEEDED")
        rows = []
        for item in result["Items"]:
            # Point-read decoder preserves the same integral-number validation as ledger records.
            key = item["PK"]["S"], item["SK"]["S"]
            if key not in {slot_key(number) for number in range(MAX_WORK_SLOTS)}:
                raise ValueError("RUN_WORK_SLOT_INVALID")
            row = self.get(key)
            if row is not None:
                rows.append(row)
        return rows
