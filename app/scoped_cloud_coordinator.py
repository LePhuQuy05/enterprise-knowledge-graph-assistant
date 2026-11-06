"""V2 outbox publication/recovery; no model, secret or asset initialization."""

import re

from app.contracts import canonical_json
from app.owned_runs import TERMINAL
from app.query_auth import ApiError, AuthContext
from app.scoped_owned_runs import ScopedOutboxDispatcher
from app.scoped_run_contracts import ScopedRunMessageV2
from app.scoped_run_service import ScopedRunService


def slot_message(slot):
    owner = re.fullmatch(r"TENANT#([a-z][a-z0-9_-]{0,63})#USER#([0-9a-f-]{36})", slot.get("owner_partition", ""))
    if slot.get("record_type") != "SCOPED_RUN_WORK_SLOT_V2" or slot.get("schema_version") != 2 or (
            slot.get("status") != "ACTIVE" or not owner):
        raise ValueError("SCOPED_WORK_SLOT_INVALID")
    return ScopedRunMessageV2(tenant_id=owner[1], owner_id=owner[2], run_id=slot["run_id"],
        method_id=slot["method_id"], admission_token=slot["admission_token"])


class ScopedSqsSender:
    def __init__(self, client, queue_url):
        self.client, self.queue_url = client, queue_url

    def __call__(self, message):
        raw = canonical_json(ScopedRunMessageV2.model_validate_json(canonical_json(message)).model_dump(mode="json"))
        return self.client.send_message(QueueUrl=self.queue_url, MessageBody=raw.decode())


class ScopedCloudCoordinator:
    def __init__(self, repository, send, authorize):
        self.repo, self.send, self.authorize = repository, send, authorize
        self.dispatcher = ScopedOutboxDispatcher(repository, send)

    def notify(self, auth, run_id):
        self.authorize(auth)
        self.repo.recover(auth, run_id)
        return self.dispatcher.publish(auth, run_id)

    def sweep(self):
        slots = self.repo.store.work_slots()
        summary = {"examined": len(slots), "pending_published": 0, "safe_redeliveries": 0,
            "terminal": 0, "unknown": 0, "errors": 0, "model_calls": 0}
        for slot in slots:
            if slot.get("status") == "FREE":
                continue
            try:
                msg = slot_message(slot)
                auth = AuthContext(msg.tenant_id, msg.owner_id)
                parent = self.repo.get(auth, msg.run_id)
                if parent["admission_token"] != msg.admission_token:
                    raise ValueError("SCOPED_WORK_SLOT_BINDING_LOST")
                try:
                    self.authorize(auth)
                except ApiError as exc:
                    if exc.status not in {401, 403}:
                        raise
                    self.repo.cancel(auth, msg.run_id)
                self.repo.recover(auth, msg.run_id)
                child = self.repo.child(auth, msg.run_id, msg.method_id)
                if child["status"] in TERMINAL:
                    summary["unknown" if child["status"] == "UNKNOWN" else "terminal"] += 1
                    continue
                outbox = next(row for row in self.repo.outboxes(auth, msg.run_id) if row["message"]["method_id"] == msg.method_id)
                if outbox["status"] == "PENDING":
                    summary["pending_published"] += self.dispatcher.publish(auth, msg.run_id)
                elif child["status"] == "QUEUED" or (child["status"] == "CLAIMED" and
                        child["lease_until"] <= int(self.repo.now().timestamp())):
                    self.send(msg.model_dump(mode="json"))
                    summary["safe_redeliveries"] += 1
            except Exception:
                summary["errors"] += 1
        return summary


class ScopedCloudService(ScopedRunService):
    def __init__(self, *args, coordinator, **kwargs):
        super().__init__(*args, **kwargs)
        self.coordinator = coordinator

    def submit(self, auth, payload, key):
        result = super().submit(auth, payload, key)
        try:
            self.coordinator.notify(auth, result["run_id"])
            result["publication"] = "ATTEMPTED_AFTER_COMMIT"
        except Exception:
            result["publication"] = "RECOVERY_PENDING"
        return result
