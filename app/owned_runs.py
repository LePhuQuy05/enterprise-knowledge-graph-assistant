"""Owned async run admission and fenced execution, independent of a queue transport."""

import re
from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import UUID, uuid4

from pydantic import Field, StringConstraints, field_validator

from app.contracts import Record, Sha256, canonical_json, sha256_bytes, stable_id
from app.ledger import Change, ConcurrentWrite
from app.query_auth import ApiError, AuthContext
from app.query_state import DAY, QueryRepository
from app.rag_methods import MethodId

RUN_ID = r"run_[0-9a-f]{64}"
TERMINAL = frozenset({"COMPLETED", "FAILED", "CANCELLED", "EXPIRED", "UNKNOWN"})
RETRIEVAL_METHODS = frozenset({"bm25", "vector", "hybrid", "hybrid_reranked"})


class OwnedRunRequestV1(Record):
    schema_version: Literal[1] = 1
    original_question: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    language: Literal["auto", "en", "vi"] = "auto"
    source_scope: Literal["local_collection"] = "local_collection"
    method_ids: Annotated[tuple[MethodId, ...], Field(min_length=1, max_length=3)] = ("hybrid",)
    run_kind: Literal["retrieval", "answer"] = "retrieval"

    @field_validator("original_question")
    @classmethod
    def trim_question(cls, value):
        value = value.replace("\r\n", "\n").replace("\r", "\n").strip()
        if not value:
            raise ValueError("EMPTY_QUESTION")
        return value

    @field_validator("method_ids")
    @classmethod
    def distinct(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("DUPLICATE_METHODS")
        return value


class OwnedRunReleaseV1(Record):
    schema_version: Literal[1] = 1
    collection_version: Annotated[str, StringConstraints(pattern=r"^collection_[0-9a-f]{64}$")]
    collection_manifest_sha256: Sha256
    method_config_sha256: Annotated[dict[MethodId, Sha256], Field(min_length=1, max_length=6)]


class RunMessageV1(Record):
    schema_version: Literal[1] = 1
    tenant_id: Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
    owner_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$")]
    run_id: Annotated[str, StringConstraints(pattern="^" + RUN_ID + "$")]
    method_id: MethodId
    admission_token: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


def transition(row, **fields):
    new = row | fields | {"version": row["version"] + 1}
    return Change((row["PK"], row["SK"]), row["version"], new)


def parent_status(children):
    states = [row["status"] for row in children]
    if "UNKNOWN" in states:
        return "UNKNOWN"
    if any(state not in TERMINAL for state in states):
        return "QUEUED" if all(state == "QUEUED" for state in states) else "RUNNING"
    if len(set(states)) == 1:
        return states[0]
    return "PARTIAL" if "COMPLETED" in states else "FAILED"


class OwnedRunRepository:
    def __init__(self, store, *, clock=lambda: datetime.now(UTC), answer_budget=None):
        self.store, self.clock, self.answer_budget = store, clock, answer_budget
        if answer_budget is not None and answer_budget.store is not store:
            raise ValueError("OWNED_BUDGET_AND_RUN_STORE_MUST_SHARE_TRANSACTION")

    def now(self):
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("RUN_CLOCK_REQUIRES_TIMEZONE")
        return now.astimezone(UTC)

    @staticmethod
    def active_keys(auth):
        return [("RUNACTIVE#kg-r72", "ALL"), ("RUNACTIVE#kg-r72", auth.partition)]

    def get(self, auth, run_id):
        if type(run_id) is not str or not re.fullmatch(RUN_ID, run_id):
            raise ApiError(404, "RUN_NOT_FOUND")
        row = self.store.get((auth.partition, "RUN#" + run_id))
        if not row or row.get("record_type") != "OWNED_RUN" or row.get("schema_version") != 1 or (
            row.get("tenant_id") != auth.tenant_id or row.get("owner_id") != auth.user_id
            or row.get("run_id") != run_id or row["expires_at"] <= int(self.now().timestamp())
        ):
            raise ApiError(404, "RUN_NOT_FOUND")
        return row

    def child(self, auth, run_id, method):
        parent = self.get(auth, run_id)
        if method not in parent["request"]["method_ids"]:
            raise ApiError(404, "RUN_NOT_FOUND")
        row = self.store.get((auth.partition, f"RUN#{run_id}#METHOD#{method}"))
        if not row or row.get("run_id") != run_id or row.get("method_id") != method or (
            row.get("record_type") != "OWNED_RUN_METHOD" or row.get("admission_token") != parent["admission_token"]
        ):
            raise ApiError(503, "RUN_STATE_UNAVAILABLE")
        return row

    def admit(self, auth, key, request, release):
        try:
            if type(key) is not str or str(UUID(key)) != key:
                raise ValueError()
            request = OwnedRunRequestV1.model_validate_json(request.model_dump_json())
            release = OwnedRunReleaseV1.model_validate_json(release.model_dump_json())
        except Exception:
            raise ApiError(400, "INVALID_REQUEST") from None
        if request.run_kind != "retrieval" and self.answer_budget is None:
            raise ApiError(400, "ANSWER_RUNS_NOT_ENABLED")
        if not set(request.method_ids) <= RETRIEVAL_METHODS & set(release.method_config_sha256):
            raise ApiError(400, "MODE_UNAVAILABLE")
        payload = request.model_dump(mode="json")
        digest = sha256_bytes(canonical_json(payload))
        run_id = stable_id("run", tenant=auth.tenant_id, owner=auth.user_id, idempotency_key=key)
        now, nonce = self.now(), uuid4().hex
        epoch = int(now.timestamp())
        idem_key = auth.partition, "RUNIDEMP#" + key
        active_keys = self.active_keys(auth)
        quota_key = QueryRepository.quota_key(now.strftime("%Y-%m"))
        for _ in range(8):
            existing = self.store.get(idem_key)
            if existing:
                if existing["expires_at"] <= epoch:
                    raise ApiError(409, "IDEMPOTENCY_KEY_EXPIRED")
                if existing["payload_sha256"] != digest:
                    raise ApiError(409, "IDEMPOTENCY_CONFLICT")
                return self.get(auth, existing["run_id"]), False
            if self.store.get((auth.partition, "RUN#" + run_id)):
                # Another transaction may have committed between the two independent reads.
                if self.store.get(idem_key):
                    continue
                raise ApiError(409, "IDEMPOTENCY_KEY_EXPIRED")
            quota = self.store.get(quota_key)
            count = quota["count"] if quota else 0
            if count + len(request.method_ids) > 300:
                raise ApiError(429, "QUERY_QUOTA_EXCEEDED")
            active = [self.store.get(k) for k in active_keys]
            if any((r["count"] if r else 0) + len(request.method_ids) > limit
                   for r, limit in zip(active, (6, 3), strict=True)):
                raise ApiError(429, "RUN_CONCURRENCY_LIMIT")
            row = {"PK": auth.partition, "SK": "RUN#" + run_id, "version": 1, "record_type": "OWNED_RUN",
                "schema_version": 1, "run_id": run_id, "tenant_id": auth.tenant_id, "owner_id": auth.user_id,
                "request": payload, "release": release.model_dump(mode="json"), "payload_sha256": digest,
                "admission_token": nonce, "created_at": now.isoformat(), "deadline_at": epoch + 600,
                "expires_at": epoch + 30 * DAY, "cancel_requested": False}
            history = {"PK": auth.partition, "SK": "RUNHIST#" + now.strftime("%Y-%m-%dT%H:%M:%S.%fZ") + "#" + run_id,
                "version": 1, "record_type": "OWNED_RUN_HISTORY", "run_id": run_id, "expires_at": row["expires_at"]}
            idem = {"PK": idem_key[0], "SK": idem_key[1], "version": 1, "run_id": run_id,
                "payload_sha256": digest, "expires_at": epoch + 7 * DAY}
            rows = [row, history, idem]
            for method in request.method_ids:
                message = RunMessageV1(tenant_id=auth.tenant_id, owner_id=auth.user_id, run_id=run_id,
                    method_id=method, admission_token=nonce).model_dump(mode="json")
                rows.extend([
                    {"PK": auth.partition, "SK": f"RUN#{run_id}#METHOD#{method}", "version": 1,
                        "record_type": "OWNED_RUN_METHOD", "run_id": run_id, "method_id": method,
                        "admission_token": nonce, "status": "QUEUED", "attempts": 0, "expires_at": row["expires_at"]},
                    {"PK": auth.partition, "SK": f"OUTBOX#{run_id}#METHOD#{method}", "version": 1,
                        "record_type": "OWNED_RUN_OUTBOX", "status": "PENDING", "message": message,
                        "expires_at": row["expires_at"]}])
            changes = [Change((r["PK"], r["SK"]), None, r) for r in rows]
            for counter, counter_key, increment in [(quota, quota_key, len(request.method_ids)),
                    *[(r, k, len(request.method_ids)) for r, k in zip(active, active_keys, strict=True)]]:
                new = {"PK": counter_key[0], "SK": counter_key[1], "version": (counter["version"] if counter else 0) + 1,
                       "count": (counter["count"] if counter else 0) + increment, "record_type": "RUN_ADMISSION_COUNTER"}
                if counter_key == quota_key:
                    # Preserve the synchronous endpoint's counter contract and global 300-attempt limit.
                    new.update(record_type="QUERY_QUOTA", expires_at=epoch + 400 * DAY)
                changes.append(Change(counter_key, counter["version"] if counter else None, new))
            if request.run_kind == "answer":
                from app.ledger import BudgetDenied

                try:
                    changes.extend(self.answer_budget.admission_changes(row))
                except BudgetDenied:
                    raise ApiError(429, "BUDGET_LIMIT") from None
            try:
                self.store.commit(changes)
                return row, True
            except ConcurrentWrite:
                continue
        raise ApiError(503, "STATE_BUSY")

    def children(self, auth, run_id):
        row = self.get(auth, run_id)
        return [self.child(auth, run_id, method) for method in row["request"]["method_ids"]]

    def outboxes(self, auth, run_id):
        row = self.get(auth, run_id)
        return [self.store.get((auth.partition, f"OUTBOX#{run_id}#METHOD#{method}"))
                for method in row["request"]["method_ids"]]

    def finish(self, auth, run_id, method, status, *, token=None, result_ref=None, error=None):
        if status not in TERMINAL:
            raise ValueError("RUN_TERMINAL_STATE_REQUIRED")
        for _ in range(8):
            row = self.child(auth, run_id, method)
            if row["status"] in TERMINAL:
                self.cleanup_budget(auth, run_id, method)
                return row
            if token is not None and row.get("lease_token") != token:
                raise ApiError(409, "RUN_LEASE_LOST")
            if token is None and row["status"] == "EXECUTING":
                raise ApiError(409, "RUN_LEASE_REQUIRED")
            if status == "COMPLETED" and (row["status"] != "EXECUTING" or result_ref is None):
                raise ValueError("RUN_RESULT_REQUIRED")
            if status == "COMPLETED":
                self.guard(auth, run_id, method, token, allow_cancelled=True)
            change = transition(row, status=status, result_ref=result_ref, error_category=error)
            changes = [change]
            # UNKNOWN retains its outstanding obligation until an explicit reconciliation.
            if status != "UNKNOWN":
                for key in self.active_keys(auth):
                    counter = self.store.get(key)
                    if not counter or counter["count"] < 1:
                        raise ApiError(503, "RUN_STATE_UNAVAILABLE")
                    changes.append(transition(counter, count=counter["count"] - 1))
            try:
                self.store.commit(changes)
                self.cleanup_budget(auth, run_id, method)
                return change.item
            except ConcurrentWrite:
                continue
        raise ApiError(503, "STATE_BUSY")

    def cleanup_budget(self, auth, run_id, method):
        if self.answer_budget is not None:
            parent = self.get(auth, run_id)
            if parent["request"]["run_kind"] == "answer":
                self.answer_budget.release_unused(auth, parent, method)
                if all(c["status"] in TERMINAL for c in self.children(auth, run_id)):
                    self.answer_budget.release_unused(auth, parent, all_slots=True)

    def claim(self, message):
        message = RunMessageV1.model_validate_json(message.model_dump_json())
        auth = AuthContext(message.tenant_id, message.owner_id)
        for _ in range(8):
            parent = self.get(auth, message.run_id)
            if message.admission_token != parent["admission_token"]:
                raise ApiError(404, "RUN_NOT_FOUND")
            row = self.child(auth, message.run_id, message.method_id)
            now = int(self.now().timestamp())
            if row["status"] in TERMINAL:
                return "ACK", None
            if row["status"] == "EXECUTING":
                if row["lease_until"] <= now:
                    self.finish(auth, message.run_id, message.method_id, "UNKNOWN", token=row["lease_token"],
                        error="RUN_RECONCILIATION_REQUIRED")
                    return "ACK", None
                return "RETRY", None
            if parent["cancel_requested"] or parent["deadline_at"] <= now:
                self.finish(auth, message.run_id, message.method_id,
                    "CANCELLED" if parent["cancel_requested"] else "EXPIRED")
                return "ACK", None
            if row["status"] == "CLAIMED" and row["lease_until"] > now:
                return "RETRY", None
            if row["attempts"] >= 3:
                self.finish(auth, message.run_id, message.method_id, "FAILED", error="RUN_CLAIM_LIMIT")
                return "ACK", None
            token = uuid4().hex
            changes = [transition(row, status="CLAIMED", lease_token=token,
                lease_until=min(now + 120, parent["deadline_at"]), attempts=row["attempts"] + 1), transition(parent)]
            try:
                self.store.commit(changes)
                return "CLAIMED", token
            except ConcurrentWrite:
                continue
        raise ApiError(503, "STATE_BUSY")

    def start(self, auth, run_id, method, token):
        parent = self.get(auth, run_id)
        row = self.child(auth, run_id, method)
        self.guard(auth, run_id, method, token, expected="CLAIMED")
        self.store.commit([transition(row, status="EXECUTING"), transition(parent)])

    def guard(self, auth, run_id, method, token, *, expected="EXECUTING", allow_cancelled=False):
        parent = self.get(auth, run_id)
        row = self.child(auth, run_id, method)
        now = int(self.now().timestamp())
        if row["status"] != expected or row.get("lease_token") != token or row["lease_until"] <= now:
            raise ApiError(409, "RUN_LEASE_LOST")
        if parent["deadline_at"] <= now or (parent["cancel_requested"] and not allow_cancelled):
            raise ApiError(409, "RUN_STOP_REQUESTED")

    def cancel(self, auth, run_id):
        for _ in range(8):
            parent = self.get(auth, run_id)
            if parent["cancel_requested"]:
                break
            try:
                self.store.commit([transition(parent, cancel_requested=True)])
                break
            except ConcurrentWrite:
                continue
        else:
            raise ApiError(503, "STATE_BUSY")
        for child in self.children(auth, run_id):
            if child["status"] in {"QUEUED", "CLAIMED"}:
                try:
                    self.finish(auth, run_id, child["method_id"], "CANCELLED")
                except ApiError as exc:
                    if exc.code != "RUN_LEASE_REQUIRED":
                        raise

    def summary(self, auth, run_id):
        parent = self.get(auth, run_id)
        children = self.children(auth, run_id)
        return {"schema_version": 1, "run_id": run_id, "status": parent_status(children),
            "request": parent["request"], "collection_version": parent["release"]["collection_version"],
            "created_at": parent["created_at"], "deadline_at": parent["deadline_at"],
            "cancel_requested": parent["cancel_requested"], "quality_metrics": None,
            "methods": [{"method_id": c["method_id"], "status": c["status"],
                "result_available": c["status"] == "COMPLETED", "error_category": c.get("error_category")}
                for c in children]}

    def recover(self, auth, run_id):
        """Explicit watchdog operation; no execution or automatic UNKNOWN reset."""
        parent = self.get(auth, run_id)
        now = int(self.now().timestamp())
        for child in self.children(auth, run_id):
            if child["status"] == "EXECUTING" and child["lease_until"] <= now:
                self.finish(auth, run_id, child["method_id"], "UNKNOWN", token=child["lease_token"],
                    error="RUN_RECONCILIATION_REQUIRED")
            elif child["status"] in {"QUEUED", "CLAIMED"} and parent["deadline_at"] <= now:
                try:
                    self.finish(auth, run_id, child["method_id"], "EXPIRED")
                except ApiError as exc:
                    if exc.code != "RUN_LEASE_REQUIRED":
                        raise
        return self.summary(auth, run_id)
