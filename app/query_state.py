"""Atomic user-owned admission/history with conservative retries and explicit expiry."""

import base64
import copy
import re
from datetime import UTC, datetime
from uuid import UUID, uuid4

from app.contracts import QueryResponse, canonical_json, sha256_bytes, stable_id
from app.external_http import strict_json
from app.ledger import Change, ConcurrentWrite, DynamoLedgerStore, MemoryLedgerStore
from app.query_auth import ApiError

DAY = 86400
REQUEST_ID = r"q_[0-9a-f]{64}"


class MemoryQueryStore(MemoryLedgerStore):
    def history_keys(self, partition, *, limit, after=None):
        with self._lock:
            keys = sorted((k for k in self._items if k[0] == partition and k[1].startswith("QUERY#")
                           and (after is None or k[1] < after)), reverse=True)
            return [copy.deepcopy(self._items[k]) for k in keys[:limit]], (
                keys[limit - 1][1] if len(keys) > limit else None)


class DynamoQueryStore(DynamoLedgerStore):
    def history_keys(self, partition, *, limit, after=None):
        args = {"TableName": self.table_name, "ConsistentRead": True, "Limit": limit, "ScanIndexForward": False,
            "KeyConditionExpression": "PK = :pk AND begins_with(SK, :prefix)",
            "ExpressionAttributeValues": {":pk": {"S": partition}, ":prefix": {"S": "QUERY#"}},
            "ProjectionExpression": "PK, SK"}
        if after is not None:
            args["ExclusiveStartKey"] = {"PK": {"S": partition}, "SK": {"S": after}}
        response = self.client.query(**args)
        rows = [{"PK": r["PK"]["S"], "SK": r["SK"]["S"]} for r in response["Items"]]
        last = response.get("LastEvaluatedKey", {}).get("SK", {}).get("S")
        return rows, last


class QueryRepository:
    def __init__(self, store, *, clock=lambda: datetime.now(UTC)):
        self.store, self.clock = store, clock

    def now(self):
        value = self.clock()
        if value.tzinfo is None:
            raise ValueError("Query clock must have timezone")
        return value.astimezone(UTC)

    @staticmethod
    def quota_key(period):
        return "QUOTA#kg-r72#" + period, "ADMITTED"

    def quota_remaining(self):
        row = self.store.get(self.quota_key(self.now().strftime("%Y-%m")))
        return max(0, 300 - (row["count"] if row else 0))

    def _request(self, auth, request_id):
        if not re.fullmatch(REQUEST_ID, request_id):
            raise ApiError(404, "QUERY_NOT_FOUND")
        row = self.store.get((auth.partition, "REQUEST#" + request_id))
        if not row or row["expires_at"] <= int(self.now().timestamp()) or (
            row.get("tenant_id") != auth.tenant_id or row.get("owner_id") != auth.user_id
            or row.get("request_id") != request_id or row.get("record_type") != "QUERY_REQUEST"
        ):
            raise ApiError(404, "QUERY_NOT_FOUND")
        return row

    def get(self, auth, request_id):
        row = self._request(auth, request_id)
        # A crashed worker is never restarted. Persisted PROCESSING expires into a safe failed view.
        if row["status"] == "PROCESSING" and row["deadline_at"] <= int(self.now().timestamp()):
            return row | {"status": "FAILED", "error_code": "RECONCILIATION_REQUIRED"}
        return row

    def admit(self, auth, key, payload, release, recipe_sha):
        try:
            if type(key) is not str or str(UUID(key)) != key:
                raise ValueError()
        except (ValueError, AttributeError):
            raise ApiError(400, "INVALID_IDEMPOTENCY_KEY") from None
        digest = sha256_bytes(canonical_json(payload))
        idem_key = auth.partition, "IDEMP#" + key
        now = self.now()
        epoch = int(now.timestamp())
        request_id = stable_id("q", tenant=auth.tenant_id, owner=auth.user_id, idempotency_key=key)
        history_sk = "QUERY#" + now.strftime("%Y-%m-%dT%H:%M:%S.%fZ") + "#" + request_id
        # DynamoDB may return success for identical transaction client tokens. A distinct
        # admission claim makes competing workers' transactions different, so CAS picks one.
        admission_token = uuid4().hex
        quota_key = self.quota_key(now.strftime("%Y-%m"))
        for _ in range(4):
            existing = self.store.get(idem_key)
            if existing:
                if existing["expires_at"] <= epoch:
                    raise ApiError(409, "IDEMPOTENCY_KEY_EXPIRED", request_id=request_id)
                if existing["payload_sha256"] != digest:
                    raise ApiError(409, "IDEMPOTENCY_CONFLICT", request_id=request_id)
                return self.get(auth, existing["request_id"]), False
            # UUIDs are single-use; physical TTL cleanup must not revive a surviving request.
            if self.store.get((auth.partition, "REQUEST#" + request_id)):
                raise ApiError(409, "IDEMPOTENCY_KEY_EXPIRED", request_id=request_id)
            quota = self.store.get(quota_key)
            count = quota["count"] if quota else 0
            if count >= 300:
                raise ApiError(429, "QUERY_QUOTA_EXCEEDED")
            row = {"PK": auth.partition, "SK": "REQUEST#" + request_id, "record_type": "QUERY_REQUEST",
                "schema_version": 1, "version": 1, "request_id": request_id, "tenant_id": auth.tenant_id,
                "owner_id": auth.user_id, "question": payload["question"], "mode": payload["mode"],
                "payload_sha256": digest, "status": "PROCESSING", "snapshot_version": release["snapshot_version"],
                "manifest_sha256": release["manifest_sha256"], "recipe_sha256": recipe_sha,
                "created_at": now.isoformat(), "deadline_at": epoch + 28, "expires_at": epoch + 30 * DAY,
                "history_sk": history_sk, "admission_token": admission_token}
            for option in ("corpus_scope", "answer_language"):
                if option in payload:
                    row[option] = payload[option]
            idem = {"PK": idem_key[0], "SK": idem_key[1], "version": 1, "record_type": "QUERY_IDEMPOTENCY",
                "request_id": request_id, "payload_sha256": digest, "expires_at": epoch + 7 * DAY,
                "snapshot_version": release["snapshot_version"], "recipe_sha256": recipe_sha}
            history = {"PK": auth.partition, "SK": history_sk, "version": 1,
                "record_type": "QUERY_HISTORY", "request_id": request_id, "expires_at": row["expires_at"]}
            counter = {"PK": quota_key[0], "SK": quota_key[1], "version": (quota["version"] if quota else 0) + 1,
                "record_type": "QUERY_QUOTA", "count": count + 1, "expires_at": epoch + 400 * DAY}
            changes = [Change((r["PK"], r["SK"]), None, r) for r in (row, idem, history)]
            changes.append(Change(quota_key, quota["version"] if quota else None, counter))
            try:
                self.store.commit(changes)
                return row, True
            except ConcurrentWrite:
                continue
        raise ApiError(503, "STATE_BUSY")

    def update(self, auth, request_id, **fields):
        if set(fields) - {"receipt", "response", "response_sha256", "status", "error_code", "allowed_chunk_ids",
                          "source_bundle", "source_bundle_sha256", "rewrite_receipt"}:
            raise ValueError("Unapproved query state mutation")
        if "status" in fields and fields["status"] not in {"COMPLETED", "FAILED"}:
            raise ValueError("Unapproved terminal status")
        row = self._request(auth, request_id)
        if row["status"] != "PROCESSING":
            raise ApiError(409, "REQUEST_ALREADY_TERMINAL", request_id=request_id)
        if "receipt" in fields and "receipt" in row:
            raise ApiError(409, "RECEIPT_ALREADY_PRESENT", request_id=request_id)
        if any(key in fields and key in row for key in ("source_bundle", "rewrite_receipt")):
            raise ApiError(409, "RECEIPT_ALREADY_PRESENT", request_id=request_id)
        if "response" in fields:
            response = QueryResponse.model_validate_json(canonical_json(fields["response"]))
            if response.request_id != request_id or response.snapshot_version != row["snapshot_version"] or (
                len(canonical_json(fields["response"])) > 64 * 1024
            ):
                raise ValueError("Response binding or size invalid")
        new = row | fields | {"version": row["version"] + 1}
        if len(canonical_json(new)) > 192 * 1024:
            raise ValueError("Query record too large")
        self.store.commit([Change((row["PK"], row["SK"]), row["version"], new)])
        return new

    def history(self, auth, *, limit=10, cursor=None):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ApiError(400, "INVALID_LIMIT")
        after = None
        if cursor is not None:
            try:
                if not isinstance(cursor, str) or len(cursor) > 2048 or not re.fullmatch(r"[A-Za-z0-9_-]+", cursor):
                    raise ValueError()
                value = strict_json(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
                if set(value) != {"PK", "SK", "exp"} or value["PK"] != auth.partition or (
                    type(value["exp"]) is not int or value["exp"] <= int(self.now().timestamp())
                    or not re.fullmatch(r"QUERY#[0-9TZ:.\-]+#q_[0-9a-f]{64}", value["SK"])
                ):
                    raise ValueError()
                after = value["SK"]
            except Exception:
                raise ApiError(400, "INVALID_CURSOR") from None
        keys, last = self.store.history_keys(auth.partition, limit=limit, after=after)
        rows = []
        for key in keys:
            request_id = key["SK"].split("#")[-1]
            try:
                rows.append(self.get(auth, request_id))
            except ApiError as exc:
                if exc.status != 404:
                    raise
        next_cursor = None
        if last:
            next_cursor = base64.urlsafe_b64encode(canonical_json({"PK": auth.partition, "SK": last,
                "exp": int(self.now().timestamp()) + 3600})).decode().rstrip("=")
        return rows, next_cursor
