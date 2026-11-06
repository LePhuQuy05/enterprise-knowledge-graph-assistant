"""Opt-in V2 admission/outbox/leases; legacy run records stay readable unchanged.

The outer record is explicitly V2, so a V1 service cannot relabel these results.
Shared quota and work counters are reused, never a second allowance of 300.
"""

from uuid import UUID, uuid4

from app.answer_qualification import policy_fingerprint
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.funding_policy import FundingPolicyError, validate_external_funding
from app.ledger import BudgetDenied, Change, ConcurrentWrite
from app.owned_runs import TERMINAL, OwnedRunRepository, transition
from app.query_auth import ApiError, AuthContext
from app.query_state import DAY, QueryRepository
from app.scoped_run_budget import ScopedRunBudgetController
from app.scoped_run_contracts import ScopedRunMessageV2, ScopedRunReleaseV2, ScopedRunRequestV2

SCOPED_RUN_CAP_MICRO_USD = 2000000


class ScopedRunRepository(OwnedRunRepository):
    def __init__(self, store, *, clock, answer_budget):
        if not isinstance(answer_budget, ScopedRunBudgetController):
            raise ValueError("SCOPED_RUN_BUDGET_REQUIRED")
        super().__init__(store, clock=clock, answer_budget=answer_budget)

    def get(self, auth, run_id):
        import re

        from app.owned_runs import RUN_ID

        if type(run_id) is not str or not re.fullmatch(RUN_ID, run_id):
            raise ApiError(404, "RUN_NOT_FOUND")
        row = self.store.get((auth.partition, "RUN#"+run_id))
        if not row or row.get("record_type") != "OWNED_SCOPED_RUN_V2" or row.get("schema_version") != 2 or (
                row.get("tenant_id") != auth.tenant_id or row.get("owner_id") != auth.user_id
                or row.get("run_id") != run_id or row["expires_at"] <= int(self.now().timestamp())):
            raise ApiError(404, "RUN_NOT_FOUND")
        return row

    def check_profile(self, release):
        controller = self.answer_budget
        if release.budget_run_id != controller.budget_run_id or release.qualification_policy_sha256 != policy_fingerprint() or any(
                c.generation_recipe.model_metadata_sha256 != controller.routes.account_model_metadata_sha256
                for c in release.configs.values()):
            raise ApiError(503, "SCOPED_RUN_POLICY_BINDING")
        if controller.live:
            # Live enabling is a release operation, not implied by constructing
            # this class. Retain the existing monthly and shared funding caps.
            caps = ((controller.ledger.run_key(release.budget_run_id), SCOPED_RUN_CAP_MICRO_USD),
                (controller.ledger.month_key("kg-r72", controller.ledger._now().strftime("%Y-%m")), 12000000))
            if release.budget_run_id != "collection319-scoped-answer-v2" or any(c.fixture_only for c in release.configs.values()):
                raise ApiError(503, "SCOPED_RUN_LIVE_RELEASE_POLICY")
            for key, cap in caps:
                row = self.store.get(key)
                if not row or row.get("cap_micro_usd") != cap or row.get("halted"):
                    raise ApiError(429, "BUDGET_LIMIT")
            try:
                validate_external_funding(controller.ledger, profile="AUDITED_2_TO_8_USD_V2")
            except FundingPolicyError:
                raise ApiError(429, "BUDGET_LIMIT") from None

    def admit(self, auth, key, request, release):
        try:
            if type(key) is not str or str(UUID(key)) != key:
                raise ValueError()
            request = ScopedRunRequestV2.model_validate_json(request.model_dump_json())
            release = ScopedRunReleaseV2.model_validate_json(release.model_dump_json())
        except Exception:
            raise ApiError(400, "INVALID_REQUEST") from None
        self.check_profile(release)
        if not set(request.method_ids) <= set(release.configs):
            raise ApiError(400, "MODE_UNAVAILABLE")
        payload = request.model_dump(mode="json")
        digest = sha256_bytes(canonical_json(payload))
        run_id = stable_id("run", protocol="qualified-scoped-v2", tenant=auth.tenant_id,
            owner=auth.user_id, idempotency_key=key)
        now, nonce = self.now(), uuid4().hex
        epoch = int(now.timestamp())
        idem_key = auth.partition, "RUNIDEMP#qualified-v2#"+key
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
            if self.store.get((auth.partition, "RUN#"+run_id)):
                if self.store.get(idem_key):
                    continue
                raise ApiError(409, "IDEMPOTENCY_KEY_EXPIRED")
            quota = self.store.get(quota_key)
            if (quota["count"] if quota else 0)+len(request.method_ids) > 300:
                raise ApiError(429, "QUERY_QUOTA_EXCEEDED")
            active = [self.store.get(k) for k in active_keys]
            if any((r["count"] if r else 0)+len(request.method_ids) > limit
                    for r, limit in zip(active, (6, 3), strict=True)):
                raise ApiError(429, "RUN_CONCURRENCY_LIMIT")
            row = {"PK": auth.partition, "SK": "RUN#"+run_id, "version": 1, "record_type": "OWNED_SCOPED_RUN_V2",
                "schema_version": 2, "run_id": run_id, "tenant_id": auth.tenant_id, "owner_id": auth.user_id,
                "request": payload, "release": release.model_dump(mode="json"), "payload_sha256": digest,
                "admission_token": nonce, "created_at": now.isoformat(), "deadline_at": epoch+600,
                "expires_at": epoch+30*DAY, "cancel_requested": False}
            rows = [row,
                {"PK": auth.partition, "SK": "RUNHIST#"+now.strftime("%Y-%m-%dT%H:%M:%S.%fZ")+"#"+run_id,
                    "version": 1, "record_type": "OWNED_RUN_HISTORY", "run_id": run_id, "expires_at": row["expires_at"]},
                {"PK": idem_key[0], "SK": idem_key[1], "version": 1, "run_id": run_id,
                    "payload_sha256": digest, "expires_at": epoch+7*DAY}]
            for method in request.method_ids:
                message = ScopedRunMessageV2(tenant_id=auth.tenant_id, owner_id=auth.user_id, run_id=run_id,
                    method_id=method, admission_token=nonce).model_dump(mode="json")
                rows.extend([
                    {"PK": auth.partition, "SK": f"RUN#{run_id}#METHOD#{method}", "version": 1,
                        "record_type": "OWNED_RUN_METHOD", "run_id": run_id, "method_id": method,
                        "admission_token": nonce, "status": "QUEUED", "attempts": 0, "expires_at": row["expires_at"]},
                    {"PK": auth.partition, "SK": f"OUTBOX#{run_id}#METHOD#{method}", "version": 1,
                        "record_type": "OWNED_RUN_OUTBOX", "status": "PENDING", "message": message,
                        "expires_at": row["expires_at"]}])
            changes = [Change((r["PK"], r["SK"]), None, r) for r in rows]
            for counter, counter_key in [(quota, quota_key), *zip(active, active_keys, strict=True)]:
                new = {"PK": counter_key[0], "SK": counter_key[1], "version": (counter["version"] if counter else 0)+1,
                    "count": (counter["count"] if counter else 0)+len(request.method_ids), "record_type": "RUN_ADMISSION_COUNTER"}
                if counter_key == quota_key:
                    new.update(record_type="QUERY_QUOTA", expires_at=epoch+400*DAY)
                changes.append(Change(counter_key, counter["version"] if counter else None, new))
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

    def claim(self, message):
        message = ScopedRunMessageV2.model_validate_json(message.model_dump_json())
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
                self.finish(auth, message.run_id, message.method_id, "CANCELLED" if parent["cancel_requested"] else "EXPIRED")
                return "ACK", None
            if row["status"] == "CLAIMED" and row["lease_until"] > now:
                return "RETRY", None
            if row["attempts"] >= 3:
                self.finish(auth, message.run_id, message.method_id, "FAILED", error="RUN_CLAIM_LIMIT")
                return "ACK", None
            token = uuid4().hex
            changes = [transition(row, status="CLAIMED", lease_token=token,
                lease_until=min(now+120, parent["deadline_at"]), attempts=row["attempts"]+1), transition(parent)]
            try:
                self.store.commit(changes)
                return "CLAIMED", token
            except ConcurrentWrite:
                continue
        raise ApiError(503, "STATE_BUSY")


class ScopedOutboxDispatcher:
    def __init__(self, repository, send):
        self.repo, self.send = repository, send

    def publish(self, auth, run_id):
        sent = 0
        for row in self.repo.outboxes(auth, run_id):
            if row["status"] != "PENDING":
                continue
            self.send(ScopedRunMessageV2.model_validate_json(canonical_json(row["message"])).model_dump(mode="json"))
            try:
                self.repo.store.commit([transition(row, status="SENT")])
            except ConcurrentWrite:
                current = self.repo.store.get((row["PK"], row["SK"]))
                if current is None or current["status"] != "SENT":
                    raise
            sent += 1
        return sent
