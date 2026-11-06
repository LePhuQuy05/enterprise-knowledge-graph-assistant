"""Atomic parent holds and single-use, receipted stages on the existing shared ledger."""

from datetime import timedelta
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import Field

from app.contracts import Nonempty, Record, canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.ledger import (
    BudgetDenied,
    Counter,
    ExternalOperationSpec,
    IdempotencyConflict,
    Ledger,
    Operation,
    _change,
)
from app.model_gateway import CallContext, ModelResult
from app.owned_runs import transition
from app.query_language import answer_language


class RunStageBusy(RuntimeError):
    pass


class RunStageBlocked(RuntimeError):
    pass


class StageReceipt(Record):
    operation_id: Nonempty
    raw_output: Annotated[str, Field(max_length=128 * 1024)]
    stop_reason: Literal["end_turn", "max_tokens"]
    usage: TokenUsage
    cost_micro_usd: Annotated[int, Field(ge=0)]


class RunBudgetController:
    def __init__(self, gateway=None, *, budget_run_id, project_id="kg-r72", ledger=None,
                 routes=None, prices=None, quote_live=True):
        if project_id != "kg-r72":
            raise ValueError("OWNED_ANSWER_APPROVED_GATEWAY_REQUIRED")
        if gateway is None:
            # API/watchdog may quote/release holds without possessing a provider key.
            if not isinstance(ledger, Ledger) or type(routes) is not ExternalRoutes or (
                type(prices) is not ExternalPriceSheet or type(quote_live) is not bool
            ):
                raise ValueError("OWNED_ADMISSION_POLICY_REQUIRED")
            self.ledger, self._routes, self._prices, self._live = ledger, routes, prices, quote_live
        else:
            if type(gateway) is not ExternalModelGateway or any(v is not None for v in (ledger, routes, prices)):
                raise ValueError("OWNED_ANSWER_APPROVED_GATEWAY_REQUIRED")
            self.ledger, self._routes, self._prices, self._live = (
                gateway.ledger, gateway.routes, gateway.prices, gateway.provider.live)
        self.gateway, self.store = gateway, self.ledger.store
        self.budget_run_id, self.project_id = budget_run_id, project_id

    @property
    def routes(self):
        return self.gateway.routes if self.gateway is not None else self._routes

    @property
    def prices(self):
        return self.gateway.prices if self.gateway is not None else self._prices

    @property
    def live(self):
        return self.gateway.provider.live if self.gateway is not None else self._live

    def bind_gateway(self, gateway):
        """Only an authorized worker attaches the live transport to the exact quoted policy."""
        if type(gateway) is not ExternalModelGateway or gateway.ledger.store is not self.store or (
            canonical_json(gateway.routes.model_dump(mode="json")) != canonical_json(self.routes.model_dump(mode="json"))
            or canonical_json(gateway.prices.model_dump(mode="json")) != canonical_json(self.prices.model_dump(mode="json"))
            or gateway.provider.live is not self.live
            or self.gateway is not None and self.gateway is not gateway
        ):
            raise ValueError("OWNED_GATEWAY_POLICY_BINDING")
        self.gateway = gateway

    @staticmethod
    def envelope_key(auth, run_id):
        return auth.partition, "RUNBUDGET#" + run_id

    @staticmethod
    def slot_for(method=None):
        return "REWRITE:shared" if method is None else "GENERATE:" + method

    def context(self, parent, slot):
        return CallContext(project_id=self.project_id, tenant_id=parent["tenant_id"], run_id=self.budget_run_id,
            request_id=stable_id("astage", run=parent["run_id"], owner=parent["owner_id"], slot=slot),
            stage=slot.split(":")[0])

    def _spec(self, parent, slot, input_sha):
        context = self.context(parent, slot)
        rate = self.prices.rate_for(self.routes.generation_model_id,
            self.routes.generation_invocation_id, "external-default", now=self.ledger._now(), live=self.live)
        output = 256 if slot.startswith("REWRITE:") else 800
        bound = rate.quote(TokenUsage(input_tokens=self.routes.generation_context_ceiling, output_tokens=output))
        return ExternalOperationSpec(**context.model_dump(), provider="DEEPSEEK", input_sha256=input_sha,
            model_id=self.routes.generation_model_id, invocation_id=self.routes.generation_invocation_id,
            source_region="external-default", price_sheet_version=self.prices.version, max_cost_micro_usd=bound)

    def admission_changes(self, parent):
        """Returned changes join the exact transaction that creates run/outbox/quota."""
        from app.query_auth import AuthContext

        auth = AuthContext(parent["tenant_id"], parent["owner_id"])
        slots = [self.slot_for(method) for method in parent["request"]["method_ids"]]
        if answer_language(parent["request"]["original_question"], parent["request"]["language"]) == "vi":
            slots.append(self.slot_for())
        specs = {slot: self._spec(parent, slot, "0" * 64) for slot in slots}
        total = sum(spec.max_cost_micro_usd for spec in specs.values())
        now = self.ledger._now()
        period = now.strftime("%Y-%m")
        counters = self.ledger._operation_counters(next(iter(specs.values())), period)
        changes = []
        for counter in counters:
            committed = counter.settled_micro_usd + counter.reserved_micro_usd
            if counter.halted or committed + total > counter.cap_micro_usd or (
                counter.gross_stop_micro_usd is not None
                and committed + total + counter.other_cost_estimate_micro_usd > counter.gross_stop_micro_usd
            ):
                raise BudgetDenied("OWNED_PARENT_BUDGET_INSUFFICIENT")
            changes.append(_change(Counter(**(counter.model_dump() | {
                "reserved_micro_usd": counter.reserved_micro_usd + total}))))
        key = self.envelope_key(auth, parent["run_id"])
        row = {"PK": key[0], "SK": key[1], "version": 1, "record_type": "OWNED_BUDGET_ENVELOPE",
            "run_id": parent["run_id"], "admission_token": parent["admission_token"], "period": period,
            "budget_run_id": self.budget_run_id, "price_sheet_version": self.prices.version,
            "counter_spec": next(iter(specs.values())).model_dump(mode="json"),
            "price_sheet_sha256": sha256_bytes(canonical_json(self.prices.model_dump(mode="json"))),
            "routes_sha256": sha256_bytes(canonical_json(self.routes.model_dump(mode="json"))),
            "max_cost_micro_usd": total, "created_at": now.isoformat(),
            "slots": {slot: {"state": "HELD", "max_cost_micro_usd": spec.max_cost_micro_usd,
                             "operation_id": spec.operation_id} for slot, spec in specs.items()}}
        from app.ledger import Change

        changes.append(Change(key, None, row))
        return changes

    def envelope(self, auth, parent):
        row = self.store.get(self.envelope_key(auth, parent["run_id"]))
        if not row or row.get("record_type") != "OWNED_BUDGET_ENVELOPE" or (
            row["admission_token"] != parent["admission_token"] or row["budget_run_id"] != self.budget_run_id
        ):
            raise BudgetDenied("OWNED_BUDGET_BINDING_INVALID")
        return row

    def assert_usable(self, auth, parent, slot):
        envelope = self.envelope(auth, parent)
        if slot not in envelope["slots"] or envelope["period"] != self.ledger._now().strftime("%Y-%m") or (
            envelope["price_sheet_sha256"] != sha256_bytes(canonical_json(self.prices.model_dump(mode="json")))
            or envelope["routes_sha256"] != sha256_bytes(canonical_json(self.routes.model_dump(mode="json")))
        ):
            raise BudgetDenied("OWNED_BUDGET_PERIOD_OR_ROUTE_CHANGED")
        spec = self._spec(parent, slot, "0" * 64)
        if any(c.halted for c in self.ledger._operation_counters(spec, envelope["period"])):
            raise BudgetDenied("OWNED_BUDGET_HALTED")
        return envelope

    def allocate(self, auth, parent, slot, spec):
        """Transfer an existing hold into an operation, without double-reserving counters."""
        from app.ledger import ConcurrentWrite

        expected = self._spec(parent, slot, spec.input_sha256)
        if type(spec) is not ExternalOperationSpec or expected.fingerprint != spec.fingerprint:
            raise IdempotencyConflict("OWNED_STAGE_SPEC_MISMATCH")
        for _ in range(8):
            envelope = self.assert_usable(auth, parent, slot)
            own = envelope["slots"][slot]
            if own["state"] == "ALLOCATED":
                operation = self.ledger.operation(own["operation_id"])
                if operation.spec_sha256 != spec.fingerprint:
                    raise IdempotencyConflict("OWNED_STAGE_INPUT_CHANGED")
                return operation
            if own["state"] != "HELD" or own["max_cost_micro_usd"] != spec.max_cost_micro_usd:
                raise BudgetDenied("OWNED_STAGE_HOLD_UNAVAILABLE")
            slots = envelope["slots"] | {slot: own | {"state": "ALLOCATED"}}
            now = self.ledger._now()
            operation = Operation(PK="OP#" + spec.operation_id, version=1, operation_id=spec.operation_id,
                spec=spec, spec_sha256=spec.fingerprint, period=envelope["period"], status="RESERVED",
                created_at=now, updated_at=now)
            try:
                self.store.commit([transition(envelope, slots=slots), _change(operation, existing=False)])
                return operation
            except ConcurrentWrite:
                continue
        raise BudgetDenied("OWNED_STAGE_ALLOCATION_CONTENTION")

    def release_unused(self, auth, parent, method=None, *, all_slots=False):
        from app.ledger import ConcurrentWrite

        for _ in range(8):
            envelope = self.envelope(auth, parent)
            selected = set(envelope["slots"]) if all_slots else {self.slot_for(method)}
            slots = {key: dict(value) for key, value in envelope["slots"].items()}
            refund = 0
            for slot in selected:
                if slot not in slots:
                    continue
                own = slots[slot]
                if own["state"] == "HELD":
                    refund += own["max_cost_micro_usd"]
                    own["state"] = "CANCELLED_BEFORE_CALL"
                elif own["state"] == "ALLOCATED":
                    operation = self.ledger.operation(own["operation_id"])
                    if operation.status == "RESERVED" and not operation.call_started:
                        # Existing ledger CAS arbitrates a race with begin_call; never refund a started call.
                        try:
                            self.ledger.cancel_before_call(operation.operation_id)
                        except Exception:
                            operation = self.ledger.operation(operation.operation_id)
                            if not operation.call_started and operation.status != "CANCELLED_BEFORE_CALL":
                                raise
            if not refund:
                return
            spec = ExternalOperationSpec.model_validate(envelope["counter_spec"])
            counters = self.ledger._operation_counters(spec, envelope["period"])
            changes = [_change(Counter(**(counter.model_dump() | {
                "reserved_micro_usd": counter.reserved_micro_usd - refund}))) for counter in counters]
            try:
                self.store.commit([*changes, transition(envelope, slots=slots)])
                return
            except ConcurrentWrite:
                continue
        raise BudgetDenied("OWNED_HOLD_RELEASE_CONTENTION")

    def safe_failure(self, auth, parent, method):
        envelope = self.envelope(auth, parent)
        relevant = {self.slot_for(method), self.slot_for()}
        for name, slot in envelope["slots"].items():
            if name in relevant and slot["state"] == "ALLOCATED":
                operation = self.ledger.operation(slot["operation_id"])
                if operation.call_started and operation.status not in {"SETTLED", "CANCELLED_BEFORE_CALL"}:
                    return False
        return True

    def view(self, auth, parent):
        envelope = self.envelope(auth, parent)
        total, unknown, held = 0, False, 0
        stages = []
        for name, slot in envelope["slots"].items():
            status, cost = slot["state"], 0
            if status == "ALLOCATED":
                op = self.ledger.operation(slot["operation_id"])
                status, cost = op.status, op.actual_micro_usd
                if cost is not None:
                    total += cost
                if op.status in {"RESERVED", "UNKNOWN_SPEND"}:
                    held += slot["max_cost_micro_usd"]
                    unknown |= op.call_started
            elif status == "HELD":
                held += slot["max_cost_micro_usd"]
            stages.append({"slot": name, "status": status, "operation_id": slot["operation_id"],
                           "cost_micro_usd": cost})
        return {"max_cost_micro_usd": envelope["max_cost_micro_usd"], "settled_micro_usd": total,
            "reserved_micro_usd": held, "cost_complete": not unknown and held == 0,
            "cost_basis": "PUBLISHED_PEAK_NO_ALLOWANCE_UPPER_BOUND_NOT_INVOICE", "stages": stages}


class EnvelopeLedger(Ledger):
    def __init__(self, controller, auth, parent, slot, guard):
        super().__init__(controller.store, clock=controller.ledger.clock)
        self.controller, self.auth, self.parent, self.slot = controller, auth, parent, slot
        self.guard = guard

    def reserve(self, spec):
        return self.controller.allocate(self.auth, self.parent, self.slot, spec)

    def begin_call(self, operation_id):
        self.guard()
        self.controller.assert_usable(self.auth, self.parent, self.slot)
        return super().begin_call(operation_id)


class ReceiptedStageGateway:
    """Store raw text/usage before application parsing; never retry an incomplete stage."""

    def __init__(self, controller, auth, parent, slot, guard):
        self.controller, self.auth, self.parent, self.slot, self.guard = controller, auth, parent, slot, guard
        self.cache_hit = False

    def generate(self, context, *, system, prompt, max_output_tokens=800):
        from app.ledger import Change, ConcurrentWrite

        if self.controller.gateway is None:
            raise RunStageBlocked("OWNED_ADMISSION_CANNOT_INVOKE_MODEL")
        expected = self.controller.context(self.parent, self.slot)
        if context != expected:
            raise IdempotencyConflict("OWNED_STAGE_CONTEXT_MISMATCH")
        digest = sha256_bytes(canonical_json([context.model_dump(), system, prompt, max_output_tokens]))
        key = self.auth.partition, f"RUNSTAGE#{self.parent['run_id']}#{self.slot}"
        now = self.controller.ledger._now()
        self.guard()
        row = self.controller.store.get(key)
        if row:
            if row.get("record_type") != "OWNED_STAGE_RECEIPT" or row.get("PK") != key[0] or row.get("SK") != key[1]:
                raise RunStageBlocked("OWNED_STAGE_RECORD_INVALID")
            if row["input_sha256"] != digest or row["admission_token"] != self.parent["admission_token"]:
                raise IdempotencyConflict("OWNED_STAGE_INPUT_CHANGED")
            if row["status"] == "RECEIVED":
                payload = row["receipt"]
                if len(canonical_json(payload)) > 128 * 1024 or sha256_bytes(canonical_json(payload)) != row["receipt_sha256"]:
                    raise RunStageBlocked("OWNED_STAGE_RECEIPT_CORRUPT")
                try:
                    StageReceipt.model_validate(payload)
                except ValueError:
                    raise RunStageBlocked("OWNED_STAGE_RECEIPT_INVALID") from None
                envelope = self.controller.envelope(self.auth, self.parent)
                own = envelope["slots"][self.slot]
                operation = self.controller.ledger.operation(payload["operation_id"])
                if own["state"] != "ALLOCATED" or own["operation_id"] != payload["operation_id"] or (
                    operation.status != "SETTLED" or operation.actual_micro_usd != payload["cost_micro_usd"]
                    or operation.usage != TokenUsage.model_validate(payload["usage"])
                    or operation.spec.request_id != expected.request_id or operation.spec.stage != expected.stage
                    or operation.spec.tenant_id != expected.tenant_id or operation.spec.run_id != expected.run_id
                    or operation.spec.project_id != expected.project_id or operation.spec.attempt != 1
                ):
                    raise RunStageBlocked("OWNED_STAGE_LEDGER_RECEIPT_MISMATCH")
                self.cache_hit = True
                return ModelResult(operation_id=payload["operation_id"], text=payload["raw_output"],
                    stop_reason=payload["stop_reason"], usage=TokenUsage.model_validate(payload["usage"]),
                    actual_micro_usd=payload["cost_micro_usd"])
            if row["status"] == "DISPATCHING" and row["lease_until"] > int(now.timestamp()):
                raise RunStageBusy("OWNED_SHARED_STAGE_IN_PROGRESS")
            raise RunStageBlocked("OWNED_STAGE_RECONCILIATION_REQUIRED")
        self.controller.assert_usable(self.auth, self.parent, self.slot)
        row = {"PK": key[0], "SK": key[1], "version": 1, "record_type": "OWNED_STAGE_RECEIPT",
            "input_sha256": digest, "admission_token": self.parent["admission_token"], "status": "DISPATCHING",
            "stage_token": uuid4().hex, "lease_until": int((now + timedelta(seconds=120)).timestamp()),
            "expires_at": self.parent["expires_at"]}
        try:
            self.controller.store.commit([Change(key, None, row)])
        except ConcurrentWrite:
            raise RunStageBusy("OWNED_STAGE_CLAIM_RACE") from None
        gateway = self.controller.gateway
        def dispatch_guard():
            self.guard()
            current = self.controller.store.get(key)
            if not current or current.get("stage_token") != row["stage_token"] or (
                current.get("status") != "DISPATCHING"
                or current.get("lease_until", 0) <= int(self.controller.ledger._now().timestamp())
            ):
                raise RunStageBlocked("OWNED_STAGE_LEASE_LOST")
        scoped = ExternalModelGateway(EnvelopeLedger(self.controller, self.auth, self.parent, self.slot, dispatch_guard),
            gateway.provider, gateway.routes, gateway.prices)
        try:
            self.guard()
            result = scoped.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
            payload = StageReceipt.model_validate({"operation_id": result.operation_id, "raw_output": result.text or "",
                "stop_reason": result.stop_reason, "usage": result.usage.model_dump(mode="json"),
                "cost_micro_usd": result.actual_micro_usd}).model_dump(mode="json")
            if len(canonical_json(payload)) > 128 * 1024:
                raise ValueError("OWNED_RECEIPT_SIZE_LIMIT")
            self.controller.store.commit([transition(row, status="RECEIVED", receipt=payload,
                receipt_sha256=sha256_bytes(canonical_json(payload)))])
            return result
        except Exception:
            # Preserve the DISPATCHING marker and underlying operation obligation for reconciliation.
            raise
