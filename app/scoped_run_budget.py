"""Opt-in V14 owned-run holds, separate from the unchanged V9/316 policy.

Generation must use the typed V14 gateway over EnvelopeLedger, not the legacy
800-token generate() entry point. This module does not enable cloud routes.
"""

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.ledger import BudgetDenied, Change, Counter, ExternalOperationSpec, _change
from app.model_gateway import CallContext
from app.query_auth import AuthContext
from app.query_language import answer_language
from app.run_budget import RunBudgetController
from app.scoped_method_flow import METHODS, Q_METHODS
from app.scoped_query_v6 import ScopeRecipeV6
from app.task_scoped_generation_v14 import TaskScopedRecipeV14

PLAN_SLOT = "PLAN_QUERY:shared-v6"


class ScopedRunBudgetController(RunBudgetController):
    def output_tokens(self, slot):
        if slot in {self.slot_for(), PLAN_SLOT}:
            return 256
        if slot.startswith("GENERATE:") and slot.split(":", 1)[1] in METHODS:
            return 1600
        raise BudgetDenied("SCOPED_RUN_STAGE_NOT_ALLOWED")

    def context(self, parent, slot):
        self.output_tokens(slot)
        recipe = (TaskScopedRecipeV14(model_metadata_sha256=self.routes.account_model_metadata_sha256).fingerprint
            if slot.startswith("GENERATE:") else ScopeRecipeV6().fingerprint if slot == PLAN_SLOT else "vi-en-aws-search-v1")
        return CallContext(project_id=self.project_id, tenant_id=parent["tenant_id"], run_id=self.budget_run_id,
            request_id=stable_id("astage-scoped-v14", run=parent["run_id"], owner=parent["owner_id"],
                slot=slot, recipe=recipe, budget_run=self.budget_run_id), stage=slot.split(":")[0])

    def admission_slots(self, parent):
        methods = parent["request"]["method_ids"]
        if not 1 <= len(methods) <= 3 or len(methods) != len(set(methods)) or any(m not in METHODS for m in methods):
            raise BudgetDenied("SCOPED_RUN_METHODS_NOT_ALLOWED")
        slots = [self.slot_for(method) for method in methods]
        if answer_language(parent["request"]["original_question"], parent["request"]["language"]) == "vi":
            slots.append(self.slot_for())
        if any(method in Q_METHODS for method in methods):
            slots.append(PLAN_SLOT)
        return slots

    def _spec(self, parent, slot, input_sha):
        context = self.context(parent, slot)
        rate = self.prices.rate_for(self.routes.generation_model_id, self.routes.generation_invocation_id,
            "external-default", now=self.ledger._now(), live=self.live)
        bound = rate.quote(TokenUsage(input_tokens=self.routes.generation_context_ceiling,
            output_tokens=self.output_tokens(slot)))
        return ExternalOperationSpec(**context.model_dump(), provider="DEEPSEEK", input_sha256=input_sha,
            model_id=self.routes.generation_model_id, invocation_id=self.routes.generation_invocation_id,
            source_region="external-default", price_sheet_version=self.prices.version, max_cost_micro_usd=bound)

    def admission_changes(self, parent):
        # Keep the frozen legacy source byte-identical. This bounded copy of
        # admission bookkeeping is versioned here; shared allocate/settle/CAS
        # still use the inherited ledger protocol and its regression tests.
        auth = AuthContext(parent["tenant_id"], parent["owner_id"])
        specs = {slot: self._spec(parent, slot, "0"*64) for slot in self.admission_slots(parent)}
        total = sum(spec.max_cost_micro_usd for spec in specs.values())
        now = self.ledger._now()
        period = now.strftime("%Y-%m")
        counters = self.ledger._operation_counters(next(iter(specs.values())), period)
        changes = []
        for counter in counters:
            committed = counter.settled_micro_usd + counter.reserved_micro_usd
            if counter.halted or committed+total > counter.cap_micro_usd or (
                    counter.gross_stop_micro_usd is not None
                    and committed+total+counter.other_cost_estimate_micro_usd > counter.gross_stop_micro_usd):
                raise BudgetDenied("SCOPED_PARENT_BUDGET_INSUFFICIENT")
            changes.append(_change(Counter(**(counter.model_dump() | {
                "reserved_micro_usd": counter.reserved_micro_usd+total}))))
        key = self.envelope_key(auth, parent["run_id"])
        row = {"PK": key[0], "SK": key[1], "version": 1, "record_type": "OWNED_BUDGET_ENVELOPE",
            "budget_protocol": "SCOPED_V14_NATIVE_PLAN_V1", "run_id": parent["run_id"],
            "admission_token": parent["admission_token"], "period": period,
            "budget_run_id": self.budget_run_id, "price_sheet_version": self.prices.version,
            "counter_spec": next(iter(specs.values())).model_dump(mode="json"),
            "price_sheet_sha256": sha256_bytes(canonical_json(self.prices.model_dump(mode="json"))),
            "routes_sha256": sha256_bytes(canonical_json(self.routes.model_dump(mode="json"))),
            "max_cost_micro_usd": total, "created_at": now.isoformat(),
            "slots": {slot: {"state": "HELD", "max_cost_micro_usd": spec.max_cost_micro_usd,
                "operation_id": spec.operation_id} for slot, spec in specs.items()}}
        return [*changes, Change(key, None, row)]

    def envelope(self, auth, parent):
        row = super().envelope(auth, parent)
        if row.get("budget_protocol") != "SCOPED_V14_NATIVE_PLAN_V1":
            raise BudgetDenied("SCOPED_RUN_ENVELOPE_VERSION")
        return row

    def safe_failure(self, auth, parent, method):
        if not super().safe_failure(auth, parent, method):
            return False
        slot = self.envelope(auth, parent)["slots"].get(PLAN_SLOT)
        if slot is not None and slot["state"] == "ALLOCATED":
            operation = self.ledger.operation(slot["operation_id"])
            if operation.call_started and operation.status not in {"SETTLED", "CANCELLED_BEFORE_CALL"}:
                return False
        return True
