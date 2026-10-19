"""Single-writer local experiment journal, additional to the managed ledger.

No retry/recovery dispatch: an unfinished intent blocks subsequent work until an
operator reconciles it. The project ledger still enforces shared concurrent caps.
"""

import json
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage


class ExperimentStopped(ValueError):
    pass


class ReleaseExperiment:
    MAX_OPERATIONS = 80
    OBLIGATION_CAP = 400_000
    BATCH_STOP = 80_000

    def __init__(self, directory, gateway):
        self.directory, self.gateway = Path(directory), gateway
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "operations.json"
        self.lock = self.directory / "writer.lock"
        self.state = json.loads(self.path.read_bytes()) if self.path.exists() else {
            "version": "recruiter-release-experiment-v1", "operations": [],
            "obligation_cap_micro_usd": self.OBLIGATION_CAP, "maximum_operations": self.MAX_OPERATIONS}

    def __enter__(self):
        try:
            with self.lock.open("x", encoding="utf-8") as handle:
                handle.write("Exclusive experiment writer; do not remove after an uncertain dispatch.\n")
        except FileExistsError:
            raise ExperimentStopped("EXPERIMENT_WRITER_OR_CRASH_REQUIRES_RECONCILIATION") from None
        if self.state.get("maximum_operations") in {50, 64}:
            self.state.setdefault("policy_revisions", []).append({
                "from_max_operations": self.state["maximum_operations"], "to_max_operations": self.MAX_OPERATIONS,
                "reason": "Retain development failures and evaluate task-scoped spans without resetting receipts",
                "money_caps_unchanged": True})
            self.state["maximum_operations"] = self.MAX_OPERATIONS
            self._save()
        return self

    def __exit__(self, *args):
        # A normally unwound error is recorded; abrupt process death retains lock.
        self.lock.unlink()

    def _save(self):
        temporary = self.path.with_suffix(".tmp")
        temporary.write_bytes(canonical_json(self.state))
        temporary.replace(self.path)

    def generate(self, context, **kwargs):
        if not self.lock.exists():
            raise ExperimentStopped("EXPERIMENT_WRITER_REQUIRED")
        rows = self.state["operations"]
        if any(row["status"] != "SETTLED" for row in rows):
            raise ExperimentStopped("EXPERIMENT_UNRESOLVED_OPERATION")
        settled = sum(row["actual_micro_usd"] for row in rows)
        if len(rows) >= self.MAX_OPERATIONS or settled >= self.BATCH_STOP:
            raise ExperimentStopped("EXPERIMENT_COUNT_OR_BATCH_STOP")
        rate = self.gateway.prices.rate_for("deepseek-flash", "deepseek-flash", "external-default",
            now=self.gateway.ledger._now(), live=self.gateway.provider.live)
        bound = rate.quote(TokenUsage(input_tokens=self.gateway.routes.generation_context_ceiling,
                                      output_tokens=kwargs.get("max_output_tokens", 800)))
        if settled + bound > self.OBLIGATION_CAP:
            raise ExperimentStopped("EXPERIMENT_OBLIGATION_LIMIT")
        operation_id = stable_id("op", **context.model_dump())
        if any(row["operation_id"] == operation_id for row in rows):
            raise ExperimentStopped("EXPERIMENT_EXISTING_INTENT_NO_RETRY")
        row = {"operation_id": operation_id, "status": "ARMED", "bound_micro_usd": bound,
            "input_sha256": sha256_bytes(canonical_json(kwargs)), "context": context.model_dump()}
        rows.append(row)
        self._save()  # Count and obligation survive failure before/after provider dispatch.
        try:
            return self.gateway.generate(context, **kwargs)
        finally:
            try:
                operation = self.gateway.ledger.operation(operation_id)
                row.update(status=operation.status, actual_micro_usd=operation.actual_micro_usd)
            except Exception:
                row["status"] = "RECONCILIATION_REQUIRED"
            self._save()
