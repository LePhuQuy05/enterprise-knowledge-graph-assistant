"""Durable operation protocol. Memory store is fixture-only; DynamoDB adapter is shared."""

import copy
import threading
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Annotated, Any, Literal, Protocol, Self

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError
from pydantic import AwareDatetime, Field, StringConstraints, TypeAdapter, model_validator

from app.contracts import (
    Nonempty,
    Record,
    Sha256,
    TenantId,
    canonical_json,
    sha256_bytes,
    stable_id,
)
from app.cost import TokenUsage

Key = tuple[str, str]
Identifier = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
Money = Annotated[int, Field(ge=0)]
EXTERNAL_FUNDING_ID = "external-r72-v1"
EXTERNAL_CAP_MICRO_USD = 10_000_000


class LedgerError(ValueError):
    pass


class BudgetDenied(LedgerError):
    pass


class IdempotencyConflict(LedgerError):
    pass


class AlreadyDispatched(LedgerError):
    pass


class ConcurrentWrite(Exception):
    pass


class Counter(Record):
    PK: str
    SK: str
    version: Annotated[int, Field(ge=1)]
    record_type: Literal["BUDGET_COUNTER"] = "BUDGET_COUNTER"
    project_id: TenantId
    tenant_id: TenantId | None
    period: str | None
    cap_micro_usd: Annotated[int, Field(gt=0, le=12_000_000)]
    reserved_micro_usd: Money = 0
    settled_micro_usd: Money = 0
    halted: bool = False
    other_cost_estimate_micro_usd: Money = 0
    gross_stop_micro_usd: Annotated[int, Field(gt=0)] | None = None


class OperationSpec(Record):
    project_id: TenantId
    tenant_id: TenantId
    run_id: Identifier
    request_id: Identifier
    stage: Literal["EMBED", "EXTRACT", "GENERATE", "EVALUATE", "SMOKE", "REWRITE"]
    attempt: Annotated[int, Field(ge=1, le=10)]
    input_sha256: Sha256
    model_id: Nonempty
    invocation_id: Nonempty
    source_region: Literal["us-east-1"]
    price_sheet_version: Nonempty
    max_cost_micro_usd: Annotated[int, Field(gt=0, le=12_000_000)]

    @property
    def operation_id(self) -> str:
        return stable_id("op", **self.model_dump(include={
            "project_id", "tenant_id", "run_id", "request_id", "stage", "attempt",
        }))

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


class ExternalOperationSpec(OperationSpec):
    """Explicit v2 shape; legacy Bedrock specs retain their exact serialized identity."""

    schema_version: Literal[2] = 2
    provider: Literal["DEEPSEEK", "VOYAGE"]
    billing_source: Literal["DIRECT_API"] = "DIRECT_API"
    funding_id: Literal["external-r72-v1"] = EXTERNAL_FUNDING_ID
    # This is an external route marker, not a claim about provider processing location.
    source_region: Literal["external-default"]


class Operation(Record):
    PK: str
    SK: Literal["STATE"] = "STATE"
    version: Annotated[int, Field(ge=1)]
    record_type: Literal["MODEL_OPERATION"] = "MODEL_OPERATION"
    operation_id: str
    spec: OperationSpec | ExternalOperationSpec
    spec_sha256: Sha256
    period: Annotated[str, StringConstraints(pattern=r"^\d{4}-(0[1-9]|1[0-2])$")]
    status: Literal["RESERVED", "SETTLED", "CANCELLED_BEFORE_CALL", "UNKNOWN_SPEND"]
    call_started: bool = False
    dispatch_id: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")] | None = None
    created_at: AwareDatetime
    updated_at: AwareDatetime
    actual_micro_usd: Money | None = None
    usage: TokenUsage | None = None
    bound_exceeded: bool = False

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if any(value.utcoffset().total_seconds() != 0 for value in (self.created_at, self.updated_at)):
            raise ValueError("Operation timestamps must be UTC")
        if self.operation_id != self.spec.operation_id or self.spec_sha256 != self.spec.fingerprint:
            raise ValueError("Operation identity or payload hash mismatch")
        if self.PK != "OP#" + self.operation_id:
            raise ValueError("Operation key mismatch")
        if self.call_started != (self.dispatch_id is not None):
            raise ValueError("Dispatched operations require a unique dispatch owner")
        if self.status == "SETTLED" and (
            not self.call_started or self.actual_micro_usd is None or self.usage is None
        ):
            raise ValueError("Settlement needs dispatched operation and usage")
        if self.status == "CANCELLED_BEFORE_CALL" and self.call_started:
            raise ValueError("Dispatched operations cannot be cancelled")
        if self.status == "UNKNOWN_SPEND" and not self.call_started:
            raise ValueError("Unknown spend requires a possibly dispatched call")
        return self


@dataclass(frozen=True)
class Change:
    key: Key
    expected_version: int | None
    item: dict[str, Any]


class LedgerStore(Protocol):
    backend: str

    def get(self, key: Key) -> dict[str, Any] | None: ...

    def commit(self, changes: Sequence[Change]) -> None: ...


def _validate_changes(changes: Sequence[Change]) -> None:
    keys = [change.key for change in changes]
    if not 1 <= len(keys) <= 100 or len(keys) != len(set(keys)):
        raise ValueError("Transaction needs unique item keys")
    for change in changes:
        if (change.item["PK"], change.item["SK"]) != change.key:
            raise ValueError("Transaction key differs from item")
        if change.item["version"] != (change.expected_version or 0) + 1:
            raise ValueError("Transaction version must increase exactly once")


class MemoryLedgerStore:
    """Atomic test reference; explicitly never sufficient for live ModelGateway."""

    backend = "memory_fixture"

    def __init__(self):
        self._items: dict[Key, dict] = {}
        self._lock = threading.RLock()

    def get(self, key: Key) -> dict | None:
        with self._lock:
            return copy.deepcopy(self._items.get(key))

    def commit(self, changes: Sequence[Change]) -> None:
        _validate_changes(changes)
        with self._lock:
            for change in changes:
                existing = self._items.get(change.key)
                if (existing is not None and existing["version"] != change.expected_version) or (
                    existing is None and change.expected_version is not None
                ):
                    raise ConcurrentWrite()
            for change in changes:
                self._items[change.key] = copy.deepcopy(change.item)


class DynamoLedgerStore:
    backend = "dynamodb"

    def __init__(self, client: Any, table_name: str):
        self.client, self.table_name = client, table_name
        self.serializer, self.deserializer = TypeSerializer(), TypeDeserializer()

    def get(self, key: Key) -> dict | None:
        response = self.client.get_item(
            TableName=self.table_name, Key={"PK": {"S": key[0]}, "SK": {"S": key[1]}},
            ConsistentRead=True,
        )
        if "Item" not in response:
            return None

        def decode(value):
            if isinstance(value, Decimal):
                if value != value.to_integral_value():
                    raise ValueError("Ledger numbers must be integral")
                return int(value)
            if isinstance(value, dict):
                return {k: decode(v) for k, v in value.items()}
            if isinstance(value, list):
                return [decode(v) for v in value]
            return value

        return decode({k: self.deserializer.deserialize(v) for k, v in response["Item"].items()})

    def commit(self, changes: Sequence[Change]) -> None:
        _validate_changes(changes)
        actions = []
        for change in changes:
            put = dict(TableName=self.table_name,
                       Item={k: self.serializer.serialize(v) for k, v in change.item.items()})
            if change.expected_version is None:
                put["ConditionExpression"] = "attribute_not_exists(PK) AND attribute_not_exists(SK)"
            else:
                put.update(ConditionExpression="#v = :expected", ExpressionAttributeNames={"#v": "version"},
                           ExpressionAttributeValues={":expected": {"N": str(change.expected_version)}})
            actions.append({"Put": put})
        token = sha256_bytes(canonical_json(actions))[:36]
        try:
            self.client.transact_write_items(TransactItems=actions, ClientRequestToken=token)
        except ClientError as exc:
            reasons = exc.response.get("CancellationReasons", [])
            if exc.response["Error"]["Code"] == "TransactionCanceledException" and any(
                reason.get("Code") in {"ConditionalCheckFailed", "TransactionConflict"}
                for reason in reasons
            ):
                raise ConcurrentWrite() from exc
            # Ambiguous transport/service failures propagate: never dispatch a model on failure.
            raise


def _record(model: type[Record], data: dict) -> Any:
    # Persisted timestamps/nested models use JSON encoding on both storage backends.
    return model.model_validate_json(canonical_json(data))


def _change(record: Record, *, existing: bool = True) -> Change:
    item = record.model_dump(mode="json")
    expected = item["version"] if existing else None
    item["version"] = (expected or 0) + 1
    return Change((item["PK"], item["SK"]), expected, item)


class Ledger:
    def __init__(self, store: LedgerStore, clock: Callable[[], datetime] | None = None):
        self.store, self.clock = store, clock or (lambda: datetime.now(UTC))

    def _now(self) -> datetime:
        now = self.clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("Ledger clock must be timezone-aware")
        return now.astimezone(UTC)

    @staticmethod
    def month_key(project: str, period: str) -> Key:
        return f"BUDGET#{project}#{period}", "MODEL"

    @staticmethod
    def run_key(run_id: str) -> Key:
        return "RUN#" + run_id, "BUDGET"

    @staticmethod
    def external_funding_key(project: str) -> Key:
        # Intentionally independent of provider, run, tenant and calendar month.
        return f"FUNDING#{project}#{EXTERNAL_FUNDING_ID}", "BUDGET"

    def _counter(self, key: Key) -> Counter:
        data = self.store.get(key)
        if data is None:
            raise BudgetDenied("Budget counter not initialized")
        counter = _record(Counter, data)
        if (counter.PK, counter.SK) != key:
            raise BudgetDenied("Budget counter key mismatch")
        return counter

    def _operation_counters(
        self, spec: OperationSpec | ExternalOperationSpec, period: str,
    ) -> list[Counter]:
        month = self._counter(self.month_key(spec.project_id, period))
        run = self._counter(self.run_key(spec.run_id))
        if (month.project_id, month.tenant_id, month.period) != (spec.project_id, None, period) or (
            run.project_id, run.tenant_id, run.period
        ) != (spec.project_id, spec.tenant_id, None):
            raise BudgetDenied("Budget ownership mismatch")
        counters = [month, run]
        if isinstance(spec, ExternalOperationSpec):
            funding = self._counter(self.external_funding_key(spec.project_id))
            if (funding.project_id, funding.tenant_id, funding.period) != (
                spec.project_id, None, None,
            ) or funding.cap_micro_usd > EXTERNAL_CAP_MICRO_USD:
                raise BudgetDenied("External funding scope or cap mismatch")
            counters.append(funding)
        return counters

    def operation(self, operation_id: str) -> Operation:
        data = self.store.get(("OP#" + operation_id, "STATE"))
        if data is None:
            raise LedgerError("Operation not found")
        return _record(Operation, data)

    def initialize_month(
        self, project: str, period: str, *, cap_micro_usd: int = 12_000_000,
        other_cost_estimate_micro_usd: int = 3_000_000, gross_stop_micro_usd: int = 15_000_000,
    ) -> None:
        if datetime.strptime(period, "%Y-%m").strftime("%Y-%m") != period:
            raise ValueError("Budget month must use canonical YYYY-MM")
        key = self.month_key(project, period)
        counter = Counter(PK=key[0], SK=key[1], version=1, project_id=project, tenant_id=None,
                          period=period, cap_micro_usd=cap_micro_usd,
                          other_cost_estimate_micro_usd=other_cost_estimate_micro_usd,
                          gross_stop_micro_usd=gross_stop_micro_usd)
        self._initialize(counter)

    def initialize_run(
        self, project: str, tenant: str, run_id: str, *, cap_micro_usd: int = 100_000,
    ) -> None:
        TypeAdapter(Identifier).validate_python(run_id, strict=True)
        key = self.run_key(run_id)
        counter = Counter(PK=key[0], SK=key[1], version=1, project_id=project, tenant_id=tenant,
                          period=None, cap_micro_usd=cap_micro_usd)
        self._initialize(counter)

    def initialize_external_funding(
        self, project: str, *, cap_micro_usd: int = EXTERNAL_CAP_MICRO_USD,
    ) -> None:
        """Create the one-time shared allowance; reinitialization never refunds spending."""
        if type(cap_micro_usd) is not int or not 0 < cap_micro_usd <= EXTERNAL_CAP_MICRO_USD:
            raise ValueError("External funding cap must be 1..10000000 integer micro-USD")
        key = self.external_funding_key(project)
        self._initialize(Counter(
            PK=key[0], SK=key[1], version=1, project_id=project, tenant_id=None,
            period=None, cap_micro_usd=cap_micro_usd,
        ))

    def _initialize(self, counter: Counter) -> None:
        try:
            self.store.commit([_change(counter, existing=False)])
        except ConcurrentWrite:
            existing = self._counter((counter.PK, counter.SK))
            fields = {"project_id", "tenant_id", "period", "cap_micro_usd",
                      "other_cost_estimate_micro_usd", "gross_stop_micro_usd"}
            if existing.model_dump(include=fields) != counter.model_dump(include=fields):
                raise IdempotencyConflict("Cannot reset or silently change an existing budget") from None

    def reserve(self, spec: OperationSpec | ExternalOperationSpec) -> Operation:
        for _ in range(8):
            existing = self.store.get(("OP#" + spec.operation_id, "STATE"))
            if existing is not None:
                operation = _record(Operation, existing)
                if operation.spec_sha256 != spec.fingerprint:
                    raise IdempotencyConflict("Operation key already has a different payload")
                return operation
            now = self._now()
            period = now.strftime("%Y-%m")
            counters = []
            for counter in self._operation_counters(spec, period):
                committed = counter.settled_micro_usd + counter.reserved_micro_usd
                prospective = committed + spec.max_cost_micro_usd
                if counter.halted or prospective > counter.cap_micro_usd or (
                    counter.gross_stop_micro_usd is not None and (
                        committed + counter.other_cost_estimate_micro_usd >= counter.gross_stop_micro_usd
                        or prospective + counter.other_cost_estimate_micro_usd > counter.gross_stop_micro_usd
                    )
                ):
                    raise BudgetDenied("Budget allowance exhausted or halted")
                counters.append(Counter(**(counter.model_dump() | {
                    "reserved_micro_usd": counter.reserved_micro_usd + spec.max_cost_micro_usd,
                })))
            operation = Operation(
                PK="OP#" + spec.operation_id, version=1, operation_id=spec.operation_id,
                spec=spec, spec_sha256=spec.fingerprint, period=period, status="RESERVED",
                created_at=now, updated_at=now,
            )
            try:
                self.store.commit([*(_change(c) for c in counters), _change(operation, existing=False)])
                return self.operation(spec.operation_id)
            except ConcurrentWrite:
                continue
        raise BudgetDenied("Concurrent updates exhausted bounded retries")

    def begin_call(self, operation_id: str) -> Operation:
        operation = self.operation(operation_id)
        if operation.status != "RESERVED" or operation.call_started:
            raise AlreadyDispatched("A second dispatch is forbidden for this operation")
        # Distinct workers must not submit identical idempotency tokens for dispatch ownership:
        # DynamoDB can replay transaction success without re-evaluating the CAS condition.
        dispatch_id = uuid.uuid4().hex
        updated = Operation(**(operation.model_dump() | {
            "call_started": True, "dispatch_id": dispatch_id, "updated_at": self._now(),
        }))
        try:
            self.store.commit([_change(updated)])
        except ConcurrentWrite as exc:
            raise AlreadyDispatched("Dispatch ownership changed; do not invoke") from exc
        confirmed = self.operation(operation_id)
        if confirmed.dispatch_id != dispatch_id or confirmed.status != "RESERVED":
            raise AlreadyDispatched("Dispatch ownership is no longer confirmed; do not invoke")
        return confirmed

    def mark_unknown(self, operation_id: str) -> Operation:
        operation = self.operation(operation_id)
        if operation.status == "UNKNOWN_SPEND":
            return operation
        if operation.status != "RESERVED" or not operation.call_started:
            raise LedgerError("Only dispatched reservations can have unknown spend")
        updated = Operation(**(operation.model_dump() | {
            "status": "UNKNOWN_SPEND", "updated_at": self._now(),
        }))
        self.store.commit([_change(updated)])
        return self.operation(operation_id)

    def cancel_before_call(self, operation_id: str) -> Operation:
        return self._finish(operation_id, None, None)

    def halt_operation_budgets(self, operation_id: str) -> None:
        """Stop new spending after a token-bound violation, even below the money bound."""
        operation = self.operation(operation_id)
        for _ in range(8):
            counters = self._operation_counters(operation.spec, operation.period)
            if all(counter.halted for counter in counters):
                return
            try:
                self.store.commit([_change(Counter(**(c.model_dump() | {"halted": True})))
                                   for c in counters])
                return
            except ConcurrentWrite:
                continue
        raise LedgerError("Could not halt counters; operator reconciliation required")

    def settle(self, operation_id: str, actual_micro_usd: int, usage: TokenUsage) -> Operation:
        if type(actual_micro_usd) is not int or actual_micro_usd < 0:
            raise ValueError("Actual charge must be nonnegative integer micro-USD")
        return self._finish(operation_id, actual_micro_usd, usage)

    def _finish(self, operation_id: str, actual: int | None, usage: TokenUsage | None) -> Operation:
        status = "CANCELLED_BEFORE_CALL" if actual is None else "SETTLED"
        for _ in range(8):
            operation = self.operation(operation_id)
            if operation.status == status:
                if status == "SETTLED" and (operation.actual_micro_usd, operation.usage) != (actual, usage):
                    raise IdempotencyConflict("Settlement payload conflicts with durable result")
                return operation
            if status == "CANCELLED_BEFORE_CALL":
                allowed = operation.status == "RESERVED" and not operation.call_started
            else:
                allowed = operation.status in {"RESERVED", "UNKNOWN_SPEND"} and operation.call_started
            if not allowed:
                raise LedgerError("Illegal operation transition; reservation cannot be refunded")
            spec = operation.spec
            overrun = actual is not None and actual > spec.max_cost_micro_usd
            counters = []
            for counter in self._operation_counters(spec, operation.period):
                counters.append(Counter(**(counter.model_dump() | {
                    "reserved_micro_usd": counter.reserved_micro_usd - spec.max_cost_micro_usd,
                    "settled_micro_usd": counter.settled_micro_usd + (actual or 0),
                    "halted": counter.halted or overrun,
                })))
            updated = Operation(**(operation.model_dump() | {
                "status": status, "updated_at": self._now(), "actual_micro_usd": actual,
                "usage": usage, "bound_exceeded": overrun,
            }))
            try:
                self.store.commit([*(_change(c) for c in counters), _change(updated)])
                return self.operation(operation_id)
            except ConcurrentWrite:
                continue
        raise LedgerError("Settlement contention: keep reservation and reconcile")
