"""Question-only shared rewrite/plans, independently pinned before retrieval.

Saved provider receipts are revalidated without a gateway. Neither references nor
expected facts appear in this contract; generated searches are never evidence.
"""

from typing import Annotated, Literal

from pydantic import Field, TypeAdapter, model_validator

from app.collection_benchmark import RetrievalReceipt
from app.contracts import Nonempty, Record, Sha256, sha256_bytes, stable_id
from app.model_gateway import CallContext
from app.query_planning import PlannerRecipe, _parse_subqueries
from app.query_planning import _ProviderReceipt as PlanReceipt
from app.rewrite_cache import RewriteCache
from app.rewrite_cache import _ProviderReceipt as RewriteReceipt

BATCH_RUN = "benchmark316-shared-v1"


class QueryInput(Record):
    scenario_id: Nonempty
    language: Literal["en", "vi"]
    question: Annotated[str, Field(min_length=5, max_length=2000)]


def query_context(task, protocol_sha256, stage):
    return CallContext(tenant_id="demo", run_id=BATCH_RUN, stage=stage,
        request_id=stable_id("benchsearch", protocol=protocol_sha256, scenario=task.scenario_id,
            language=task.language, question_sha256=sha256_bytes(task.question.encode("utf-8"))))


class SharedQueryRow(Record):
    schema_version: Literal[1] = 1
    preparation_status: Literal["READY"] = "READY"
    scenario_id: Nonempty
    language: Literal["en", "vi"]
    question_sha256: Sha256
    search_query: Annotated[str, Field(min_length=1, max_length=2000)]
    subqueries: Annotated[tuple[str, ...], Field(max_length=2)]
    planner_recipe_sha256: Sha256
    planner_receipt: PlanReceipt
    rewrite_receipt: RewriteReceipt | None


class FailedSharedQueryRow(SharedQueryRow):
    preparation_status: Literal["PLAN_FAILED"]
    failure_code: Literal["QUERY_PLAN_OUTPUT_INVALID"]
    failure_review_sha256: Sha256


def parse_shared_row(raw):
    return TypeAdapter(SharedQueryRow | FailedSharedQueryRow).validate_json(raw)


class SharedQueryBundle(Record):
    schema_version: Literal[1] = 1
    kind: Literal["PINNED_SHARED_QUERY_PREPARATION"] = "PINNED_SHARED_QUERY_PREPARATION"
    protocol_sha256: Sha256
    model_metadata_sha256: Sha256
    budget_run_id: Literal["benchmark316-shared-v1"] = BATCH_RUN
    rows: Annotated[tuple[SharedQueryRow | FailedSharedQueryRow, ...], Field(min_length=1, max_length=200)]

    @model_validator(mode="after")
    def coherent(self):
        if len({(r.scenario_id, r.language) for r in self.rows}) != len(self.rows):
            raise ValueError("BENCHMARK_SHARED_DUPLICATE_INPUT")
        operations = [r.planner_receipt.operation_id for r in self.rows]
        operations += [r.rewrite_receipt.operation_id for r in self.rows if r.rewrite_receipt]
        if len(set(operations)) != len(operations):
            raise ValueError("BENCHMARK_SHARED_DUPLICATE_OPERATION")
        return self


def validate_shared_row(row, task, protocol_sha256):
    if (row.scenario_id != task.scenario_id or row.language != task.language
            or row.question_sha256 != sha256_bytes(task.question.encode("utf-8"))
            or row.planner_recipe_sha256 != PlannerRecipe().fingerprint):
        raise ValueError("BENCHMARK_SHARED_QUESTION_OR_RECIPE_MISMATCH")
    if task.language == "en":
        if row.search_query != task.question or row.rewrite_receipt is not None:
            raise ValueError("BENCHMARK_EN_MUST_USE_ORIGINAL_QUERY")
    else:
        if row.rewrite_receipt is None or RewriteCache._validate(row.rewrite_receipt,
                query_context(task, protocol_sha256, "REWRITE"), task.question) != row.search_query:
            raise ValueError("BENCHMARK_SHARED_REWRITE_RECEIPT_MISMATCH")
    if isinstance(row, FailedSharedQueryRow):
        receipt = row.planner_receipt
        raw = receipt.raw_output.encode()
        if (receipt.stop_reason != "end_turn" or not receipt.raw_output_complete
                or receipt.raw_output_bytes != len(raw) or receipt.raw_output_sha256 != sha256_bytes(raw)
                or row.subqueries):
            raise ValueError("BENCHMARK_FAILED_PLAN_NOT_SETTLED_COMPLETE_RECEIPT")
        try:
            _parse_subqueries(receipt, query_context(task, protocol_sha256, "PLAN_QUERY"), row.search_query, PlannerRecipe())
        except ValueError as exc:
            if str(exc) != "QUERY_PLAN_OUTPUT_INVALID":
                raise
            return row
        raise ValueError("BENCHMARK_VALID_PLAN_CANNOT_BE_RELABELLED_FAILED")
    actual = _parse_subqueries(row.planner_receipt, query_context(task, protocol_sha256, "PLAN_QUERY"), row.search_query, PlannerRecipe())
    if actual != row.subqueries:
        raise ValueError("BENCHMARK_SHARED_PLAN_RECEIPT_MISMATCH")
    return row


def validate_shared_bundle(bundle, tasks, *, protocol_sha256, metadata_sha256, failure_reviews=None):
    if bundle.protocol_sha256 != protocol_sha256 or bundle.model_metadata_sha256 != metadata_sha256:
        raise ValueError("BENCHMARK_SHARED_PROTOCOL_OR_MODEL_MISMATCH")
    expected = {(t.scenario_id, t.language): t for t in tasks}
    if {(r.scenario_id, r.language) for r in bundle.rows} != set(expected):
        raise ValueError("BENCHMARK_SHARED_INPUT_SET_MISMATCH")
    for row in bundle.rows:
        validate_shared_row(row, expected[(row.scenario_id, row.language)], protocol_sha256)
        if isinstance(row, FailedSharedQueryRow):
            review = (failure_reviews or {}).get(row.failure_review_sha256)
            if (not review or review.get("status") != "SETTLED_VALIDATION_FAILURE_RETAIN_NO_RETRY"
                    or review.get("operation_id") != row.planner_receipt.operation_id
                    or review.get("raw_output_sha256") != row.planner_receipt.raw_output_sha256
                    or review.get("ledger_state") != "SETTLED"
                    or review.get("cost_micro_usd") != row.planner_receipt.cost_micro_usd):
                raise ValueError("BENCHMARK_FAILED_PLAN_REVIEW_REQUIRED")
    return bundle


class AugmentedRetrievalReceipt(RetrievalReceipt):
    query_track: Literal["EN_ORIGINAL", "VI_SHARED_REWRITE"]
    search_query_sha256: Sha256
    shared_bundle_sha256: Sha256
    planner_operation_id: Nonempty | None
    rewrite_operation_id: Nonempty | None
    subquery_count: Annotated[int, Field(ge=0, le=2)] = 0
    context_selection_used: bool = False
    execution_source: Literal["NEW_OFFLINE_RETRIEVAL", "REUSED_PINNED_CORE_EN"] = "NEW_OFFLINE_RETRIEVAL"
    preparation_error: Literal["QUERY_PLAN_OUTPUT_INVALID"] | None = None

    @model_validator(mode="after")
    def shared_identity(self):
        planned = self.requested_method in {"hybrid_multi_query", "hybrid_multi_query_balanced"}
        if bool(self.planner_operation_id) != planned or bool(self.rewrite_operation_id) != (self.language == "vi"):
            raise ValueError("BENCHMARK_SHARED_STAGE_IDENTITY_REQUIRED")
        if (self.subquery_count and not planned) or self.context_selection_used != (
            self.status == "COMPLETED" and self.requested_method == "hybrid_multi_query_balanced"
        ):
            raise ValueError("BENCHMARK_SHARED_SELECTION_METADATA_MISMATCH")
        if self.preparation_error and (not planned or self.status != "FAILED"):
            raise ValueError("BENCHMARK_SHARED_ERROR_CANNOT_BE_SUCCESS")
        return self
