"""Strict method outcomes, independent of legacy response shapes and fallbacks."""

from dataclasses import dataclass
from typing import Literal

from pydantic import Field, model_validator

from app.contracts import Record

StrictMethod = Literal["graph_1hop", "graph_2hop", "hybrid_multi_query", "hybrid_multi_query_balanced"]
MechanismStatus = Literal["USED_GRAPH", "USED_MULTI_QUERY", "NO_DECOMPOSITION_NEEDED", "NO_LINKED_ENTITY",
    "NO_SUPPORTED_RELATION", "PATH_INSUFFICIENT", "BUDGET_EXCEEDED", "FAILED"]


class MethodExecutionV2(Record):
    schema_version: Literal[2] = 2
    requested_method: StrictMethod
    effective_method: StrictMethod | None = None
    mechanism_status: MechanismStatus
    executed_stages: tuple[str, ...] = ()
    graph_path_used: bool = False
    planning_status: Literal["NOT_REQUIRED", "VALID", "FAILED"] = "NOT_REQUIRED"
    subquery_count: int = Field(default=0, ge=0, le=2)
    context_selection_used: bool = False
    context_chunk_ids: tuple[str, ...] = Field(default=(), max_length=8)
    context_assertion_ids: tuple[str, ...] = Field(default=(), max_length=200)
    answer_status: Literal["READY", "NOT_REQUESTED"] = "NOT_REQUESTED"
    failure_reason: str | None = Field(default=None, max_length=160)

    @model_validator(mode="after")
    def consistent_execution(self):
        graph = self.requested_method.startswith("graph_")
        ready = self.mechanism_status in {"USED_GRAPH", "USED_MULTI_QUERY", "NO_DECOMPOSITION_NEEDED"}
        if ready != (self.answer_status == "READY") or (ready and self.effective_method != self.requested_method):
            raise ValueError("STRICT_METHOD_READY_BINDING")
        if not ready and (self.effective_method is not None or self.context_chunk_ids or self.context_assertion_ids
                          or self.graph_path_used or self.context_selection_used):
            raise ValueError("STRICT_METHOD_UNSUPPORTED_CANNOT_GENERATE")
        if graph:
            if (self.planning_status != "NOT_REQUIRED" or self.subquery_count or self.context_selection_used
                    or ready != self.graph_path_used or ready != bool(self.context_assertion_ids)
                    or ready and (self.mechanism_status != "USED_GRAPH" or not self.context_chunk_ids)):
                raise ValueError("STRICT_GRAPH_EVIDENCE_REQUIRED")
        elif self.graph_path_used or self.context_assertion_ids or self.mechanism_status == "USED_GRAPH":
            raise ValueError("STRICT_QUERY_CANNOT_CLAIM_GRAPH")
        elif ready:
            if self.planning_status != "VALID" or (self.subquery_count > 0) != (self.mechanism_status == "USED_MULTI_QUERY"):
                raise ValueError("STRICT_QUERY_PLAN_REQUIRED")
            if self.context_selection_used != (self.requested_method == "hybrid_multi_query_balanced"):
                raise ValueError("STRICT_QUERY_SELECTION_REQUIRED")
        if len(set(self.context_chunk_ids)) != len(self.context_chunk_ids):
            raise ValueError("STRICT_CONTEXT_DUPLICATES")
        return self


@dataclass(frozen=True)
class StrictPreparedMethod:
    outcome: "MethodExecutionV2 | ReviewedGraphExecutionV3"
    view: object | None
    prepared: object | None
    trace: dict
    answer_language: Literal["en", "vi"] = "en"

    def __post_init__(self):
        if self.answer_language not in {"en", "vi"}:
            raise ValueError("STRICT_ANSWER_LANGUAGE")
        if (self.prepared is not None) != (self.outcome.answer_status == "READY"):
            raise ValueError("STRICT_PREPARED_OUTCOME_MISMATCH")
        if self.prepared is not None and (
            self.prepared.binding.mode_requested != self.outcome.requested_method
            or self.prepared.binding.mode_used != self.outcome.effective_method
            or self.prepared.binding.used_graph != self.outcome.graph_path_used
            or self.prepared.packed.chunk_ids != self.outcome.context_chunk_ids
            or self.prepared.packed.assertion_ids != self.outcome.context_assertion_ids
        ):
            raise ValueError("STRICT_PREPARED_EVIDENCE_MISMATCH")


class ReviewedGraphExecutionV3(Record):
    """A completed graph search can add zero evidence without claiming graph use.

    V2 receipts retain their original schema and unsupported-result semantics.
    Data/availability failures never construct this successful outcome.
    """

    schema_version: Literal[3] = 3
    requested_method: Literal["graph_1hop", "graph_2hop"]
    effective_method: Literal["graph_1hop", "graph_2hop"]
    mechanism_status: Literal["USED_GRAPH", "GRAPH_ZERO_EXPANSION"]
    executed_stages: tuple[str, ...]
    graph_path_used: bool
    context_chunk_ids: tuple[str, ...] = Field(max_length=8)
    context_assertion_ids: tuple[str, ...] = Field(max_length=200)
    answer_status: Literal["READY"] = "READY"
    failure_reason: None = None
    zero_expansion_reason: Literal["NO_LINKED_ENTITY", "NO_RELEVANT_PATH", "NO_PATH_FITS_CONTEXT"] | None = None
    release_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def consistent_graph_execution(self):
        used = self.mechanism_status == "USED_GRAPH"
        if (self.requested_method != self.effective_method or used != self.graph_path_used
                or used != bool(self.context_assertion_ids) or used == bool(self.zero_expansion_reason)
                or used and not self.context_chunk_ids
                or "TRAVERSE_GRAPH" not in self.executed_stages
                or len(set(self.context_chunk_ids)) != len(self.context_chunk_ids)):
            raise ValueError("REVIEWED_GRAPH_EXECUTION_BINDING")
        return self


def answer_strict_prepared(result, cache, *, gateway=None):
    """The sole generation seam for strict results; unsupported never calls cache/provider."""
    from app.generation import GenerationError

    if type(result) is not StrictPreparedMethod or result.prepared is None:
        raise GenerationError("STRICT_METHOD_NOT_READY_FOR_GENERATION")
    prepared = result.prepared
    return cache.resolve(prepared.context, question=prepared.question, snapshot=result.view, packed=prepared.packed,
        binding=prepared.binding, recipe=prepared.recipe, gateway=gateway, answer_language=result.answer_language)
