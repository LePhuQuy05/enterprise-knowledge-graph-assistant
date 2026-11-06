"""Offline benchmark receipts. Labels are scored after the runtime task completes.

No provider, account, credential or cloud client belongs in this module.
"""

import json
from collections import Counter
from dataclasses import dataclass
from itertools import combinations
from typing import Annotated, Literal, Self

from pydantic import Field, model_validator

from app.collection_evaluation import CollectionEvaluationDataset, score_collection_retrieval
from app.contracts import Nonempty, Record, Sha256, sha256_bytes
from app.evaluation import span_covered

CORE_METHODS = ("bm25", "vector", "hybrid", "hybrid_reranked", "graph_1hop", "graph_2hop")
ALL_METHODS = (*CORE_METHODS, "hybrid_multi_query", "hybrid_multi_query_balanced")


class ExposureCase(Record):
    scenario_id: Nonempty
    intent_group_id: Nonempty
    grouping_reason: Nonempty
    prior_related_inputs: tuple[Nonempty, ...]
    exposure: Literal["PRIOR_RELATED", "NO_MATCH_IN_REVIEWED_INPUTS"]
    split: Literal["DEVELOPMENT", "EVALUATION"]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if bool(self.prior_related_inputs) != (self.exposure == "PRIOR_RELATED"):
            raise ValueError("EXPOSURE_REFERENCE_MISMATCH")
        if self.exposure == "PRIOR_RELATED" and self.split != "DEVELOPMENT":
            raise ValueError("PREVIOUSLY_RELATED_CASE_IN_EVALUATION")
        return self


class BenchmarkProtocol(Record):
    schema_version: Literal[1]
    protocol_id: Nonempty
    status: Literal["FROZEN_AI_DRAFT_NONBLIND"]
    authoring_dataset_sha256: Sha256
    frozen_dataset_sha256: Sha256
    exposure_inventory_sha256: Sha256
    source_manifest_sha256: Sha256
    independent_holdout: Literal[False]
    all_cases_seen_by_annotation_author: Literal[True]
    exposure_search_exhaustive: Literal[False]
    reviewer_type: Literal["AI"]
    k: Literal[5]
    cases: Annotated[tuple[ExposureCase, ...], Field(min_length=1, max_length=100)]

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if len({c.scenario_id for c in self.cases}) != len(self.cases):
            raise ValueError("PROTOCOL_DUPLICATE_CASE")
        groups = {}
        for case in self.cases:
            groups.setdefault(case.intent_group_id, set()).add(case.split)
        if any(len(splits) != 1 for splits in groups.values()):
            raise ValueError("PROTOCOL_GROUP_LEAKAGE")
        return self


def validate_protocol(protocol, frozen_bytes, inventory_bytes, *, authoring_bytes):
    if (protocol.frozen_dataset_sha256 != sha256_bytes(frozen_bytes)
            or protocol.exposure_inventory_sha256 != sha256_bytes(inventory_bytes)
            or protocol.authoring_dataset_sha256 != sha256_bytes(authoring_bytes)):
        raise ValueError("BENCHMARK_PROTOCOL_HASH_MISMATCH")
    dataset = CollectionEvaluationDataset.model_validate_json(frozen_bytes)
    authoring = CollectionEvaluationDataset.model_validate_json(authoring_bytes)
    if dataset.split_status != "GROUPED_FROZEN" or dataset.source_manifest_sha256 != protocol.source_manifest_sha256:
        raise ValueError("BENCHMARK_PROTOCOL_DATASET_STATE")
    cases = {c.scenario_id: c for c in protocol.cases}
    inventory_refs = {q["reference"] for q in json.loads(inventory_bytes)["questions"]}
    if any(ref not in inventory_refs for c in cases.values() for ref in c.prior_related_inputs):
        raise ValueError("BENCHMARK_EXPOSURE_REFERENCE_UNKNOWN")
    if set(cases) != {s.scenario_id for s in dataset.scenarios}:
        raise ValueError("BENCHMARK_PROTOCOL_CASE_SET")
    for scenario in dataset.scenarios:
        case = cases[scenario.scenario_id]
        if scenario.split != case.split or scenario.intent_group_id != case.intent_group_id:
            raise ValueError("BENCHMARK_PROTOCOL_ASSIGNMENT_MISMATCH")
    excluded = {"dataset_id", "split_status", "scenarios"}
    if dataset.model_dump(exclude=excluded) != authoring.model_dump(exclude=excluded):
        raise ValueError("BENCHMARK_FROZEN_CONTENT_CHANGED")
    source_cases = {s.scenario_id: s for s in authoring.scenarios}
    for scenario in dataset.scenarios:
        source = source_cases.get(scenario.scenario_id)
        if source is None or scenario.model_dump(exclude={"split", "intent_group_id"}) != source.model_dump(
                exclude={"split", "intent_group_id"}):
            raise ValueError("BENCHMARK_FROZEN_CONTENT_CHANGED")
    return dataset


@dataclass(frozen=True)
class RetrievalTask:
    """The complete question-only interface; no reference answers, spans or rubrics."""

    scenario_id: str
    language: str
    question: str


class RetrievalReceipt(Record):
    schema_version: Literal[1]
    scenario_id: Nonempty
    language: Literal["en", "vi"]
    query_track: Literal["EN_ORIGINAL", "VI_RAW_DIAGNOSTIC"]
    question_sha256: Sha256
    requested_method: Literal[*ALL_METHODS]
    effective_method: Literal[*ALL_METHODS] | None
    status: Literal["COMPLETED", "FAILED", "NOT_RUN"]
    ranked_chunk_ids: tuple[Nonempty, ...] = ()
    context_chunk_ids: tuple[Nonempty, ...] = ()
    context_assertion_ids: tuple[Nonempty, ...] = ()
    used_graph: bool = False
    warnings: tuple[str, ...] = ()
    error_category: Nonempty | None = None
    duration_ms: Annotated[float, Field(ge=0)] | None = None
    model_calls: Literal[0] = 0

    @model_validator(mode="after")
    def coherent(self) -> Self:
        if (self.language == "en") != (self.query_track == "EN_ORIGINAL"):
            raise ValueError("BENCHMARK_QUERY_TRACK_MISMATCH")
        if self.status == "COMPLETED":
            if self.effective_method is None or self.duration_ms is None or self.error_category:
                raise ValueError("BENCHMARK_COMPLETED_RECEIPT_INVALID")
            expected = self.requested_method if self.used_graph or not self.requested_method.startswith("graph_") else "hybrid"
            if self.effective_method != expected:
                raise ValueError("BENCHMARK_EFFECTIVE_METHOD_MISMATCH")
        elif self.ranked_chunk_ids or self.context_chunk_ids or self.used_graph or self.context_assertion_ids:
            raise ValueError("BENCHMARK_NONCOMPLETED_HAS_OUTPUTS")
        elif self.status == "FAILED" and not self.error_category:
            raise ValueError("BENCHMARK_FAILURE_WITHOUT_REASON")
        if self.status != "COMPLETED" and self.effective_method is not None:
            raise ValueError("BENCHMARK_NONCOMPLETED_EFFECTIVE_METHOD")
        if (len(set(self.ranked_chunk_ids)) != len(self.ranked_chunk_ids)
                or len(set(self.context_chunk_ids)) != len(self.context_chunk_ids)
                or not set(self.context_chunk_ids) <= set(self.ranked_chunk_ids)):
            raise ValueError("BENCHMARK_RECEIPT_RANKING_INVALID")
        if self.used_graph and (self.requested_method not in {"graph_1hop", "graph_2hop"}
                               or self.effective_method != self.requested_method
                               or not self.context_assertion_ids):
            raise ValueError("BENCHMARK_GRAPH_EVIDENCE_REQUIRED")
        return self


def score_receipts(dataset, receipts, index):
    """Retain failures in the attempted denominator; unfinished matrices stay null."""
    cases = {s.scenario_id: s for s in dataset.scenarios}
    seen, scored = set(), []
    for receipt in receipts:
        key = (receipt.scenario_id, receipt.language, receipt.requested_method)
        if key in seen:
            raise ValueError("BENCHMARK_DUPLICATE_METHOD_CASE")
        seen.add(key)
        scenario = cases[receipt.scenario_id]
        question = getattr(scenario, "question_" + receipt.language)
        if receipt.question_sha256 != sha256_bytes(question.encode("utf-8")):
            raise ValueError("BENCHMARK_RECEIPT_QUESTION_MISMATCH")
        metrics = None
        if receipt.status != "NOT_RUN":
            metrics = score_collection_retrieval(scenario, receipt.ranked_chunk_ids,
                                                receipt.context_chunk_ids, index, k=5)
        scored.append(receipt.model_dump(mode="json") | {"split": scenario.split,
            "intent_group_id": scenario.intent_group_id, "answerable": scenario.answerable,
            "metrics": metrics})
    return scored


def summarize_receipts(dataset, scored, *, languages, methods):
    """A missing row is NOT_RUN, never a silent success or zero."""
    summaries = []
    for language in languages:
        for method in methods:
            for split in ("DEVELOPMENT", "EVALUATION", "ALL"):
                expected = [s for s in dataset.scenarios if split == "ALL" or s.split == split]
                allowed = {s.scenario_id for s in expected}
                rows = [r for r in scored if r["language"] == language and r["requested_method"] == method
                        and r["scenario_id"] in allowed]
                attempted = [r for r in rows if r["status"] != "NOT_RUN"]
                answerable = [r for r in attempted if r["answerable"]]
                missing = len(expected) - len(attempted)
                complete = missing == 0
                metrics = {key: (sum(r["metrics"][key] for r in answerable) / len(answerable)
                                 if complete and answerable else None)
                           for key in ("recall_at_k", "complete_evidence_at_k", "context_complete")}
                summaries.append({"language": language, "method": method, "split": split,
                    "expected_scenarios": len(expected), "attempted": len(attempted),
                    "failed": sum(r["status"] == "FAILED" for r in attempted), "not_run": missing,
                    "answerable_denominator": len(answerable), "controls": len(attempted) - len(answerable),
                    "intent_groups": len({s.intent_group_id for s in expected}),
                    "graph_used": sum(r["used_graph"] for r in attempted),
                    "graph_fallback": sum(r["effective_method"] == "hybrid"
                                          for r in attempted) if method.startswith("graph_") else 0,
                    "complete_matrix": complete, **metrics})
    return summaries


def evidence_at_k_feasibility(scenario, index, *, k=5, max_combinations=100_000):
    """Bounded oracle over overlapping chunks; UNKNOWN is never called impossible.

    This uses labels exclusively for evaluator diagnostics, never runtime search.
    """
    if not scenario.answerable:
        return {"status": "CONTROL_NOT_APPLICABLE", "min_chunks": None, "checked": 0}
    alternatives = [a for g in scenario.evidence_groups for a in g.alternatives]
    candidates = sorted(c.chunk_id for c in index.chunks.values() if any(
        c.document_version_id == a.document_version_id and c.start_char < a.end_char
        and c.end_char > a.start_char for a in alternatives))
    checked = 0
    for count in range(1, min(k, len(candidates)) + 1):
        for ids in combinations(candidates, count):
            if checked >= max_combinations:
                return {"status": "UNKNOWN_SEARCH_LIMIT", "min_chunks": None, "checked": checked}
            checked += 1
            chunks = [index.chunks[key] for key in ids]
            if all(any(span_covered(a, chunks) for a in g.alternatives) for g in scenario.evidence_groups):
                return {"status": "REACHABLE_AT_K", "min_chunks": count, "checked": checked}
    return {"status": "NOT_REACHABLE_AT_K_FOR_CURRENT_SPANS", "min_chunks": None, "checked": checked}


def assignment_counts(dataset):
    return {split: {"scenarios": len(cases), "answerable": sum(s.answerable for s in cases),
                    "controls": sum(not s.answerable for s in cases),
                    "intent_groups": len({s.intent_group_id for s in cases}),
                    "kinds": dict(Counter(s.kind for s in cases))}
            for split in ("DEVELOPMENT", "EVALUATION")
            if (cases := [s for s in dataset.scenarios if s.split == split])}
