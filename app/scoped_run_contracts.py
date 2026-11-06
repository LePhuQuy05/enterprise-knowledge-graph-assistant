"""Versioned owned-run contracts for the source-bound V14 pipeline.

The legacy 316/V9 contracts remain unchanged, including their cap-eight wire
shape. These records carry execution state separately from display eligibility.
"""

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from app.answer_qualification import policy_fingerprint
from app.contracts import (
    Mode,
    Nonempty,
    QueryResponse,
    Record,
    Sha256,
    canonical_json,
    sha256_bytes,
    stable_id,
)
from app.cost import TokenUsage
from app.owned_runs import RUN_ID, OwnedRunRequestV1
from app.profiled_source_context import SourceContextProfile
from app.query_auth import SUB, TENANT
from app.scoped_query_v6 import ScopeRecipeV6
from app.task_scoped_generation_v14 import TaskScopedRecipeV14

Q_METHODS = {"hybrid_multi_query", "hybrid_multi_query_balanced"}


class ScopedRunRequestV2(OwnedRunRequestV1):
    schema_version: Literal[2] = 2
    method_ids: Annotated[tuple[Mode, ...], Field(min_length=1, max_length=3)] = ("hybrid",)
    run_kind: Literal["answer"] = "answer"


class ScopedMethodConfigV2(Record):
    schema_version: Literal[2] = 2
    implementation: Literal["scoped-method-flow-v1-v14-qualified"] = "scoped-method-flow-v1-v14-qualified"
    method_id: Mode
    collection_manifest_sha256: Sha256
    lexical_index_version: Nonempty
    document_recipe_sha256: Sha256
    query_recipe_sha256: Sha256 | None = None
    reranker_recipe_sha256: Sha256 | None = None
    graph_release_sha256: Sha256 | None = None
    planner_recipe_sha256: Sha256 | None = None
    generation_recipe: TaskScopedRecipeV14
    source_profile: SourceContextProfile
    fixture_only: bool

    @model_validator(mode="after")
    def integrity(self) -> Self:
        self.generation_recipe.fingerprint
        if self.source_profile.max_chunks != 12:
            raise ValueError("SCOPED_RELEASE_C12_REQUIRED")
        if (self.method_id != "bm25") != (self.query_recipe_sha256 is not None):
            raise ValueError("SCOPED_QUERY_ENCODER_CONFIG")
        if (self.method_id == "hybrid_reranked") != (self.reranker_recipe_sha256 is not None):
            raise ValueError("SCOPED_RERANKER_CONFIG")
        if self.method_id.startswith("graph_") != (self.graph_release_sha256 is not None):
            raise ValueError("SCOPED_GRAPH_RELEASE_CONFIG")
        expected = ScopeRecipeV6().fingerprint if self.method_id in Q_METHODS else None
        if self.planner_recipe_sha256 != expected:
            raise ValueError("SCOPED_NATIVE_FACETS_CONFIG")
        return self

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


class ScopedRunReleaseV2(Record):
    schema_version: Literal[2] = 2
    collection_version: Annotated[str, StringConstraints(pattern=r"^collection_[0-9a-f]{64}$")]
    collection_manifest_sha256: Sha256
    budget_run_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
    document_count: Annotated[int, Field(gt=0)]
    chunk_count: Annotated[int, Field(gt=0)]
    graph_fact_count: Annotated[int, Field(ge=0)] = 0
    graph_source_document_count: Annotated[int, Field(ge=0)] = 0
    configs: Annotated[dict[Mode, ScopedMethodConfigV2], Field(min_length=1, max_length=8)]
    method_config_sha256: dict[Mode, Sha256]
    qualification_policy_sha256: Sha256

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if set(self.configs) != set(self.method_config_sha256) or any(
                key != config.method_id or config.collection_manifest_sha256 != self.collection_manifest_sha256
                or self.method_config_sha256[key] != config.fingerprint for key, config in self.configs.items()):
            raise ValueError("SCOPED_RELEASE_CONFIG_BINDING")
        if self.graph_source_document_count > self.document_count:
            raise ValueError("SCOPED_GRAPH_COVERAGE_COUNT")
        return self


class ScopedRunMessageV2(Record):
    schema_version: Literal[2] = 2
    tenant_id: Annotated[str, StringConstraints(pattern="^"+TENANT+"$")]
    owner_id: Annotated[str, StringConstraints(pattern="^"+SUB+"$")]
    run_id: Annotated[str, StringConstraints(pattern="^"+RUN_ID+"$")]
    method_id: Mode
    admission_token: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$")]


class ScopedSourceV2(Record):
    source_id: Annotated[str, StringConstraints(pattern=r"^s(?:[1-9]|1[0-2])$")]
    chunk_id: Nonempty
    document_version_id: Nonempty
    shard_id: Nonempty
    shard_manifest_sha256: Sha256
    source_uri: Nonempty
    title: Nonempty
    section_path: tuple[str, ...]
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(gt=0)]
    text_sha256: Sha256
    text: Annotated[str, StringConstraints(min_length=1, max_length=20000)]

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.end_char-self.start_char != len(self.text) or sha256_bytes(self.text.encode()) != self.text_sha256:
            raise ValueError("SCOPED_SOURCE_BYTES")
        return self


class ScopedReviewFlagV2(Record):
    claim_id: Annotated[str, StringConstraints(pattern=r"^c[1-8]$")]
    rule_id: Literal["INVOCATION_PREMISE_NOT_EXPLICIT_IN_CITATIONS", "ANTECEDENT_AVAILABLE_BUT_NOT_CITED",
        "MISSING_ANTECEDENT_IN_CONTEXT", "AMBIGUOUS_QUOTE_LOCATION_REQUIRES_REVIEW"]
    citation_ids: tuple[Annotated[str, StringConstraints(pattern=r"^profiled-cite-v12_[0-9a-f]{64}$")], ...]


class ScopedBundleV2(Record):
    bundle_id: Nonempty
    chunk_ids: Annotated[tuple[Nonempty, ...], Field(min_length=1, max_length=12)]
    assertion_ids: Annotated[tuple[Nonempty, ...], Field(max_length=2)] = ()
    selection: Literal["SELECTED", "MAX_CHUNKS", "TOKEN_BUDGET"]


class ScopedRetrievalTraceV2(Record):
    mechanism_status: Literal["READY", "READY_NO_EVIDENCE_ABSTENTION", "NOT_READY_NO_DISTINCT_NATIVE_FACETS",
        "NOT_READY_GRAPH_ZERO_EXPANSION", "NOT_READY_GRAPH_PATH_NOT_PACKED"]
    queries: Annotated[tuple[Annotated[str, StringConstraints(min_length=1, max_length=2000)], ...], Field(min_length=1, max_length=4)]
    query_rankings: Annotated[tuple[Annotated[tuple[Nonempty, ...], Field(max_length=40)], ...], Field(max_length=4)] = ()
    pool_fingerprint: Sha256
    bundles: Annotated[tuple[ScopedBundleV2, ...], Field(max_length=40)]
    balanced_allocator_used: bool
    planner_origin: Literal["FIXTURE", "NATIVE_V6"] | None = None
    planner_operation_id: Nonempty | None = None
    planner_receipt_sha256: Sha256 | None = None
    hidden_fallbacks: Literal[0] = 0
    partial_source_truncations: Literal[0] = 0


class ScopedStageEventV2(Record):
    stage: Literal["shared_search_translation", "shared_native_task_facets", "retrieval_and_atomic_source_packing",
        "generation_and_display_qualification", "mechanism_not_ready_no_dispatch"]
    duration_ms: Annotated[float, Field(ge=0)]
    status: Literal["COMPLETED"] = "COMPLETED"
    input_sha256: Sha256
    output_count: Annotated[int, Field(ge=0)]


class ScopedGraphNodeV2(Record):
    node_id: Nonempty
    name: Annotated[str, StringConstraints(min_length=1, max_length=2000)]
    identity_scope: Nonempty
    role: str | None = None


class ScopedGraphFactV2(Record):
    assertion_id: Nonempty
    chunk_id: Nonempty
    kind: Nonempty
    clause: Annotated[str, StringConstraints(min_length=1, max_length=20000)]
    nodes: Annotated[tuple[ScopedGraphNodeV2, ...], Field(min_length=2, max_length=12)]
    links: Annotated[tuple[tuple[Nonempty, Nonempty, Nonempty], ...], Field(min_length=1, max_length=132)]
    unresolved_references_present: bool
    permission_granted: Literal[False] = False
    interpretation: Literal["RETRIEVAL_SOURCE_ASSOCIATIONS_NOT_ACCESS_GRANTS"] = "RETRIEVAL_SOURCE_ASSOCIATIONS_NOT_ACCESS_GRANTS"


class ScopedMethodResultV2(Record):
    schema_version: Literal[2] = 2
    status: Literal["COMPLETED"] = "COMPLETED"
    tenant_id: Annotated[str, StringConstraints(pattern="^"+TENANT+"$")]
    owner_id: Annotated[str, StringConstraints(pattern="^"+SUB+"$")]
    run_id: Annotated[str, StringConstraints(pattern="^"+RUN_ID+"$")]
    request_id: Nonempty
    original_question_sha256: Sha256
    answer_language: Literal["en", "vi"]
    requested_method: Mode
    effective_method: Mode
    config: ScopedMethodConfigV2
    config_sha256: Sha256
    collection_version: Nonempty
    ranked_chunk_ids: Annotated[tuple[Nonempty, ...], Field(max_length=40)]
    context_chunk_ids: Annotated[tuple[Nonempty, ...], Field(max_length=12)]
    context_proxy_tokens: Annotated[int, Field(ge=0, le=4096)]
    context_sources: Annotated[tuple[ScopedSourceV2, ...], Field(max_length=12)]
    source_context_sha256: Sha256
    qualification_policy_sha256: Sha256
    qualification: Literal["NO_MECHANICAL_FLAGS", "REVIEW_REQUIRED", "NOT_READY"]
    response: QueryResponse | None = None
    native_answer_status: Literal["ANSWERED", "PARTIAL", "ABSTAINED"] | None = None
    review_flags: tuple[ScopedReviewFlagV2, ...] = ()
    used_graph: bool = False
    answer_origin: Literal["NATIVE_RECEIPT", "SERVER_DETERMINISTIC_ABSTENTION", "NOT_READY"]
    semantic_verified: Literal[False] = False
    cache_hit: bool = False
    new_cost_micro_usd: Annotated[int, Field(ge=0)] = 0
    original_receipt_cost_micro_usd: Annotated[int, Field(ge=0)] | None = None
    usage: TokenUsage | None = None
    trace: Annotated[tuple[ScopedStageEventV2, ...], Field(max_length=5)]
    retrieval: ScopedRetrievalTraceV2
    graph_evidence: Annotated[tuple[ScopedGraphFactV2, ...], Field(max_length=40)] = ()

    @model_validator(mode="after")
    def integrity(self) -> Self:
        if self.requested_method != self.effective_method or self.config.method_id != self.effective_method:
            raise ValueError("SCOPED_RESULT_METHOD_CHANGED")
        if self.config_sha256 != self.config.fingerprint:
            raise ValueError("SCOPED_RESULT_CONFIG_HASH")
        if self.qualification_policy_sha256 != policy_fingerprint():
            raise ValueError("SCOPED_RESULT_POLICY_STALE")
        if len(set(self.ranked_chunk_ids)) != len(self.ranked_chunk_ids) or (
                tuple(source.chunk_id for source in self.context_sources) != self.context_chunk_ids
                or len(set(self.context_chunk_ids)) != len(self.context_chunk_ids)
                or not set(self.context_chunk_ids) <= set(self.ranked_chunk_ids)):
            raise ValueError("SCOPED_RESULT_SOURCE_ORDER")
        if any(source.source_id != f"s{i}" for i, source in enumerate(self.context_sources, 1)):
            raise ValueError("SCOPED_RESULT_SOURCE_IDS")
        flagged = self.qualification == "REVIEW_REQUIRED"
        ready = self.qualification != "NOT_READY"
        if flagged != bool(self.review_flags) or (self.response is not None) != (ready and not flagged):
            raise ValueError("SCOPED_RESULT_DISPLAY_STATE")
        if not ready and (self.answer_origin != "NOT_READY" or self.native_answer_status is not None
                or self.usage is not None or self.new_cost_micro_usd or self.original_receipt_cost_micro_usd is not None):
            raise ValueError("SCOPED_RESULT_NOT_READY_GENERATION")
        if ready != self.retrieval.mechanism_status.startswith("READY") or ready and (
                self.answer_origin == "NOT_READY" or self.native_answer_status is None):
            raise ValueError("SCOPED_RESULT_MECHANISM_STATE")
        if self.cache_hit and self.new_cost_micro_usd:
            raise ValueError("SCOPED_RESULT_REPLAY_COST")
        for bundle in self.retrieval.bundles:
            if len(set(bundle.chunk_ids)) != len(bundle.chunk_ids) or not set(bundle.chunk_ids) <= set(self.ranked_chunk_ids):
                raise ValueError("SCOPED_RESULT_BUNDLE_POOL")
            if bundle.selection == "SELECTED" and not set(bundle.chunk_ids) <= set(self.context_chunk_ids):
                raise ValueError("SCOPED_RESULT_BUNDLE_ATOMIC_SUPPORT")
            if bundle.assertion_ids and not self.effective_method.startswith("graph_") or (
                    self.effective_method == "graph_1hop" and len(bundle.assertion_ids) > 1):
                raise ValueError("SCOPED_RESULT_GRAPH_HOP_OR_METHOD")
        if self.used_graph != any(b.assertion_ids and b.selection == "SELECTED" for b in self.retrieval.bundles):
            raise ValueError("SCOPED_RESULT_GRAPH_CONTRIBUTION")
        expected_facts = {key for b in self.retrieval.bundles if b.selection == "SELECTED" for key in b.assertion_ids}
        if len({f.assertion_id for f in self.graph_evidence}) != len(self.graph_evidence) or (
                {f.assertion_id for f in self.graph_evidence} != expected_facts):
            raise ValueError("SCOPED_RESULT_GRAPH_MAP_FACTS")
        for fact in self.graph_evidence:
            source = next((source for source in self.context_sources if source.chunk_id == fact.chunk_id), None)
            ids = {node.node_id for node in fact.nodes}
            if source is None or fact.clause not in source.text or len(ids) != len(fact.nodes) or any(
                    left not in ids or right not in ids for left, right, _ in fact.links):
                raise ValueError("SCOPED_RESULT_GRAPH_MAP_SOURCE")
        if ready and self.effective_method.startswith("graph_") and not self.used_graph:
            raise ValueError("SCOPED_RESULT_GRAPH_NO_FALLBACK")
        if ready and self.effective_method in Q_METHODS and len(self.retrieval.queries) < 2:
            raise ValueError("SCOPED_RESULT_NATIVE_FACETS_REQUIRED")
        if self.effective_method in Q_METHODS and (self.retrieval.planner_origin is None
                or self.retrieval.planner_operation_id is None or self.retrieval.planner_receipt_sha256 is None
                or not self.config.fixture_only and self.retrieval.planner_origin != "NATIVE_V6"):
            raise ValueError("SCOPED_RESULT_NATIVE_FACET_RECEIPT")
        if self.retrieval.balanced_allocator_used != (self.effective_method == "hybrid_multi_query_balanced"):
            raise ValueError("SCOPED_RESULT_ALLOCATOR")
        if self.response is not None:
            response = self.response
            if (response.request_id != self.request_id or response.mode_requested != self.effective_method
                    or response.mode_used != self.effective_method or response.used_graph != self.used_graph
                    or response.evidence_bundle_sha256 != self.source_context_sha256
                    or response.answer_language != self.answer_language or response.answer_status != self.native_answer_status):
                raise ValueError("SCOPED_RESULT_RESPONSE_BINDING")
            sources = {source.chunk_id: source for source in self.context_sources}
            for citation in response.citations:
                source = sources.get(citation.chunk_id)
                if source is None or source.text.count(citation.quote) != 1 or (
                        source.document_version_id != citation.document_version_id or source.title != citation.title
                        or source.source_uri != citation.source_uri or source.section_path != citation.section_path):
                    raise ValueError("SCOPED_RESULT_CITATION_SOURCE")
                start = source.start_char+source.text.index(citation.quote)
                expected = stable_id("profiled-cite-v12", tenant=self.tenant_id, owner=self.owner_id,
                    request=self.request_id, context=self.source_context_sha256, chunk=source.chunk_id,
                    start=start, end=start+len(citation.quote))
                if citation.citation_id != expected:
                    raise ValueError("SCOPED_RESULT_CITATION_OWNER")
        return self


def scoped_request_id(parent, method):
    config = ScopedRunReleaseV2.model_validate_json(canonical_json(parent["release"])).configs[method]
    return stable_id("astage-scoped-v14", run=parent["run_id"], owner=parent["owner_id"],
        slot="GENERATE:"+method, recipe=config.generation_recipe.fingerprint,
        budget_run=parent["release"]["budget_run_id"])


def validate_scoped_result(parent, method, raw):
    result = ScopedMethodResultV2.model_validate_json(canonical_json(raw))
    release = ScopedRunReleaseV2.model_validate_json(canonical_json(parent["release"]))
    request = ScopedRunRequestV2.model_validate_json(canonical_json(parent["request"]))
    from app.query_language import answer_language

    if (method not in request.method_ids or result.effective_method != method
            or result.run_id != parent["run_id"] or result.owner_id != parent["owner_id"]
            or result.tenant_id != parent["tenant_id"] or result.request_id != scoped_request_id(parent, method)
            or result.original_question_sha256 != sha256_bytes(request.original_question.encode())
            or result.answer_language != answer_language(request.original_question, request.language)
            or result.config != release.configs[method] or result.config_sha256 != release.method_config_sha256[method]
            or result.collection_version != release.collection_version
            or result.qualification_policy_sha256 != release.qualification_policy_sha256):
        raise ValueError("SCOPED_RUN_RESULT_PARENT_BINDING")
    return result
