"""Fixed local context ordering within an already authorized multi-query pool.

Round-robin is an allocation heuristic, not a relevance scorer or a proof of
evidence coverage. Both policies first compute the same complete ID ordering.
The scoped exact-text policy then suppresses aliases only within one document
version and section; it never merges similar/overlapping passages or qualifiers.
No retrieval, model invocation, source expansion or token packing happens here.
"""

from collections.abc import Sequence
from typing import Annotated, Literal, Self

from pydantic import Field, TypeAdapter, model_validator

from app.contracts import Chunk, Record, Sha256, TenantId, canonical_json, sha256_bytes
from app.multi_query import MultiQueryTraceV1

SELECTION_RECIPE_VERSION = "bounded-context-round-robin-v1"
SelectionPolicy = Literal["RR_ID_ONLY", "RR_SCOPED_EXACT_DEDUP"]
PositiveRank = Annotated[int, Field(ge=1, le=40)]
QueryIndex = Annotated[int, Field(ge=0, le=2)]


class ContextSelectionRecipe(Record):
    version: Literal["bounded-context-round-robin-v1"] = SELECTION_RECIPE_VERSION
    policy: SelectionPolicy = "RR_ID_ONLY"
    query_order: Literal["BASE_THEN_SUBQUERY_INPUT_ORDER"] = "BASE_THEN_SUBQUERY_INPUT_ORDER"
    cursor_policy: Literal["SKIP_SEEN_IDS_AND_OUTSIDE_FROZEN_POOL"] = "SKIP_SEEN_IDS_AND_OUTSIDE_FROZEN_POOL"
    allocation: Literal["ONE_UNSEEN_POOL_ID_PER_QUERY_PER_ROUND"] = "ONE_UNSEEN_POOL_ID_PER_QUERY_PER_ROUND"
    exhaustion: Literal["ALL_FROZEN_POOL_IDS_REQUIRED_NO_FALLBACK"] = "ALL_FROZEN_POOL_IDS_REQUIRED_NO_FALLBACK"
    duplicate_scope: Literal["SAME_DOCUMENT_VERSION_SECTION_EXACT_UTF8"] = "SAME_DOCUMENT_VERSION_SECTION_EXACT_UTF8"
    duplicate_representative: Literal["FIRST_IN_COMPLETE_ID_ROUND_ROBIN"] = "FIRST_IN_COMPLETE_ID_ROUND_ROBIN"
    duplicate_filter_stage: Literal["AFTER_COMPLETE_ID_ROUND_ROBIN"] = "AFTER_COMPLETE_ID_ROUND_ROBIN"
    max_pool_candidates: Literal[40] = 40
    max_queries: Literal[3] = 3
    max_candidates_per_query: Literal[40] = 40
    max_transient_occurrences: Literal[120] = 120
    query_index_basis: Literal[0] = 0
    rank_and_round_basis: Literal[1] = 1
    source_offsets: Literal["ORIGINAL_DOCUMENT_UNICODE"] = "ORIGINAL_DOCUMENT_UNICODE"

    @property
    def fingerprint(self) -> str:
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


class QueryOccurrence(Record):
    query_index: QueryIndex
    query_rank: PositiveRank
    kind: Literal["BASE", "SUBQUERY"]


class CandidateProvenance(Record):
    chunk_id: str
    pool_rank: PositiveRank
    text_sha256: Sha256
    document_version_id: str
    section_path: tuple[str, ...]
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(gt=0)]
    query_occurrences: Annotated[tuple[QueryOccurrence, ...], Field(min_length=1, max_length=3)]
    representative_chunk_id: str


class SelectionDecision(Record):
    chunk_id: str
    pool_rank: PositiveRank
    round_robin_rank: PositiveRank
    round_index: PositiveRank
    query_index: QueryIndex
    query_rank: PositiveRank
    output_rank: PositiveRank | None
    representative_chunk_id: str


class DuplicateGroup(Record):
    representative_chunk_id: str
    alias_chunk_ids: Annotated[tuple[str, ...], Field(min_length=1, max_length=39)]
    document_version_id: str
    section_path: tuple[str, ...]
    text_sha256: Sha256


class OutsidePoolOccurrence(Record):
    chunk_id: str
    query_index: QueryIndex
    query_rank: PositiveRank


class ContextSelectionResult(Record):
    schema_version: Literal[1] = 1
    recipe: ContextSelectionRecipe
    recipe_sha256: Sha256
    source_trace_sha256: Sha256
    original_pool_ids: Annotated[tuple[str, ...], Field(max_length=40)]
    round_robin_chunk_ids: Annotated[tuple[str, ...], Field(max_length=40)]
    ordered_chunk_ids: Annotated[tuple[str, ...], Field(max_length=40)]
    candidates: Annotated[tuple[CandidateProvenance, ...], Field(max_length=40)]
    decisions: Annotated[tuple[SelectionDecision, ...], Field(max_length=40)]
    duplicates: Annotated[tuple[DuplicateGroup, ...], Field(max_length=20)]
    skipped_outside_pool: Annotated[tuple[OutsidePoolOccurrence, ...], Field(max_length=120)]

    @model_validator(mode="after")
    def identity_and_provenance(self) -> Self:
        pool, rr, ordered = self.original_pool_ids, self.round_robin_chunk_ids, self.ordered_chunk_ids
        if (self.recipe_sha256 != self.recipe.fingerprint or len(pool) != len(set(pool))
                or len(rr) != len(set(rr)) or set(rr) != set(pool)
                or len(ordered) != len(set(ordered)) or not set(ordered).issubset(pool)
                or tuple(row.chunk_id for row in self.candidates) != pool
                or tuple(row.chunk_id for row in self.decisions) != rr):
            raise ValueError("CONTEXT_SELECTION_POOL_OR_RECIPE_MISMATCH")
        candidates = {row.chunk_id: row for row in self.candidates}
        representatives = {}
        for row in self.candidates:
            representative = candidates.get(row.representative_chunk_id)
            if (row.pool_rank != pool.index(row.chunk_id) + 1 or representative is None
                    or representative.chunk_id not in ordered or row.end_char <= row.start_char):
                raise ValueError("CONTEXT_SELECTION_CANDIDATE_PROVENANCE_MISMATCH")
            if row.chunk_id != representative.chunk_id:
                if (self.recipe.policy != "RR_SCOPED_EXACT_DEDUP"
                        or (row.document_version_id, row.section_path, row.text_sha256)
                        != (representative.document_version_id, representative.section_path, representative.text_sha256)):
                    raise ValueError("CONTEXT_SELECTION_DUPLICATE_SCOPE_MISMATCH")
            representatives[row.chunk_id] = representative.chunk_id
        expected_order = tuple(key for key in rr if representatives[key] == key)
        if ordered != expected_order:
            raise ValueError("CONTEXT_SELECTION_OUTPUT_ORDER_MISMATCH")
        for rank, decision in enumerate(self.decisions, 1):
            candidate = candidates[decision.chunk_id]
            occurrences = {(row.query_index, row.query_rank) for row in candidate.query_occurrences}
            expected_output = ordered.index(decision.chunk_id) + 1 if decision.chunk_id in ordered else None
            if (decision.pool_rank != candidate.pool_rank or decision.round_robin_rank != rank
                    or (decision.query_index, decision.query_rank) not in occurrences
                    or decision.representative_chunk_id != representatives[decision.chunk_id]
                    or decision.output_rank != expected_output):
                raise ValueError("CONTEXT_SELECTION_DECISION_MISMATCH")
        expected_duplicates = []
        for key in ordered:
            aliases = tuple(other for other in rr if other != key and representatives[other] == key)
            if aliases:
                own = candidates[key]
                expected_duplicates.append(DuplicateGroup(representative_chunk_id=key, alias_chunk_ids=aliases,
                    document_version_id=own.document_version_id, section_path=own.section_path,
                    text_sha256=own.text_sha256))
        if tuple(expected_duplicates) != self.duplicates:
            raise ValueError("CONTEXT_SELECTION_DUPLICATE_TRACE_MISMATCH")
        if any(row.chunk_id in pool for row in self.skipped_outside_pool):
            raise ValueError("CONTEXT_SELECTION_OUTSIDE_POOL_MISMATCH")
        return self


def select_context_order(trace: MultiQueryTraceV1, chunks: Sequence[Chunk], *, tenant_id: str,
                         recipe: ContextSelectionRecipe | None = None) -> ContextSelectionResult:
    """Order exactly the frozen Q pool, then apply the explicitly chosen policy.

    Query indices start at zero; ranks and rounds start at one. Input chunk
    order has no effect. Source URI/title/document text validation remains the
    caller's responsibility; original Chunk identities and text are rechecked.
    """
    if not isinstance(trace, MultiQueryTraceV1):
        raise ValueError("CONTEXT_SELECTION_TRACE_REQUIRED")
    trace = MultiQueryTraceV1.model_validate(trace.model_dump())
    if recipe is None:
        recipe = ContextSelectionRecipe()
    if not isinstance(recipe, ContextSelectionRecipe):
        raise ValueError("CONTEXT_SELECTION_RECIPE_REQUIRED")
    recipe = ContextSelectionRecipe.model_validate(recipe.model_dump())
    tenant_id = TypeAdapter(TenantId).validate_python(tenant_id, strict=True)
    if not isinstance(chunks, Sequence) or isinstance(chunks, (str, bytes)) or len(chunks) > 40:
        raise ValueError("CONTEXT_SELECTION_CHUNK_INPUT_BOUND")
    pool = trace.ranked_chunk_ids
    by_id = {}
    for chunk in chunks:
        if not isinstance(chunk, Chunk):
            raise ValueError("CONTEXT_SELECTION_CHUNK_REQUIRED")
        chunk = Chunk.model_validate(chunk.model_dump())
        if chunk.tenant_id != tenant_id:
            raise ValueError("CONTEXT_SELECTION_TENANT_MISMATCH")
        if chunk.chunk_id in by_id:
            raise ValueError("CONTEXT_SELECTION_DUPLICATE_CHUNK_ID")
        by_id[chunk.chunk_id] = chunk
    if set(by_id) != set(pool):
        raise ValueError("CONTEXT_SELECTION_AUTHORIZED_POOL_MISMATCH")
    occurrences = {key: [] for key in pool}
    outside = []
    for query_index, query in enumerate(trace.queries):
        for query_rank, key in enumerate(query.ranked_chunk_ids, 1):
            if key in occurrences:
                occurrences[key].append(QueryOccurrence(query_index=query_index, query_rank=query_rank, kind=query.kind))
            else:
                outside.append(OutsidePoolOccurrence(chunk_id=key, query_index=query_index, query_rank=query_rank))
    if any(not own for own in occurrences.values()):
        raise ValueError("CONTEXT_SELECTION_POOL_NOT_COVERED_BY_QUERIES")

    cursors = [0] * len(trace.queries)
    seen, rr, raw_decisions = set(), [], []
    round_index = 0
    while len(rr) < len(pool):
        round_index += 1
        progressed = False
        for query_index, query in enumerate(trace.queries):
            while cursors[query_index] < len(query.ranked_chunk_ids):
                query_rank = cursors[query_index] + 1
                key = query.ranked_chunk_ids[cursors[query_index]]
                cursors[query_index] += 1
                if key not in by_id or key in seen:
                    continue
                seen.add(key)
                rr.append(key)
                raw_decisions.append((key, round_index, query_index, query_rank))
                progressed = True
                break
        if not progressed:
            raise ValueError("CONTEXT_SELECTION_POOL_EXHAUSTED_WITHOUT_COVERAGE")

    representatives, first_by_scope = {}, {}
    for key in rr:
        chunk = by_id[key]
        # Exact bytes, not merely a hash match: no normalization or fuzzy test.
        scope = (chunk.document_version_id, chunk.section_path, chunk.text.encode("utf-8"))
        representative = first_by_scope.setdefault(scope, key) if recipe.policy == "RR_SCOPED_EXACT_DEDUP" else key
        representatives[key] = representative
    ordered = tuple(key for key in rr if representatives[key] == key)
    candidates = tuple(CandidateProvenance(chunk_id=key, pool_rank=index, text_sha256=by_id[key].text_sha256,
        document_version_id=by_id[key].document_version_id, section_path=by_id[key].section_path,
        start_char=by_id[key].start_char, end_char=by_id[key].end_char,
        query_occurrences=tuple(occurrences[key]), representative_chunk_id=representatives[key])
        for index, key in enumerate(pool, 1))
    decisions = tuple(SelectionDecision(chunk_id=key, pool_rank=pool.index(key) + 1,
        round_robin_rank=index, round_index=round_number, query_index=query_index, query_rank=query_rank,
        output_rank=ordered.index(key) + 1 if key in ordered else None, representative_chunk_id=representatives[key])
        for index, (key, round_number, query_index, query_rank) in enumerate(raw_decisions, 1))
    duplicates = tuple(DuplicateGroup(representative_chunk_id=key,
        alias_chunk_ids=tuple(other for other in rr if other != key and representatives[other] == key),
        document_version_id=by_id[key].document_version_id, section_path=by_id[key].section_path,
        text_sha256=by_id[key].text_sha256)
        for key in ordered if any(other != key and representatives[other] == key for other in rr))
    return ContextSelectionResult(recipe=recipe, recipe_sha256=recipe.fingerprint,
        source_trace_sha256=sha256_bytes(canonical_json(trace.model_dump(mode="json"))), original_pool_ids=pool,
        round_robin_chunk_ids=tuple(rr), ordered_chunk_ids=ordered, candidates=candidates,
        decisions=decisions, duplicates=duplicates, skipped_outside_pool=tuple(outside))
