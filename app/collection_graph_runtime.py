"""Explicitly experimental, source-reviewed graph retrieval over a pinned collection.

The overlay is not a replacement snapshot. Sources/vectors remain in the original
collection. AI curation is never represented as independent human gold labels.
"""

from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Annotated, Literal

from pydantic import Field

from app.answer_pipeline import PreparedAnswer
from app.collection_answering import _evidence_view, validate_collection_lexical_binding
from app.collection_graph import expand_graph, graph_fingerprint
from app.context import pack_context
from app.contracts import (
    Entity,
    Mention,
    QueryRequest,
    Record,
    RelationAssertion,
    Sha256,
    canonical_json,
    sha256_bytes,
    stable_id,
)
from app.external_http import strict_json
from app.extraction import normalize_alias
from app.generation import AnswerBinding, GenerationError
from app.graph_retrieval import GraphCandidate, GraphRetrievalResult
from app.model_gateway import CallContext
from app.retrieval import reciprocal_rank_fusion
from app.snapshot import SnapshotContent


class CollectionGraphOverlayV1(Record):
    schema_version: Literal[1] = 1
    kind: Literal["COLLECTION_GRAPH_AI_REVIEWED_EXPERIMENTAL"] = "COLLECTION_GRAPH_AI_REVIEWED_EXPERIMENTAL"
    tenant_id: str
    collection_version: str
    collection_manifest_sha256: Sha256
    graph_sha256: Sha256
    curation_sha256: Sha256
    upstream_review_queue_sha256: Sha256
    human_reviewed: Literal[False] = False
    full_relation_extraction: Literal[False] = False
    quality_metrics: None = None
    entities: Annotated[tuple[Entity, ...], Field(max_length=1000)]
    mentions: Annotated[tuple[Mention, ...], Field(max_length=100000)]
    assertions: Annotated[tuple[RelationAssertion, ...], Field(max_length=5000)]


@dataclass(frozen=True)
class ValidatedCollectionGraph:
    overlay: CollectionGraphOverlayV1
    file_sha256: str


def validate_graph_sources(overlay, collection):
    if overlay.tenant_id != collection.manifest.tenant_id or (
        overlay.collection_manifest_sha256 != collection.manifest_sha256
        or overlay.collection_version != collection.manifest.collection_version
    ):
        raise GenerationError("COLLECTION_GRAPH_RELEASE_BINDING")
    if overlay.graph_sha256 != graph_fingerprint(collection.manifest_sha256,
        overlay.entities, overlay.mentions, overlay.assertions):
        raise GenerationError("COLLECTION_GRAPH_CONTENT_FINGERPRINT")
    records = ((overlay.entities, "entity_id"), (overlay.mentions, "mention_id"), (overlay.assertions, "assertion_id"))
    for rows, field in records:
        if len({getattr(row, field) for row in rows}) != len(rows) or any(row.tenant_id != overlay.tenant_id for row in rows):
            raise GenerationError("COLLECTION_GRAPH_RECORD_ID_OR_TENANT")
    entities = {row.entity_id: row for row in overlay.entities}
    aliases = {key: {normalize_alias(value) for value in (*e.aliases, e.canonical_name)} for key, e in entities.items()}
    docs, chunks = {}, {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        docs.update(shard.documents)
        chunks.update(shard.chunks)
    for mention in overlay.mentions:
        chunk = chunks.get(mention.chunk_id)
        if chunk is None or mention.entity_id not in entities or chunk.document_version_id != mention.document_version_id:
            raise GenerationError("COLLECTION_GRAPH_MENTION_BINDING")
        if not chunk.start_char <= mention.start_char < mention.end_char <= chunk.end_char or (
            docs[mention.document_version_id].normalized_text[mention.start_char:mention.end_char] != mention.quote
        ):
            raise GenerationError("COLLECTION_GRAPH_MENTION_SOURCE")
        if normalize_alias(mention.quote) not in aliases[mention.entity_id]:
            raise GenerationError("COLLECTION_GRAPH_MENTION_ALIAS")
    for assertion in overlay.assertions:
        chunk = chunks.get(assertion.chunk_id)
        evidence = assertion.evidence
        if chunk is None or chunk.document_version_id != assertion.document_version_id or (
            {assertion.subject_id, assertion.object_id} - entities.keys()
        ):
            raise GenerationError("COLLECTION_GRAPH_ASSERTION_BINDING")
        if not chunk.start_char <= evidence.start_char < evidence.end_char <= chunk.end_char or (
            docs[assertion.document_version_id].normalized_text[evidence.start_char:evidence.end_char] != evidence.quote
            or any(q.text not in evidence.quote for q in assertion.qualifiers)
        ):
            raise GenerationError("COLLECTION_GRAPH_ASSERTION_SOURCE")
        for endpoint in (assertion.subject_id, assertion.object_id):
            if not any(m.entity_id == endpoint and m.chunk_id == assertion.chunk_id
                and evidence.start_char <= m.start_char < m.end_char <= evidence.end_char for m in overlay.mentions):
                raise GenerationError("COLLECTION_GRAPH_ENDPOINT_SOURCE")


def load_collection_graph(path: Path, collection, *, trusted_sha256: str, allow_experimental=False):
    if allow_experimental is not True:
        raise GenerationError("COLLECTION_GRAPH_EXPERIMENTAL_OPT_IN_REQUIRED")
    if path.is_symlink() or not path.is_file() or not 1 <= path.stat().st_size <= 32 * 1024 * 1024:
        raise GenerationError("COLLECTION_GRAPH_FILE_BOUND")
    with path.open("rb") as handle:
        raw = handle.read(32 * 1024 * 1024 + 1)
    if len(raw) > 32 * 1024 * 1024:
        raise GenerationError("COLLECTION_GRAPH_FILE_BOUND")
    if sha256_bytes(raw) != trusted_sha256:
        raise GenerationError("COLLECTION_GRAPH_FILE_CHECKSUM")
    overlay = CollectionGraphOverlayV1.model_validate_json(canonical_json(strict_json(raw)))
    validate_graph_sources(overlay, collection)
    return ValidatedCollectionGraph(overlay, trusted_sha256)


def prepare_collection_graph_answer(collection, lexical, graph, question, query_vector, *, tenant_id,
    owner_id, run_id, recipe, hops=1, search_query=None, allow_experimental=False, time_budget_ms=300, clock=None):
    if allow_experimental is not True or type(graph) is not ValidatedCollectionGraph:
        raise GenerationError("COLLECTION_GRAPH_EXPERIMENTAL_OPT_IN_REQUIRED")
    if type(hops) is not int or hops not in (1, 2):
        raise GenerationError("COLLECTION_GRAPH_HOP_BOUND")
    requested = f"graph_{hops}hop"
    request = QueryRequest(question=question, mode=requested)
    search = request.question if search_query is None else QueryRequest(question=search_query, mode=requested).question
    overlay = graph.overlay
    if tenant_id != collection.manifest.tenant_id or overlay.tenant_id != tenant_id or (
        overlay.collection_manifest_sha256 != collection.manifest_sha256
        or overlay.collection_version != collection.manifest.collection_version
    ):
        raise GenerationError("COLLECTION_GRAPH_RELEASE_BINDING")
    validate_collection_lexical_binding(collection, lexical)
    vector_ids = tuple(hit.chunk_id for hit in collection.search_vector(query_vector, k=20))
    lexical_ids = tuple(key for key, _ in lexical.search(search, k=20))
    baseline = tuple(reciprocal_rank_fusion([vector_ids, lexical_ids]))
    scores = {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        scores.update((key, float(value)) for key, value in zip(shard.chunk_ids, shard.vectors @ query_vector, strict=True))
    extra = {"clock": clock} if clock is not None else {}
    trace = expand_graph(overlay.entities, overlay.mentions, overlay.assertions,
        baseline, scores, hops=hops, time_budget_ms=time_budget_ms, **extra)
    has_paths = any(c["assertion_path"] for c in trace["graph_candidates"])
    warnings = ["GRAPH_AI_REVIEWED_EXPERIMENTAL", "GRAPH_INCOMPLETE_RELATION_COVERAGE"]
    if trace["timed_out"] or not has_paths:
        warnings.append("GRAPH_TIMEOUT" if trace["timed_out"] else "GRAPH_NO_RELATION_PATH")
        ranked, candidates, effective = baseline, (), "hybrid"
    else:
        # Three independent rank lists: Hybrid is not counted as a single vote.
        ranked = tuple(reciprocal_rank_fusion([vector_ids, lexical_ids,
            [c["chunk_id"] for c in trace["graph_candidates"]]]))[:40]
        candidates = tuple(GraphCandidate(**c) for c in trace["graph_candidates"]
            if c["chunk_id"] in ranked and set(c["supporting_chunk_ids"]) <= set(ranked))
        effective = requested
    if trace["truncated"]:
        warnings.append("GRAPH_TRUNCATED")
    view = _evidence_view(collection, lexical, ranked)
    view = replace(view, entities=MappingProxyType({e.entity_id: e for e in overlay.entities}),
        content=SnapshotContent((), (), overlay.entities, overlay.mentions, overlay.assertions))
    result = GraphRetrievalResult(ranked, baseline, candidates, tuple(trace["seed_entity_ids"]),
        effective, effective != "hybrid", tuple(warnings), trace["assertion_visits"], trace["visited_entities"])
    packed = pack_context(view, result)
    if effective != "hybrid" and not packed.assertion_ids:
        # No relation evidence survived packing: return the exact Hybrid baseline.
        warnings.append("GRAPH_PATH_NOT_PACKED")
        effective, ranked = "hybrid", baseline
        view = _evidence_view(collection, lexical, ranked)
        result = GraphRetrievalResult(ranked, baseline, (), tuple(trace["seed_entity_ids"]),
            effective, False, tuple(warnings), trace["assertion_visits"], trace["visited_entities"])
        packed = pack_context(view, result)
    request_id = stable_id("qgraph", tenant=tenant_id, owner=owner_id, run_id=run_id,
        question=request.question, search_query=search, requested_method=requested,
        graph_file_sha256=graph.file_sha256, lexical_index=lexical.manifest.index_version,
        time_budget_ms=time_budget_ms, query_vector_sha256=sha256_bytes(query_vector.tobytes(order="C")), generation_recipe=recipe.fingerprint)
    binding = AnswerBinding(request_id=request_id, tenant_id=tenant_id, owner_id=owner_id,
        mode_requested=requested, mode_used=effective, used_graph=effective != "hybrid", warnings=tuple(warnings) + packed.warnings)
    prepared = PreparedAnswer(CallContext(tenant_id=tenant_id, run_id=run_id, request_id=request_id, stage="GENERATE"),
        binding, packed, request.question, recipe, ranked)
    trace.update(vector_ranking=vector_ids, lexical_ranking=lexical_ids, baseline_ranking=baseline,
        final_ranking=ranked, context_chunk_ids=packed.chunk_ids, context_assertion_ids=packed.assertion_ids,
        context_proxy_tokens=packed.proxy_tokens, warnings=tuple(warnings) + packed.warnings,
        requested_method=requested, mode_used=effective, used_graph=binding.used_graph,
        graph_sha256=overlay.graph_sha256, graph_file_sha256=graph.file_sha256,
        question=request.question, search_query=search)
    return view, prepared, trace
