"""Bounded title-to-chunk recall and atomic original-document windows.

No source annotations, gold IDs, service-specific boosts, invented graph edges,
or LLM answers enter retrieval. Q/QB share the exact same prepared pool; only
bundle ordering differs. This is an opt-in local recipe, not a serving default.
"""

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from app.answer_pipeline import PreparedAnswer
from app.collection_answering import _evidence_view, validate_collection_lexical_binding
from app.context import PackedContext
from app.contracts import QueryRequest, Record, canonical_json, sha256_bytes, stable_id
from app.faceted_query_v4 import revalidate_plan
from app.generation import AnswerBinding, GenerationError, generation_request
from app.graph_collection_migration import collection_records
from app.ingestion.text import ProxyTokenizer
from app.model_gateway import CallContext
from app.retrieval import lexical_tokens, reciprocal_rank_fusion
from app.strict_collection_methods import STOPWORDS

POLICY = "LITERAL_FACETS_TITLE_RECALL_ATOMIC_DOCUMENT_WINDOWS_V1"


class DocumentRecallRecipe(Record):
    version: Literal["bounded-document-recall-v1"] = "bounded-document-recall-v1"
    radius: Literal[1, 2] = 1
    title_depth_per_query: Literal[4] = 4
    max_document_windows: Literal[6] = 6
    base_pool_reserved: Literal[16] = 16
    max_pool: Literal[40] = 40
    max_chunks: Literal[8] = 8
    user_message_proxy_tokens: Literal[4096] = 4096
    window_selection: Literal["FACETS_FIRST_ROUND_ROBIN_DOC_RANK"] = "FACETS_FIRST_ROUND_ROBIN_DOC_RANK"
    anchor_selection: Literal["BEST_DOCUMENT_CHUNK_COSINE_FOR_OWN_QUERY"] = "BEST_DOCUMENT_CHUNK_COSINE_FOR_OWN_QUERY"
    singleton_policy: Literal["NO_PARTIAL_WINDOW_MEMBERS_AS_SINGLETONS"] = "NO_PARTIAL_WINDOW_MEMBERS_AS_SINGLETONS"

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


@dataclass(frozen=True)
class RecallPool:
    collection_manifest_sha256: str
    question: str
    search: str
    language: str
    recipe: DocumentRecallRecipe
    pool_ids: tuple[str, ...]
    bundles_json: str
    trace_json: str

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json({"collection": self.collection_manifest_sha256, "question": self.question,
            "search": self.search, "language": self.language, "recipe": self.recipe.model_dump(mode="json"),
            "pool_ids": self.pool_ids, "bundles": json.loads(self.bundles_json), "trace": json.loads(self.trace_json)}))


def round_robin_bundles(bundles, query_order):
    groups = {query: [b for b in bundles if b["query_index"] == query] for query in query_order}
    if len(query_order) != len(set(query_order)) or any(b["query_index"] not in groups for b in bundles):
        raise GenerationError("DOCUMENT_RECALL_QUERY_ORDER")
    result = []
    for rank in range(max((len(group) for group in groups.values()), default=0)):
        for query in query_order:
            if rank < len(groups[query]):
                result.append(groups[query][rank])
    return result


def document_window(chunks, anchor_id, radius):
    if type(radius) is not int or radius not in (1, 2) or anchor_id not in chunks:
        raise GenerationError("DOCUMENT_WINDOW_BOUND_OR_ANCHOR")
    anchor = chunks[anchor_id]
    own = sorted((c for c in chunks.values() if c.document_version_id == anchor.document_version_id),
        key=lambda c: (c.start_char, c.end_char, c.chunk_id))
    index = next(i for i, chunk in enumerate(own) if chunk.chunk_id == anchor_id)
    return tuple(c.chunk_id for c in own[max(0, index - radius):index + radius + 1])


def _vector(vector, dimension):
    if (not isinstance(vector, np.ndarray) or vector.dtype != np.dtype("float32") or vector.shape != (dimension,)
            or not np.isfinite(vector).all() or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5, rtol=0)):
        raise GenerationError("DOCUMENT_RECALL_QUERY_VECTOR")


def build_recall_pool(collection, lexical, *, question, search, language, plan, vectors, recipe=None):
    if QueryRequest(question=question).question != question or QueryRequest(question=search).question != search:
        raise GenerationError("DOCUMENT_RECALL_CANONICAL_QUERY_REQUIRED")
    recipe = DocumentRecallRecipe.model_validate_json((recipe or DocumentRecallRecipe()).model_dump_json())
    plan = revalidate_plan(plan, collection=collection, question=question, search=search, language=language)
    validate_collection_lexical_binding(collection, lexical)
    queries = (search, *(f.search_query for f in plan.facets))
    if len(vectors) != len(queries):
        raise GenerationError("DOCUMENT_RECALL_QUERY_VECTOR_COUNT")
    for vector in vectors:
        _vector(vector, collection.manifest.embedding_recipe.dimension)
    docs, chunks, document_vectors = collection_records(collection)
    doc_ids = sorted(docs)
    title_tokens = [tuple(t for t in lexical_tokens(docs[key].title) if t not in STOPWORDS) for key in doc_ids]
    title_bm25 = BM25Okapi(title_tokens) if any(title_tokens) else None
    title_sets = dict(zip(doc_ids, map(set, title_tokens), strict=True))
    query_rows, title_rows = [], []
    for i, (query, vector) in enumerate(zip(queries, vectors, strict=True)):
        vector_ids = tuple(hit.chunk_id for hit in collection.search_vector(vector, k=20))
        lexical_ids = tuple(key for key, _ in lexical.search(query, k=20))
        fused = tuple(reciprocal_rank_fusion([vector_ids, lexical_ids]))
        query_rows.append({"query_index": i, "query": query, "vector_ranking": vector_ids,
            "lexical_ranking": lexical_ids, "ranking": fused,
            "query_vector_sha256": sha256_bytes(vector.tobytes(order="C"))})
        terms = tuple(t for t in lexical_tokens(query) if t not in STOPWORDS)
        scores = title_bm25.get_scores(terms) if title_bm25 else np.zeros(len(doc_ids))
        eligible = [j for j, key in enumerate(doc_ids) if title_sets[key] & set(terms)]
        title_order = sorted(eligible, key=lambda j: (-float(scores[j]), doc_ids[j]))[:recipe.title_depth_per_query]
        title_rows.append({"query_index": i, "query": query, "document_ranking": [doc_ids[j] for j in title_order]})
    base = tuple(reciprocal_rank_fusion([row["ranking"] for row in query_rows]))[:40]
    query_order = (*range(1, len(queries)), 0)
    proposals, duplicate_windows = [], set()
    for doc_rank in range(recipe.title_depth_per_query):
        for qi in query_order:
            ranked = title_rows[qi]["document_ranking"]
            if doc_rank >= len(ranked):
                continue
            doc_id = ranked[doc_rank]
            own = [c for c in chunks.values() if c.document_version_id == doc_id]
            anchor = min(own, key=lambda c: (-float(document_vectors[c.chunk_id] @ vectors[qi]), c.chunk_id))
            window = document_window(chunks, anchor.chunk_id, recipe.radius)
            if window in duplicate_windows:
                continue
            duplicate_windows.add(window)
            proposals.append({"bundle_id": stable_id("original-document-window-v1", collection=collection.manifest_sha256,
                query=queries[qi], anchor=anchor.chunk_id, radius=recipe.radius, chunks=window),
                "kind": "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT", "query_index": qi, "document_rank": doc_rank + 1,
                "document_version_id": doc_id, "anchor_chunk_id": anchor.chunk_id, "chunk_ids": window})
    selected = list(base[:recipe.base_pool_reserved])
    bundles, decisions = [], []
    for proposed in proposals:
        combined = tuple(dict.fromkeys((*selected, *proposed["chunk_ids"])))
        admitted = len(bundles) < recipe.max_document_windows and len(combined) <= recipe.max_pool
        decisions.append({"bundle_id": proposed["bundle_id"], "admitted": admitted, "proposed_pool_size": len(combined)})
        if admitted:
            selected = list(combined)
            bundles.append(proposed)
    for key in base:
        if key not in selected and len(selected) < recipe.max_pool:
            selected.append(key)
    window_members = {key for bundle in bundles for key in bundle["chunk_ids"]}
    for key in selected:
        if key not in window_members:
            owner = min((row for row in query_rows if key in row["ranking"]),
                key=lambda row: (row["ranking"].index(key), row["query_index"]))["query_index"]
            bundles.append({"bundle_id": stable_id("recall-singleton", chunk=key), "kind": "RANKED_ORIGINAL_CHUNK",
                "query_index": owner, "chunk_ids": (key,), "anchor_chunk_id": key,
                "document_version_id": chunks[key].document_version_id, "document_rank": 99})
    # Stable fusion order is Q; QB uses the same bundles/pool in query rounds.
    scores = {key: rank for rank, key in enumerate(reciprocal_rank_fusion([r["ranking"] for r in query_rows]), 1)}
    bundles.sort(key=lambda b: (min((scores.get(key, 161) for key in b["chunk_ids"]), default=161), b["bundle_id"]))
    trace = {"policy": POLICY, "recipe": recipe.model_dump(mode="json"), "plan": plan.model_dump(mode="json"),
        "query_rankings": query_rows, "title_rankings": title_rows, "query_order": query_order,
        "original_fused_pool": base, "pool_chunk_ids": selected, "pool_admission": decisions,
        "source_annotation_inputs": False, "max_transient_query_occurrences": 160,
        "title_index_sha256": sha256_bytes(canonical_json([[k, docs[k].title] for k in doc_ids])),
        "new_graph_records": 0, "hidden_fallbacks": 0, "human_reviewed": 0}
    if (len(selected) > 40 or len(set(selected)) != len(selected)
            or any(not set(b["chunk_ids"]) <= set(selected) for b in bundles)):
        raise GenerationError("DOCUMENT_RECALL_POOL_OR_WINDOW_ADMISSION")
    return RecallPool(collection.manifest_sha256, question, search, language, recipe, tuple(selected),
        canonical_json(bundles).decode(), canonical_json(trace).decode())


def _full_rendering(view, keys, prompt):
    sources = {s["source_id"]: s for s in json.loads(prompt)["untrusted_sources"]}
    if set(sources) != {"s" + str(i + 1) for i in range(len(keys))}:
        return False
    for i, key in enumerate(keys):
        text, cursor = view.chunks[key].text, 0
        for span in sources["s" + str(i + 1)]["spans"]:
            start = text.find(span["text"], cursor)
            if start < 0 or text[cursor:start].strip():
                return False
            cursor = start + len(span["text"])
        if text[cursor:].strip():
            return False
    return True


def prepare_recall_answer(collection, lexical, pool, *, balanced, recipe, run_id, owner_id):
    if type(pool) is not RecallPool or type(balanced) is not bool or pool.collection_manifest_sha256 != collection.manifest_sha256:
        raise GenerationError("DOCUMENT_RECALL_POOL_BINDING")
    view = _evidence_view(collection, lexical, pool.pool_ids)
    trace = json.loads(pool.trace_json)
    original = json.loads(pool.bundles_json)
    validate_collection_lexical_binding(collection, lexical)
    if (len(pool.pool_ids) > 40 or len(set(pool.pool_ids)) != len(pool.pool_ids)
            or trace["pool_chunk_ids"] != list(pool.pool_ids)
            or not isinstance(original, list) or len(original) > 40
            or len({b["bundle_id"] for b in original}) != len(original)
            or any(not b["chunk_ids"] or len(set(b["chunk_ids"])) != len(b["chunk_ids"])
                or not set(b["chunk_ids"]) <= set(pool.pool_ids) for b in original)):
        raise GenerationError("DOCUMENT_RECALL_PACK_POOL_OR_BUNDLE_BINDING")
    windows = [b for b in original if b["kind"] == "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT"]
    window_members = {key for b in windows for key in b["chunk_ids"]}
    all_chunks = collection_records(collection)[1]
    for bundle in original:
        if bundle["kind"] == "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT":
            expected = document_window(all_chunks, bundle["anchor_chunk_id"], pool.recipe.radius)
            if tuple(bundle["chunk_ids"]) != expected:
                raise GenerationError("DOCUMENT_RECALL_NEIGHBOR_WINDOW_CHANGED")
        elif (bundle["kind"] != "RANKED_ORIGINAL_CHUNK" or len(bundle["chunk_ids"]) != 1
                or bundle["chunk_ids"][0] in window_members):
            raise GenerationError("DOCUMENT_RECALL_PARTIAL_WINDOW_SINGLETON")
    bundles = round_robin_bundles(original, trace["query_order"]) if balanced else original
    tokenizer = ProxyTokenizer()
    selected, admitted, decisions = (), [], []

    def render(keys):
        packed = PackedContext(keys, tuple("s" + str(i + 1) for i in range(len(keys))), (), "", 0, ())
        system, prompt = generation_request(pool.question, view, packed, collection.manifest.tenant_id, recipe, answer_language=pool.language)
        return system, prompt, tokenizer.count(prompt)

    if render(())[2] > pool.recipe.user_message_proxy_tokens:
        raise GenerationError("DOCUMENT_RECALL_EMPTY_ENVELOPE_BUDGET")
    for bundle in bundles:
        proposal = tuple(dict.fromkeys((*selected, *bundle["chunk_ids"])))
        reason = "MAX_CHUNKS"
        if len(proposal) <= 8:
            _, prompt, count = render(proposal)
            reason = "TOKEN_BUDGET" if count > 4096 else "RENDERING_INCOMPLETE" if not _full_rendering(view, proposal, prompt) else "SELECTED"
            if reason == "SELECTED":
                selected = proposal
                admitted.append(bundle["bundle_id"])
        decisions.append({"bundle_id": bundle["bundle_id"], "query_index": bundle["query_index"],
            "chunk_ids": bundle["chunk_ids"], "proposed_chunk_count": len(proposal), "reason": reason})
    system, prompt, tokens = render(selected)
    mode = "hybrid_multi_query_balanced" if balanced else "hybrid_multi_query"
    identity = stable_id("faceted-document-recall-answer-v1", pool=pool.fingerprint, recipe=recipe.fingerprint,
        mode=mode, language=pool.language, owner=owner_id, run=run_id, user_message_sha256=sha256_bytes(prompt.encode()))
    binding = AnswerBinding(request_id=identity, tenant_id=collection.manifest.tenant_id, owner_id=owner_id,
        mode_requested=mode, mode_used=mode, used_graph=False,
        warnings=("EXPERIMENTAL_NATIVE_FACETS_V4_DOCUMENT_WINDOWS", "DOCUMENT_WINDOW_NOT_SEMANTIC_PROOF"))
    prepared = PreparedAnswer(CallContext(tenant_id=collection.manifest.tenant_id, run_id=run_id, request_id=identity,
        stage="GENERATE"), binding, PackedContext(selected, tuple("s" + str(i + 1) for i in range(len(selected))), (),
            prompt, tokens, (), tokenizer=tokenizer.version), pool.question, recipe, pool.pool_ids)
    audit = {"policy": POLICY, "pool_fingerprint": pool.fingerprint, "context_chunk_ids": selected,
        "context_assertion_ids": [], "used_graph": False, "context_selection_used": balanced,
        "bundle_order": [b["bundle_id"] for b in bundles], "admitted_bundle_ids": admitted, "decisions": decisions,
        "sources": [{"chunk_id": key, "document_version_id": view.chunks[key].document_version_id,
            "start_char": view.chunks[key].start_char, "end_char": view.chunks[key].end_char,
            "text_sha256": view.chunks[key].text_sha256, "source_uri": view.documents[view.chunks[key].document_version_id].source_uri}
            for key in selected], "source_message_sha256": sha256_bytes(prompt.encode()), "system_sha256": sha256_bytes(system.encode()),
        "user_message_proxy_tokens": tokens, "requested_method": mode, "effective_method": mode,
        "mechanism_status": "READY_DOCUMENT_RECALL_V1", "generation_executed": False, "hidden_fallbacks": 0}
    return view, prepared, audit
