"""Collapse section hits to distinct parent documents BEFORE the depth limit.

Shares source-window packing with section_recall; changes candidate retrieval
only. Parent aggregation is a retrieval heuristic, not a document-quality label.
"""

from typing import Literal

import numpy as np
from rank_bm25 import BM25Okapi

from app.collection_answering import validate_collection_lexical_binding
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.document_recall_v2 import document_window
from app.generation import GenerationError
from app.graph_collection_migration import collection_records
from app.retrieval import lexical_tokens, reciprocal_rank_fusion
from app.section_recall import (
    SectionIndex,
    SectionPool,
    SectionRecipe,
    _diverse_order,
    literal_probes,
)
from app.strict_collection_methods import STOPWORDS

POLICY = "PARENT_COLLAPSED_SECTION_FIELDS_V1"


class ParentSectionRecipe(SectionRecipe):
    version: Literal["parent-section-recall-v1"] = "parent-section-recall-v1"
    candidate_policy: Literal["TOP20_DISTINCT_DOCUMENTS_BEFORE_FIELD_AND_QUERY_RRF"] = "TOP20_DISTINCT_DOCUMENTS_BEFORE_FIELD_AND_QUERY_RRF"


def distinct_parents(index, ranked_indices):
    result, seen = [], set()
    for i in ranked_indices:
        doc = index.rows[i]["document_version_id"]
        if doc not in seen:
            seen.add(doc)
            result.append(i)
        if len(result) == 20:
            break
    return result


def parent_rankings(index, query, vector, mode, chunks, body_vectors):
    if mode not in {"BM25", "DENSE", "RRF"}:
        raise GenerationError("PARENT_SECTION_FIELD_MODE")
    tokens = [tuple(t for t in lexical_tokens(r["metadata_text"]) if t not in STOPWORDS) for r in index.rows]
    terms = tuple(t for t in lexical_tokens(query) if t not in STOPWORDS)
    scores = BM25Okapi(tokens).get_scores(terms) if any(tokens) else np.zeros(len(tokens))
    cosine = index.vectors @ vector
    lexical = distinct_parents(index, sorted((i for i, own in enumerate(tokens) if set(own) & set(terms)),
        key=lambda i: (-float(scores[i]), index.rows[i]["section_id"])))
    dense = distinct_parents(index, sorted(range(len(tokens)), key=lambda i: (-float(cosine[i]), index.rows[i]["section_id"])))
    b, d = ([index.rows[i]["document_version_id"] for i in order] for order in (lexical, dense))
    ranked = b if mode == "BM25" else d if mode == "DENSE" else reciprocal_rank_fusion((b, d))[:20]
    anchors, sections = {}, {}
    for doc in ranked:
        relevant = lexical if mode == "BM25" else dense if mode == "DENSE" else [*lexical, *dense]
        candidate_sections = [index.rows[i] for i in relevant if index.rows[i]["document_version_id"] == doc]
        keys = {k for section in candidate_sections for k in section["chunk_ids"]}
        anchor = min(keys, key=lambda k: (-float(body_vectors[k] @ vector), k))
        anchors[doc] = anchor
        sections[doc] = sorted(s["section_id"] for s in candidate_sections if anchor in s["chunk_ids"])[0]
        if chunks[anchor].document_version_id != doc:
            raise GenerationError("PARENT_SECTION_ANCHOR_DOCUMENT")
    return {"query": query, "lexical_document_ranking": b, "dense_document_ranking": d,
        "ranking": ranked, "anchors": anchors, "sections": sections,
        "query_vector_sha256": sha256_bytes(vector.tobytes(order="C"))}


def build_parent_section_pool(collection, lexical, index, *, question, search, language, plan, vectors, base_ids, recipe):
    validate_collection_lexical_binding(collection, lexical)
    recipe = ParentSectionRecipe.model_validate_json(recipe.model_dump_json())
    queries = literal_probes(collection, question, search, language, plan)
    if type(index) is not SectionIndex or index.collection_sha256 != collection.manifest_sha256 or len(vectors) != len(queries):
        raise GenerationError("PARENT_SECTION_INDEX_OR_QUERY_BINDING")
    expected = sha256_bytes(canonical_json({"collection": collection.manifest_sha256,
        "projection": "DEDUPLICATED_TITLE_AND_SECTION_PATH_ONLY_V1", "rows": index.rows,
        "vectors_sha256": sha256_bytes(index.vectors.tobytes(order="C"))}))
    if expected != index.fingerprint:
        raise GenerationError("PARENT_SECTION_INDEX_MUTATED")
    for vector in vectors:
        if (not isinstance(vector, np.ndarray) or vector.dtype != np.dtype("float32")
                or vector.shape != (index.vectors.shape[1],) or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5, rtol=0)):
            raise GenerationError("PARENT_SECTION_QUERY_VECTOR")
    _, chunks, body_vectors = collection_records(collection)
    if not 1 <= len(base_ids) <= 40 or len(set(base_ids)) != len(base_ids) or not set(base_ids) <= set(chunks):
        raise GenerationError("PARENT_SECTION_BASE_POOL")
    rankings = [parent_rankings(index, q, v, recipe.field_mode, chunks, body_vectors) for q, v in zip(queries, vectors, strict=True)]
    parents = reciprocal_rank_fusion([r["ranking"] for r in rankings])
    proposals, unique = [], set()
    for doc in parents:
        owner = min((i for i, r in enumerate(rankings) if doc in r["ranking"]),
            key=lambda i: (rankings[i]["ranking"].index(doc), i))
        anchor = rankings[owner]["anchors"][doc]
        window = document_window(chunks, anchor, recipe.radius)
        if window in unique:
            continue
        unique.add(window)
        proposals.append({"bundle_id": stable_id("parent-section-source-window-v1", collection=collection.manifest_sha256,
            query=queries[owner], anchor=anchor, radius=recipe.radius), "kind": "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT",
            "query_index": owner, "document_version_id": doc, "section_id": rankings[owner]["sections"][doc],
            "anchor_chunk_id": anchor, "chunk_ids": window})
    if recipe.admission == "DOCUMENT_DIVERSE":
        proposals = _diverse_order(proposals)
    selected = list(base_ids[:16])
    windows, decisions = [], []
    for proposed in proposals:
        combined = tuple(dict.fromkeys((*selected, *proposed["chunk_ids"])))
        admitted = len(windows) < 6 and len(combined) <= 40
        decisions.append({"bundle_id": proposed["bundle_id"], "admitted": admitted, "proposed_pool_size": len(combined),
            "reason": "ADMITTED" if admitted else "WINDOW_CAP" if len(windows) >= 6 else "POOL_CAP"})
        if admitted:
            selected = list(combined)
            windows.append(proposed)
    for key in base_ids:
        if key not in selected and len(selected) < 40:
            selected.append(key)
    members = {k for b in windows for k in b["chunk_ids"]}
    singletons = [{"bundle_id": stable_id("parent-section-base-singleton", chunk=k), "kind": "RANKED_ORIGINAL_CHUNK",
        "query_index": 0, "document_version_id": chunks[k].document_version_id, "anchor_chunk_id": k,
        "chunk_ids": (k,)} for k in selected if k not in members]
    trace = {"policy": POLICY, "recipe": recipe.model_dump(mode="json"), "index_fingerprint": index.fingerprint,
        "native_v5_plan": plan.model_dump(mode="json"), "query_track": "DERIVED_LITERAL_RECALL_PROBES_NOT_NATIVE_REWRITES",
        "queries": queries, "rankings": rankings, "fused_document_ranking": parents, "proposals": proposals,
        "admission_decisions": decisions, "pool_ids": selected, "query_order": (*range(1, len(queries)), 0),
        "source_annotations_used_for_ranking": False, "new_graph_records": 0, "hidden_fallbacks": 0,
        "packing_component": "SECTION_RECALL_V1_ATOMIC_PACKER_SHARED_UNCHANGED"}
    return SectionPool(collection.manifest_sha256, question, search, language, recipe, tuple(base_ids), tuple(selected),
        canonical_json(windows+singletons).decode(), canonical_json(trace).decode())
