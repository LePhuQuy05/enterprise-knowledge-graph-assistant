"""Local field/section recall; metadata is a search surface, never evidence.

Only original chunks reach the answer message. No annotation, target URI,
required fact, policy grant or graph edge is accepted by this module.
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
from app.document_recall_v2 import _full_rendering, document_window, round_robin_bundles
from app.faceted_query_v5 import FacetPlanV5, revalidate_plan
from app.generation import AnswerBinding, GenerationError, generation_request
from app.graph_collection_migration import collection_records
from app.ingestion.text import ProxyTokenizer
from app.model_gateway import CallContext
from app.retrieval import lexical_tokens, reciprocal_rank_fusion
from app.strict_collection_methods import STOPWORDS

POLICY = "LITERAL_FOCUS_SECTION_METADATA_RECALL_V1"


class SectionRecipe(Record):
    version: Literal["section-recall-v1"] = "section-recall-v1"
    field_mode: Literal["BM25", "DENSE", "RRF"] = "RRF"
    admission: Literal["RANKED", "DOCUMENT_DIVERSE"] = "DOCUMENT_DIVERSE"
    radius: Literal[1, 2] = 1
    metadata_depth: Literal[20] = 20
    max_windows: Literal[6] = 6
    base_reserved: Literal[16] = 16
    max_pool: Literal[40] = 40
    context_chunks: Literal[8] = 8
    user_message_proxy_tokens: Literal[4096] = 4096

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


def metadata_rows(documents, chunks, tenant_id):
    groups = {}
    for key in sorted(chunks):
        chunk = chunks[key]
        doc = documents.get(chunk.document_version_id)
        if (doc is None or chunk.tenant_id != tenant_id or doc.tenant_id != tenant_id
                or key != chunk.chunk_id or sha256_bytes(chunk.text.encode()) != chunk.text_sha256
                or doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
            raise GenerationError("SECTION_METADATA_SOURCE_OR_TENANT")
        headings = tuple(dict.fromkeys((doc.title, *chunk.section_path)))
        text = "\n\n".join(headings)
        identity = stable_id("source-section-metadata-v1", document=doc.document_version_id, headings=headings)
        group = groups.setdefault(identity, {"section_id": identity, "document_version_id": doc.document_version_id,
            "headings": headings, "metadata_text": text, "chunk_ids": []})
        group["chunk_ids"].append(key)
    rows = []
    for key in sorted(groups):
        row = groups[key]
        row["chunk_ids"] = tuple(sorted(row["chunk_ids"], key=lambda k: (chunks[k].start_char, chunks[k].end_char, k)))
        rows.append(row)
    if not 1 <= len(rows) <= 5000:
        raise GenerationError("SECTION_METADATA_UNITS_BOUND")
    return tuple(rows)


@dataclass(frozen=True)
class SectionIndex:
    collection_sha256: str
    rows: tuple[dict, ...]
    vectors: np.ndarray
    fingerprint: str


def make_index(collection, rows, vectors):
    docs, chunks, _ = collection_records(collection)
    expected = metadata_rows(docs, chunks, collection.manifest.tenant_id)
    if canonical_json(rows) != canonical_json(expected):
        raise GenerationError("SECTION_METADATA_ROWS_BINDING")
    dimension = collection.manifest.embedding_recipe.dimension
    if (not isinstance(vectors, np.ndarray) or vectors.dtype != np.dtype("float32")
            or vectors.shape != (len(rows), dimension) or not np.isfinite(vectors).all()
            or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=1e-5, rtol=0)):
        raise GenerationError("SECTION_METADATA_VECTOR_BINDING")
    fingerprint = sha256_bytes(canonical_json({"collection": collection.manifest_sha256,
        "projection": "DEDUPLICATED_TITLE_AND_SECTION_PATH_ONLY_V1", "rows": rows,
        "vectors_sha256": sha256_bytes(vectors.tobytes(order="C"))}))
    matrix = vectors.copy()
    matrix.flags.writeable = False
    return SectionIndex(collection.manifest_sha256, expected, matrix, fingerprint)


def literal_probes(collection, question, search, language, plan):
    if QueryRequest(question=question).question != question or QueryRequest(question=search).question != search:
        raise GenerationError("SECTION_CANONICAL_QUERY")
    plan = revalidate_plan(plan, collection=collection, question=question, search=search, language=language)
    return (search, *(facet.focus_quote for facet in plan.facets))


def _rank_fields(index, query, vector, mode):
    if mode not in {"BM25", "DENSE", "RRF"}:
        raise GenerationError("SECTION_FIELD_MODE")
    tokens = [tuple(t for t in lexical_tokens(row["metadata_text"]) if t not in STOPWORDS) for row in index.rows]
    terms = tuple(t for t in lexical_tokens(query) if t not in STOPWORDS)
    bm25 = BM25Okapi(tokens) if any(tokens) else None
    scores = bm25.get_scores(terms) if bm25 else np.zeros(len(tokens))
    lexical = sorted((i for i, own in enumerate(tokens) if set(own) & set(terms)),
        key=lambda i: (-float(scores[i]), index.rows[i]["section_id"]))[:20]
    cosine = index.vectors @ vector
    dense = sorted(range(len(index.rows)), key=lambda i: (-float(cosine[i]), index.rows[i]["section_id"]))[:20]
    b, d = ([index.rows[i]["section_id"] for i in ranking] for ranking in (lexical, dense))
    ranking = b if mode == "BM25" else d if mode == "DENSE" else reciprocal_rank_fusion((b, d))[:20]
    return {"query": query, "lexical_ranking": b, "dense_ranking": d, "ranking": ranking,
        "query_vector_sha256": sha256_bytes(vector.tobytes(order="C"))}


def _diverse_order(proposals):
    seen, first, deferred = set(), [], []
    for item in proposals:
        key = item["document_version_id"]
        if key in seen:
            deferred.append(item)
        else:
            seen.add(key)
            first.append(item)
    return first + deferred


@dataclass(frozen=True)
class SectionPool:
    collection_sha256: str
    question: str
    search: str
    language: str
    recipe: SectionRecipe
    base_ids: tuple[str, ...]
    pool_ids: tuple[str, ...]
    bundles_json: str
    trace_json: str

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json({k: getattr(self, k) for k in (
            "collection_sha256", "question", "search", "language", "base_ids", "pool_ids", "bundles_json", "trace_json")}
            | {"recipe": self.recipe.model_dump(mode="json")}))


def build_section_pool(collection, lexical, index, *, question, search, language, plan, vectors, base_ids, recipe):
    validate_collection_lexical_binding(collection, lexical)
    recipe = SectionRecipe.model_validate_json(recipe.model_dump_json())
    queries = literal_probes(collection, question, search, language, plan)
    if type(index) is not SectionIndex or index.collection_sha256 != collection.manifest_sha256 or len(vectors) != len(queries):
        raise GenerationError("SECTION_INDEX_OR_QUERY_BINDING")
    expected_index = sha256_bytes(canonical_json({"collection": collection.manifest_sha256,
        "projection": "DEDUPLICATED_TITLE_AND_SECTION_PATH_ONLY_V1", "rows": index.rows,
        "vectors_sha256": sha256_bytes(index.vectors.tobytes(order="C"))}))
    if expected_index != index.fingerprint:
        raise GenerationError("SECTION_INDEX_MUTATED")
    for vector in vectors:
        if (not isinstance(vector, np.ndarray) or vector.dtype != np.dtype("float32")
                or vector.shape != (index.vectors.shape[1],) or not np.isfinite(vector).all()
                or not np.isclose(np.linalg.norm(vector), 1, atol=1e-5, rtol=0)):
            raise GenerationError("SECTION_QUERY_VECTOR")
    docs, chunks, body_vectors = collection_records(collection)
    if (not 1 <= len(base_ids) <= 40 or len(set(base_ids)) != len(base_ids)
            or not set(base_ids) <= set(chunks)):
        raise GenerationError("SECTION_BASE_POOL")
    rankings = [_rank_fields(index, query, vector, recipe.field_mode) for query, vector in zip(queries, vectors, strict=True)]
    by_id = {r["section_id"]: r for r in index.rows}
    fused_sections = tuple(reciprocal_rank_fusion([r["ranking"] for r in rankings]))
    proposals, unique = [], set()
    for section_id in fused_sections:
        section = by_id[section_id]
        owner = min((i for i, r in enumerate(rankings) if section_id in r["ranking"]),
            key=lambda i: (rankings[i]["ranking"].index(section_id), i))
        anchor = min(section["chunk_ids"], key=lambda k: (-float(body_vectors[k] @ vectors[owner]), k))
        window = document_window(chunks, anchor, recipe.radius)
        if window in unique:
            continue
        unique.add(window)
        proposals.append({"bundle_id": stable_id("section-source-window-v1", collection=collection.manifest_sha256,
            section=section_id, query=queries[owner], anchor=anchor, radius=recipe.radius),
            "kind": "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT", "query_index": owner,
            "section_id": section_id, "document_version_id": section["document_version_id"],
            "anchor_chunk_id": anchor, "chunk_ids": window})
    if recipe.admission == "DOCUMENT_DIVERSE":
        proposals = _diverse_order(proposals)
    selected = list(base_ids[:16])
    windows, decisions = [], []
    for item in proposals:
        combined = tuple(dict.fromkeys((*selected, *item["chunk_ids"])))
        admitted = len(windows) < 6 and len(combined) <= 40
        decisions.append({"bundle_id": item["bundle_id"], "admitted": admitted, "proposed_pool_size": len(combined),
            "reason": "ADMITTED" if admitted else "WINDOW_CAP" if len(windows) >= 6 else "POOL_CAP"})
        if admitted:
            selected = list(combined)
            windows.append(item)
    for key in base_ids:
        if key not in selected and len(selected) < 40:
            selected.append(key)
    covered = {k for b in windows for k in b["chunk_ids"]}
    singletons = [{"bundle_id": stable_id("section-base-singleton", chunk=k), "kind": "RANKED_ORIGINAL_CHUNK",
        "query_index": 0, "document_version_id": chunks[k].document_version_id,
        "anchor_chunk_id": k, "chunk_ids": (k,)} for k in selected if k not in covered]
    # Q keeps metadata proposal order, then the preserved baseline tail. QB interleaves the exact same bundles.
    bundles = windows + singletons
    trace = {"policy": POLICY, "recipe": recipe.model_dump(mode="json"), "index_fingerprint": index.fingerprint,
        "native_v5_plan": plan.model_dump(mode="json"), "query_track": "DERIVED_LITERAL_RECALL_PROBES_NOT_NATIVE_REWRITES",
        "queries": queries, "rankings": rankings, "fused_section_ranking": fused_sections,
        "proposals": proposals, "admission_decisions": decisions, "pool_ids": selected,
        "query_order": (*range(1, len(queries)), 0), "source_annotations_used_for_ranking": False,
        "new_graph_records": 0, "hidden_fallbacks": 0}
    return SectionPool(collection.manifest_sha256, question, search, language, recipe, tuple(base_ids), tuple(selected),
        canonical_json(bundles).decode(), canonical_json(trace).decode())


def pack_section_pool(collection, lexical, pool, *, balanced, generation_recipe, run_id, owner_id):
    if type(pool) is not SectionPool or type(balanced) is not bool or pool.collection_sha256 != collection.manifest_sha256:
        raise GenerationError("SECTION_PACK_POOL_BINDING")
    validate_collection_lexical_binding(collection, lexical)
    trace, bundles = json.loads(pool.trace_json), json.loads(pool.bundles_json)
    if trace["recipe"] != pool.recipe.model_dump(mode="json"):
        raise GenerationError("SECTION_PACK_RECIPE_CHANGED")
    plan = FacetPlanV5.model_validate_json(canonical_json(trace["native_v5_plan"]))
    queries = literal_probes(collection, pool.question, pool.search, pool.language, plan)
    if tuple(trace["queries"]) != queries or tuple(trace["query_order"]) != (*range(1, len(queries)), 0):
        raise GenerationError("SECTION_PACK_TASK_CHANGED")
    _, chunks, _ = collection_records(collection)
    if (len(pool.pool_ids) > 40 or len(set(pool.pool_ids)) != len(pool.pool_ids)
            or trace["pool_ids"] != list(pool.pool_ids) or len(bundles) > 40
            or len({b["bundle_id"] for b in bundles}) != len(bundles)
            or not set(pool.base_ids[:16]) <= set(pool.pool_ids)):
        raise GenerationError("SECTION_PACK_POOL_OR_BASE_PREFIX")
    windows = [b for b in bundles if b["kind"] == "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT"]
    members = {k for b in windows for k in b["chunk_ids"]}
    if len(windows) > 6:
        raise GenerationError("SECTION_PACK_WINDOW_CAP")
    for b in bundles:
        if not b["chunk_ids"] or not set(b["chunk_ids"]) <= set(pool.pool_ids):
            raise GenerationError("SECTION_PACK_BUNDLE")
        if b["kind"] == "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT":
            if tuple(b["chunk_ids"]) != document_window(chunks, b["anchor_chunk_id"], pool.recipe.radius):
                raise GenerationError("SECTION_PACK_NEIGHBORS_CHANGED")
        elif b["kind"] != "RANKED_ORIGINAL_CHUNK" or len(b["chunk_ids"]) != 1 or b["chunk_ids"][0] in members:
            raise GenerationError("SECTION_PACK_PARTIAL_WINDOW")
    ordered = round_robin_bundles(bundles, trace["query_order"]) if balanced else bundles
    view = _evidence_view(collection, lexical, pool.pool_ids)
    tokenizer = ProxyTokenizer()

    def render(keys):
        provisional = PackedContext(keys, tuple(f"s{i+1}" for i in range(len(keys))), (), "", 0, ())
        system, prompt = generation_request(pool.question, view, provisional, collection.manifest.tenant_id,
            generation_recipe, answer_language=pool.language)
        return system, prompt, tokenizer.count(prompt)

    if render(())[2] > 4096:
        raise GenerationError("SECTION_PACK_EMPTY_BUDGET")
    selected, decisions = (), []
    for bundle in ordered:
        proposed = tuple(dict.fromkeys((*selected, *bundle["chunk_ids"])))
        reason = "MAX_CHUNKS"
        if len(proposed) <= 8:
            _, prompt, tokens = render(proposed)
            reason = "TOKEN_BUDGET" if tokens > 4096 else "RENDERING_INCOMPLETE" if not _full_rendering(view, proposed, prompt) else "SELECTED"
            if reason == "SELECTED":
                selected = proposed
        decisions.append({"bundle_id": bundle["bundle_id"], "chunk_ids": bundle["chunk_ids"], "reason": reason})
    system, prompt, tokens = render(selected)
    mode = "hybrid_multi_query_balanced" if balanced else "hybrid_multi_query"
    identity = stable_id("section-recall-context-v1", pool=pool.fingerprint, recipe=generation_recipe.fingerprint,
        mode=mode, run=run_id, owner=owner_id, prompt_sha256=sha256_bytes(prompt.encode()))
    binding = AnswerBinding(request_id=identity, tenant_id=collection.manifest.tenant_id, owner_id=owner_id,
        mode_requested=mode, mode_used=mode, used_graph=False,
        warnings=("EXPERIMENTAL_SECTION_RECALL_DERIVED_LITERAL_PROBES", "METADATA_NOT_EVIDENCE_OR_GRAPH_FACT"))
    prepared = PreparedAnswer(CallContext(tenant_id=collection.manifest.tenant_id, run_id=run_id, request_id=identity,
        stage="GENERATE"), binding, PackedContext(selected, tuple(f"s{i+1}" for i in range(len(selected))), (),
            prompt, tokens, (), tokenizer=tokenizer.version), pool.question, generation_recipe, pool.pool_ids)
    audit = {"policy": POLICY, "pool_fingerprint": pool.fingerprint, "context_selection_used": balanced,
        "source_message_sha256": sha256_bytes(prompt.encode()), "system_sha256": sha256_bytes(system.encode()),
        "context_chunk_ids": selected, "context_assertion_ids": [], "used_graph": False, "hidden_fallbacks": 0,
        "bundle_order": [b["bundle_id"] for b in ordered], "decisions": decisions, "generation_executed": False}
    return prepared, audit
