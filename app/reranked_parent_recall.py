"""Bounded local reranking over parent anchor chunks, with no method fallback."""

import json
from dataclasses import replace
from typing import Literal

from app.contracts import canonical_json, sha256_bytes
from app.generation import GenerationError
from app.graph_collection_migration import collection_records
from app.parent_section_recall import ParentSectionRecipe, build_parent_section_pool
from app.reranking import RerankResult


class RerankedParentRecipe(ParentSectionRecipe):
    version: Literal["reranked-parent-section-recall-v1"] = "reranked-parent-section-recall-v1"
    parent_reranking: Literal["FULL_ORIGINAL_ENGLISH_SEARCH_ON_MAX40_ORIGINAL_ANCHOR_CHUNKS"] = "FULL_ORIGINAL_ENGLISH_SEARCH_ON_MAX40_ORIGINAL_ANCHOR_CHUNKS"


def build_reranked_parent_pool(collection, lexical, index, *, question, search, language, plan, vectors, base_ids, recipe, reranker):
    recipe = RerankedParentRecipe.model_validate_json(recipe.model_dump_json())
    parent_recipe = ParentSectionRecipe(field_mode=recipe.field_mode, admission=recipe.admission, radius=recipe.radius)
    parent = build_parent_section_pool(collection, lexical, index, question=question, search=search,
        language=language, plan=plan, vectors=vectors, base_ids=base_ids, recipe=parent_recipe)
    trace = json.loads(parent.trace_json)
    candidates = trace["proposals"][:40]
    _, chunks, _ = collection_records(collection)
    anchors = tuple(b["anchor_chunk_id"] for b in candidates)
    if getattr(reranker, "fixture_only", None) is not (collection.manifest.purpose == "FIXTURE"):
        raise GenerationError("PARENT_RERANKER_NAMESPACE")
    result = reranker.rerank(search, tuple(chunks[k] for k in anchors), collection.manifest.tenant_id)
    if (type(result) is not RerankResult or len(result.ranked_chunk_ids) != len(anchors)
            or set(result.ranked_chunk_ids) != set(anchors) or len(set(result.ranked_chunk_ids)) != len(anchors)
            or result.recipe_sha256 != reranker.recipe.fingerprint):
        raise GenerationError("PARENT_RERANKER_PERMUTATION_OR_RECIPE")
    by_anchor = {b["anchor_chunk_id"]: b for b in candidates}
    proposals = [by_anchor[k] for k in result.ranked_chunk_ids]
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
    original_singletons = {b["anchor_chunk_id"]: b for b in json.loads(parent.bundles_json)
        if b["kind"] == "RANKED_ORIGINAL_CHUNK"}
    singletons = []
    for key in selected:
        if key in members:
            continue
        singletons.append(original_singletons.get(key) or {
            "bundle_id": "reranked-parent-singleton-" + key, "kind": "RANKED_ORIGINAL_CHUNK", "query_index": 0,
            "document_version_id": chunks[key].document_version_id, "anchor_chunk_id": key, "chunk_ids": (key,)})
    updated = trace | {"policy": "RERANKED_PARENT_SECTION_FIELDS_V1", "recipe": recipe.model_dump(mode="json"),
        "parent_candidate_recipe": parent_recipe.model_dump(mode="json"), "parent_candidate_pool_fingerprint": parent.fingerprint,
        "proposals": proposals, "admission_decisions": decisions, "pool_ids": selected,
        "reranking": {"recipe_sha256": result.recipe_sha256, "recipe": reranker.recipe.model_dump(mode="json"),
            "candidate_anchor_ids": anchors, "ranked_anchor_ids": result.ranked_chunk_ids,
            "query_sha256": sha256_bytes(search.encode()),
            "scores": [{"chunk_id": s.chunk_id, "raw_logit": s.raw_logit, "original_rank": s.original_rank,
                "window_count": s.window_count} for s in result.scores],
            "windows": [{"chunk_id": w.chunk_id, "start_char": w.start_char, "end_char": w.end_char,
                "input_tokens": w.input_tokens, "raw_logit": w.raw_logit} for w in result.windows],
            "scores_are_not_factual_confidence": True, "fallback_executed": False}}
    return replace(parent, recipe=recipe, pool_ids=tuple(selected), bundles_json=canonical_json(windows+singletons).decode(),
        trace_json=canonical_json(updated).decode())
