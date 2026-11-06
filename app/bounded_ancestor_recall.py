"""Bound parent context by retaining the actual retrieved child in its window."""

import json
from dataclasses import replace
from typing import Literal

from app.ancestor_section_recall import (
    AncestorSectionRecipe,
    ancestor_lead,
    build_ancestor_section_pool,
)
from app.contracts import canonical_json, stable_id
from app.document_recall_v2 import document_window
from app.graph_collection_migration import collection_records


class BoundedAncestorRecipe(AncestorSectionRecipe):
    version: Literal["bounded-ancestor-recall-v1"] = "bounded-ancestor-recall-v1"
    source_retention: Literal["PARENT_CENTER_ONLY_IF_RETRIEVED_CHILD_REMAINS_IN_WINDOW"] = "PARENT_CENTER_ONLY_IF_RETRIEVED_CHILD_REMAINS_IN_WINDOW"


def bounded_anchor(chunks, child_id, radius):
    parent_id = ancestor_lead(chunks, child_id)
    if child_id not in document_window(chunks, parent_id, radius):
        return child_id, "PARENT_TOO_FAR_RETAIN_CHILD_CENTER"
    return parent_id, "PARENT_WITHIN_WINDOW" if parent_id != child_id else "NO_NONROOT_PARENT"


def build_bounded_ancestor_pool(collection, lexical, index, *, question, search, language, plan, vectors, base_ids, recipe):
    recipe = BoundedAncestorRecipe.model_validate_json(recipe.model_dump_json())
    prior_recipe = AncestorSectionRecipe(field_mode=recipe.field_mode, admission=recipe.admission, radius=recipe.radius)
    prior = build_ancestor_section_pool(collection, lexical, index, question=question, search=search, language=language,
        plan=plan, vectors=vectors, base_ids=base_ids, recipe=prior_recipe)
    trace = json.loads(prior.trace_json)
    _, chunks, _ = collection_records(collection)
    proposals = []
    for original in trace["proposals"]:
        child = original["retrieved_child_anchor_id"]
        anchor, reason = bounded_anchor(chunks, child, recipe.radius)
        window = document_window(chunks, anchor, recipe.radius)
        proposals.append(original | {"bundle_id": stable_id("bounded-source-ancestor-window-v1",
            prior_bundle=original["bundle_id"], anchor=anchor, radius=recipe.radius),
            "anchor_chunk_id": anchor, "anchor_moved_to_source_parent": anchor != child,
            "source_retention_decision": reason, "chunk_ids": window})
    selected, windows, decisions = list(base_ids[:16]), [], []
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
    members = {k for b in windows for k in b["chunk_ids"]}
    singles = [{"bundle_id": stable_id("bounded-ancestor-singleton", chunk=k), "kind": "RANKED_ORIGINAL_CHUNK",
        "query_index": 0, "document_version_id": chunks[k].document_version_id, "anchor_chunk_id": k,
        "chunk_ids": (k,)} for k in selected if k not in members]
    updated = trace | {"policy": "CHILD_RETAINED_ANCESTOR_SECTION_FIELDS_V1", "recipe": recipe.model_dump(mode="json"),
        "prior_ancestor_recipe": prior_recipe.model_dump(mode="json"), "prior_ancestor_pool_fingerprint": prior.fingerprint,
        "proposals": proposals, "admission_decisions": decisions, "pool_ids": selected,
        "every_proposal_retains_retrieved_child": all(p["retrieved_child_anchor_id"] in p["chunk_ids"] for p in proposals)}
    return replace(prior, recipe=recipe, pool_ids=tuple(selected), bundles_json=canonical_json(windows+singles).decode(),
        trace_json=canonical_json(updated).decode())
