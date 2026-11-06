"""Center source windows on an existing nearest parent-section lead.

The original retrieved child anchor stays in the trace. No parent relation is
interpreted as an AWS fact. Root-document jumps and cross-document jumps are
forbidden; windows keep original source chunk identities and the same bounds.
"""

import json
from dataclasses import replace
from typing import Literal

from app.contracts import canonical_json, stable_id
from app.document_recall_v2 import document_window
from app.graph_collection_migration import collection_records
from app.parent_section_recall import ParentSectionRecipe, build_parent_section_pool


class AncestorSectionRecipe(ParentSectionRecipe):
    version: Literal["ancestor-section-recall-v1"] = "ancestor-section-recall-v1"
    anchor_context: Literal["NEAREST_EXISTING_NONROOT_ANCESTOR_LEAD_WITH_SAME_DOCUMENT"] = "NEAREST_EXISTING_NONROOT_ANCESTOR_LEAD_WITH_SAME_DOCUMENT"


def ancestor_lead(chunks, anchor_id):
    anchor = chunks[anchor_id]
    for depth in range(len(anchor.section_path)-1, 1, -1):
        path = anchor.section_path[:depth]
        own = [c for c in chunks.values() if c.document_version_id == anchor.document_version_id
            and c.section_path == path and c.start_char <= anchor.start_char]
        if own:
            return min(own, key=lambda c: (c.start_char, c.chunk_id)).chunk_id
    return anchor_id


def build_ancestor_section_pool(collection, lexical, index, *, question, search, language, plan, vectors, base_ids, recipe):
    recipe = AncestorSectionRecipe.model_validate_json(recipe.model_dump_json())
    parent_recipe = ParentSectionRecipe(field_mode=recipe.field_mode, admission=recipe.admission, radius=recipe.radius)
    parent = build_parent_section_pool(collection, lexical, index, question=question, search=search,
        language=language, plan=plan, vectors=vectors, base_ids=base_ids, recipe=parent_recipe)
    trace = json.loads(parent.trace_json)
    _, chunks, _ = collection_records(collection)
    proposals = []
    for original in trace["proposals"]:
        anchor = ancestor_lead(chunks, original["anchor_chunk_id"])
        window = document_window(chunks, anchor, recipe.radius)
        proposals.append(original | {"bundle_id": stable_id("ancestor-section-window-v1",
            parent_bundle=original["bundle_id"], anchor=anchor, radius=recipe.radius),
            "retrieved_child_anchor_id": original["anchor_chunk_id"], "anchor_chunk_id": anchor,
            "anchor_moved_to_source_parent": anchor != original["anchor_chunk_id"], "chunk_ids": window})
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
    singletons = [{"bundle_id": stable_id("ancestor-section-singleton", chunk=k), "kind": "RANKED_ORIGINAL_CHUNK",
        "query_index": 0, "document_version_id": chunks[k].document_version_id, "anchor_chunk_id": k,
        "chunk_ids": (k,)} for k in selected if k not in members]
    updated = trace | {"policy": "SOURCE_ANCESTOR_LEAD_SECTION_FIELDS_V1", "recipe": recipe.model_dump(mode="json"),
        "parent_candidate_recipe": parent_recipe.model_dump(mode="json"), "parent_candidate_pool_fingerprint": parent.fingerprint,
        "proposals": proposals, "admission_decisions": decisions, "pool_ids": selected,
        "metadata_ancestry_is_not_semantic_dependency": True}
    return replace(parent, recipe=recipe, pool_ids=tuple(selected), bundles_json=canonical_json(windows+singletons).decode(),
        trace_json=canonical_json(updated).decode())
