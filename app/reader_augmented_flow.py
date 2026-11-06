"""Explicit reader-augmented variants of actual Graph/Q/QB retrieval.

This is a new admission recipe, not a relabelled baseline or a fallback.
"""

import json
from dataclasses import dataclass, replace

from app.answer_cache import AnswerReceipt
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.literal_source_catalog import SourceCatalog
from app.profiled_source_context import pack_profiled_context
from app.routed_reader_json_v2 import READ_SYSTEM
from app.routed_source_reader import parse_ids
from app.scoped_method_flow import PreparedScopedFlow


@dataclass(frozen=True)
class ReaderSelection:
    catalog: SourceCatalog
    receipt: AnswerReceipt
    selected_ids: tuple[str, ...]

    @property
    def fingerprint(self):
        return sha256_bytes(canonical_json({"catalog": self.catalog.fingerprint,
            "receipt": self.receipt.model_dump(mode="json"), "selected": self.selected_ids}))


def validate_selection(catalog, receipt, collection):
    if type(catalog) is not SourceCatalog or type(receipt) is not AnswerReceipt:
        raise GenerationError("READER_SELECTION_TYPED_PROVENANCE_REQUIRED")
    expected = sha256_bytes(canonical_json({"binding": catalog.fingerprint, "recipe": sha256_bytes(READ_SYSTEM.encode()),
        "system": READ_SYSTEM, "prompt": catalog.message, "max_output_tokens": 768}))
    if (receipt.binding_sha256 != expected or receipt.recipe_sha256 != sha256_bytes(READ_SYSTEM.encode())
            or catalog.collection_sha256 != collection.manifest_sha256
            or receipt.namespace != ("FIXTURE" if collection.manifest.purpose == "FIXTURE" else "LIVE")):
        raise GenerationError("READER_SELECTION_CATALOG_RECIPE_OR_NAMESPACE")
    selected = parse_ids(receipt.raw_output, receipt.stop_reason, "source_ids", "r", catalog.source_ids, 12)
    return ReaderSelection(catalog, receipt, selected)


def augment_with_reader(flow, selection, collection, graph):
    if type(flow) is not PreparedScopedFlow or type(selection) is not ReaderSelection:
        raise GenerationError("READER_AUGMENT_TYPED_FLOW_REQUIRED")
    selection = validate_selection(selection.catalog, selection.receipt, collection)
    cat = selection.catalog
    if (cat.question != flow.context.question or cat.language != flow.context.language
            or cat.search != flow.trace["english_search_query"]):
        raise GenerationError("READER_AUGMENT_ORIGINAL_TASK_CHANGED")
    # Revalidate catalog bytes against the same immutable source records.
    payload = json.loads(cat.message)
    if (payload["untrusted_question"] != cat.question or payload["untrusted_english_search"] != cat.search
            or payload["language"] != cat.language or len(payload["untrusted_sources"]) != len(cat.source_ids)):
        raise GenerationError("READER_AUGMENT_CATALOG_TASK_BINDING")
    for i, (key, source) in enumerate(zip(cat.source_ids, payload["untrusted_sources"], strict=True), 1):
        chunk = flow.chunks[key]
        doc = flow.documents[chunk.document_version_id]
        if (source["source_id"] != f"r{i}" or source["untrusted_text"] != chunk.text
                or source["untrusted_title"] != doc.title or doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text
                or chunk.tenant_id != flow.context.tenant_id):
            raise GenerationError("READER_AUGMENT_ORIGINAL_SOURCE_BYTES")
    paths = [b for b in flow.trace["bundles"] if b.get("assertion_ids")][:2] if flow.method.startswith("graph_") else []
    support = tuple(dict.fromkeys(key for b in paths for key in b["chunk_ids"]))
    pool = tuple(dict.fromkeys((*support, *selection.selected_ids, *flow.trace["pool_ids"])))[:40]
    paths = [b for b in paths if set(b["chunk_ids"]) <= set(pool)]
    covered = {key for b in paths for key in b["chunk_ids"]}
    owner = {key: b["query_index"] for b in flow.trace["bundles"] for key in b["chunk_ids"]}
    bundles = paths + [{"bundle_id": "reader-augmented-" + key, "chunk_ids": [key],
        "query_index": owner.get(key, 0)} for key in pool if key not in covered]
    records = {f.assertion_id: json.loads(f.context_json) for f in graph.facts} if paths else {}
    context, packing = pack_profiled_context(flow.documents, flow.chunks, pool, bundles,
        question=cat.question, language=cat.language, tenant_id=flow.context.tenant_id,
        collection_sha256=collection.manifest_sha256, profile=flow.context.profile,
        balanced=flow.method == "hybrid_multi_query_balanced", query_order=flow.trace["query_order"], graph_records=records)
    status = flow.status
    if not selection.selected_ids:
        status = "NOT_READY_READER_NO_SELECTED_EVIDENCE"
    if flow.method.startswith("graph_") and not packing["used_graph"]:
        status = "NOT_READY_GRAPH_PATH_NOT_PACKED" if paths else "NOT_READY_GRAPH_ZERO_EXPANSION"
    fingerprint = sha256_bytes(canonical_json({"original_pool": flow.trace["pool_fingerprint"],
        "reader_selection": selection.fingerprint, "pool": pool, "bundles": bundles}))
    request = stable_id("reader-augmented-flow-v1", parent=flow.binding.request_id,
        reader=selection.fingerprint, context=context.fingerprint)
    trace = flow.trace | {"variant": "METHOD_PLUS_NATIVE_ROUTED_READER_V1_NOT_PURE_BASELINE",
        "original_pool_fingerprint": flow.trace["pool_fingerprint"], "reader_selection_sha256": selection.fingerprint,
        "reader_catalog_sha256": cat.fingerprint, "reader_selected_ids": selection.selected_ids,
        "reader_receipt_operation_id": selection.receipt.operation_id,
        "pool_ids": pool, "pool_fingerprint": fingerprint, "bundles": bundles,
        "context_chunk_ids": context.chunk_ids, "context_fingerprint": context.fingerprint,
        "context_proxy_tokens": context.proxy_tokens, "packing": packing, "mechanism_status": status,
        "source_annotations_used_for_ranking": False, "hidden_fallbacks": 0}
    binding = flow.binding.model_copy(update={"request_id": request, "used_graph": packing["used_graph"],
        "warnings": (*flow.binding.warnings, "NATIVE_ROUTED_READER_AUGMENTATION_NOT_PURE_BASELINE")})
    return replace(flow, context=context, status=status, trace=trace, binding=binding,
        call_context=flow.call_context.model_copy(update={"request_id": request}))
