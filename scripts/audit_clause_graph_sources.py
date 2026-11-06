"""Bind native V9 source judgments and produce a local V2 candidate export."""

import json
from collections import Counter

from app.clause_graph_records import canonical_recipe, canonicalize_relation
from app.collection_graph import graph_fingerprint
from app.collection_graph_v2 import CollectionGraphOverlayV2, load_collection_graph_v2
from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction_cache_v9 import SemanticExtractionCache
from app.source_entity_boundary import check_reference, policy_recipe
from scripts.run_clause_graph_qualification import DIRECTORY, ROOT, RUN, freeze, immutable

CONFIG = ROOT / "configs/extraction/semantic-v9-source-review-v1.json"
OUTPUT = DIRECTORY / "source-review-v1"


def main():
    plan, chunks, native_recipe = freeze()
    from scripts.run_semantic_graph_canary import load_collection_inputs

    collection, _, _ = load_collection_inputs()
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v9", namespace="LIVE")
    rows = []
    for unit in plan["selection"]:
        row = json.loads((DIRECTORY / "units" / (unit["v9_unit_id"] + ".json")).read_bytes())
        from app.source_graph_jobs_v9 import context_for

        text = chunks[unit["occurrences"][0]["chunk_id"]].text
        lookup = cache.resolve(context_for(unit, RUN), text=text, recipe=native_recipe)
        result = lookup.validation
        if (row["text_sha256"] != sha256_bytes(text.encode()) or canonical_json(lookup.receipt) != canonical_json(row["receipt"])
                or canonical_json(result.accepted_relationships) != canonical_json(row["accepted_relationships"])
                or canonical_json(result.accepted_permissions) != canonical_json(row["accepted_permissions"])
                or canonical_json(result.deferred_statements) != canonical_json(row["deferred_statements"])
                or canonical_json(result.rejected_records) != canonical_json(row["rejected_records"])):
            raise ValueError("V9_REVIEW_NATIVE_SOURCE_REPLAY_CHANGED")
        rows.append(row)
    config = json.loads(CONFIG.read_bytes())
    candidates = {(r["unit_id"], kind, c["index"]): (r, c) for r in rows
        for kind, field in (("RELATION", "accepted_relationships"), ("PERMISSION", "accepted_permissions")) for c in r[field]}
    judgments, seen = [], set()
    for judgment in config["reviews"]:
        key = judgment["unit_id"], judgment["kind"], judgment["index"]
        if key in seen or key not in candidates:
            raise ValueError("V9_REVIEW_UNKNOWN_OR_DUPLICATE_CANDIDATE")
        row, candidate = candidates[key]
        if judgment["candidate_sha256"] != sha256_bytes(canonical_json(candidate)):
            raise ValueError("V9_REVIEW_CANDIDATE_CHECKSUM")
        seen.add(key)
        allowed, reason = True, None
        if judgment["kind"] == "RELATION":
            from app.semantic_extraction_v9 import RelationV9

            record = RelationV9.model_validate_json(canonical_json(candidate["record"]))
            for endpoint in (record.subject, record.object):
                allowed, reason = check_reference(endpoint, record.clause_quote)
                if not allowed:
                    break
        if judgment["status"] == "AI_SOURCE_SUPPORTED_SOURCE_SCOPED" and not allowed:
            raise ValueError("V9_REVIEW_SUPPORTED_LABEL_CONFLICTS_WITH_REFERENCE_GUARD")
        judgments.append(dict(judgment, text_sha256=row["text_sha256"], clause_sha256=sha256_bytes(candidate["record"]["clause_quote"].encode()),
            source_uris=sorted({o["source_uri"] for o in row["occurrences"]}), reference_guard_admitted=allowed, reference_guard_reason=reason))
    if seen != set(candidates):
        raise ValueError("V9_REVIEW_INCOMPLETE")
    documents = {}
    for ref in collection.manifest.shards:
        documents.update(collection.get_shard(ref.shard_id).documents)
    entities, mentions, assertions, sidecar = {}, {}, {}, []
    for judgment in judgments:
        if judgment["status"] != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED" or judgment["kind"] != "RELATION":
            continue
        row, candidate = candidates[judgment["unit_id"], judgment["kind"], judgment["index"]]
        for occurrence in row["occurrences"]:
            chunk = chunks[occurrence["chunk_id"]]
            if chunk.text_sha256 != row["text_sha256"]:
                raise ValueError("V9_REVIEW_OCCURRENCE_SOURCE_CHANGED")
            es, ms, assertion, metadata = canonicalize_relation(candidate, judgment, chunk=chunk,
                document=documents[chunk.document_version_id], native_recipe=native_recipe)
            for entity in es:
                if entity.entity_id in entities and entities[entity.entity_id] != entity:
                    raise ValueError("V9_REVIEW_ENTITY_IDENTITY_CONFLICT")
                entities[entity.entity_id] = entity
            mentions.update((m.mention_id, m) for m in ms)
            assertions[assertion.assertion_id] = assertion
            sidecar.append(metadata | {"receipt_sha256": sha256_bytes(canonical_json(row["receipt"])),
                "source_uri": occurrence["source_uri"], "operation_id": row["receipt"]["operation_id"]})
    OUTPUT.mkdir(exist_ok=True)
    semantic_sidecar = {"recipe": canonical_recipe(native_recipe).model_dump(mode="json"), "assertions": sidecar,
        "reference_promotion_policy": policy_recipe(), "native_v9_inference_is_separate_from_promotion_guard": True}
    sidecar_raw = canonical_json(semantic_sidecar) + b"\n"
    es, ms, rs = (tuple(values[k] for k in sorted(values)) for values in (entities, mentions, assertions))
    inventory = ROOT / "results/method-correctness-v2/full-graph-inventory-v2/source-units.jsonl"
    overlay = CollectionGraphOverlayV2(tenant_id="demo", collection_version=collection.manifest.collection_version,
        collection_manifest_sha256=collection.manifest_sha256, graph_sha256=graph_fingerprint(collection.manifest_sha256, es, ms, rs),
        source_inventory_sha256=sha256_bytes(inventory.read_bytes()), extraction_recipe_sha256=canonical_recipe(native_recipe).fingerprint,
        aliases_sha256=native_recipe.parameters["typed_catalog_sha256"], curation_sha256=sha256_bytes(CONFIG.read_bytes()),
        upstream_review_queue_sha256=sha256_bytes(sidecar_raw), inventory_units=2897, processed_units=32,
        entities=es, mentions=ms, assertions=rs)
    graph_raw = canonical_json(overlay.model_dump(mode="json")) + b"\n"
    immutable(OUTPUT / "graph.json", graph_raw)
    immutable(OUTPUT / "sidecar.json", sidecar_raw)
    loaded = load_collection_graph_v2(OUTPUT / "graph.json", collection, trusted_sha256=sha256_bytes(graph_raw), allow_experimental=True)
    if canonical_json(loaded.overlay.model_dump(mode="json")) != canonical_json(overlay.model_dump(mode="json")):
        raise ValueError("V9_REVIEW_EXPORT_RELOAD_CHANGED")
    report = {"status": "ALL_NATIVE_CANDIDATES_REVIEWED_LOCAL_EXPORT_MATCH_NOT_FULL_BATCH_ACCEPTANCE",
        "processed_native_v9_sources": 32, "fresh_source_texts": 24, "known_development_texts": 8,
        "native_candidates_reviewed": len(judgments), "judgments": judgments,
        "judgment_statuses": dict(Counter(j["status"] for j in judgments)),
        "entities": len(es), "mentions": len(ms), "assertions": len(rs), "graph_bytes": len(graph_raw), "graph_sha256": sha256_bytes(graph_raw),
        "native_v9_pending_texts": 2865, "deferred_statements_are_not_edges": True,
        "native_v9_semantic_acceptance": "PARTIAL_OVER_DEFERRAL_AND_KNOWN_SOURCE_BINDING_REJECTIONS",
        "human_reviewed": 0, "quality_metrics": None, "serving_records_added": 0, "provider_calls": 0, "aws_calls": 0}
    immutable(OUTPUT / "report.json", canonical_json(report) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "judgments"}))


if __name__ == "__main__":
    main()
