"""Materialize AI-reviewed relations as a local experimental V2 overlay.

Keeps permissions as source-bound triads; records type/modality in sidecars.
No endpoint, predicate, raw output or serving release is silently rewritten.
"""

import json
from collections import Counter, defaultdict

from app.collection_graph import graph_fingerprint
from app.collection_graph_v2 import CollectionGraphOverlayV2, load_collection_graph_v2
from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes
from app.semantic_extraction_cache_v7 import SemanticExtractionCache
from app.semantic_extraction_v8_review import review_output, review_recipe
from app.source_candidate_policy import policy_recipe
from app.source_graph_jobs import context_for
from app.typed_graph_records import canonical_recipe, canonicalize_reviewed
from scripts.audit_source_graph_tranche import audit, completed_rows
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

OUTPUT = ROOT / "results/semantic-graph-v7/typed-export-v1"


def immutable(path, raw):
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("TYPED_EXPORT_IMMUTABLE_CONTENT_CHANGED:" + path.name)
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(raw)


def export():
    collection, chunks, _ = load_collection_inputs()
    native_dir = ROOT / "results/semantic-graph-v7/qualification-v1"
    fresh_dir = ROOT / "results/semantic-graph-v7/composite-qualification-v1"
    plan = json.loads((fresh_dir / "plan.json").read_bytes())
    recipe = ExtractionRecipe.model_validate_json(canonical_json(plan["recipe"]))
    composite = {"native_recipe_sha256": recipe.fingerprint, "review": review_recipe(), "retention": policy_recipe()}
    if composite != plan["composite_recipe"]:
        raise ValueError("TYPED_EXPORT_REVIEW_RECIPE_CHANGED")
    cache = SemanticExtractionCache(ROOT / "data/cache/semantic-extraction-v7", namespace="LIVE")
    native_rows = [json.loads(line) for line in (native_dir / "v8-review-v1/decisions.jsonl").read_bytes().splitlines()]
    native_by_id = {r["unit_id"]: r for r in native_rows}
    for unit in json.loads((native_dir / "plan.json").read_bytes())["selection"]:
        row = native_by_id[unit["v7_unit_id"]]
        lookup = cache.resolve(context_for(unit, "collection316-semantic-v7-qualification-v1"),
            text=chunks[unit["occurrences"][0]["chunk_id"]].text, recipe=recipe)
        if (row["native_receipt_sha256"] != sha256_bytes(canonical_json(lookup.receipt))
                or canonical_json(row["decisions"]) != canonical_json(review_output(lookup.validation.output, chunks[unit["occurrences"][0]["chunk_id"]].text))):
            raise ValueError("TYPED_EXPORT_NATIVE_REVIEW_REPLAY")
        row["receipt"] = lookup.receipt
    fresh_rows = []
    for unit in plan["selection"]:
        row = json.loads((fresh_dir / "units" / (unit["v7_unit_id"] + ".json")).read_bytes())
        lookup = cache.resolve(context_for(unit, plan["run_id"]), text=chunks[unit["occurrences"][0]["chunk_id"]].text, recipe=recipe)
        if canonical_json(row["receipt"]) != canonical_json(lookup.receipt):
            raise ValueError("TYPED_EXPORT_FRESH_RECEIPT_REPLAY")
        fresh_rows.append(row)
    tranche_review = audit(version="v2")
    tranche_rows, _, _ = completed_rows(include_recovery=True)
    rows = native_rows + fresh_rows + tranche_rows
    by_id = {r["unit_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError("TYPED_EXPORT_DUPLICATE_UNIT")
    report_paths = [native_dir / "v8-review-v1/source-review.json", fresh_dir / "source-review.json",
        ROOT / "results/semantic-graph-v7/tranche-001/review-completed-v2/source-review.json"]
    reports = [json.loads(path.read_bytes()) for path in report_paths[:2]] + [tranche_review]
    judgments = [j for report in reports for j in report["judgments"]]
    entities, mentions, assertions, metadata, permissions = {}, {}, {}, [], []
    documents = {}
    for ref in collection.manifest.shards:
        documents.update(collection.get_shard(ref.shard_id).documents)
    for judgment in judgments:
        if judgment["status"] != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED":
            continue
        row = by_id[judgment["unit_id"]]
        decision = next(d for d in row["decisions"] if (d["kind"], d["index"]) == (judgment["kind"], judgment["index"]))
        if (judgment["candidate_sha256"] != sha256_bytes(canonical_json(decision))
                or row["text_sha256"] != judgment["text_sha256"]):
            raise ValueError("TYPED_EXPORT_JUDGMENT_BINDING")
        for occurrence in row["occurrences"]:
            chunk = chunks[occurrence["chunk_id"]]
            document = documents[chunk.document_version_id]
            if chunk.text_sha256 != row["text_sha256"]:
                raise ValueError("TYPED_EXPORT_SOURCE_OCCURRENCE_BINDING")
            if judgment["kind"] == "PERMISSION":
                record = decision["normalized_record"]
                quote = record["evidence_quote"]
                start = chunk.start_char + chunk.text.index(quote)
                if document.normalized_text[start:start + len(quote)] != quote:
                    raise ValueError("TYPED_EXPORT_PERMISSION_SOURCE")
                permissions.append({"record": record, "candidate_sha256": judgment["candidate_sha256"],
                    "source_review": judgment, "document_version_id": document.document_version_id,
                    "chunk_id": chunk.chunk_id, "evidence_start_char": start, "evidence_end_char": start + len(quote),
                    "receipt_sha256": sha256_bytes(canonical_json(row["receipt"])), "binary_edge_projected": False})
                continue
            es, ms, assertion, info = canonicalize_reviewed(decision, judgment, chunk=chunk,
                document=document, native_recipe=recipe, composite=composite)
            for entity in es:
                if entity.entity_id in entities and entities[entity.entity_id] != entity:
                    raise ValueError("TYPED_EXPORT_ENTITY_IDENTITY_CONFLICT")
                entities[entity.entity_id] = entity
            mentions.update((m.mention_id, m) for m in ms)
            assertions[assertion.assertion_id] = assertion
            metadata.append(dict(info, unit_id=row["unit_id"], source_uri=document.source_uri,
                receipt_sha256=sha256_bytes(canonical_json(row["receipt"])), operation_id=row["receipt"]["operation_id"]))
    OUTPUT.mkdir(exist_ok=True)
    inventory_path = native_dir / "source-inventory-v7.jsonl"
    inventory = [json.loads(line) for line in inventory_path.read_bytes().splitlines()]
    done = {r["text_sha256"]: r for r in rows}
    statuses = [dict(u, status="NATIVE_V7_RECEIPT_READY" if u["text_sha256"] in done else "PENDING_NATIVE_V7",
        coverage_note=done[u["text_sha256"]]["coverage_note"] if u["text_sha256"] in done else None) for u in inventory]
    es, ms, rs = (tuple(values[k] for k in sorted(values)) for values in (entities, mentions, assertions))
    sidecar = {"recipe": canonical_recipe(recipe, composite).model_dump(mode="json"), "assertions": metadata,
        "permissions": permissions, "review_reports": {str(p.relative_to(ROOT)): sha256_bytes(p.read_bytes()) for p in report_paths},
        "human_reviewed": 0, "serving_published": False}
    sidecar_raw = canonical_json(sidecar) + b"\n"
    overlay = CollectionGraphOverlayV2(tenant_id="demo", collection_version=collection.manifest.collection_version,
        collection_manifest_sha256=collection.manifest_sha256, graph_sha256=graph_fingerprint(collection.manifest_sha256, es, ms, rs),
        source_inventory_sha256=sha256_bytes(inventory_path.read_bytes()), extraction_recipe_sha256=canonical_recipe(recipe, composite).fingerprint,
        aliases_sha256=recipe.parameters["typed_catalog_sha256"], curation_sha256=sha256_bytes(canonical_json(judgments)),
        upstream_review_queue_sha256=sha256_bytes(sidecar_raw), inventory_units=len(inventory), processed_units=len(rows),
        entities=es, mentions=ms, assertions=rs)
    raw = canonical_json(overlay.model_dump(mode="json")) + b"\n"
    immutable(OUTPUT / "source-review-sidecar.json", sidecar_raw)
    immutable(OUTPUT / "source-statuses.jsonl", b"".join(canonical_json(u) + b"\n" for u in statuses))
    immutable(OUTPUT / "graph.json", raw)
    loaded = load_collection_graph_v2(OUTPUT / "graph.json", collection, trusted_sha256=sha256_bytes(raw), allow_experimental=True)
    if canonical_json(loaded.overlay.model_dump(mode="json")) != canonical_json(overlay.model_dump(mode="json")):
        raise ValueError("TYPED_EXPORT_ROUNDTRIP_MISMATCH")
    adjacency = defaultdict(list)
    for assertion in rs:
        adjacency[assertion.subject_id].append({"assertion_id": assertion.assertion_id, "neighbor": assertion.object_id,
            "direction": "OUT", "chunk_id": assertion.chunk_id})
        adjacency[assertion.object_id].append({"assertion_id": assertion.assertion_id, "neighbor": assertion.subject_id,
            "direction": "IN", "chunk_id": assertion.chunk_id})
    immutable(OUTPUT / "adjacency.json", canonical_json(dict(adjacency)) + b"\n")
    report = {"status": "LOCAL_SOURCE_EXACT_V2_EXPORT_RELOAD_MATCH_NOT_SERVING", "graph_sha256": sha256_bytes(raw),
        "bytes": len(raw), "entities": len(es), "mentions": len(ms), "assertions": len(rs), "permission_triads": len(permissions),
        "documents_with_assertions": len({a.document_version_id for a in rs}), "processed_units": len(rows), "inventory_units": len(inventory),
        "pending_units": len(inventory) - len(rows), "coverage_notes": dict(Counter(r["coverage_note"] for r in rows)),
        "processed_uri_groups": len({o["service_group"] for r in rows for o in r["occurrences"]}),
        "own_receipt_cost_micro_usd": sum(r["receipt"]["cost_micro_usd"] for r in rows),
        "provider_calls": 0, "aws_calls": 0, "human_reviewed": 0, "serving_records_added": 0, "full_graph": False,
        "neo4j_roundtrip": "NOT_RUN", "recall": "NOT_ESTIMATED", "answer_quality": "NOT_MEASURED"}
    immutable(OUTPUT / "report.json", canonical_json(report) + b"\n")
    print(json.dumps(report))
    return report


if __name__ == "__main__":
    export()
