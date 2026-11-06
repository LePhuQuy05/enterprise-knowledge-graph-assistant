"""Export reviewed composite relations/actions with source checks and deduplication."""

import json
from collections import Counter

from app.collection_graph import graph_fingerprint
from app.collection_graph_v2 import CollectionGraphOverlayV2, load_collection_graph_v2
from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes
from app.source_action_graph import SourceActionGraph
from app.source_syntax import SyntaxDocument
from app.syntax_graph_records import canonical_recipe, canonicalize
from scripts.freeze_source_syntax_queue import OUTPUT as QUEUE
from scripts.inventory_source_actions import OUTPUT as INVENTORY
from scripts.inventory_source_actions import PARSED, sentence_document
from scripts.run_clause_graph_qualification import immutable
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs

OUTPUT = ROOT / "results/source-syntax-reviewed-v1"
CONFIG = ROOT / "configs/extraction/source-syntax-source-review-v1.json"


def frame_signature(candidate, parsed):
    clause = candidate["record"]["clause_quote"]
    offset = candidate["evidence"]["quote"].index(clause)
    # Root source positions permit dedup of 'bucket' versus 'the bucket' without
    # merging different documents, assertions, branches, polarities or payloads.
    return (candidate["source_sha256"], candidate["kind"],
        candidate["record"].get("predicate", candidate["record"].get("action_family")),
        offset + candidate["verb_token"]["start"], candidate["record"]["polarity"], candidate["record"]["modality"],
        offset + parsed.tokens[candidate["actor"]["root_token"]]["start"],
        offset + parsed.tokens[candidate["target"]["root_token"]]["start"],
        candidate.get("payload_span", {}).get("quote") if candidate.get("payload_span") else None)


def main():
    collection, chunks, _ = load_collection_inputs()
    native_plan = json.loads((ROOT / "results/semantic-graph-v10/qualification-v1/plan.json").read_bytes())
    native_recipe = ExtractionRecipe.model_validate_json(canonical_json(native_plan["recipe"]))
    queue = json.loads((QUEUE / "queue.json").read_bytes())
    occurrences = {r["unit_id"]: r["occurrences"] for r in queue["rows"]}
    inputs = {}
    for row in json.loads((QUEUE / "comparison-v3/report.json").read_bytes())["rows"]:
        if row["syntax_guard_admitted"]:
            inputs["COMPOSITE", row["unit_id"], row["candidate_sha256"]] = row["candidate"], occurrences[row["unit_id"]]
    for row in map(json.loads, (INVENTORY / "proposals.jsonl").read_bytes().splitlines()):
        inputs["INVENTORY", row["source_unit_id"], row["candidate_sha256"]] = row["candidate"], row["occurrences"]
    config = json.loads(CONFIG.read_bytes())
    judgments = {(j["origin"], j["source_unit_id"], j["candidate_sha256"]): j for j in config["reviews"]}
    if len(judgments) != len(config["reviews"]) or judgments.keys() != inputs.keys() or config["human_reviewed"] != 0:
        raise ValueError("SYNTAX_EXPORT_REVIEW_INCOMPLETE_DUPLICATE_OR_MISLABELED")
    parsed_artifact = json.loads((QUEUE / "parsed.json").read_bytes())
    installation, parses = parsed_artifact["installation"], {}
    for d in parsed_artifact["documents"]:
        parses[sha256_bytes(canonical_json({"text": d["text"], "tokens": d["tokens"]}))] = SyntaxDocument(d)
    needed_sources = {candidate["source_sha256"] for (origin, _, _), (candidate, _) in inputs.items() if origin == "INVENTORY"}
    with PARSED.open(encoding="utf-8") as handle:
        for line in handle:
            data = json.loads(line)
            if "document" not in data:
                continue
            d = data["document"]
            if sha256_bytes(d["text"].encode()) not in needed_sources:
                continue
            for sentence in d["sentences"]:
                parsed = sentence_document(d, sentence["start"], sentence["end"])
                key = sha256_bytes(canonical_json({"text": parsed.text, "tokens": parsed.tokens}))
                parses[key] = parsed
    documents = {}
    for ref in collection.manifest.shards:
        documents.update(collection.get_shard(ref.shard_id).documents)
    entities, mentions, relations, actions, metadata, duplicates, seen = {}, {}, {}, {}, [], [], set()
    for key in sorted(inputs, key=lambda k: (k[0] != "COMPOSITE", k)):
        candidate, copies = inputs[key]
        judgment = judgments[key]
        if (sha256_bytes(canonical_json(candidate)) != judgment["candidate_sha256"] or candidate["source_sha256"] != judgment["source_sha256"]):
            raise ValueError("SYNTAX_EXPORT_REVIEW_CANDIDATE_OR_SOURCE_CHANGED")
        if judgment["status"] != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED":
            continue
        signature = frame_signature(candidate, parses[candidate["parse_sha256"]])
        if signature in seen:
            duplicates.append({"candidate_sha256": key[2], "reason": "SAME_SOURCE_VERB_ARGUMENT_ROOTS_POLARITY_MODALITY_PAYLOAD_PREFER_COMPOSITE"})
            continue
        seen.add(signature)
        for occurrence in copies:
            chunk = chunks[occurrence["chunk_id"]]
            es, ms, assertion, sidecar = canonicalize(candidate, judgment, chunk=chunk, document=documents[chunk.document_version_id],
                parsed=parses[candidate["parse_sha256"]], native_recipe=native_recipe, installation=installation)
            for entity in es:
                if entity.entity_id in entities and entities[entity.entity_id] != entity:
                    raise ValueError("SYNTAX_EXPORT_ENTITY_IDENTITY_COLLISION")
                entities[entity.entity_id] = entity
            mentions.update((m.mention_id, m) for m in ms)
            if candidate["kind"] == "RELATION":
                relations[assertion.assertion_id] = assertion
            else:
                actions[assertion["assertion_id"]] = assertion
            metadata.append(sidecar)
    recipe = canonical_recipe(native_recipe, installation)
    es, ms, rs = tuple(entities.values()), tuple(mentions.values()), tuple(relations.values())
    overlay = CollectionGraphOverlayV2(tenant_id=collection.manifest.tenant_id,
        collection_version=collection.manifest.collection_version, collection_manifest_sha256=collection.manifest_sha256,
        graph_sha256=graph_fingerprint(collection.manifest_sha256, es, ms, rs),
        source_inventory_sha256=sha256_bytes((INVENTORY / "plan.json").read_bytes()), extraction_recipe_sha256=recipe.fingerprint,
        aliases_sha256=sha256_bytes((ROOT / "configs/extraction/aws-approved-aliases-v2.json").read_bytes()),
        curation_sha256=sha256_bytes(CONFIG.read_bytes()), upstream_review_queue_sha256=sha256_bytes((QUEUE / "queue.json").read_bytes()),
        inventory_units=2897, processed_units=len({j["source_sha256"] for j in judgments.values()}),
        entities=es, mentions=ms, assertions=rs)
    OUTPUT.mkdir(exist_ok=True)
    immutable(OUTPUT / "relations-v2.json", canonical_json(overlay.model_dump(mode="json")) + b"\n")
    loaded = load_collection_graph_v2(OUTPUT / "relations-v2.json", collection,
        trusted_sha256=sha256_bytes((OUTPUT / "relations-v2.json").read_bytes()), allow_experimental=True)
    action_graph = SourceActionGraph(list(actions.values()))
    trace = [r for seed in sorted(action_graph.adjacency) for r in action_graph.evidence_paths(seed, hops=2, max_edges=100)]
    action_artifact = {"version": "source-actions-reviewed-composite-v1", "recipe": recipe.model_dump(mode="json"),
        "source_manifest_sha256": collection.manifest_sha256, "assertions": list(actions.values()), "serving_published": False, "human_reviewed": 0}
    immutable(OUTPUT / "actions.json", canonical_json(action_artifact) + b"\n")
    reloaded = SourceActionGraph(json.loads((OUTPUT / "actions.json").read_bytes())["assertions"])
    if canonical_json(trace) != canonical_json([r for seed in sorted(reloaded.adjacency) for r in reloaded.evidence_paths(seed, hops=2, max_edges=100)]):
        raise ValueError("SYNTAX_EXPORT_ACTION_RELOAD_TRACE_MISMATCH")
    report = {"status": "AI_SOURCE_REVIEWED_LOCAL_EXPORT_SOURCE_RELOAD_AND_TRACE_PASS_NOT_SERVING",
        "source_manifest_sha256": collection.manifest_sha256, "reviews": len(judgments),
        "review_statuses": dict(Counter(j["status"] for j in judgments.values())), "duplicates_not_materialized": duplicates,
        "entities": len(es), "mentions": len(ms), "relation_assertions": len(loaded.overlay.assertions),
        "source_action_assertions": len(actions), "action_families": dict(Counter(a["record"]["action_family"] for a in actions.values())),
        "action_negative_assertions": sum(a["record"]["polarity"] == "NEGATIVE" for a in actions.values()),
        "trace_results": len(trace), "two_hop_trace_results": sum(len(t["path"]) == 2 for t in trace),
        "reviewed_candidate_source_units": overlay.processed_units,
        "documents_with_retained_assertions": len({m.document_version_id for m in ms}), "parser_inventory_units": 2897,
        "native_v10_processing_units": 32, "full_relation_extraction": False, "semantic_accuracy_measured": False,
        "human_reviewed": 0, "model_api_calls": 0, "aws_calls": 0, "serving_records_added": 0,
        "relations_file_sha256": sha256_bytes((OUTPUT / "relations-v2.json").read_bytes()),
        "actions_file_sha256": sha256_bytes((OUTPUT / "actions.json").read_bytes())}
    for name, content in (("sidecar.json", metadata), ("traces.json", trace), ("report.json", report)):
        immutable(OUTPUT / name, canonical_json(content) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
