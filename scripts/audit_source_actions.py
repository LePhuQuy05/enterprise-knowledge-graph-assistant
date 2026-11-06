"""Bind explicit AI source reviews, export a local typed action and trace it."""

import json
from collections import Counter

from app.contracts import canonical_json, sha256_bytes
from app.source_action_graph import SourceActionGraph, canonicalize_action
from scripts.run_clause_graph_qualification import immutable
from scripts.run_semantic_graph_canary import load_collection_inputs
from scripts.run_source_actions_qualification import DIRECTORY, ROOT, freeze

CONFIG = ROOT / "configs/extraction/semantic-v10-source-review-v1.json"


def main():
    plan, chunks, recipe = freeze()
    collection, _, _ = load_collection_inputs()
    rows = {unit["v10_unit_id"]: json.loads((DIRECTORY / "units" / (unit["v10_unit_id"] + ".json")).read_bytes()) for unit in plan["selection"]}
    config = json.loads(CONFIG.read_bytes())
    candidates = {(unit, kind, c["index"]): (row, c) for unit, row in rows.items()
        for kind, key in (("RELATION", "accepted_relationships"), ("PERMISSION", "accepted_permissions"), ("ACTION", "accepted_actions"))
        for c in row[key]}
    judgments = {}
    for judgment in config["reviews"]:
        key = judgment["unit_id"], judgment["kind"], judgment["index"]
        if key not in candidates or key in judgments:
            raise ValueError("ACTION_REVIEW_UNKNOWN_OR_DUPLICATE_RECORD")
        row, candidate = candidates[key]
        if (judgment["candidate_sha256"] != sha256_bytes(canonical_json(candidate))
                or judgment["source_sha256"] != row["text_sha256"] or not judgment["reason_vi"].strip()):
            raise ValueError("ACTION_REVIEW_SOURCE_OR_CANDIDATE_BINDING")
        judgments[key] = judgment
    if judgments.keys() != candidates.keys() or config["human_reviewed"] != 0:
        raise ValueError("ACTION_REVIEW_INCOMPLETE_OR_MISLABELED")
    documents = {}
    for ref in collection.manifest.shards:
        documents.update(collection.get_shard(ref.shard_id).documents)
    assertions = []
    for key, judgment in judgments.items():
        if key[1] != "ACTION" or judgment["status"] != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED":
            continue
        row, candidate = candidates[key]
        for occurrence in row["occurrences"]:
            chunk = chunks[occurrence["chunk_id"]]
            assertions.append(canonicalize_action(candidate, judgment, chunk=chunk,
                document=documents[chunk.document_version_id], native_recipe=recipe))
    graph = SourceActionGraph(assertions)
    traces = [trace for assertion in assertions for trace in graph.evidence_paths(assertion["nodes"][0]["node_id"], hops=2)]
    # Reload serialized content before checking the exported consumer, not just
    # an in-memory candidate dictionary.
    artifact = {"version": "reviewed-source-actions-experimental-v1", "source_manifest_sha256": collection.manifest_sha256,
        "recipe_sha256": recipe.fingerprint, "assertions": assertions, "serving_published": False, "human_reviewed": 0}
    reloaded = SourceActionGraph(json.loads(canonical_json(artifact))["assertions"])
    expected = [trace for assertion in assertions for trace in reloaded.evidence_paths(assertion["nodes"][0]["node_id"], hops=2)]
    if canonical_json(traces) != canonical_json(expected):
        raise ValueError("ACTION_GRAPH_RELOAD_TRACE_MISMATCH")
    report = {"status": "AI_SOURCE_REVIEW_AND_LOCAL_ACTION_TRACE_COMPLETE_NOT_SERVING", "source_units": len(rows),
        "guard_pass_records_reviewed": len(judgments), "review_statuses": dict(Counter(j["status"] for j in judgments.values())),
        "by_kind": dict(Counter(key[1] for key in judgments)), "action_assertions_exported": len(assertions), "trace_results": len(traces),
        "action_graph_sha256": sha256_bytes(canonical_json(artifact) + b"\n"), "config_sha256": sha256_bytes(CONFIG.read_bytes()),
        "model_calls": 0, "aws_calls": 0, "human_reviewed": 0, "serving_records_added": 0,
        "accuracy_or_recall_estimated": False, "full_batch_gate": "NOT_READY_SOURCE_SYNTAX_AND_POSITIVE_CONTROLS_PENDING"}
    output = DIRECTORY / "source-review-v1"
    output.mkdir(exist_ok=True)
    for name, value in (("actions.json", artifact), ("traces.json", traces), ("report.json", report)):
        immutable(output / name, canonical_json(value) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
