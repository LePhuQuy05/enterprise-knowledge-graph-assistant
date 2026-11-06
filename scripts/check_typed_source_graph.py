"""Offline integrity, provenance and bounded size checks on the real export."""

import json
from collections import Counter, defaultdict

from app.collection_graph import graph_fingerprint
from app.collection_graph_v2 import load_collection_graph_v2, validate_graph_sources_v2
from app.contracts import canonical_json, sha256_bytes
from scripts.run_semantic_graph_canary import ROOT, load_collection_inputs


def main():
    collection, _, _ = load_collection_inputs()
    base = ROOT / "results/semantic-graph-v7"
    directory = base / "typed-export-v1"
    report = json.loads((directory / "report.json").read_bytes())
    graph = load_collection_graph_v2(directory / "graph.json", collection,
        trusted_sha256=report["graph_sha256"], allow_experimental=True).overlay
    sidecar_raw = (directory / "source-review-sidecar.json").read_bytes()
    if sha256_bytes(sidecar_raw) != graph.upstream_review_queue_sha256:
        raise ValueError("TYPED_CHECK_SIDECAR_CHANGED")
    sidecar = json.loads(sidecar_raw)
    recipe = sidecar["recipe"]
    if (sha256_bytes(canonical_json(recipe)) != graph.extraction_recipe_sha256
            or recipe["parameters"]["canonicalizer_implementation_sha256"] != sha256_bytes((ROOT / "app/typed_graph_records.py").read_bytes())):
        raise ValueError("TYPED_CHECK_CANONICAL_RECIPE_CHANGED")
    for stage in ("qualification-v1", "composite-qualification-v1", "tranche-001/job-1", "tranche-001/job-2", "tranche-001/job-3"):
        plan = json.loads((base / stage / "plan.json").read_bytes())
        for path, expected in plan["code_sha256"].items():
            if sha256_bytes((ROOT / path).read_bytes()) != expected:
                raise ValueError("TYPED_CHECK_FROZEN_IMPLEMENTATION_CHANGED:" + path)
    for path, expected in sidecar["review_reports"].items():
        if sha256_bytes((ROOT / path).read_bytes()) != expected:
            raise ValueError("TYPED_CHECK_SOURCE_REVIEW_CHANGED")
    rows = [json.loads(line) for line in (directory / "source-statuses.jsonl").read_bytes().splitlines()]
    statuses = Counter(row["status"] for row in rows)
    if len(rows) != graph.inventory_units or statuses["NATIVE_V7_RECEIPT_READY"] != graph.processed_units:
        raise ValueError("TYPED_CHECK_PROCESSING_STATUS_COUNTS")
    ids = {a.assertion_id for a in graph.assertions}
    if ids != {m["assertion_id"] for m in sidecar["assertions"]}:
        raise ValueError("TYPED_CHECK_ASSERTION_METADATA_COVERAGE")
    for assertion in graph.assertions:
        metadata = next(m for m in sidecar["assertions"] if m["assertion_id"] == assertion.assertion_id)
        if (metadata["serving_published"] or metadata["human_reviewed"]
                or not any(q.text == assertion.evidence.quote for q in assertion.qualifiers)):
            raise ValueError("TYPED_CHECK_SOURCE_SCOPE_OR_REVIEW_STATE")
    adjacency = json.loads((directory / "adjacency.json").read_bytes())
    expected = defaultdict(list)
    for assertion in graph.assertions:
        expected[assertion.subject_id].append({"assertion_id": assertion.assertion_id, "neighbor": assertion.object_id,
            "direction": "OUT", "chunk_id": assertion.chunk_id})
        expected[assertion.object_id].append({"assertion_id": assertion.assertion_id, "neighbor": assertion.subject_id,
            "direction": "IN", "chunk_id": assertion.chunk_id})
    if canonical_json(adjacency) != canonical_json(dict(expected)):
        raise ValueError("TYPED_CHECK_ADJACENCY_PARITY")
    negatives = {}
    try:
        load_collection_graph_v2(directory / "graph.json", collection, trusted_sha256="0" * 64, allow_experimental=True)
    except ValueError:
        negatives["bad_checksum_rejected"] = True
    # Valid structural fingerprint must not conceal a missing endpoint mention.
    first = graph.assertions[0]
    removed = graph.model_copy(update={"mentions": tuple(m for m in graph.mentions
        if (m.chunk_id, m.entity_id) != (first.chunk_id, first.subject_id))})
    removed = removed.model_copy(update={"graph_sha256": graph_fingerprint(graph.collection_manifest_sha256,
        removed.entities, removed.mentions, removed.assertions)})
    try:
        validate_graph_sources_v2(removed, collection)
    except ValueError:
        negatives["missing_endpoint_mention_rejected"] = True
    if len(negatives) != 2:
        raise ValueError("TYPED_CHECK_NEGATIVE_PROVENANCE_TEST_FAILED")
    neighbors = {e.entity_id: set() for e in graph.entities}
    for a in graph.assertions:
        neighbors[a.subject_id].add(a.object_id)
        neighbors[a.object_id].add(a.subject_id)
    unseen, components = set(neighbors), []
    while unseen:
        stack, component = [min(unseen)], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(neighbors[current] - component)
        unseen -= component
        components.append(len(component))
    output = {"status": "REAL_LOCAL_EXPORT_INTEGRITY_SOURCE_AND_NEGATIVE_CHECKS_PASS",
        "graph_sha256": report["graph_sha256"], "source_statuses": dict(statuses),
        "weakly_connected_component_sizes": sorted(components, reverse=True),
        "max_distinct_neighbor_degree": max(map(len, neighbors.values()), default=0),
        "adjacency_entries": sum(map(len, adjacency.values())), "negative_checks": negatives,
        "frozen_plan_code_checks": "MATCH", "source_review_and_scope_checks": "MATCH",
        "graph_json_bytes": (directory / "graph.json").stat().st_size,
        "source_docs_with_native_processing": len({o["document_version_id"] for r in rows
            if r["status"] == "NATIVE_V7_RECEIPT_READY" for o in r["occurrences"]}),
        "provider_calls": 0, "aws_calls": 0, "neo4j_roundtrip": "NOT_RUN", "process_rss": "NOT_MEASURED"}
    path = base / "verification-2026-10-27/artifact-integrity.json"
    raw = canonical_json(output) + b"\n"
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("TYPED_CHECK_IMMUTABLE_RESULT_CHANGED")
    path.write_bytes(raw)
    print(json.dumps(output))


if __name__ == "__main__":
    main()
