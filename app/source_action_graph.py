"""Reviewed action assertions and bounded evidence traversal, separate from V2.

Projection edges are retrieval hints into complete records, never standalone
binary facts or proof that an indirect path is a valid AWS workflow.
"""

from collections import deque
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.extraction import ExtractionError, check_source
from app.semantic_extraction_v10 import (
    SourceActionV1,
    semantic_raw_recipe,
    validate_semantic_output,
)


def canonicalize_action(candidate, judgment, *, chunk, document, native_recipe):
    check_source(chunk, document)
    semantic_raw_recipe(native_recipe)
    if (judgment.get("status") != "AI_SOURCE_SUPPORTED_SOURCE_SCOPED"
            or judgment.get("candidate_sha256") != sha256_bytes(canonical_json(candidate))):
        raise ExtractionError("ACTION_GRAPH_BOUND_REVIEW_REQUIRED")
    raw = candidate["normalization"]["raw_record"]
    validation = validate_semantic_output(canonical_json({"relationships": [], "permission_requirements": [],
        "actions": [raw], "deferred_statements": [], "coverage_note": "HAS_CANDIDATES"}).decode(), chunk.text, stop_reason="end_turn")
    if len(validation.accepted_actions) != 1:
        raise ExtractionError("ACTION_GRAPH_REVALIDATION_REJECTED")
    expected = validation.accepted_actions[0] | {"index": candidate["index"]}
    if canonical_json(expected) != canonical_json(candidate):
        raise ExtractionError("ACTION_GRAPH_CANDIDATE_OR_SOURCE_CHANGED")
    record = SourceActionV1.model_validate_json(canonical_json(candidate["record"]))
    nodes = []
    for endpoint, identity in zip((record.actor, record.target, record.enabler), candidate["endpoint_identities"]):
        key = identity["canonical_key"]
        if identity["scope"] == "SOURCE_LOCAL":
            key = stable_id("action-source-local", key=key, document=document.document_version_id,
                chunk=chunk.chunk_id, clause_start=candidate["clause_span"]["start_char"])
        node_id = stable_id("action-node", key=key)
        start, end = chunk.start_char + identity["start_char"], chunk.start_char + identity["end_char"]
        if document.normalized_text[start:end] != endpoint.name:
            raise ExtractionError("ACTION_GRAPH_MENTION_SOURCE")
        nodes.append({"node_id": node_id, "name": endpoint.name, "semantic_type": endpoint.semantic_type,
            "identity_scope": identity["scope"], "source_span": {"start_char": start, "end_char": end, "quote": endpoint.name},
            "head_quote": endpoint.head_quote, "restrictions": list(endpoint.restrictions), "scope_qualifiers": list(endpoint.scope_qualifiers)})
    content = {"version": "source-action-assertion-v1", "native_recipe_sha256": native_recipe.fingerprint,
        "adapter_sha256": sha256_bytes(Path(__file__).read_bytes()), "document_version_id": document.document_version_id,
        "chunk_id": chunk.chunk_id, "source_uri": document.source_uri, "record": record.model_dump(mode="json"),
        "nodes": nodes, "evidence": {"start_char": chunk.start_char, "end_char": chunk.end_char, "quote": chunk.text},
        "clause_span": {"start_char": chunk.start_char + candidate["clause_span"]["start_char"],
            "end_char": chunk.start_char + candidate["clause_span"]["end_char"], "quote": record.clause_quote},
        "source_review": judgment, "candidate_sha256": judgment["candidate_sha256"], "human_reviewed": False,
        "serving_published": False, "context_policy": candidate["context_policy"]}
    return {"assertion_id": stable_id("source-action", **content)} | content


class SourceActionGraph:
    """Return evidence paths only; caller must still judge scope and answer support."""

    def __init__(self, assertions):
        if len(assertions) > 20000:
            raise ValueError("ACTION_GRAPH_RECORD_BOUND")
        self.records, self.adjacency = {}, {}
        for assertion in assertions:
            body = {k: v for k, v in assertion.items() if k != "assertion_id"}
            if assertion["assertion_id"] != stable_id("source-action", **body):
                raise ValueError("ACTION_GRAPH_CHECKSUM")
            if assertion["assertion_id"] in self.records:
                raise ValueError("ACTION_GRAPH_DUPLICATE")
            self.records[assertion["assertion_id"]] = assertion
            # Negated assertions can be displayed as source records, not traversed
            # as if a delivery/invocation actually exists.
            if assertion["record"]["polarity"] != "POSITIVE":
                continue
            actor, target = (n["node_id"] for n in assertion["nodes"][:2])
            self.adjacency.setdefault(actor, []).append({"target": target, "assertion_id": assertion["assertion_id"],
                "role": assertion["record"]["target_role"], "family": assertion["record"]["action_family"]})

    def evidence_paths(self, seed, *, hops=1, max_edges=20):
        if type(hops) is not int or hops not in {0, 1, 2} or type(max_edges) is not int or not 1 <= max_edges <= 100:
            raise ValueError("ACTION_GRAPH_TRAVERSAL_BOUND")
        queue, visited, results = deque([(seed, ())]), {seed}, []
        while queue and len(results) < max_edges:
            node, path = queue.popleft()
            if len(path) >= hops:
                continue
            for edge in sorted(self.adjacency.get(node, []), key=lambda e: e["assertion_id"]):
                full_path = path + (edge["assertion_id"],)
                results.append({"path": list(full_path), "edge": edge, "assertion": self.records[edge["assertion_id"]],
                    "interpretation": "RETRIEVAL_EVIDENCE_NOT_TRANSITIVE_WORKFLOW_PROOF"})
                if len(results) == max_edges:
                    break
                if edge["target"] not in visited:
                    visited.add(edge["target"])
                    queue.append((edge["target"], full_path))
        return results
