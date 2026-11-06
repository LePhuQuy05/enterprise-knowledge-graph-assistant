from copy import deepcopy

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.semantic_extraction_v10 import semantic_recipe
from app.source_action_graph import SourceActionGraph, canonicalize_action
from tests.test_extraction import aliases, source
from tests.test_semantic_extraction_v10 import action, ep, validate


def fixture(*, negative=False, suffix="a"):
    clause = "CloudTrail does not deliver log files to an S3 bucket." if negative else "CloudTrail delivers log files to an S3 bucket."
    text = "If this configuration is enabled, the documented behavior applies. " + clause
    row = action(clause, ep("CloudTrail", "SERVICE_PRODUCT"), ep("S3 bucket", "RESOURCE_TYPE", "bucket"),
        action_quote="does not deliver" if negative else "delivers", polarity="NEGATIVE" if negative else "POSITIVE")
    candidate = validate(text, actions=[row]).accepted_actions[0]
    judgment = {"status": "AI_SOURCE_SUPPORTED_SOURCE_SCOPED", "candidate_sha256": sha256_bytes(canonical_json(candidate)),
        "reason_vi": "Synthetic mechanism fixture, not a real AWS source judgment."}
    doc, chunk = source(text, suffix=suffix)
    recipe = semantic_recipe(aliases(), metadata_sha256="0" * 64)
    assertion = canonicalize_action(candidate, judgment, chunk=chunk, document=doc, native_recipe=recipe)
    return assertion, candidate, judgment, doc, chunk, recipe


def test_traversal_returns_complete_payload_roles_context_and_source_spans():
    assertion, *_ = fixture()
    graph = SourceActionGraph([assertion])
    actor = assertion["nodes"][0]["node_id"]
    trace = graph.evidence_paths(actor, hops=1)[0]
    assert trace["assertion"]["record"]["payload_quote"] == "log files"
    assert trace["edge"]["role"] == "DESTINATION"
    assert trace["assertion"]["evidence"]["quote"].startswith("If this configuration")
    assert trace["interpretation"] == "RETRIEVAL_EVIDENCE_NOT_TRANSITIVE_WORKFLOW_PROOF"
    assert graph.evidence_paths(actor, hops=0) == []
    assert graph.evidence_paths(assertion["nodes"][1]["node_id"], hops=1) == []


def test_source_local_resource_nodes_do_not_bridge_different_documents_by_name():
    a, *_ = fixture(suffix="a")
    b, *_ = fixture(suffix="b")
    assert a["nodes"][0]["node_id"] == b["nodes"][0]["node_id"]
    assert a["nodes"][1]["node_id"] != b["nodes"][1]["node_id"]


@pytest.mark.parametrize("change", ["payload", "scope", "review", "normalization"])
def test_rehashed_tampered_candidates_or_unreviewed_records_are_rejected(change):
    _, candidate, judgment, doc, chunk, recipe = fixture()
    candidate, judgment = deepcopy(candidate), deepcopy(judgment)
    if change == "payload":
        candidate["record"]["payload_quote"] = "logs"
    elif change == "scope":
        candidate["evidence"]["quote"] = candidate["record"]["clause_quote"]
    elif change == "review":
        judgment["status"] = "PENDING"
    else:
        candidate["normalization"]["edits"] = [{"reason": "INVENTED"}]
    judgment["candidate_sha256"] = sha256_bytes(canonical_json(candidate))
    with pytest.raises(ValueError):
        canonicalize_action(candidate, judgment, chunk=chunk, document=doc, native_recipe=recipe)


def test_negative_records_are_visible_but_never_positive_traversal_edges():
    assertion, *_ = fixture(negative=True)
    graph = SourceActionGraph([assertion])
    assert len(graph.records) == 1
    assert graph.evidence_paths(assertion["nodes"][0]["node_id"]) == []


@pytest.mark.parametrize("kwargs", [dict(hops=3), dict(hops=True), dict(max_edges=101), dict(max_edges=0)])
def test_traversal_is_bounded(kwargs):
    assertion, *_ = fixture()
    with pytest.raises(ValueError, match="BOUND"):
        SourceActionGraph([assertion]).evidence_paths(assertion["nodes"][0]["node_id"], **kwargs)
