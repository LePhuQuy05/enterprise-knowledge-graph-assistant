import json
from pathlib import Path

import pytest

from app.contracts import sha256_bytes

ROOT = Path(__file__).resolve().parents[1]


def test_every_method_and_source_belongs_to_collection316_with_sparse_graph_scope():
    data = json.loads((ROOT / "frontend/tracer/methods-data.json").read_bytes())
    graph = data["graph"]
    assert graph["documents"] == 316 and graph["chunks"] == 2978
    assert graph["review"] == "PENDING" and not graph["full_relation_extraction"]
    assert graph["relation_source_documents"] < 316
    assert all(e["collection_version"] == graph["snapshot_version"] for e in data["examples"])
    assert all(e["config"]["collection_manifest_sha256"] == graph["manifest_sha256"] for e in data["examples"])
    for source in data["sources"].values():
        assert sha256_bytes(source["text"].encode()) == source["text_sha256"]
    assert len((ROOT / "frontend/tracer/methods-data.json").read_bytes()) <= 524288
    assert graph["legacy_graph_reused"] is False and graph["human_reviewed"] is False
    assert graph["source_review"] == "AI_PROVISIONAL_NOT_HUMAN_GOLD"
    for mode in graph["modes"]:
        assert mode["response"]["claims"] and mode["response"]["citations"]
        for citation in mode["response"]["citations"]:
            assert citation["chunk_id"] in mode["context_chunk_ids"]
            assert citation["quote"] in data["sources"][citation["chunk_id"]]["text"]


def test_method_source_offsets_match_local_collection():
    path = ROOT / "results/workbench-cloud/assets-build-v1.json"
    if not path.is_file():
        pytest.skip("Requires the local, uncommitted collection316 source bundle")
    pipeline = json.loads(path.read_bytes())
    directory = Path(pipeline["bundle_directory"]) / "collection"
    if not directory.is_dir():
        pytest.skip("The recorded collection316 source bundle is not available on this machine")
    documents = {}
    for shard in directory.iterdir():
        if shard.is_dir():
            documents.update((d["document_version_id"], d) for d in map(
                json.loads, (shard / "documents.jsonl").read_text(encoding="utf-8").splitlines()))
    assert len(documents) == 316
    data = json.loads((ROOT / "frontend/tracer/methods-data.json").read_bytes())
    for source in data["sources"].values():
        doc = documents[source["document_version_id"]]
        assert doc["normalized_text"][source["start_char"]:source["end_char"]] == source["text"]


def test_public_projection_excludes_private_provider_owner_and_funding_payloads():
    raw = (ROOT / "frontend/tracer/methods-data.json").read_text(encoding="utf-8")
    for key in ('"raw_output"', '"owner_id"', '"idempotency_key"', '"preflight_counters"',
                '"operation_id"', '"temporary_password"', '"access_token"', '"funding_id"', 'C:\\'):
        assert key not in raw
    html = (ROOT / "frontend/tracer/index.html").read_text(encoding="utf-8")
    for method in ("bm25", "vector", "hybrid", "hybrid_reranked", "hybrid_multi_query", "hybrid_multi_query_balanced", "graph_1hop", "graph_2hop"):
        assert f'data-page="method-{method}"' in html
        assert f'id="page-method-{method}"' in html
    assert 'data-page="benchmark"' not in html
