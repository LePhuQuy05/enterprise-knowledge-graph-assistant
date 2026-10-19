import dataclasses
import json

import pytest

from app.context import pack_context
from app.fixtures import fixture_vector
from app.graph_retrieval import GraphIndex


def path_result(loaded):
    index = GraphIndex(loaded)
    entity = next(e.entity_id for e in loaded.content.entities if e.canonical_name == "Service A")
    index.seeds = lambda *args: (entity,)
    result = index.retrieve("Service A", fixture_vector("Service A"), hops=2)
    candidate = next(c for c in result.graph_candidates if len(c.assertion_path) == 2 and
                     len(set((c.chunk_id, *c.supporting_chunk_ids))) >= 2)
    return dataclasses.replace(result, ranked_chunk_ids=(candidate.chunk_id,), graph_candidates=(candidate,)), candidate


def test_two_hop_evidence_is_all_present_and_source_ids_are_trusted_slices(loaded):
    result, candidate = path_result(loaded)
    packed = pack_context(loaded, result)
    assert set(candidate.supporting_chunk_ids).issubset(packed.chunk_ids)
    assert set(candidate.assertion_path) == set(packed.assertion_ids)
    value = json.loads(packed.serialized)
    for source in value["untrusted_sources"]:
        doc = loaded.documents[source["document_version_id"]]
        assert doc.normalized_text[source["start_char"]:source["end_char"]] == source["untrusted_text"]
    assert packed.proxy_tokens <= 4096 and not packed.provider_exact_token_count


def test_path_bundle_too_large_is_skipped_whole_with_warning(loaded):
    result, _ = path_result(loaded)
    packed = pack_context(loaded, result, max_chunks=1)
    assert not packed.chunk_ids and not packed.assertion_ids
    assert packed.warnings == ("CONTEXT_BUNDLE_SKIPPED",)


def test_baseline_and_graph_use_same_packer_and_header_budget(loaded):
    result = GraphIndex(loaded).retrieve("Service A", fixture_vector("Service A"), hops=0)
    packed = pack_context(loaded, result, proxy_token_budget=500)
    assert len(packed.chunk_ids) <= 8 and packed.proxy_tokens <= 500
    assert not packed.assertion_ids


def test_invalid_support_is_an_error_not_a_fallback(loaded):
    result, candidate = path_result(loaded)
    bad = dataclasses.replace(candidate, supporting_chunk_ids=())
    with pytest.raises(ValueError, match="SUPPORT_MISMATCH"):
        pack_context(loaded, dataclasses.replace(result, graph_candidates=(bad,)))
