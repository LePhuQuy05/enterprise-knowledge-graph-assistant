"""Adversarial synthetic examples: balanced allocation is not a quality guarantee.

The relevance/necessity labels below are fixture construction, never AWS gold,
actual pilot targets or measured answer accuracy. No model or network is used.
"""

from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

from app.context import pack_context
from app.context_selection import select_context_order
from app.contracts import Chunk, DocumentVersion, text_hash
from app.graph_retrieval import GraphRetrievalResult
from app.snapshot import SnapshotContent
from tests.test_context import path_result
from tests.test_context_selection import trace_for

TENANT = "tradeoff-fixture"


def source(label, text=None):
    text = text if text is not None else f"Synthetic source {label}."
    document = DocumentVersion.create(tenant_id=TENANT,
        source_uri=f"https://example.invalid/tradeoffs/{label}", title=f"Synthetic {label}",
        fetched_at=datetime(2026, 10, 25, tzinfo=UTC), raw_sha256=text_hash(text),
        normalizer_version="synthetic-text-v1", normalized_text_path=f"synthetic/{label}.txt",
        normalized_text=text, source_kind="TEXT")
    chunk = Chunk.create(tenant_id=TENANT, document_version_id=document.document_version_id,
        chunking_version="synthetic-one-source-v1", section_path=("Synthetic",), position=0,
        start_char=0, end_char=len(text), text=text, token_count=max(1, len(text.split())),
        tokenizer_version="synthetic-words-v1")
    return document, chunk


def view(records):
    documents, chunks = zip(*records, strict=True)
    return SimpleNamespace(documents={row.document_version_id: row for row in documents},
        chunks={row.chunk_id: row for row in chunks}, entities={},
        content=SnapshotContent(tuple(documents), tuple(chunks)))


def pack(evidence, ids, **bounds):
    return pack_context(evidence, GraphRetrievalResult(tuple(ids), tuple(ids), (), (),
        "hybrid_multi_query_balanced", False, (), 0, 0), **bounds)


def allocate(trace, evidence, tenant=TENANT):
    return select_context_order(trace, tuple(evidence.chunks[key] for key in trace.ranked_chunk_ids),
                                tenant_id=tenant)


def test_consensus_critical_rank_four_can_be_lost_after_nine_unique_round_robin_distractors():
    critical = source("critical", "Synthetic requirement: both conditions must hold.")
    noise = [source(f"noise-{query}-{rank}") for query in range(3) for rank in range(3)]
    evidence = view([critical, *noise])
    trace = trace_for(*(tuple(row[1] for row in noise[index * 3:(index + 1) * 3]) + (critical[1],)
                        for index in range(3)))
    assert all(query.ranked_chunk_ids.index(critical[1].chunk_id) == 3 for query in trace.queries)
    assert trace.ranked_chunk_ids[0] == critical[1].chunk_id  # Three consensus votes.
    selection = allocate(trace, evidence)
    fused = pack(evidence, trace.ranked_chunk_ids)
    balanced = pack(evidence, selection.ordered_chunk_ids)
    assert critical[1].chunk_id in fused.chunk_ids
    assert selection.ordered_chunk_ids.index(critical[1].chunk_id) == 9
    assert len(balanced.chunk_ids) == 8 and critical[1].chunk_id not in balanced.chunk_ids
    assert set(selection.ordered_chunk_ids) == set(trace.ranked_chunk_ids)  # No retrieval loss.


def test_drifting_subquery_can_displace_known_relevant_sources_despite_same_candidate_pool():
    relevant = [source(f"required-{index}", f"Synthetic task requirement {index}.") for index in range(8)]
    unrelated = [source(f"drift-{index}", f"Unrelated synthetic gardening fact {index}.") for index in range(8)]
    evidence = view([*relevant, *unrelated])
    task_ids = {row[1].chunk_id for row in relevant}
    drift_ids = {row[1].chunk_id for row in unrelated}
    trace = trace_for(tuple(row[1] for row in relevant), tuple(row[1] for row in relevant),
                      tuple(row[1] for row in unrelated))
    selection = allocate(trace, evidence)
    fused = pack(evidence, trace.ranked_chunk_ids)
    balanced = pack(evidence, selection.ordered_chunk_ids)
    assert set(fused.chunk_ids) == task_ids and not set(fused.chunk_ids) & drift_ids
    assert set(balanced.chunk_ids) & drift_ids
    assert len(set(balanced.chunk_ids) & task_ids) < len(task_ids)
    assert len(fused.chunk_ids) == len(balanced.chunk_ids) == 8
    assert set(selection.ordered_chunk_ids) == set(trace.ranked_chunk_ids)


def test_equal_query_turns_do_not_guarantee_packed_query_coverage_under_real_token_budget():
    short = source("short", "Synthetic compact base-query evidence.")
    long = source("long", "Synthetic lengthy condition. " * 700)
    evidence = view([short, long])
    trace = trace_for((short[1],), (long[1],))
    selection = allocate(trace, evidence)
    # The long source fits on its own, including its real serialized metadata.
    long_alone = pack(evidence, (long[1].chunk_id,))
    budget = long_alone.proxy_tokens
    assert 0 < budget <= 4096
    assert pack(evidence, (short[1].chunk_id,)).proxy_tokens < budget
    assert pack(evidence, selection.ordered_chunk_ids).proxy_tokens > budget
    packed = pack(evidence, selection.ordered_chunk_ids, proxy_token_budget=budget)
    assert {decision.query_index for decision in selection.decisions} == {0, 1}
    assert packed.chunk_ids == (short[1].chunk_id,)
    assert "CONTEXT_BUNDLE_SKIPPED" in packed.warnings
    assert packed.proxy_tokens <= budget


def test_noise_before_atomic_graph_bundle_skips_whole_bundle_instead_of_partial_path(loaded):
    # QB does not enable graph answering. This isolates the existing packer's
    # bundle invariant when a hypothetical selected order spends slots on noise.
    graph, candidate = path_result(loaded)
    bundle = {candidate.chunk_id, *candidate.supporting_chunk_ids}
    noise = next(chunk for chunk in loaded.content.chunks if chunk.chunk_id not in bundle)
    trace = trace_for((noise,), (loaded.chunks[candidate.chunk_id],))
    selection = allocate(trace, loaded, tenant=loaded.manifest.tenant_id)
    noisy = replace(graph, ranked_chunk_ids=selection.ordered_chunk_ids)
    budget = len(bundle)
    assert 2 <= budget <= 8
    available = pack_context(loaded, graph, max_chunks=budget)
    blocked = pack_context(loaded, noisy, max_chunks=budget)
    assert set(available.chunk_ids) == bundle
    assert set(available.assertion_ids) == set(candidate.assertion_path)
    assert blocked.chunk_ids == (noise.chunk_id,)
    assert not set(blocked.chunk_ids) & bundle and not blocked.assertion_ids
    assert "CONTEXT_BUNDLE_SKIPPED" in blocked.warnings


def test_needed_source_beyond_frozen_forty_pool_is_not_recoverable_by_allocation():
    records = [source(f"pool-{index}") for index in range(120)]
    corpus = view(records)
    chunks = tuple(row[1] for row in records)
    needed = chunks[-1]  # Deliberately rare rank40 in only one synthetic query.
    trace = trace_for(chunks[:40], chunks[40:80], chunks[80:])
    assert needed.chunk_id in corpus.chunks and needed.chunk_id not in trace.ranked_chunk_ids
    assert trace.unique_union_count == 120 and trace.truncated_count == 80
    selection = allocate(trace, corpus)
    packed = pack(corpus, selection.ordered_chunk_ids)
    assert len(selection.ordered_chunk_ids) == 40
    assert any(row.chunk_id == needed.chunk_id and row.query_rank == 40
               for row in selection.skipped_outside_pool)
    assert needed.chunk_id not in selection.ordered_chunk_ids and needed.chunk_id not in packed.chunk_ids
