import copy
import json
from datetime import UTC, datetime

import pytest

from app.contracts import (
    Chunk,
    DocumentVersion,
    canonical_json,
    embedding_input,
    sha256_bytes,
    text_hash,
)
from app.evaluation import DevelopmentDataset, EvalItem, EvidenceGroup, GoldSpan, score_retrieval
from app.fixtures import FIXTURE_EMBEDDING_MODEL, FIXTURE_TENANT, fixture_recipes, fixture_vector
from app.snapshot import SnapshotContent, build_snapshot, load_snapshot
from scripts import export_dashboard
from scripts.evaluate_development import SYSTEMS, summarize


@pytest.fixture
def frozen(tmp_path):
    # Synthetic text at official-shaped URLs; no AWS content, private receipts or model assets.
    documents, chunks = [], []
    for number in range(2):
        lines = [f"Synthetic service {number} has fixture fact {i}." for i in range(4)]
        text = "\n".join(lines)
        doc = DocumentVersion.create(tenant_id=FIXTURE_TENANT,
            source_uri=f"https://docs.aws.amazon.com/fixture-{number}/guide.html",
            title=f"Synthetic document {number}", fetched_at=datetime(2026, 10, 16, tzinfo=UTC),
            raw_sha256=text_hash(text), normalizer_version="normalize-v1", normalized_text=text,
            normalized_text_path=f"sources/fixture-{number}.txt", source_kind="TEXT")
        documents.append(doc)
        start = 0
        for i, line in enumerate(lines):
            chunks.append(Chunk.create(tenant_id=FIXTURE_TENANT, document_version_id=doc.document_version_id,
                chunking_version="chunk-v1", section_path=("Synthetic fixture",), position=i,
                start_char=start, end_char=start + len(line), text=line, token_count=8,
                tokenizer_version="fixture-words-v1"))
            start += len(line) + 1
    extraction, embedding = fixture_recipes()
    by_id = {d.document_version_id: d for d in documents}
    vectors = {c.chunk_id: fixture_vector(embedding_input(by_id[c.document_version_id], c)) for c in chunks}
    directory = build_snapshot(tmp_path / "snapshot", SnapshotContent(tuple(documents), tuple(chunks), (), (), ()),
        vectors, tenant_id=FIXTURE_TENANT, purpose="FIXTURE", extraction_recipe=extraction,
        embedding_recipe=embedding, corpus_version="synthetic-v1", normalizer_version="normalize-v1",
        chunking_version="chunk-v1", tokenizer_version="fixture-words-v1")
    baseline = load_snapshot(directory, tenant_id=FIXTURE_TENANT,
                             embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True)
    first, last = chunks[0], chunks[-1]
    groups = tuple(EvidenceGroup(group_id=f"g{i}", required_fact=f"Synthetic fact {i}", alternatives=(
        GoldSpan(document_version_id=c.document_version_id, start_char=c.start_char,
                 end_char=c.end_char, quote=c.text),)) for i, c in enumerate((first, last)))
    items = (EvalItem(question_id="synthetic-multi", intent_group_id="fixture", category="MULTI_HOP",
        question="Which two synthetic facts connect the fixture documents?", answerable=True,
        reference_answer="Two synthetic facts.", review_status="AI_DRAFT", review_note="Synthetic fixture.",
        multi_hop_rationale="The two required facts are in different fixture documents.", evidence_groups=groups),
        EvalItem(question_id="synthetic-unanswerable", intent_group_id="absent", category="UNANSWERABLE",
        question="What unrecorded fact is absent?", answerable=False, reference_answer="Not available.",
        review_status="AI_DRAFT", review_note="Synthetic fixture.", unanswerable_reason="Not in fixture.",
        evidence_groups=()))
    dataset = DevelopmentDataset(schema_version=1, dataset_id="synthetic-dashboard", split="DEVELOPMENT",
        language="en", source_snapshot_version=baseline.manifest.snapshot_version,
        source_manifest_sha256=baseline.manifest_sha256,
        annotation_policy="SOURCE_FIRST_AI_DRAFT_REQUIRES_HUMAN_REVIEW_V1", items=items)
    ranked = [c.chunk_id for c in chunks if c != last] + [last.chunk_id]
    rows = [{"question_id": item.question_id, "system": system, "ranked_chunk_ids": ranked,
        "context_chunk_ids": ranked[:5], "context_proxy_tokens": 40, "used_graph": False, "warnings": [],
        "metrics": score_retrieval(item, ranked, ranked[:5], baseline, k=5)}
        for item in items for system in SYSTEMS]
    manifest = {"status": "COMPLETE_DIAGNOSTIC_ONLY", "k": 5, "graph_release_approved": False,
        "source_manifest_sha256": baseline.manifest_sha256, "graph_manifest_sha256": baseline.manifest_sha256,
        "run_id": "synthetic-dashboard", "completed_at": "2026-10-16T00:00:00+00:00",
        "dataset_sha256": sha256_bytes(canonical_json(dataset.model_dump(mode="json"))),
        "retrieval_sha256": sha256_bytes(b"".join(canonical_json(r) + b"\n" for r in rows)),
        "summary_sha256": sha256_bytes(canonical_json(summarize(rows, dataset))),
        "context_max_chunks": 8, "context_proxy_tokens": 4096, "rrf_constant": 60}
    return dataset, baseline, baseline, manifest, rows


def test_export_replays_all_cases_and_discloses_recorded_scope(frozen):
    value = export_dashboard.project(*frozen)
    assert value["scope"] == "PUBLIC_RECORDED_DEVELOPMENT"
    assert (len(value["documents"]), len(value["chunks"]), len(value["cases"])) == (2, 8, 2)
    assert value["run"]["generation_calls"] == 0
    assert value["run"]["graph_serving"] is False
    assert all(s["approved"]["complete_evidence_at_k"] is None for s in value["summaries"].values())
    # The synthetic candidate pool has both facts, but top5/context miss one.
    case = next(c for c in value["cases"] if c["id"] == "synthetic-multi")
    coverage = case["methods"]["B2_HYBRID"]["coverage"]
    assert all(coverage["corpus"]) and all(coverage["candidates"])
    assert not all(coverage["top_k"]) and not all(coverage["context"])
    unanswerable = next(c for c in value["cases"] if not c["answerable"])
    assert all(v is None for v in unanswerable["methods"]["B2_HYBRID"]["coverage"].values())


def test_export_is_closed_public_projection(frozen):
    dataset, baseline, graph, manifest, rows = frozen
    poisoned = copy.deepcopy(rows)
    for row in poisoned:
        row["access_token"] = "NEVER_EXPORT"
        row["local_path"] = "NEVER_EXPORT"
    value = export_dashboard.project(dataset, baseline, graph, manifest, poisoned)
    raw = json.dumps(value)
    assert "NEVER_EXPORT" not in raw
    assert "normalized_text_path" not in raw and "tenant_id" not in raw
    assert all(set(d) == {"id", "title", "url", "service", "fetched_at", "chunks"} for d in value["documents"])
    assert all(len(c["preview"]) <= 280 for c in value["chunks"])


@pytest.mark.parametrize("change", ["duplicate", "metric", "chunk", "identity"])
def test_export_fails_closed_on_corrupt_experiment(frozen, change):
    dataset, baseline, graph, manifest, rows = frozen
    manifest, rows = copy.deepcopy(manifest), copy.deepcopy(rows)
    if change == "duplicate":
        rows[-1] = rows[0]
    elif change == "metric":
        rows[0]["metrics"]["complete_evidence_at_k"] = 1 - rows[0]["metrics"]["complete_evidence_at_k"]
    elif change == "chunk":
        rows[0]["ranked_chunk_ids"][0] = "not-a-real-chunk"
    else:
        manifest["source_manifest_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        export_dashboard.project(dataset, baseline, graph, manifest, rows)


@pytest.mark.parametrize("url", ["http://docs.aws.amazon.com/x", "https://docs.aws.amazon.com.evil.test/x",
    "https://user:password@docs.aws.amazon.com/x", "javascript:alert(1)", "https://docs.aws.amazon.com:444/x"])
def test_export_rejects_unsafe_source_links(url):
    with pytest.raises(ValueError, match="UNAPPROVED"):
        export_dashboard.official_url(url)


def test_projection_file_round_trip_checks_frozen_input_hashes(frozen, tmp_path, monkeypatch):
    dataset, baseline, graph, manifest, rows = frozen
    (tmp_path / "dataset.json").write_bytes(canonical_json(dataset.model_dump(mode="json")))
    (tmp_path / "retrieval.jsonl").write_bytes(b"".join(canonical_json(r) + b"\n" for r in rows))
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(export_dashboard, "load_inputs", lambda _: (baseline, graph))
    raw = export_dashboard.build_projection(tmp_path)
    assert len(raw) <= export_dashboard.MAX_BYTES
    assert json.loads(raw) == export_dashboard.project(*frozen)
    (tmp_path / "retrieval.jsonl").write_bytes(b"{}\n")
    with pytest.raises(ValueError, match="FROZEN_INPUT_HASH_MISMATCH"):
        export_dashboard.build_projection(tmp_path)


def test_committed_display_projection_is_bounded_and_explicitly_draft():
    raw = export_dashboard.OUTPUT.read_bytes()
    value = json.loads(raw)
    assert len(raw) <= export_dashboard.MAX_BYTES
    assert value["scope"] == "PUBLIC_RECORDED_DEVELOPMENT"
    assert value["run"]["label_status"] == "AI_DRAFT" and value["run"]["graph_serving"] is False
    assert (len(value["documents"]), len(value["chunks"]), len(value["cases"])) == (20, 236, 15)
