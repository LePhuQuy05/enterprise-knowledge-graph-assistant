import copy

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.evaluation import DevelopmentDataset, EvalItem, EvidenceGroup, GoldSpan, score_retrieval
from scripts.evaluate_development import read_dataset, save
from scripts.evaluate_expanded import ROOT, SYSTEMS, read_pair, replay, summarize, validate_pairs


def test_bilingual_catalog_keeps_labels_pairs_and_legacy_examples():
    datasets, hashes = read_pair()
    old, _ = read_dataset(ROOT / "configs/evaluation/r72-development-v1.json")
    assert len(hashes) == 2 and len(SYSTEMS) == 5
    assert datasets["en"].items[:15] == old.items
    for ds in datasets.values():
        assert len(ds.items) == 50 and sum(i.answerable for i in ds.items) == 40
        assert {i.review_status for i in ds.items} == {"AI_DRAFT"}
    wrong = datasets["vi"].model_copy(update={"items": (
        datasets["vi"].items[0].model_copy(update={"intent_group_id": "false-independent-group"}),
        *datasets["vi"].items[1:])})
    with pytest.raises(ValueError, match="PAIR_LABEL_MISMATCH"):
        validate_pairs({"en": datasets["en"], "vi": wrong})


def fixture_pair(loaded):
    chunk = loaded.content.chunks[0]
    span = GoldSpan(document_version_id=chunk.document_version_id, start_char=chunk.start_char,
                    end_char=chunk.end_char, quote=chunk.text)
    item = EvalItem(question_id="q1", intent_group_id="group1", category="FACTUAL", question="What is the source?",
        answerable=True, reference_answer="Fixture fact", review_status="AI_DRAFT", review_note="Fixture only",
        evidence_groups=(EvidenceGroup(group_id="g1", required_fact="Fixture fact", alternatives=(span,)),))
    en = DevelopmentDataset(schema_version=1, dataset_id="fixture-en", split="DEVELOPMENT", language="en",
        source_snapshot_version=loaded.manifest.snapshot_version, source_manifest_sha256=loaded.manifest_sha256,
        annotation_policy="SOURCE_FIRST_AI_DRAFT_REQUIRES_HUMAN_REVIEW_V1", items=(item,))
    vi = en.model_copy(update={"dataset_id": "fixture-vi", "language": "vi", "items": (
        item.model_copy(update={"question_id": "q1-vi", "question": "Nguồn tài liệu là gì?"}),)})
    rows = [{"language": lang, "question_id": ds.items[0].question_id, "system": system,
             "question_sha256": sha256_bytes(ds.items[0].question.encode()),
             "ranked_chunk_ids": [chunk.chunk_id], "context_chunk_ids": [],
             "metrics": score_retrieval(ds.items[0], [chunk.chunk_id], [], loaded),
             "used_graph": False, "warnings": [], "retrieval_ms": 1.0}
            for lang, ds in {"en": en, "vi": vi}.items() for system in SYSTEMS]
    return {"en": en, "vi": vi}, rows


def test_summary_denominator_is_not_language_or_system_row_count(loaded):
    datasets, rows = fixture_pair(loaded)
    summary = summarize(rows, datasets)
    assert summary["unique_scenarios"] == 1 and summary["language_inputs"] == 2 and len(rows) == 10
    assert summary["answerable_intent_groups"] == 1
    for lang in ("en", "vi"):
        for system in SYSTEMS:
            result = summary["languages"][lang]["systems"][system]
            assert result["provisional_ai_draft"]["answerable_denominator"] == 1
            assert result["human_approved_only"]["complete_evidence_at_k"] is None
    for bad in (rows[:-1], rows + [rows[0]]):
        with pytest.raises(ValueError, match="INCOMPLETE_OR_DUPLICATE"):
            summarize(bad, datasets)


def test_expanded_replay_detects_wrong_score_even_if_row_hash_updated(loaded, tmp_path):
    datasets, rows = fixture_pair(loaded)
    hashes = {}
    for lang, ds in datasets.items():
        raw = canonical_json(ds.model_dump(mode="json"))
        (tmp_path / f"dataset-{lang}.json").write_bytes(raw)
        hashes[lang] = sha256_bytes(raw)
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    (tmp_path / "retrieval.jsonl").write_bytes(raw)
    summary = summarize(rows, datasets)
    save(tmp_path / "summary.json", summary)
    manifest = {"status": "COMPLETE_DIAGNOSTIC_ONLY", "dataset_sha256": hashes,
        "graph_manifest_sha256": loaded.manifest_sha256, "systems": list(SYSTEMS),
        "retrieval_sha256": sha256_bytes(raw), "summary_sha256": sha256_bytes(canonical_json(summary))}
    save(tmp_path / "manifest.json", manifest)
    assert replay(tmp_path, datasets, hashes, loaded)["rows"] == 10
    wrong = copy.deepcopy(rows)
    wrong[0]["metrics"]["recall_at_k"] = 0.0
    raw = b"".join(canonical_json(row) + b"\n" for row in wrong)
    (tmp_path / "retrieval.jsonl").write_bytes(raw)
    manifest["retrieval_sha256"] = sha256_bytes(raw)
    manifest["summary_sha256"] = sha256_bytes(canonical_json(summarize(wrong, datasets)))
    save(tmp_path / "manifest.json", manifest)
    with pytest.raises(ValueError, match="REPLAY_METRICS_CHANGED"):
        replay(tmp_path, datasets, hashes, loaded)
