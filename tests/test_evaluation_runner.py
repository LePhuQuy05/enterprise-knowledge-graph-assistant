import json
from pathlib import Path

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.evaluation import DevelopmentDataset, EvalItem, EvidenceGroup, GoldSpan, score_retrieval
from scripts.evaluate_development import SYSTEMS, read_dataset, replay, summarize


def test_dataset_loader_rejects_duplicate_keys_and_non_development_split(tmp_path):
    path = tmp_path / "dataset.json"
    path.write_bytes(b'{"split":"DEVELOPMENT","split":"HOLDOUT"}')
    with pytest.raises(ValueError, match="DUPLICATE_DATASET_JSON_KEY"):
        read_dataset(path)
    path.write_bytes(b'{"split":"PILOT_EVAL","items":["must not be evaluated"]}')
    with pytest.raises(ValueError, match="CANNOT_OPEN_OTHER_SPLITS"):
        read_dataset(path)


def test_committed_development_draft_keeps_categories_and_no_automatic_approval():
    path = Path(__file__).resolve().parents[1] / "configs/evaluation/r72-development-v1.json"
    dataset, fingerprint = read_dataset(path)
    assert len(dataset.items) == 15 and len(fingerprint) == 64
    assert sum(i.answerable for i in dataset.items) == 12
    assert {i.review_status for i in dataset.items} == {"AI_DRAFT"}


def test_replay_recomputes_scores_and_detects_tampering(loaded, tmp_path):
    chunk = loaded.content.chunks[0]
    anchor = GoldSpan(document_version_id=chunk.document_version_id, start_char=chunk.start_char,
                      end_char=chunk.end_char, quote=chunk.text)
    item = EvalItem(question_id="q1", intent_group_id="intent1", category="FACTUAL", question="What is used?",
        answerable=True, reference_answer="A synthetic fact", review_status="AI_DRAFT", review_note="Fixture.",
        evidence_groups=(EvidenceGroup(group_id="g1", required_fact="Fact", alternatives=(anchor,)),))
    dataset = DevelopmentDataset(schema_version=1, dataset_id="fixture-dev", split="DEVELOPMENT", language="en",
        source_snapshot_version=loaded.manifest.snapshot_version, source_manifest_sha256=loaded.manifest_sha256,
        annotation_policy="SOURCE_FIRST_AI_DRAFT_REQUIRES_HUMAN_REVIEW_V1", items=(item,))
    rows = [{"question_id":"q1", "system":system, "ranked_chunk_ids":[chunk.chunk_id], "context_chunk_ids":[],
             "metrics":score_retrieval(item,[chunk.chunk_id],[],loaded), "used_graph":False,"warnings":[]}
            for system in SYSTEMS]
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    path = tmp_path / "retrieval.jsonl"
    path.write_bytes(raw)
    manifest = {"status":"COMPLETE_DIAGNOSTIC_ONLY", "dataset_sha256":"a"*64,
        "graph_manifest_sha256":loaded.manifest_sha256, "retrieval_sha256":sha256_bytes(raw),
        "summary_sha256":sha256_bytes(canonical_json(summarize(rows,dataset)))}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    result = replay(tmp_path,dataset,"a"*64,loaded)
    assert result["status"] == "PASS_RECOMPUTED_FROM_RANKINGS"
    assert result["new_local_onnx_forwards"] == result["model_api_calls"] == result["aws_calls"] == 0
    path.write_bytes(raw.replace(b'"recall_at_k":1.0',b'"recall_at_k":0.0'))
    with pytest.raises(ValueError, match="REPLAY_INPUT_HASH_MISMATCH"):
        replay(tmp_path,dataset,"a"*64,loaded)

    # Even if the local checksum is updated, wrong saved scores must be rejected.
    manifest["retrieval_sha256"] = sha256_bytes(path.read_bytes())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(ValueError, match="REPLAY_METRICS_CHANGED"):
        replay(tmp_path,dataset,"a"*64,loaded)
