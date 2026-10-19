import json

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.evaluation import DevelopmentDataset, EvalItem, EvidenceGroup, GoldSpan, score_retrieval
from scripts.evaluate_service_mentions import SYSTEMS, summarize, verify_replay


@pytest.fixture
def saved_mention_run(loaded, tmp_path):
    chunk = loaded.content.chunks[0]
    span = GoldSpan(document_version_id=chunk.document_version_id,start_char=chunk.start_char,
                    end_char=chunk.end_char,quote=chunk.text)
    item = EvalItem(question_id="q1",intent_group_id="intent1",category="FACTUAL",question="Which service?",
        answerable=True,reference_answer="A fixture fact",review_status="AI_DRAFT",review_note="Fixture only",
        evidence_groups=(EvidenceGroup(group_id="g1",required_fact="A fact",alternatives=(span,)),))
    dataset = DevelopmentDataset(schema_version=1,dataset_id="fixture-dev",split="DEVELOPMENT",language="en",
        source_snapshot_version=loaded.manifest.snapshot_version,source_manifest_sha256=loaded.manifest_sha256,
        annotation_policy="SOURCE_FIRST_AI_DRAFT_REQUIRES_HUMAN_REVIEW_V1",items=(item,))
    rows = [{"question_id":"q1","system":system,"ranked_chunk_ids":[chunk.chunk_id],"context_chunk_ids":[],
             "metrics":score_retrieval(item,[chunk.chunk_id],[],loaded),"used_mentions":system==SYSTEMS[-1],"warnings":[]}
            for system in SYSTEMS]
    raw = b"".join(canonical_json(row)+b"\n" for row in rows)
    manifest = {"status":"COMPLETE_DIAGNOSTIC_ONLY","dataset_sha256":"b"*64,
        "mention_manifest_sha256":loaded.manifest_sha256,"retrieval_sha256":sha256_bytes(raw),
        "summary_sha256":sha256_bytes(canonical_json(summarize(rows,dataset)))}
    (tmp_path/"retrieval.jsonl").write_bytes(raw)
    (tmp_path/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    return tmp_path,dataset,loaded,rows,manifest


def test_mention_replay_recomputes_without_encoder_and_keeps_human_metrics_null(saved_mention_run):
    path,dataset,loaded,rows,_ = saved_mention_run
    result = verify_replay(path,dataset,"b"*64,loaded)
    assert result["status"] == "PASS_RECOMPUTED_FROM_RANKINGS"
    assert result["new_onnx_forwards"] == result["model_api_calls"] == result["aws_calls"] == 0
    summary = summarize(rows,dataset)
    assert summary["systems"][SYSTEMS[-1]]["human_approved_only"]["complete_evidence_at_k"] is None
    assert summary["mention_vs_hybrid_provisional"]["ties"] == 1


@pytest.mark.parametrize("mutation",["dataset-hash","raw-bytes","wrong-metric-with-new-checksum","duplicate-system"])
def test_mention_replay_rejects_changed_or_incomplete_evidence(saved_mention_run,mutation):
    path,dataset,loaded,rows,manifest = saved_mention_run
    if mutation == "dataset-hash":
        manifest["dataset_sha256"] = "c"*64
    elif mutation == "raw-bytes":
        (path/"retrieval.jsonl").write_bytes(b"changed")
    else:
        if mutation == "wrong-metric-with-new-checksum":
            rows[-1]["metrics"]["complete_evidence_at_k"] = 0
        else:
            rows[-1]["system"] = "B1_VECTOR"
        raw = b"".join(canonical_json(row)+b"\n" for row in rows)
        (path/"retrieval.jsonl").write_bytes(raw)
        manifest["retrieval_sha256"] = sha256_bytes(raw)
    (path/"manifest.json").write_text(json.dumps(manifest),encoding="utf-8")
    with pytest.raises(ValueError):
        verify_replay(path,dataset,"b"*64,loaded)
