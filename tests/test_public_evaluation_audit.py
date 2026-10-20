import json
import shutil

import pytest

from app.contracts import sha256_bytes
from scripts.export_expanded_evaluation import PUBLIC, audit


def test_public_range_audit_needs_no_models_credentials_or_corpus():
    result = audit(PUBLIC)
    assert result["status"] == "PASS_RANGE_AUDIT" and result["rows"] == 500
    assert result["component_rows"] == 300
    assert result["model_api_calls"] == result["aws_calls"] == 0


def test_public_range_audit_rejects_changed_ranking(tmp_path):
    for path in PUBLIC.iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    path = tmp_path / "retrieval.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    rows[0]["ranked_chunk_ids"] = []
    raw = b"".join(json.dumps(row).encode() + b"\n" for row in rows)
    path.write_bytes(raw)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["retrieval_sha256"] = sha256_bytes(raw)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="PUBLIC_AUDIT_METRIC_MISMATCH"):
        audit(tmp_path)


def test_public_answer_audit_rejects_inflated_pass_count_even_with_updated_hash(tmp_path):
    for path in PUBLIC.iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    report_path = tmp_path / "answer-summary.json"
    report = json.loads(report_path.read_bytes())
    report["languages"]["vi"]["structural_pass"] = 50
    raw = json.dumps(report).encode()
    report_path.write_bytes(raw)
    manifest_path = tmp_path / "answer-manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["report_sha256"] = sha256_bytes(raw)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="PUBLIC_ANSWER_SUMMARY_INVALID"):
        audit(tmp_path)
