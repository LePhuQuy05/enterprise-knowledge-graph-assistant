import json
from pathlib import Path

import pytest

from app.contracts import canonical_json
from scripts.project_benchmark_inventory import project

ROOT = Path(__file__).resolve().parents[1]


def test_projection_cannot_leak_extra_fields_nested_in_audit():
    public = json.loads((ROOT / "frontend/tracer/benchmark-data.json").read_bytes())
    public["forecast"] = {"provider_balance": 123, "private_path": "private"}
    public["collection"]["private_path"] = "private"
    public["methods"][0]["raw_response"] = "private"
    public["pilot"]["results"][0]["en"]["reviewer_email"] = "private"
    raw = canonical_json(public)
    exported = canonical_json(project(raw))
    assert b"private" not in exported and b"reviewer_email" not in exported and b"provider_balance" not in exported
    assert json.loads(exported)["collection_quality"] is None


def test_readiness_projection_does_not_accept_unreviewed_quality_relabeling():
    public = json.loads((ROOT / "frontend/tracer/benchmark-data.json").read_bytes())
    public["collection_quality"] = .99
    with pytest.raises(ValueError, match="READINESS_NOT_QUALITY"):
        project(canonical_json(public))
