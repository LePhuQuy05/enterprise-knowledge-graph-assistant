import json
import sys
from pathlib import Path

import pytest

from scripts import run_collection_graph_pilot

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(not (ROOT / "results/collection-graph/extraction-plan-v1.json").is_file(),
                    reason="Requires the local, uncommitted collection316 extraction plan")
def test_graph_pilot_source_selection_and_limits_are_frozen_without_labels_or_questions():
    plan = json.loads((ROOT / "results/collection-graph/extraction-plan-v1.json").read_bytes())
    assert plan["corpus_documents"] == 316 and plan["corpus_chunks"] == 2978
    assert len(plan["selection"]) == len({s["chunk_id"] for s in plan["selection"]}) == 48
    groups = {s["group"] for s in plan["selection"]}
    assert len(groups) == 24
    for group in groups:
        rows = [s for s in plan["selection"] if s["group"] == group]
        assert len(rows) == len({s["document_version_id"] for s in rows}) == 2
    assert not plan["run_initialized"] and not plan["full_batch_authorized_by_this_file"]
    assert plan["max_dispatches"] == 48 and plan["run_cap_proposed_micro_usd"] == 500000
    assert not plan["format_repair"] and not plan["funding_cap_raise"]


def test_default_graph_pilot_command_cannot_create_aws_client_or_call_provider(monkeypatch, capsys, tmp_path):
    # Exercise the no-execute boundary with synthetic inputs, not an owner's results directory.
    plan_path = tmp_path / "results/collection-graph/extraction-plan-v1.json"
    plan_path.parent.mkdir(parents=True)
    plan_path.write_text(json.dumps({"selection": [{"chunk_id": "synthetic-plan-source"}]}), encoding="utf-8")
    monkeypatch.setattr(run_collection_graph_pilot, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["run_collection_graph_pilot"])
    monkeypatch.setattr(run_collection_graph_pilot.boto3, "Session", lambda **kw: (_ for _ in ()).throw(AssertionError("AWS_NOT_ALLOWED")))
    run_collection_graph_pilot.main()
    assert json.loads(capsys.readouterr().out)["status"] == "PLANNED_NO_MODEL_OR_AWS_CALL"
