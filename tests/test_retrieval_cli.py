import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from app.cli import main
from app.fixtures import fixture_vector
from app.retrieval import reciprocal_rank_fusion, retrieve_baseline


def test_rrf_uses_ranks_deduplicates_and_breaks_ties_by_id():
    assert reciprocal_rank_fusion([["b", "b", "a"], ["a", "b"]]) == ["a", "b"]
    assert reciprocal_rank_fusion([["b"], ["a"]]) == ["a", "b"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], constant=0)


@pytest.mark.parametrize("mode", ["vector", "hybrid"])
def test_baselines_find_exact_fixture_and_respect_caps(loaded, mode):
    question = "Service A uses Service B."
    result = retrieve_baseline(loaded, question, fixture_vector(question), mode=mode,
                               vector_k=1, lexical_k=1)
    assert loaded.chunks[result[0]].text == question
    assert len(result) <= 2


@pytest.mark.parametrize("bad", [
    np.zeros(1024, dtype=np.float32), np.ones(512, dtype=np.float32),
    np.full(1024, np.nan, dtype=np.float32), np.zeros(1024, dtype=np.float64),
])
def test_query_vector_rejected(loaded, bad):
    with pytest.raises(ValueError):
        retrieve_baseline(loaded, "question", bad)


def test_unimplemented_modes_are_explicit(loaded):
    with pytest.raises(ValueError):
        retrieve_baseline(loaded, "question", fixture_vector("question"), mode="graph_1hop")


def test_offline_cli_labels_synthetic_and_returns_cited_excerpt(tmp_path, capsys):
    assert main(["fixture-demo", "--output-root", str(tmp_path)]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["fixture_only"] is True
    assert data["response"]["used_graph"] is False
    assert data["response"]["warnings"] == ["FIXTURE_ONLY_NO_MODEL"]
    assert data["response"]["answer"] == "Service A uses Service B."
    assert data["response"]["citations"][0]["source_uri"].startswith("https://example.invalid/")


def test_cli_errors_do_not_echo_input(tmp_path, capsys):
    private = "private-tenant-name"
    assert main(["snapshot-validate", "--snapshot-dir", str(tmp_path), "--tenant", private,
                 "--embedding-model", "absent"]) == 1
    output = capsys.readouterr().out
    assert private not in output
    assert json.loads(output)["error"]["code"] == "SnapshotError"


@pytest.mark.parametrize("embedding,input_tokens,output_tokens,expected", [
    (False, 2000, 500, 4950), (True, 8192, 0, 164),
])
def test_cost_quote_is_forecast_only_and_never_creates_aws_client(
    embedding, input_tokens, output_tokens, expected, monkeypatch, capsys,
):
    from app.cost import PriceSheet

    sheet = PriceSheet.model_validate_json(
        Path("configs/prices/aws-us-east-1-2026-10-16.json").read_bytes()
    )

    class FixedDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return sheet.checked_at

    monkeypatch.setattr("app.cli.datetime", FixedDatetime)
    monkeypatch.setattr("boto3.Session", lambda **kwargs: pytest.fail("Forecast must stay offline"))
    args = ["cost-quote", "--input-tokens", str(input_tokens), "--output-tokens", str(output_tokens)]
    assert main(args + (["--embedding"] if embedding else [])) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "FORECAST_ONLY" and result["inference_calls"] == 0
    assert result["estimated_micro_usd"] == expected
