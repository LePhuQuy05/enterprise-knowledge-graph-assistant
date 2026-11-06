from decimal import Decimal

import pytest

from app.benchmark_inventory import cost_scenario, readiness_methods


def test_graph_data_never_implies_serving_or_quality_approval():
    for count in (0, 14, 10000):
        rows = readiness_methods(assertion_count=count, recorded_cloud_release=True)
        assert len(rows) == len({r["id"] for r in rows}) == 8
        assert all(r["collection_quality"] is None and r["cloud_answer"] == "NOT_RELEASED" for r in rows)
        for row in rows:
            if row["id"].startswith("graph_"):
                assert row["cloud_retrieval"] == "NOT_RELEASED"
                assert row["next_gate"] == ("BUILD_SOURCE_BACKED_COLLECTION_GRAPH" if count == 0
                                             else "COLLECTION_GRAPH_REVIEW_AND_RUNNER")
        assert sum(r["cloud_retrieval"] == "RECORDED_DEPLOYED" for r in rows) == 4


def test_missing_cloud_receipt_does_not_advertise_local_methods_as_deployed():
    assert all(r["cloud_retrieval"] == "NOT_RELEASED"
               for r in readiness_methods(assertion_count=0, recorded_cloud_release=False))


def test_forecast_counts_extra_attempts_and_rounds_up_not_down():
    args = {"input_micro_per_million": 300000, "output_micro_per_million": 1200000}
    value = cost_scenario(48, input_tokens=2500, output_tokens=1000,
                          extra_attempt_fraction=Decimal("0.2"), **args)
    assert value["assumed_attempts"] == 58
    assert value["estimated_micro_usd"] == 113100
    assert value["kind"] == "ASSUMPTION_NOT_INVOICE_OR_AUTHORIZATION"
    reservation = cost_scenario(1, input_tokens=1048576, output_tokens=2048, **args)
    assert reservation["estimated_micro_usd"] == 317031


@pytest.mark.parametrize("bad", [-1, True, 1.5])
def test_invalid_counts_rejected(bad):
    with pytest.raises(ValueError):
        readiness_methods(assertion_count=bad, recorded_cloud_release=True)
