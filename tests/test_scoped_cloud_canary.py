import copy

import pytest

from scripts.run_scoped_cloud_canary import checked_cases


def plan():
    groups = [("bm25", "vector", "hybrid"), ("hybrid_reranked", "graph_1hop", "graph_2hop"),
              ("hybrid_multi_query", "hybrid_multi_query_balanced")]
    return {"canary": {"max_new_model_calls": 21, "cases": [
        {"idempotency_key": f"frozen-{lang}-{i}", "request": {"original_question": "Which policies permit KMS use?",
            "language": lang, "method_ids": list(ids), "run_kind": "answer"},
         "max_model_calls": len(ids)+int(lang == "vi")+int(i == 2)}
        for lang in ("en", "vi") for i, ids in enumerate(groups)]}}


def test_frozen_json_lists_validate_without_weakening_strict_runtime_models():
    assert len(checked_cases(plan())) == 6


@pytest.mark.parametrize("change", ["duplicate_key", "understated_calls", "missing_parent", "duplicate_method"])
def test_unapproved_or_underbudgeted_canary_matrix_fails_before_aws(change):
    payload = copy.deepcopy(plan())
    cases = payload["canary"]["cases"]
    if change == "duplicate_key":
        cases[1]["idempotency_key"] = cases[0]["idempotency_key"]
    elif change == "understated_calls":
        cases[-1]["max_model_calls"] -= 1
    elif change == "missing_parent":
        cases.pop()
    else:
        cases[1]["request"]["method_ids"] = cases[0]["request"]["method_ids"]
    with pytest.raises(ValueError):
        checked_cases(payload)
