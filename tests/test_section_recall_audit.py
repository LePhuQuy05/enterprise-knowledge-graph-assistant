"""Adversarial checks for reranker source coverage in the read-only auditor."""

from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from scripts.audit_section_recall import audit_reranker


def fixture():
    record = {"candidate_anchor_ids": ["a"], "ranked_anchor_ids": ["a"], "fallback_executed": False,
        "windows": [{"chunk_id": "a", "start_char": 0, "end_char": 5, "input_tokens": 10, "raw_logit": 1.0},
            {"chunk_id": "a", "start_char": 5, "end_char": 10, "input_tokens": 10, "raw_logit": 2.0}]}
    return {"reranking": record}, {"a": NS(text="abcdefghij")}


def test_reranker_audit_accepts_full_source_window_union():
    trace, chunks = fixture()
    assert audit_reranker(trace, chunks) == 2


@pytest.mark.parametrize("fault", ["gap", "truncation", "overflow", "nan", "unknown", "permutation", "fallback", "too_many"])
def test_reranker_audit_rejects_invalid_window_or_method(fault):
    trace, chunks = deepcopy(fixture())
    record = trace["reranking"]
    if fault == "gap":
        record["windows"][1]["start_char"] = 6
    elif fault == "truncation":
        record["windows"] = record["windows"][:1]
    elif fault == "overflow":
        record["windows"][0]["input_tokens"] = 513
    elif fault == "nan":
        record["windows"][0]["raw_logit"] = float("nan")
    elif fault == "unknown":
        record["windows"].append(record["windows"][0] | {"chunk_id": "unknown"})
    elif fault == "permutation":
        record["ranked_anchor_ids"] = []
    elif fault == "fallback":
        record["fallback_executed"] = True
    else:
        record["windows"] *= 5
    with pytest.raises(ValueError):
        audit_reranker(trace, chunks)
