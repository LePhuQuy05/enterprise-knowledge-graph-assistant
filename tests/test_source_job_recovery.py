from copy import deepcopy

import pytest

from app.contracts import stable_id
from app.source_graph_jobs import context_for
from app.source_job_recovery import validate_zero_dispatch_proof


def fixture():
    plan = {"run_id": "fixture-recovery", "selection": [{"v7_unit_id": "one"}]}
    context = context_for(plan["selection"][0], plan["run_id"])
    proof = {"context": context.model_dump(mode="json"), "operation_id": stable_id("op", **context.model_dump()),
        "operation": None, "failed_cache": [{"state": "FAILED", "error": "BudgetDenied", "payload_present": False,
        "payload_sha256": None, "binding_sha256": "a" * 64, "cache_key": "raw-key"}],
        "runs": {"3": {"PK": "RUN#fixture-recovery", "reserved_micro_usd": 0, "halted": False}}}
    return plan, proof


def test_exact_zero_dispatch_reservation_denial_is_eligible():
    plan, proof = fixture()
    assert validate_zero_dispatch_proof(proof, plan, 0) == proof["operation_id"]


@pytest.mark.parametrize("mutation", ["missing_operation", "reserved", "unknown", "payload", "error", "context", "run", "halted"])
def test_ambiguous_or_wrong_proofs_cannot_authorize_a_model_call(mutation):
    plan, original = fixture()
    proof = deepcopy(original)
    if mutation == "missing_operation":
        del proof["operation"]
    elif mutation == "reserved":
        proof["runs"]["3"]["reserved_micro_usd"] = 1
    elif mutation == "unknown":
        proof["operation"] = {"status": "UNKNOWN_SPEND", "call_started": True}
    elif mutation == "payload":
        proof["failed_cache"][0]["payload_present"] = True
    elif mutation == "error":
        proof["failed_cache"][0]["error"] = "ExternalCallUncertain"
    elif mutation == "context":
        proof["context"]["request_id"] = "other"
    elif mutation == "run":
        proof["runs"]["3"]["PK"] = "RUN#other"
    else:
        proof["runs"]["3"]["halted"] = True
    with pytest.raises(ValueError, match="KNOWN_ZERO_DISPATCH"):
        validate_zero_dispatch_proof(proof, plan, 0)
