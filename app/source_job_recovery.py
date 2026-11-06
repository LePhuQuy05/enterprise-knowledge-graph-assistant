"""A narrow operator proof for one known pre-dispatch reservation denial.

Never authorizes recovery of transport, parsing, started or uncertain calls.
The original FAILED claim is preserved in its original cache database.
"""

from app.contracts import stable_id
from app.source_graph_jobs import context_for


def validate_zero_dispatch_proof(proof, plan, failed_index):
    context = context_for(plan["selection"][failed_index], plan["run_id"])
    expected_op = stable_id("op", **context.model_dump())
    failed = proof.get("failed_cache", [])
    run = proof.get("runs", {}).get("3", {})
    if (proof.get("context") != context.model_dump(mode="json") or proof.get("operation_id") != expected_op
            or "operation" not in proof or proof["operation"] is not None or len(failed) != 1
            or failed[0].get("state") != "FAILED" or failed[0].get("error") != "BudgetDenied"
            or failed[0].get("payload_present") is not False or failed[0].get("payload_sha256") is not None
            or not failed[0].get("binding_sha256") or not failed[0].get("cache_key")
            or run.get("PK") != "RUN#" + plan["run_id"] or run.get("reserved_micro_usd") != 0
            or run.get("halted") is not False):
        raise ValueError("SOURCE_RECOVERY_KNOWN_ZERO_DISPATCH_PROOF_REQUIRED")
    return expected_op
