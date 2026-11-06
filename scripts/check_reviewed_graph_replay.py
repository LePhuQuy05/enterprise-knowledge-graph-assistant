"""Check the current graph code against immutable paid receipts, with networking forbidden."""

import json
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes
from scripts.answer_local import ROOT
from scripts.curate_collection_graph import write_immutable
from scripts.pilot_reviewed_graph_answers import FILES
from scripts.reconcile_collection_graph_pilot import require
from scripts.run_reviewed_collection_graph import main as run_case


def main():
    target = ROOT / "results/collection-graph/current-code-replay-v2.json"
    require(not target.exists(), "GRAPH_CURRENT_REPLAY_EXISTS")
    plan = json.loads((ROOT / "results/collection-graph/answer-pilot-plan-v1.json").read_bytes())
    audit_raw = (ROOT / "results/collection-graph/answer-pilot-audit-v1.json").read_bytes()
    require(json.loads(audit_raw)["status"] == "PASS_SOURCE_RECEIPT_INPUT_USAGE_LEDGER_AND_NETWORK_FREE_REPLAY",
        "GRAPH_CURRENT_REPLAY_ORIGINAL_AUDIT")
    cases = []
    with patch("boto3.Session", side_effect=AssertionError("REPLAY_AWS_FORBIDDEN")), patch(
        "app.external_http.OfficialHttpsTransport.request", side_effect=AssertionError("REPLAY_PROVIDER_FORBIDDEN")), patch(
        "socket.socket.connect", side_effect=AssertionError("REPLAY_NETWORK_FORBIDDEN")):
        for case in plan["cases"]:
            original_path = ROOT / "results/collection-graph" / ("answer-" + case["id"] + "-v1.json")
            replay_path = ROOT / "results/collection-graph" / ("replay-" + case["id"] + "-v3.json")
            original_raw = original_path.read_bytes()
            original = json.loads(original_raw)
            run_case([case["question"], "--hops", str(case["hops"]), "--language", case["language"], "--run-kind", "answer",
                "--allow-experimental", "--idempotency-key", "graph-demo-" + case["id"] + "-v1", "--output", str(replay_path)])
            replay = json.loads(replay_path.read_bytes())
            for field in ("response", "trace", "context_sources", "request_id", "search_query", "effective_method"):
                require(canonical_json(replay[field]) == canonical_json(original[field]), "GRAPH_CURRENT_REPLAY_" + field.upper())
            require(replay["answer_cache_hit"] and replay["new_cost_micro_usd"] == 0
                and replay["model_dispatch_attempts_this_invocation"] == 0, "GRAPH_CURRENT_REPLAY_NO_DISPATCH")
            cases.append({"id": case["id"], "original_sha256": sha256_bytes(original_raw),
                "replay_sha256": sha256_bytes(replay_path.read_bytes()), "answer_trace_source_parity": True})
    current = {name: sha256_bytes((ROOT / name).read_bytes()) for name in FILES}
    report = {"status": "PASS_CURRENT_CODE_NETWORK_FORBIDDEN_REPLAY", "cases": cases,
        "paid_audit_sha256": sha256_bytes(audit_raw), "current_implementation_sha256": current,
        "changed_since_paid_plan": [name for name in FILES if current[name] != plan["implementation_sha256"][name]],
        "changes": "Graph loader now binds mention quotes to entity aliases and bounds the actual file read.",
        "aws_calls": 0, "model_calls": 0, "new_cost_micro_usd": 0,
        "human_quality_approval": False, "production_deployed": False}
    write_immutable(target, canonical_json(report) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
