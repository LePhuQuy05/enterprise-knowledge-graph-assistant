"""Locate six exposed source-control chunks in frozen rankings/context.

These passages are diagnostic targets, not question-specific gold labels. Their
absence is not automatically a retrieval error or a failed quality metric.
"""

import json

from app.contracts import canonical_json, sha256_bytes
from scripts.prepare_authorization_evidence import ROOT


def diagnose():
    root = ROOT / "results/source-message-answers-v2"
    plan_path = root / "answer-plan.json"
    facts_path = ROOT / "results/authorization-evidence-v1/facts.json"
    plan = json.loads(plan_path.read_bytes())
    facts = json.loads(facts_path.read_bytes())["records"]
    rows = []
    for case in plan["cases"]:
        trace = case["trace"]
        packing = trace["source_message_packing"]
        decisions = packing["decisions"]
        chunks = {source["chunk_id"] for source in packing["audit"]["sources"]}
        for fact in facts:
            key, candidate = fact["assertion_id"], fact["candidate"]
            chunk_id = candidate["chunk_id"]
            decision = next((d for d in decisions if d["candidate_chunk_id"] == chunk_id), None)
            rank = next((i for i, d in enumerate(decisions, 1) if d["candidate_chunk_id"] == chunk_id), None)
            paths = [p for p in trace.get("paths", []) if key in p["assertion_ids"]]
            selected = chunk_id in chunks
            rows.append({"case": case["id"], "method": case["method"], "language": case["language"],
                "diagnostic_fact_id": key, "operation_quote": candidate["operation"]["quote"], "source_uri": candidate["source_uri"],
                "chunk_id": chunk_id, "candidate_selection_rank": rank, "selected_in_context": selected,
                "eligible_graph_paths": len(paths), "graph_projection_selected": key in trace.get("context_assertion_ids", []),
                "status": "SOURCE_SELECTED" if selected else "OUTSIDE_FROZEN_POOL" if rank is None else "FOUND_NOT_SELECTED",
                "packer_decision": decision, "is_question_specific_gold": False})
    focus = [r for r in rows if r["case"].startswith("s3-permission-layers-") and r["method"].startswith("graph_")
        and r["operation_quote"] == "kms:Decrypt"]
    report = {"version": "source-message-source-location-diagnostic-v1", "status": "EXPOSED_TARGET_LOCATION_NOT_QUALITY_SCORE",
        "plan_sha256": sha256_bytes(plan_path.read_bytes()), "source_facts_sha256": sha256_bytes(facts_path.read_bytes()),
        "rows": rows, "s3_decrypt_graph_focus": focus, "new_model_calls": 0, "aws_calls": 0,
        "new_retrieval_calls": 0, "human_reviewed": 0, "diagnostic_targets_are_gold_labels": False}
    (root / "source-location-diagnostic.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({"status": report["status"], "rows": len(rows), "s3_decrypt_graph_focus": [{k: r[k] for k in
        ("case", "eligible_graph_paths", "candidate_selection_rank", "selected_in_context", "status")} for r in focus]}))


if __name__ == "__main__":
    diagnose()
