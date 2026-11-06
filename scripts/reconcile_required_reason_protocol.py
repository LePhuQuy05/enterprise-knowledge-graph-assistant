"""Read-only paired protocol/source/payload audit, preserving every failed cell."""

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes
from app.generation import parse_generated
from app.ledger import DynamoLedgerStore, Ledger
from scripts.compare_required_reason_protocol import (
    CACHE,
    CODE,
    OUTPUT,
    PARENT,
    RUN,
    protocol_cases,
)
from scripts.prepare_authorization_evidence import ROOT
from scripts.reconcile_reviewed_source_answers import provider_body


def audit():
    import boto3
    from botocore.config import Config

    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
        raise ValueError("REASON_PROTOCOL_AUDIT_CODE_CHANGED")
    with patch("socket.socket", side_effect=AssertionError("REASON_PROTOCOL_REPLAY_NETWORK_DISABLED")):
        _, cases = protocol_cases()
    parent_cases = {c["id"]: c for c in json.loads((PARENT / "answer-plan.json").read_bytes())["cases"]}
    operations, summaries, reviews = [], [], []
    reason_missing, unknown_keys = 0, []
    for case in cases:
        p = case["prepared"]
        answer_path = OUTPUT / "answers" / (case["id"] + ".json")
        failed = not answer_path.exists()
        row = json.loads((OUTPUT / "failed-answers" / (case["id"] + ".json") if failed else answer_path).read_bytes())
        receipt = row["receipt"]
        if (row["repaired"] or row["retried"] or receipt["recipe_sha256"] != p.recipe.fingerprint
                or receipt["binding_sha256"] != sha256_bytes(canonical_json(p.binding.model_dump(mode="json")))):
            raise ValueError("REASON_PROTOCOL_AUDIT_BINDING")
        if sha256_bytes(case["prompt"].encode()) != parent_cases[case["id"]]["prompt_sha256"]:
            raise ValueError("REASON_PROTOCOL_AUDIT_FIXED_CONTEXT_CHANGED")
        raw = json.loads(receipt["raw_output"])
        reason_missing += "reason" not in raw
        extra = set(raw) - {"reason", "answer_status", "claims"}
        if extra:
            unknown_keys.append({"case": case["id"], "extra_keys": sorted(extra)})
        if failed:
            try:
                parse_generated(receipt["raw_output"], receipt["stop_reason"], p.recipe)
            except ValueError:
                pass
            else:
                raise ValueError("REASON_PROTOCOL_AUDIT_FAILED_SCHEMA_NOT_REPRODUCED")
            status, claims, citations = "FAILED_VALIDATION", [], []
        else:
            response = row["response"]
            if (response["request_id"] != p.binding.request_id or response["mode_used"] != case["method"]
                    or response["answer_language"] != case["language"] or response["used_graph"] != p.binding.used_graph):
                raise ValueError("REASON_PROTOCOL_AUDIT_RESPONSE_METHOD")
            status, claims, citations = response["answer_status"], response["claims"], response["citations"]
            by_id = {c["citation_id"]: c for c in citations}
            for citation in citations:
                chunk = case["view"].chunks.get(citation["chunk_id"])
                if chunk is None or chunk.chunk_id not in p.packed.chunk_ids:
                    raise ValueError("REASON_PROTOCOL_AUDIT_CITATION_NOT_IN_CONTEXT")
                doc = case["view"].documents[chunk.document_version_id]
                if (citation["document_version_id"] != doc.document_version_id or citation["source_uri"] != doc.source_uri
                        or citation["quote"] not in chunk.text or doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
                    raise ValueError("REASON_PROTOCOL_AUDIT_CITATION_SOURCE")
            for claim in claims:
                if not claim["citation_ids"] or not set(claim["citation_ids"]) <= by_id.keys():
                    raise ValueError("REASON_PROTOCOL_AUDIT_CLAIM_CITATION")
                flags = []
                if case["intent"] == "s3-permission-layers" and any(w in claim["text"].casefold() for w in ("ec2", "network interface", "giao diện mạng")):
                    flags.append("CROSS_TASK_ANALOGY_SOURCE_REVIEW_REQUIRED")
                reviews.append({"case": case["id"], "claim_id": claim["claim_id"], "claim": claim["text"],
                    "citations": [by_id[k] for k in claim["citation_ids"]], "review_status": "PENDING_NOT_INDEPENDENT_GOLD",
                    "review_flags_not_verdicts": flags})
        operations.append((receipt, sha256_bytes(provider_body(case["provider_system"], case["prompt"], 800))))
        summaries.append({"case": case["id"], "method": case["method"], "intent": case["intent"], "language": case["language"],
            "status": status, "claims": len(claims), "citations": len(citations),
            "baseline_mechanism_status": case["outcome"]["mechanism_status"], "context_unchanged": True})
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("REASON_PROTOCOL_AUDIT_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    settled = []
    for receipt, digest in operations:
        op = ledger.operation(receipt["operation_id"])
        if (op.status != "SETTLED" or not op.call_started or op.bound_exceeded or op.spec.run_id != RUN
                or op.spec.stage != "GENERATE" or op.spec.input_sha256 != digest
                or op.actual_micro_usd != receipt["original_cost_micro_usd"] or op.usage.model_dump(mode="json") != receipt["usage"]):
            raise ValueError("REASON_PROTOCOL_AUDIT_OPERATION_PAYLOAD_USAGE_COST")
        settled.append({"operation_id": op.operation_id, "status": op.status, "payload_sha256": digest,
            "usage": receipt["usage"], "cost_micro_usd": receipt["original_cost_micro_usd"]})
    total = sum(r["cost_micro_usd"] for r in settled)
    counter = ledger.store.get(ledger.run_key(RUN))
    funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
    if counter["reserved_micro_usd"] or counter["halted"] or counter["settled_micro_usd"] != total:
        raise ValueError("REASON_PROTOCOL_AUDIT_COUNTER")
    with sqlite3.connect(f"file:{(CACHE / 'answers.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        states = dict(db.execute("SELECT state,COUNT(*) FROM answers GROUP BY state"))
    expected = dict(Counter("FAILED" if s["status"] == "FAILED_VALIDATION" else "READY" for s in summaries))
    if states != expected or len(settled) != 32 or len({s["operation_id"] for s in settled}) != 32:
        raise ValueError("REASON_PROTOCOL_AUDIT_DURABLE_MATRIX")
    baseline = json.loads((PARENT / "audit.json").read_bytes())
    result = {"status": "PAIRED_FIXED_CONTEXT_PROTOCOL_AND_SOURCE_OPERATION_AUDIT_PASS",
        "checked_at": datetime.now(UTC).isoformat(), "matrix_cells": 32, "operations": settled,
        "format_statuses": dict(Counter(s["status"] for s in summaries)), "cases": summaries,
        "baseline_failed_validation_cells": baseline["failed_validation_cells"],
        "candidate_failed_validation_cells": sum(s["status"] == "FAILED_VALIDATION" for s in summaries),
        "candidate_missing_reason_cells": reason_missing, "candidate_unknown_key_cells": unknown_keys,
        "valid_claims": len(reviews), "citation_appearances": sum(s["citations"] for s in summaries),
        "source_review_flag_count": sum(bool(r["review_flags_not_verdicts"]) for r in reviews),
        "baseline_model_cost_micro_usd_including_preparation": baseline["total_cost_micro_usd"],
        "candidate_model_cost_micro_usd": total, "total_new_this_turn_micro_usd": baseline["total_cost_micro_usd"] + total,
        "funding_counter": funding, "run_counter": counter, "durable_states": states,
        "headroom_micro_usd": funding["cap_micro_usd"] - funding["settled_micro_usd"] - funding["reserved_micro_usd"],
        "network_blocked_for_rebuild": True, "model_calls_during_audit": 0, "aws_writes": 0,
        "independent_quality_accuracy_measured": False, "human_reviewed": 0, "serving_published": False}
    (OUTPUT / "claim-source-review-queue.json").write_bytes(canonical_json({"claims": reviews, "human_reviewed": 0}) + b"\n")
    (OUTPUT / "audit.json").write_bytes(canonical_json(result) + b"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"operations", "cases", "funding_counter", "run_counter"}}))


if __name__ == "__main__":
    audit()
