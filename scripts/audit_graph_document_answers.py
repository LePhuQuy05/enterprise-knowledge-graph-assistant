"""Failure-inclusive cache/source replay and optional read-only ledger audit."""

import argparse
import json
import sqlite3
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from unittest.mock import patch

from app.answer_cache import AnswerCache, AnswerReceipt
from app.contracts import canonical_json, sha256_bytes
from app.generation import parse_generated
from scripts.pilot_graph_document_answers import BASELINE, CACHE, CODE, OUTPUT, RUN, allocated_cases
from scripts.prepare_authorization_evidence import ROOT
from scripts.reconcile_reviewed_source_answers import provider_body


def audit(*, ledger_read=False):
    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
        raise ValueError("DOCUMENT_ANSWER_AUDIT_PAID_CODE_CHANGED")
    summaries, operations, reviews = [], [], []
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=AssertionError("DOCUMENT_ANSWER_REPLAY_NETWORK_BLOCKED")))
        cases, _ = allocated_cases()
        cache = AnswerCache(CACHE, namespace="LIVE")
        for case in cases:
            p = case["prepared"]
            success_path = OUTPUT / "answers" / (case["id"] + ".json")
            failed = not success_path.exists()
            row = json.loads((OUTPUT / "failed-answers" / (case["id"] + ".json") if failed else success_path).read_bytes())
            receipt = AnswerReceipt.model_validate_json(canonical_json(row["receipt"]))
            if (row["retried"] or row["repaired"] or receipt.namespace != "LIVE"
                    or receipt.recipe_sha256 != p.recipe.fingerprint
                    or receipt.binding_sha256 != sha256_bytes(canonical_json(p.binding.model_dump(mode="json")))):
                raise ValueError("DOCUMENT_ANSWER_AUDIT_RECEIPT_BINDING")
            intent = canonical_json({"context": p.context.model_dump(mode="json"), "binding_sha256": receipt.binding_sha256,
                "recipe_sha256": receipt.recipe_sha256})
            with sqlite3.connect(f"file:{(CACHE / 'answers.sqlite3').as_posix()}?mode=ro", uri=True) as db:
                cached = db.execute("SELECT state,payload,payload_sha256 FROM answers WHERE namespace='LIVE' AND intent=?", (intent,)).fetchone()
            if (cached is None or cached[0] != ("FAILED" if failed else "READY") or cached[1] is None
                    or sha256_bytes(cached[1]) != cached[2]
                    or canonical_json(AnswerReceipt.model_validate_json(cached[1]).model_dump(mode="json")) != canonical_json(receipt.model_dump(mode="json"))):
                raise ValueError("DOCUMENT_ANSWER_AUDIT_DURABLE_STATE")
            if failed:
                try:
                    parse_generated(receipt.raw_output, receipt.stop_reason, p.recipe)
                except ValueError:
                    pass
                else:
                    # A FAILED row could instead have invalid source references;
                    # never pretend this was a schema failure if not reproduced.
                    raise ValueError("DOCUMENT_ANSWER_FAILED_SCHEMA_NEEDS_SOURCE_FAILURE_AUDIT")
                status, claims, citations = "FAILED_VALIDATION", [], []
            else:
                lookup = cache.resolve(p.context, question=p.question, snapshot=case["view"], packed=p.packed,
                    binding=p.binding, recipe=p.recipe, gateway=None, answer_language=case["language"])
                if not lookup.cache_hit or lookup.response.model_dump(mode="json") != row["response"]:
                    raise ValueError("DOCUMENT_ANSWER_AUDIT_RESPONSE_REPLAY")
                response = row["response"]
                if (response["mode_used"] != case["method"] or not response["used_graph"]
                        or response["answer_language"] != case["language"] or response["request_id"] != p.binding.request_id):
                    raise ValueError("DOCUMENT_ANSWER_AUDIT_METHOD_OR_LANGUAGE")
                status, claims, citations = response["answer_status"], response["claims"], response["citations"]
                by_id = {c["citation_id"]: c for c in citations}
                for c in citations:
                    chunk = case["view"].chunks[c["chunk_id"]]
                    doc = case["view"].documents[chunk.document_version_id]
                    if (chunk.chunk_id not in p.packed.chunk_ids or c["document_version_id"] != doc.document_version_id
                            or c["source_uri"] != doc.source_uri or c["quote"] not in chunk.text
                            or doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
                        raise ValueError("DOCUMENT_ANSWER_AUDIT_ORIGINAL_SOURCE")
                for claim in claims:
                    if not claim["citation_ids"] or not set(claim["citation_ids"]) <= by_id.keys():
                        raise ValueError("DOCUMENT_ANSWER_AUDIT_CLAIM_CITATION")
                    reviews.append({"case": case["id"], "claim_id": claim["claim_id"], "claim": claim["text"],
                        "citations": [by_id[k] for k in claim["citation_ids"]],
                        "review_status": "PENDING_INDEPENDENT_ADJUDICATION_NOT_GOLD"})
            old = json.loads((BASELINE / "answers" / (case["id"] + ".json")).read_bytes())
            old_response = old["response"]
            if old_response["request_id"] != case["baseline_request_id"]:
                raise ValueError("DOCUMENT_ANSWER_AUDIT_BASELINE_ID")
            summaries.append({"id": case["id"], "language": case["language"], "method": case["method"],
                "baseline_status_self_report": old_response["answer_status"], "candidate_status_self_report": status,
                "baseline_claims": len(old_response["claims"]), "candidate_claims": len(claims),
                "candidate_citations": len(citations), "context_user_message_changed": case["baseline_prompt_sha256"] != sha256_bytes(case["prompt"].encode()),
                "used_graph": p.binding.used_graph, "hidden_fallback": False, "quality_correctness_not_inferred": True})
            operations.append({"operation_id": receipt.operation_id,
                "payload_sha256": sha256_bytes(provider_body(case["provider_system"], case["prompt"], 800)),
                "cost_micro_usd": receipt.original_cost_micro_usd, "usage": receipt.usage.model_dump(mode="json")})
    with sqlite3.connect(f"file:{(CACHE / 'answers.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        states = dict(db.execute("SELECT state,COUNT(*) FROM answers WHERE namespace='LIVE' GROUP BY state"))
    expected = dict(Counter("FAILED" if r["candidate_status_self_report"] == "FAILED_VALIDATION" else "READY" for r in summaries))
    total = sum(o["cost_micro_usd"] for o in operations)
    if states != expected or len(operations) != 8 or len({o["operation_id"] for o in operations}) != 8:
        raise ValueError("DOCUMENT_ANSWER_AUDIT_EIGHT_DISTINCT_OPERATIONS")
    result = {"status": "EIGHT_CONTEXTS_SOURCES_AND_FAILURE_INCLUSIVE_RESPONSE_REPLAY_PASS", "cases": summaries,
        "matrix_cells": 8, "operations": operations, "durable_states": states,
        "total_cost_micro_usd": total, "valid_claims": len(reviews), "human_reviewed": 0,
        "citation_appearances": sum(s["candidate_citations"] for s in summaries),
        "new_model_calls_during_audit": 0, "network_blocked_for_rebuild_and_replay": True,
        "quality_accuracy_measured": False, "serving_published": False}
    if ledger_read:
        import boto3
        from botocore.config import Config

        from app.ledger import DynamoLedgerStore, Ledger

        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
            raise ValueError("DOCUMENT_ANSWER_AUDIT_OPERATOR")
        ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
        for row in operations:
            op = ledger.operation(row["operation_id"])
            if (op.status != "SETTLED" or op.bound_exceeded or not op.call_started
                    or op.spec.run_id != RUN or op.spec.stage != "GENERATE" or op.spec.input_sha256 != row["payload_sha256"]
                    or op.usage.model_dump(mode="json") != row["usage"] or op.actual_micro_usd != row["cost_micro_usd"]):
                raise ValueError("DOCUMENT_ANSWER_AUDIT_LEDGER_PAYLOAD_USAGE_COST")
            row["ledger_status"] = op.status
        counter = ledger.store.get(ledger.run_key(RUN))
        funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
        if counter["reserved_micro_usd"] or counter["halted"] or counter["settled_micro_usd"] != total:
            raise ValueError("DOCUMENT_ANSWER_AUDIT_COUNTER")
        result.update(status="EIGHT_NATIVE_GRAPH_SOURCE_PAYLOAD_USAGE_COST_LEDGER_AUDIT_PASS",
            checked_at=datetime.now(UTC).isoformat(), run_counter=counter, funding_counter=funding,
            headroom_micro_usd=funding["cap_micro_usd"] - funding["settled_micro_usd"] - funding["reserved_micro_usd"],
            aws_writes_during_audit=0)
    (OUTPUT / "claim-source-review-queue.json").write_bytes(canonical_json({"claims": reviews, "human_reviewed": 0}) + b"\n")
    (OUTPUT / ("audit.json" if ledger_read else "replay.json")).write_bytes(canonical_json(result) + b"\n")
    print(json.dumps({k: v for k, v in result.items() if k not in {"cases", "operations", "funding_counter", "run_counter"}}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", action="store_true")
    audit(ledger_read=parser.parse_args().ledger)
