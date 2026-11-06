"""Read-only payload, usage, source and network-blocked replay audit of V2."""

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.authorization_evidence import load_authorization_facts
from app.authorization_retrieval import combine_evidence_graph
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.generation import parse_generated
from app.ingestion.text import ProxyTokenizer
from app.ledger import DynamoLedgerStore, Ledger
from app.query_preparation_v3 import PLANNER_SYSTEM_V3, REWRITE_SYSTEM_V3
from app.reviewed_source_graph import load_reviewed_source_graph
from scripts.continue_source_message_answers import restore_recorded_rerank_timings
from scripts.pilot_source_message_answers_v2 import (
    CACHE,
    OUTPUT,
    QUESTIONS,
    RUN,
    code_hashes,
    prepare_cases,
)
from scripts.prepare_authorization_evidence import ROOT, load_sources
from scripts.reconcile_reviewed_source_answers import provider_body
from scripts.run_collection_benchmark import OfflineRuntime, load_inputs


def audit():
    import boto3
    from botocore.config import Config

    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    if code_hashes() != scope["code_sha256"]:
        raise ValueError("SOURCE_MESSAGE_AUDIT_FROZEN_CODE_CHANGED")
    collection, chunks, documents = load_sources()
    _, _, _, _, pipeline = load_inputs()
    selection = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    base = load_reviewed_source_graph(ROOT / selection["release_path"], collection,
        trusted_sha256=selection["release_sha256"], allow_experimental=True)
    auth_selection = json.loads((ROOT / "configs/graph/authorization-evidence-local-v1.json").read_bytes())
    auth = load_authorization_facts(ROOT / auth_selection["release_path"], collection, chunks, documents,
        trusted_sha256=auth_selection["release_sha256"], allow_experimental=True)
    graph = combine_evidence_graph(collection, base, auth)
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
        with patch("socket.socket", side_effect=AssertionError("SOURCE_MESSAGE_REBUILD_NETWORK_FORBIDDEN")):
            cases, preparations = prepare_cases(collection, lexical, graph, runtime, None)
    if canonical_json(preparations) != canonical_json(json.loads((OUTPUT / "preparations.json").read_bytes())):
        raise ValueError("SOURCE_MESSAGE_AUDIT_PREPARATION_CHANGED")
    frozen_cases = {c["id"]: c for c in json.loads((OUTPUT / "answer-plan.json").read_bytes())["cases"]}
    for case in cases:
        case["trace"] = restore_recorded_rerank_timings(case["trace"], frozen_cases[case["id"]]["trace"])
    operations, summaries, quotes, evidence = [], [], set(), []
    t = ProxyTokenizer()
    for case in cases:
        answer_path = OUTPUT / "answers" / (case["id"] + ".json")
        failed_path = OUTPUT / "failed-answers" / (case["id"] + ".json")
        if not answer_path.exists():
            row = json.loads(failed_path.read_bytes())
            p, receipt = case["prepared"], row["receipt"]
            if (row["answer_status"] != "FAILED_VALIDATION" or row["repaired"] or row["retried"]
                    or canonical_json(row["trace"]) != canonical_json(case["trace"])
                    or receipt["binding_sha256"] != sha256_bytes(canonical_json(p.binding.model_dump(mode="json")))
                    or receipt["recipe_sha256"] != p.recipe.fingerprint):
                raise ValueError("SOURCE_MESSAGE_AUDIT_FAILED_CELL_BINDING")
            try:
                parse_generated(receipt["raw_output"], receipt["stop_reason"], p.recipe)
            except ValueError as exc:
                if str(exc) != row["validation_error"]:
                    raise ValueError("SOURCE_MESSAGE_AUDIT_FAILED_CELL_DIAGNOSTIC") from None
            else:
                raise ValueError("SOURCE_MESSAGE_AUDIT_EXPECTED_SCHEMA_FAILURE_NOT_REPRODUCED")
            operations.append((receipt["operation_id"], "GENERATE", receipt["usage"], receipt["original_cost_micro_usd"],
                sha256_bytes(provider_body(case["system"], case["prompt"], 800))))
            summaries.append({"case": case["id"], "intent": case["intent"], "language": case["language"],
                "method": case["method"], "status": "FAILED_VALIDATION", "claims": 0, "citations": 0,
                "source_chunks": len(p.packed.chunk_ids), "user_message_proxy_tokens": p.packed.proxy_tokens,
                "mechanism_status": row["outcome"]["mechanism_status"], "max_packed_hops": 0,
                "validation_error": row["validation_error"], "included_in_attempted_method_denominator": True})
            continue
        row = json.loads(answer_path.read_bytes())
        p, response, receipt = case["prepared"], row["response"], row["receipt"]
        if (canonical_json(row["outcome"]) != canonical_json(case["outcome"])
                or canonical_json(row["trace"]) != canonical_json(case["trace"])
                or response["request_id"] != p.binding.request_id or response["mode_used"] != case["method"]
                or response["answer_language"] != case["language"] or response["used_graph"] != p.binding.used_graph
                or receipt["binding_sha256"] != sha256_bytes(canonical_json(p.binding.model_dump(mode="json")))
                or receipt["recipe_sha256"] != p.recipe.fingerprint or p.packed.serialized != case["prompt"]
                or t.count(case["prompt"]) != p.packed.proxy_tokens or p.packed.proxy_tokens > 4096):
            raise ValueError("SOURCE_MESSAGE_AUDIT_RESPONSE_PAYLOAD_BINDING")
        by_citation = {c["citation_id"]: c for c in response["citations"]}
        for citation in response["citations"]:
            chunk = case["view"].chunks.get(citation["chunk_id"])
            if chunk is None or chunk.chunk_id not in p.packed.chunk_ids:
                raise ValueError("SOURCE_MESSAGE_AUDIT_CITATION_NOT_IN_CONTEXT")
            document = case["view"].documents[chunk.document_version_id]
            if (citation["document_version_id"] != document.document_version_id or citation["source_uri"] != document.source_uri
                    or citation["quote"] not in chunk.text or document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
                raise ValueError("SOURCE_MESSAGE_AUDIT_CITATION_SOURCE")
            quotes.add(sha256_bytes(citation["quote"].encode()))
        for claim in response["claims"]:
            if not claim["citation_ids"] or not set(claim["citation_ids"]) <= by_citation.keys():
                raise ValueError("SOURCE_MESSAGE_AUDIT_CLAIM_CITATION_IDS")
            evidence.append({"case": case["id"], "intent": case["intent"], "language": case["language"],
                "method": case["method"], "claim_id": claim["claim_id"], "claim": claim["text"],
                "citations": [by_citation[key] for key in claim["citation_ids"]],
                "review_status": "PENDING_SOURCE_SUPPORT_REVIEW_NOT_GOLD"})
        operations.append((receipt["operation_id"], "GENERATE", receipt["usage"], receipt["original_cost_micro_usd"],
            sha256_bytes(provider_body(case["system"], case["prompt"], 800))))
        summaries.append({"case": case["id"], "intent": case["intent"], "language": case["language"], "method": case["method"],
            "status": response["answer_status"], "claims": len(response["claims"]), "citations": len(response["citations"]),
            "source_chunks": len(p.packed.chunk_ids), "user_message_proxy_tokens": p.packed.proxy_tokens,
            "mechanism_status": row["outcome"]["mechanism_status"],
            "max_packed_hops": max((p["hop"] for p in case["trace"].get("packed_paths", [])), default=0)})
    searches = {(intent, "en"): languages["en"] for intent, languages in QUESTIONS.items()}
    searches.update({(p["intent"], p["language"]): p["validated"]["search_query"] for p in preparations if p["stage"] == "REWRITE"})
    for prep in preparations:
        intent, language, stage = prep["intent"], prep["language"], prep["stage"]
        receipt, question = prep["receipt"], QUESTIONS[intent][language]
        prompt = canonical_json({"untrusted_original_question": question,
            "untrusted_english_search_query": question if stage == "REWRITE" else searches[(intent, language)],
            "original_language": language}).decode()
        system = REWRITE_SYSTEM_V3 if stage == "REWRITE" else PLANNER_SYSTEM_V3
        operations.append((receipt["operation_id"], stage, receipt["usage"], receipt["cost_micro_usd"],
            sha256_bytes(provider_body(system, prompt, 256))))
    if len(operations) != 38 or len({op[0] for op in operations}) != 38:
        raise ValueError("SOURCE_MESSAGE_AUDIT_OPERATION_COUNT")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("SOURCE_MESSAGE_AUDIT_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    settled = []
    for operation_id, stage, usage, cost, payload_sha in operations:
        op = ledger.operation(operation_id)
        if (op.status != "SETTLED" or not op.call_started or op.bound_exceeded or op.spec.run_id != RUN
                or op.spec.stage != stage or op.spec.input_sha256 != payload_sha or op.actual_micro_usd != cost
                or op.usage.model_dump(mode="json") != usage):
            raise ValueError("SOURCE_MESSAGE_AUDIT_OPERATION_USAGE_COST_OR_PAYLOAD")
        settled.append({"operation_id": operation_id, "stage": stage, "status": op.status, "usage": usage,
            "cost_micro_usd": cost, "payload_sha256": payload_sha})
    total = sum(op["cost_micro_usd"] for op in settled)
    counter = ledger.store.get(ledger.run_key(RUN))
    funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
    if (counter["reserved_micro_usd"] or counter["halted"] or counter["settled_micro_usd"] != total
            or funding["cap_micro_usd"] != 8000000 or funding["halted"]):
        raise ValueError("SOURCE_MESSAGE_AUDIT_COUNTERS")
    states = {}
    for filename, table in (("answers.sqlite3", "answers"), ("query_plans_v3.sqlite3", "preparations"), ("query_rewrites_v3.sqlite3", "preparations")):
        with sqlite3.connect(f"file:{(CACHE / filename).as_posix()}?mode=ro", uri=True) as db:
            states[filename] = dict(db.execute(f"SELECT state,COUNT(*) FROM {table} GROUP BY state"))
    expected_answers = dict(Counter("FAILED" if s["status"] == "FAILED_VALIDATION" else "READY" for s in summaries))
    if states != {"answers.sqlite3": expected_answers, "query_plans_v3.sqlite3": {"READY": 4}, "query_rewrites_v3.sqlite3": {"READY": 2}}:
        raise ValueError("SOURCE_MESSAGE_AUDIT_DURABLE_STATES")
    (OUTPUT / "claim-source-review-queue.json").write_bytes(canonical_json({"version": "source-message-claim-review-queue-v1",
        "claim_count": len(evidence), "claims": evidence, "human_reviewed": 0}) + b"\n")
    report = {"status": "NATIVE_SOURCE_MESSAGE_PAYLOAD_USAGE_SOURCE_AND_NETWORK_BLOCKED_REBUILD_PASS",
        "checked_at": datetime.now(UTC).isoformat(), "operations": settled, "total_cost_micro_usd": total,
        "matrix_cells": len(summaries), "valid_answer_count": sum(s["status"] != "FAILED_VALIDATION" for s in summaries),
        "failed_validation_cells": sum(s["status"] == "FAILED_VALIDATION" for s in summaries),
        "claim_count": len(evidence), "citation_appearances": sum(s["citations"] for s in summaries),
        "unique_quote_hashes": len(quotes), "answer_statuses": dict(Counter(s["status"] for s in summaries)),
        "cases": summaries, "durable_states": states, "run_counter": counter, "funding_counter": funding,
        "headroom_micro_usd": funding["cap_micro_usd"] - funding["settled_micro_usd"] - funding["reserved_micro_usd"],
        "network_blocked_for_rebuild": True, "new_model_calls_during_audit": 0, "aws_writes": 0,
        "human_reviewed": 0, "quality_accuracy_measured": False}
    (OUTPUT / "audit.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k not in {"operations", "cases", "funding_counter", "run_counter"}}))


if __name__ == "__main__":
    audit()
