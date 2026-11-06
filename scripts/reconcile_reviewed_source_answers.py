"""Read-only operation/source audit; preparation rebuild explicitly blocks sockets."""

import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.ledger import DynamoLedgerStore, Ledger
from app.reviewed_source_graph import load_reviewed_source_graph
from scripts.pilot_reviewed_source_answers import CODE, OUTPUT, QUESTIONS, RUN, prepare_cases
from scripts.run_collection_benchmark import ROOT, OfflineRuntime, load_inputs


def provider_body(system, prompt, tokens):
    return canonical_json({"model": "deepseek-flash", "messages": [{"role": "system", "content": system},
        {"role": "user", "content": prompt}], "thinking": {"type": "disabled"}, "temperature": 0,
        "max_tokens": tokens, "stream": False, "response_format": {"type": "json_object"}})


def audit():
    import boto3
    from botocore.config import Config

    from app.query_preparation_v3 import PLANNER_SYSTEM_V3, REWRITE_SYSTEM_V3

    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    if any(sha256_bytes((ROOT / path).read_bytes()) != scope["code_sha256"][path] for path in CODE):
        raise ValueError("SOURCE_ANSWER_AUDIT_FROZEN_CODE_CHANGED")
    _, _, collection, _, pipeline = load_inputs()
    selection = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    graph = load_reviewed_source_graph(ROOT / selection["release_path"], collection,
        trusted_sha256=selection["release_sha256"], allow_experimental=True)
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
        with patch("socket.socket", side_effect=AssertionError("SOURCE_ANSWER_REBUILD_NETWORK_FORBIDDEN")):
            cases, preparations = prepare_cases(collection, lexical, graph, runtime, None)
    saved_preparations = json.loads((OUTPUT / "preparations.json").read_bytes())
    if canonical_json(preparations) != canonical_json(saved_preparations):
        raise ValueError("SOURCE_ANSWER_AUDIT_NATIVE_PREPARATION_CHANGED")
    operations, sources, summaries = [], 0, []
    quotes = set()
    for case in cases:
        row = json.loads((OUTPUT / "answers" / (case["id"] + ".json")).read_bytes())
        if canonical_json(row["outcome"]) != canonical_json(case["outcome"]) or canonical_json(row["trace"]) != canonical_json(case["trace"]):
            raise ValueError("SOURCE_ANSWER_AUDIT_RETRIEVAL_CHANGED")
        response, receipt = row["response"], row["receipt"]
        p = case["prepared"]
        if (response["request_id"] != p.binding.request_id or response["mode_used"] != case["method"]
                or receipt["binding_sha256"] != sha256_bytes(canonical_json(p.binding.model_dump(mode="json")))
                or receipt["recipe_sha256"] != p.recipe.fingerprint):
            raise ValueError("SOURCE_ANSWER_AUDIT_RESPONSE_BINDING")
        for citation in response["citations"]:
            chunk = case["view"].chunks.get(citation["chunk_id"])
            if chunk is None:
                raise ValueError("SOURCE_ANSWER_AUDIT_CITATION_OUTSIDE_CONTEXT")
            document = case["view"].documents[chunk.document_version_id]
            if (citation["document_version_id"] != document.document_version_id or citation["source_uri"] != document.source_uri
                    or citation["quote"] not in chunk.text or document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text
                    or chunk.chunk_id not in p.packed.chunk_ids):
                raise ValueError("SOURCE_ANSWER_AUDIT_CITATION_SOURCE_CHANGED")
            sources += 1
            quotes.add(sha256_bytes(citation["quote"].encode()))
        operations.append((receipt["operation_id"], "GENERATE", receipt["usage"], receipt["original_cost_micro_usd"],
            sha256_bytes(provider_body(case["system"], case["prompt"], 800))))
        summaries.append({"case": case["id"], "language": case["language"], "method": case["method"],
            "status": response["answer_status"], "claims": len(response["claims"]), "citations": len(response["citations"]),
            "mechanism_status": row["outcome"]["mechanism_status"],
            "max_packed_hops": max((p["hop"] for p in row["trace"].get("packed_paths", [])), default=0)})
    searches = {"en": QUESTIONS["en"]}
    searches["vi"] = next(p["validated"]["search_query"] for p in preparations if p["stage"] == "REWRITE")
    for p in preparations:
        language, stage, receipt = p["language"], p["stage"], p["receipt"]
        prompt = canonical_json({"untrusted_original_question": QUESTIONS[language],
            "untrusted_english_search_query": QUESTIONS[language] if stage == "REWRITE" else searches[language],
            "original_language": language}).decode()
        system = REWRITE_SYSTEM_V3 if stage == "REWRITE" else PLANNER_SYSTEM_V3
        operations.append((receipt["operation_id"], stage, receipt["usage"], receipt["cost_micro_usd"],
            sha256_bytes(provider_body(system, prompt, 256))))
    if len(operations) != 19 or len({r[0] for r in operations}) != 19:
        raise ValueError("SOURCE_ANSWER_AUDIT_OPERATION_MATRIX")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("SOURCE_ANSWER_AUDIT_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    settled = []
    for operation_id, stage, usage, cost, payload_sha in operations:
        op = ledger.operation(operation_id)
        if (op.status != "SETTLED" or not op.call_started or op.bound_exceeded or op.spec.run_id != RUN
                or op.spec.stage != stage or op.spec.input_sha256 != payload_sha or op.actual_micro_usd != cost
                or op.usage.model_dump(mode="json") != usage):
            raise ValueError("SOURCE_ANSWER_AUDIT_OPERATION_USAGE_OR_PAYLOAD")
        settled.append({"operation_id": operation_id, "stage": stage, "status": op.status,
            "usage": usage, "cost_micro_usd": cost, "payload_sha256": payload_sha})
    total = sum(r["cost_micro_usd"] for r in settled)
    run = ledger.store.get(ledger.run_key(RUN))
    funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
    if (run["reserved_micro_usd"] or run["halted"] or run["settled_micro_usd"] != total
            or funding["cap_micro_usd"] != 8000000 or funding["halted"]):
        raise ValueError("SOURCE_ANSWER_AUDIT_COUNTERS")
    states = {}
    for filename, table in (("answers.sqlite3", "answers"), ("query_plans_v3.sqlite3", "preparations"), ("query_rewrites_v3.sqlite3", "preparations")):
        path = ROOT / "data/cache/reviewed-source-answers-v1" / filename
        with sqlite3.connect(f"file:{path.as_posix()}?mode=ro", uri=True) as db:
            states[filename] = dict(db.execute(f"SELECT state,COUNT(*) FROM {table} GROUP BY state"))
    if states != {"answers.sqlite3": {"READY": 16}, "query_plans_v3.sqlite3": {"READY": 2}, "query_rewrites_v3.sqlite3": {"READY": 1}}:
        raise ValueError("SOURCE_ANSWER_AUDIT_DURABLE_STATES")
    report = {"status": "SOURCE_OPERATION_PAYLOAD_USAGE_COST_AND_NETWORK_BLOCKED_REBUILD_PASS",
        "checked_at": datetime.now(UTC).isoformat(), "operations": settled, "total_cost_micro_usd": total,
        "answer_count": len(summaries), "claim_count": sum(s["claims"] for s in summaries), "citation_appearances": sources,
        "unique_quote_hashes": len(quotes), "cases": summaries, "durable_states": states,
        "answer_statuses": dict(Counter(s["status"] for s in summaries)), "run_counter": run, "funding_counter": funding,
        "headroom_micro_usd": funding["cap_micro_usd"] - funding["settled_micro_usd"] - funding["reserved_micro_usd"],
        "network_blocked_for_rebuild": True, "aws_writes": 0, "new_model_calls": 0,
        "human_reviewed": 0, "semantic_accuracy_measured": False, "quality_judgments": "REQUIRE_SOURCE_REVIEW_NOT_MECHANICAL_AUDIT"}
    (OUTPUT / "audit.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "total_cost_micro_usd", "claim_count", "citation_appearances", "headroom_micro_usd")}))


if __name__ == "__main__":
    audit()
