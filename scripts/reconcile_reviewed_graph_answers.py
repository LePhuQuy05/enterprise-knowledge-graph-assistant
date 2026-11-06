"""Read-only managed/source/receipt audit plus enforced network-free graph replay."""

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.answer_cache import AnswerReceipt
from app.collection_graph_runtime import load_collection_graph, prepare_collection_graph_answer
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import canonical_json, sha256_bytes
from app.cost import TokenUsage
from app.embeddings import query_embedding_input
from app.generation import GenerationRecipeV9, generation_request
from app.ledger import Operation
from app.local_bge import BgeEncoder
from app.query_language import rewrite_for_search
from scripts.answer_local import ROOT, RUN, managed_gateway
from scripts.curate_collection_graph import write_immutable
from scripts.pilot_reviewed_graph_answers import FILES
from scripts.reconcile_collection_graph_pilot import require
from scripts.run_reviewed_collection_graph import main as run_case


def body(system, prompt, max_tokens):
    return canonical_json({"model": "deepseek-flash", "messages": [{"role": "system", "content": system},
        {"role": "user", "content": prompt}], "thinking": {"type": "disabled"}, "temperature": 0,
        "max_tokens": max_tokens, "stream": False, "response_format": {"type": "json_object"}})


def main():
    output = ROOT / "results/collection-graph/answer-pilot-audit-v1.json"
    require(not output.exists(), "GRAPH_ANSWER_AUDIT_EXISTS")
    plan_raw = (ROOT / "results/collection-graph/answer-pilot-plan-v1.json").read_bytes()
    plan = json.loads(plan_raw)
    marker = json.loads((ROOT / "results/collection-graph/answer-pilot-started-v1.json").read_bytes())
    summary_raw = (ROOT / "results/collection-graph/answer-pilot-summary-v1.json").read_bytes()
    summary = json.loads(summary_raw)
    require(marker["plan_sha256"] == summary["plan_sha256"] == sha256_bytes(plan_raw)
        and summary["status"] == "COMPLETED_SOURCE_AND_COST_RECONCILIATION_PENDING", "GRAPH_ANSWER_AUDIT_PLAN")
    require(all(sha256_bytes((ROOT / name).read_bytes()) == plan["implementation_sha256"][name] for name in FILES),
        "GRAPH_ANSWER_AUDIT_IMPLEMENTATION")
    gateway = managed_gateway(prices_path=ROOT / "configs/prices/external-direct-2026-10-24.json")
    ledger = gateway.ledger
    pipeline = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    receipt_raw = (ROOT / "results/collection-graph/curated-subset-v1.json").read_bytes()
    receipt = json.loads(receipt_raw)
    require(sha256_bytes(receipt_raw) == plan["graph_receipt_sha256"], "GRAPH_ANSWER_AUDIT_GRAPH_RECEIPT")
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=plan["collection_manifest_sha256"])
    graph = load_collection_graph(Path(receipt["graph_path"]), collection,
        trusted_sha256=plan["graph_file_sha256"], allow_experimental=True)
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    require(encoder.recipe("document") == collection.manifest.embedding_recipe, "GRAPH_ANSWER_AUDIT_ENCODER")
    prices = gateway.prices
    total, operations, sources_count, cases = 0, [], 0, []

    def check_operation(operation_id, usage, cost, expected_body, stage):
        nonlocal total
        operation = Operation.model_validate_json(canonical_json(ledger.store.get(("OP#" + operation_id, "STATE"))))
        rate = prices.rate_for(operation.spec.model_id, operation.spec.invocation_id, operation.spec.source_region,
            now=operation.created_at, live=True)
        require(operation.status == "SETTLED" and operation.call_started and operation.spec.run_id == RUN
            and operation.spec.stage == stage and operation.spec.input_sha256 == sha256_bytes(expected_body)
            and operation.spec.price_sheet_version == prices.version and operation.usage == usage
            and operation.actual_micro_usd == cost == rate.quote(usage), "GRAPH_ANSWER_AUDIT_OPERATION_USAGE_INPUT")
        require(operation.spec.max_cost_micro_usd == rate.quote(TokenUsage(input_tokens=1048576,
            output_tokens=800 if stage == "GENERATE" else 256)), "GRAPH_ANSWER_AUDIT_RESERVATION_BOUND")
        total += cost
        operations.append({"operation_id": operation_id, "stage": stage, "status": operation.status,
            "cost_micro_usd": cost, "input_sha256": operation.spec.input_sha256, "usage": usage.model_dump(mode="json")})

    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        for case, entry in zip(plan["cases"], summary["cases"], strict=True):
            path = ROOT / "results/collection-graph" / ("answer-" + case["id"] + "-v1.json")
            raw = path.read_bytes()
            row = json.loads(raw)
            require(sha256_bytes(raw) == entry["receipt_sha256"] and row["status"] == "COMPLETED"
                and row["graph_file_sha256"] == graph.file_sha256, "GRAPH_ANSWER_AUDIT_CASE_BINDING")
            replay_path = ROOT / "results/collection-graph" / ("replay-" + case["id"] + "-v1.json")
            with patch("boto3.Session", side_effect=AssertionError("REPLAY_AWS_FORBIDDEN")), patch(
                "app.external_http.OfficialHttpsTransport.request", side_effect=AssertionError("REPLAY_PROVIDER_FORBIDDEN")):
                run_case([case["question"], "--hops", str(case["hops"]), "--language", case["language"], "--run-kind", "answer",
                    "--allow-experimental", "--idempotency-key", "graph-demo-" + case["id"] + "-v1", "--output", str(replay_path)])
            replay = json.loads(replay_path.read_bytes())
            require(replay["response"] == row["response"] and replay["answer_cache_hit"] and replay["new_cost_micro_usd"] == 0
                and replay["model_dispatch_attempts_this_invocation"] == 0 and replay["trace"] == row["trace"], "GRAPH_ANSWER_AUDIT_OFFLINE_REPLAY")
            for source in row["context_sources"]:
                shard = collection.get_shard(source["shard_id"])
                c, d = shard.chunks[source["chunk_id"]], shard.documents[source["document_version_id"]]
                require(source["shard_manifest_sha256"] == shard.manifest_sha256 and source["text"] == c.text
                    == d.normalized_text[source["start_char"]:source["end_char"]] and source["source_uri"] == d.source_uri
                    and source["text_sha256"] == c.text_sha256, "GRAPH_ANSWER_AUDIT_CURRENT_SOURCE")
                sources_count += 1
            import numpy as np

            vector = np.asarray(encoder.encode([query_embedding_input(row["search_query"])], role="query")[0], dtype=np.float32)
            recipe = GenerationRecipeV9(model_metadata_sha256=gateway.routes.account_model_metadata_sha256)
            view, prepared, trace = prepare_collection_graph_answer(collection, lexical, graph, case["question"], vector,
                tenant_id="demo", owner_id="local-graph-owner", run_id=RUN, recipe=recipe, hops=case["hops"],
                search_query=row["search_query"], allow_experimental=True)
            require(prepared.binding.request_id == row["request_id"] and canonical_json(trace) == canonical_json(row["trace"]),
                "GRAPH_ANSWER_AUDIT_RETRIEVAL_REBUILD")
            matches = []
            db_path = ROOT / "data/cache/collection-graph-answers/answers.sqlite3"
            with sqlite3.connect("file:" + str(db_path) + "?mode=ro", uri=True) as db:
                for payload, digest in db.execute("SELECT payload,payload_sha256 FROM answers WHERE namespace='LIVE' AND state='READY'"):
                    cached = AnswerReceipt.model_validate_json(payload)
                    if cached.operation_id == row["answer_operation_id"]:
                        require(sha256_bytes(payload) == digest, "GRAPH_ANSWER_AUDIT_RECEIPT_HASH")
                        matches.append(cached)
            require(len(matches) == 1, "GRAPH_ANSWER_AUDIT_UNIQUE_ANSWER_RECEIPT")
            cached = matches[0]
            require(cached.binding_sha256 == sha256_bytes(canonical_json(prepared.binding.model_dump(mode="json")))
                and cached.recipe_sha256 == recipe.fingerprint, "GRAPH_ANSWER_AUDIT_RECEIPT_BINDING")
            system, prompt = generation_request(case["question"], view, prepared.packed, "demo", recipe, answer_language=case["language"])
            check_operation(cached.operation_id, cached.usage, cached.original_cost_micro_usd, body(system, prompt, 800), "GENERATE")
            rewrite = row["rewrite_receipt"]
            if rewrite:
                class ReceiptGateway:
                    def generate(self, context, *, system, prompt, max_output_tokens):
                        self.context, self.body = context, body(system, prompt, max_output_tokens)
                        return SimpleNamespace(operation_id=rewrite["operation_id"], text=rewrite["raw_output"],
                            usage=TokenUsage.model_validate(rewrite["usage"]), actual_micro_usd=rewrite["cost_micro_usd"], stop_reason=rewrite["stop_reason"])

                fake = ReceiptGateway()
                operation = ledger.operation(rewrite["operation_id"])
                search, _ = rewrite_for_search(case["question"], gateway=fake, tenant_id="demo", run_id=RUN,
                    request_id=operation.spec.request_id)
                require(search == row["search_query"] and fake.context.stage == "REWRITE", "GRAPH_ANSWER_AUDIT_REWRITE_REPLAY")
                check_operation(rewrite["operation_id"], TokenUsage.model_validate(rewrite["usage"]),
                    rewrite["cost_micro_usd"], fake.body, "REWRITE")
            cases.append({"id": case["id"], "receipt_sha256": sha256_bytes(raw), "replay_sha256": sha256_bytes(replay_path.read_bytes()),
                "context_assertions": list(prepared.packed.assertion_ids), "requested": row["requested_method"],
                "effective": row["effective_method"], "claims": len(row["response"]["claims"]),
                "citations": len(row["response"]["citations"]), "offline_replay_model_calls": 0, "offline_replay_aws_calls": 0})
    require(total == summary["new_cost_micro_usd"] and len(operations) == summary["model_dispatches"] == 3, "GRAPH_ANSWER_AUDIT_TOTAL")
    after = {name: ledger.store.get((row["PK"], row["SK"])) for name, row in marker["before"].items()}
    for name, before in marker["before"].items():
        current = after[name]
        require(current["cap_micro_usd"] == before["cap_micro_usd"] and current["reserved_micro_usd"] == before["reserved_micro_usd"]
            and current["settled_micro_usd"] >= before["settled_micro_usd"] + total, "GRAPH_ANSWER_AUDIT_COUNTERS_OR_OBLIGATIONS")
    report = {"status": "PASS_SOURCE_RECEIPT_INPUT_USAGE_LEDGER_AND_NETWORK_FREE_REPLAY", "plan_sha256": sha256_bytes(plan_raw),
        "summary_sha256": sha256_bytes(summary_raw), "collection_manifest_sha256": collection.manifest_sha256,
        "graph_file_sha256": graph.file_sha256, "cases": cases, "operations": operations, "source_appearances": sources_count,
        "settled_micro_usd": total, "counters": after, "new_model_calls": 0, "aws_reads": True, "aws_writes": False,
        "human_reviewed": False, "quality_metrics": None, "cloud_deployed": False}
    write_immutable(output, canonical_json(report) + b"\n")
    print(json.dumps({"status": report["status"], "settled_micro_usd": total, "operations": len(operations), "source_appearances": sources_count}))


if __name__ == "__main__":
    main()
