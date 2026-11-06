"""Read-only H/Q audit: cached vectors/receipts and STS/DynamoDB reads, no model calls."""

import argparse
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config

from app.answer_cache import AnswerReceipt
from app.collection_answering import _evidence_view
from app.collection_lexical import load_global_lexical_index, retrieve_collection_hybrid
from app.collections import load_collection
from app.context import pack_context
from app.contracts import (
    LocalBgeRecipe,
    canonical_json,
    embedding_cache_key,
    sha256_bytes,
    stable_id,
)
from app.cost import TokenUsage
from app.embeddings import CachedEmbedding, query_embedding_input
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport
from app.generation import (
    AnswerBinding,
    GenerationRecipeV9,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.graph_retrieval import GraphRetrievalResult
from app.ledger import DynamoLedgerStore, Ledger
from app.local_bge import BgeEncoder
from app.model_gateway import CallContext, ModelResult, safe_failure_code
from app.multi_query import retrieve_multi_query
from app.query_language import rewrite_for_search
from app.query_planning import PLANNER_SYSTEM, PlannerRecipe
from app.query_planning import _parse_subqueries as parse_saved_plan
from app.query_planning import _Receipt as PlannerReceipt
from app.query_planning import _route_identity as route_identity
from app.rag_methods import MethodResultV1
from app.reranking import OnnxCrossEncoder
from app.rewrite_cache import PROMPT_VERSION, RECIPE
from app.rewrite_cache import _Receipt as RewriteReceipt
from scripts.answer_local import ROOT, RUN
from scripts.replay_collection_run import assert_replay
from scripts.run_multi_query_pilot import check_pair
from scripts.run_reranker_answer_pilot import write_once

DIRECTORY = ROOT / "results/multi-query/pilot-v1"
OWNER = "local-ldh-owner"


def require(condition, category):
    if not condition:
        raise ValueError(category)


def read_json(path, maximum=4 * 1024 * 1024):
    require(path.is_file() and not path.is_symlink() and 1 <= path.stat().st_size <= maximum,
            "Q_AUDIT_ARTIFACT_MISSING_UNSAFE_OR_OVERSIZED")
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def connection(path):
    require(path.is_file() and not path.is_symlink(), "Q_AUDIT_LOCAL_CACHE_PATH")
    return sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)


def body(system, prompt, output):
    return canonical_json({"model": "deepseek-flash", "messages": [
        {"role": "system", "content": system}, {"role": "user", "content": prompt}],
        "thinking": {"type": "disabled"}, "temperature": 0, "max_tokens": output,
        "stream": False, "response_format": {"type": "json_object"}})


def cached_row(db, table, key, expected_intent, model, context_key=None):
    require(table in {"answers", "query_plans", "rewrites"}, "Q_AUDIT_CACHE_TABLE")
    row = db.execute(f"SELECT state,intent,payload,payload_sha256 FROM {table} "
                     "WHERE namespace='LIVE' AND cache_key=?", (key,)).fetchone()
    require(row is not None and row[0] == "READY" and row[1] == canonical_json(expected_intent),
            "Q_AUDIT_CACHE_STATE_OR_INTENT")
    require(isinstance(row[2], bytes) and 1 <= len(row[2]) <= 128 * 1024
            and sha256_bytes(row[2]) == row[3], "Q_AUDIT_CACHE_PAYLOAD_CHECKSUM")
    if context_key is not None:
        require(db.execute(f"SELECT context_key FROM {table} WHERE namespace='LIVE' AND cache_key=?",
                           (key,)).fetchone()[0] == context_key, "Q_AUDIT_CACHE_CONTEXT_KEY")
    saved = model.model_validate_json(row[2])
    require(saved.namespace == "LIVE" and saved.cache_key == key, "Q_AUDIT_CACHE_NAMESPACE_OR_KEY")
    return saved


def cached_vector(db, query, recipe):
    text = query_embedding_input(query)
    key = embedding_cache_key("demo", text, recipe)
    row = db.execute("SELECT state,payload,payload_sha256 FROM embeddings "
                     "WHERE namespace='LIVE' AND cache_key=?", (key,)).fetchone()
    require(row is not None and row[0] == "READY" and sha256_bytes(row[1]) == row[2],
            "Q_AUDIT_EMBEDDING_CACHE_STATE_OR_CHECKSUM")
    value = CachedEmbedding.model_validate_json(row[1])
    require(value.namespace == "LIVE" and value.tenant_id == "demo" and value.cache_key == key
            and value.recipe == recipe and value.input_text == text, "Q_AUDIT_EMBEDDING_IDENTITY")
    return np.asarray(value.vector, dtype=np.float32)


def expectation(context, request_body, receipt, *, case, method, output):
    require(receipt["operation_id"] == stable_id("op", **context.model_dump(mode="json")),
            "Q_AUDIT_RECEIPT_OPERATION_ID")
    return {"case": case, "method": method, "operation_id": receipt["operation_id"],
            "context": context.model_dump(mode="json"), "input_sha256": sha256_bytes(request_body),
            "cost_micro_usd": receipt["cost_micro_usd"], "usage": receipt["usage"], "output_bound": output}


def old_r_hashes():
    directory = ROOT / "results/reranker-answering/pilot-v1"
    audit, _ = read_json(directory / "reconciliation.json")
    require(audit["status"] == "PASS_SOURCE_CACHE_LEDGER_REPLAY", "Q_AUDIT_PREVIOUS_R_NOT_VERIFIED")
    for row in audit["cases"]:
        _, digest = read_json(directory / f"case-{row['case']}.json")
        require(digest == row["live_sha256"], "Q_AUDIT_OLD_R_LIVE_ARTIFACT_CHANGED")
    names = ("plan.json", "summary.json", "reconciliation.json", *(f"case-{n}.json" for n in range(1, 4)),
             *(f"case-{n}-replay.json" for n in range(1, 4)))
    return {name: read_json(directory / name)[1] for name in names}


def audit_local(report):
    plan, plan_hash = read_json(DIRECTORY / "plan.json")
    marker, _ = read_json(DIRECTORY / "execution-started.json")
    summary, summary_hash = read_json(DIRECTORY / "summary.json")
    require(summary["status"] == "COMPLETED" and summary["cost_complete"] and summary["case_reports"] == 3,
            "Q_AUDIT_PILOT_INCOMPLETE_NO_RETRY")
    preflight_path = Path(marker["preflight_path"]).resolve()
    require(preflight_path.parent == DIRECTORY.resolve() and preflight_path.name.startswith("preflight-"),
            "Q_AUDIT_PREFLIGHT_PATH")
    preflight, preflight_hash = read_json(preflight_path)
    require(marker["plan_sha256"] == plan_hash and marker["preflight_sha256"] == preflight_hash
            and marker["resume_allowed"] is False and preflight["status"] == "READY_READ_ONLY_NO_INFERENCE",
            "Q_AUDIT_EXECUTION_MARKER_OR_PREFLIGHT")
    paths, pipeline_hash = read_json(ROOT / "results/corpus-expansion-v2/pipeline.json")
    _, cases_hash = read_json(ROOT / "configs/evaluation/multi-query-answer-pilot-v1.json")
    prices_path = ROOT / "configs/prices/external-direct-2026-10-24.json"
    prices = ExternalPriceSheet.model_validate_json(prices_path.read_bytes())
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    planner_recipe = PlannerRecipe.model_validate_json(canonical_json(plan["planner_recipe"]))
    require(plan["pipeline_sha256"] == pipeline_hash and plan["cases_sha256"] == cases_hash
            and plan["price_sheet_sha256"] == sha256_bytes(prices_path.read_bytes()) and plan["run_id"] == RUN
            and plan["generation_recipe_sha256"] == recipe.fingerprint
            and plan["planner_recipe_sha256"] == planner_recipe.fingerprint, "Q_AUDIT_PINNED_PLAN_MISMATCH")
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    require(collection.manifest_sha256 == plan["collection_sha256"], "Q_AUDIT_COLLECTION_PIN")
    query_recipe = LocalBgeRecipe.model_validate_json(canonical_json(collection.manifest.embedding_recipe.model_dump(
        mode="json") | {"input_role": "query", "input_recipe_version": "question-text-v1"}))
    require(query_recipe.fingerprint == plan["query_recipe_sha256"], "Q_AUDIT_QUERY_RECIPE")
    pending, cases, source_count, citation_count = [], [], 0, 0
    old_r = old_r_hashes()
    with connection(ROOT / "data/cache/collection-answers/answers.sqlite3") as answers, \
            connection(ROOT / "data/cache/collection-query-plans/query_plans.sqlite3") as plans, \
            connection(ROOT / "data/cache/collection-rewrites/rewrites.sqlite3") as rewrites, \
            connection(ROOT / "data/cache/local-bge/embeddings.sqlite3") as embeddings, \
            load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        for number in range(1, 4):
            live, live_hash = read_json(DIRECTORY / f"case-{number}.json")
            replay, replay_hash = read_json(DIRECTORY / f"case-{number}-replay.json")
            check_pair(live)
            assert_replay(live, replay)
            require(replay["replay_verified_against_sha256"] == live_hash
                    and replay["model_dispatch_attempts_this_invocation"] == replay["aws_api_calls_this_invocation"] == 0
                    and replay["local_query_onnx_calls_this_invocation"] == 0, "Q_AUDIT_REPLAY_CALLS_OR_HASH")
            require(live["input"] == plan["requests"][number - 1] and live["pilot_case_id"] == plan["case_ids"][number - 1]
                    and live["planner_recipe_sha256"] == planner_recipe.fingerprint, "Q_AUDIT_CASE_INPUT_OR_RECIPE")
            identity = stable_id("ldh", tenant="demo", owner=OWNER, key=live["input"]["idempotency_key"])
            require(live["run_identity"] == identity, "Q_AUDIT_RUN_IDENTITY")
            base_vector = cached_vector(embeddings, live["search_query"], query_recipe)
            per_case = []
            for method in live["methods"]:
                parsed = MethodResultV1.model_validate_json(canonical_json(method))
                mode = parsed.effective_method
                require(parsed.config.collection_manifest_sha256 == collection.manifest_sha256
                        and parsed.config.lexical_index_version == lexical.manifest.index_version
                        and parsed.config.document_recipe_sha256 == collection.manifest.embedding_recipe.fingerprint
                        and parsed.config.query_recipe_sha256 == query_recipe.fingerprint
                        and parsed.config.generation_recipe_sha256 == recipe.fingerprint,
                        "Q_AUDIT_METHOD_RECIPE_BINDING")
                if mode == "hybrid_multi_query":
                    planning = parsed.planning_trace
                    context = CallContext(tenant_id="demo", run_id=RUN, request_id=stable_id(
                        "plan", run=identity, config=parsed.config_sha256), stage="PLAN_QUERY")
                    intent = {"context": context.model_dump(mode="json"),
                        "question_sha256": sha256_bytes(live["input"]["original_question"].encode()),
                        "search_query_sha256": sha256_bytes(live["search_query"].encode()),
                        "language": live["answer_language"], "route": route_identity(routes),
                        "recipe_sha256": planner_recipe.fingerprint}
                    binding_sha = sha256_bytes(canonical_json(intent))
                    key = stable_id("query-plan", binding_sha256=binding_sha)
                    saved = cached_row(plans, "query_plans", key, intent, PlannerReceipt,
                                       stable_id("query-plan-context", **context.model_dump(mode="json")))
                    require(saved.binding_sha256 == binding_sha and not planning.cache_hit
                            and saved.provider_receipt.model_dump(mode="json") == planning.receipt
                            and parse_saved_plan(saved.provider_receipt, context, live["search_query"], planner_recipe)
                            == planning.subqueries, "Q_AUDIT_PLANNER_RECEIPT_OR_BINDING")
                    sub_vectors = tuple(cached_vector(embeddings, query, query_recipe) for query in planning.subqueries)
                    actual = retrieve_multi_query(collection, lexical, live["search_query"], base_vector,
                        subqueries=planning.subqueries, subquery_vectors=sub_vectors)
                    require(actual == parsed.multi_query_trace and actual.queries[0].ranked_chunk_ids
                            == tuple(live["methods"][0]["ranked_chunk_ids"]), "Q_AUDIT_ACTUAL_MULTI_QUERY_RETRIEVAL")
                    prompt = canonical_json({"untrusted_original_question": live["input"]["original_question"],
                        "untrusted_english_search_query": live["search_query"],
                        "original_language": live["answer_language"]}).decode()
                    per_case.append(expectation(context, body(PLANNER_SYSTEM, prompt, 256), planning.receipt,
                                                case=number, method=mode, output=256))
                    require(planning.new_cost_micro_usd == planning.receipt["cost_micro_usd"],
                            "Q_AUDIT_NEW_PLANNER_COST")
                else:
                    require(tuple(retrieve_collection_hybrid(collection, lexical, live["search_query"], base_vector,
                        vector_k=20, lexical_k=20)) == parsed.ranked_chunk_ids, "Q_AUDIT_ACTUAL_BASE_H_RETRIEVAL")
                ids = parsed.ranked_chunk_ids
                view = _evidence_view(collection, lexical, ids)
                packed = pack_context(view, GraphRetrievalResult(ids, ids, (), (), mode, False, (), 0, 0))
                require(packed.chunk_ids == parsed.context_chunk_ids and packed.proxy_tokens == parsed.context_proxy_tokens,
                        "Q_AUDIT_CONTEXT_PACKING")
                require(len(parsed.context_sources) == len(packed.chunk_ids)
                        and tuple(source["source_id"] for source in parsed.context_sources) == packed.source_ids
                        and tuple(source["chunk_id"] for source in parsed.context_sources) == packed.chunk_ids,
                        "Q_AUDIT_CONTEXT_SOURCE_ORDER")
                for source in parsed.context_sources:
                    chunk = view.chunks[source["chunk_id"]]
                    document, ref = view.documents[chunk.document_version_id], view.source_refs[chunk.chunk_id]
                    require(source["start_char"] == chunk.start_char and source["end_char"] == chunk.end_char
                            and source["text"] == chunk.text == document.normalized_text[chunk.start_char:chunk.end_char]
                            and sha256_bytes(document.normalized_text.encode()) == document.normalized_sha256
                            and source["text_sha256"] == chunk.text_sha256 == sha256_bytes(chunk.text.encode())
                            and source["document_version_id"] == document.document_version_id
                            and source["source_uri"] == document.source_uri and source["title"] == document.title
                            and source["section_path"] == list(chunk.section_path) and source["shard_id"] == ref.shard_id
                            and source["shard_manifest_sha256"] == ref.manifest_sha256, "Q_AUDIT_SOURCE_PROVENANCE")
                    source_count += 1
                response = parsed.response
                request_id = stable_id("method", run=identity, config=parsed.config_sha256)
                require(response.request_id == request_id and not parsed.cache_hit, "Q_AUDIT_NEW_GENERATION_ID")
                binding = AnswerBinding(request_id=request_id, tenant_id="demo", owner_id=OWNER,
                                        mode_requested=mode, mode_used=mode, used_graph=False)
                context = CallContext(tenant_id="demo", run_id=RUN, request_id=request_id, stage="GENERATE")
                system, prompt = generation_request(live["input"]["original_question"], view, packed, "demo", recipe,
                                                    answer_language=live["answer_language"])
                binding_sha = sha256_bytes(canonical_json(binding.model_dump(mode="json")))
                key = stable_id("answer", context=context.model_dump(mode="json"), binding_sha=binding_sha,
                    request_sha=sha256_bytes(canonical_json([system, prompt])), recipe=recipe.fingerprint,
                    snapshot_sha=view.manifest_sha256)
                intent = {"context": context.model_dump(mode="json"), "binding_sha256": binding_sha,
                          "recipe_sha256": recipe.fingerprint}
                receipt = cached_row(answers, "answers", key, intent, AnswerReceipt)
                require(receipt.binding_sha256 == binding_sha and receipt.recipe_sha256 == recipe.fingerprint
                        and receipt.original_cost_micro_usd == parsed.original_receipt_cost_micro_usd
                        and receipt.usage.model_dump(mode="json") == parsed.usage, "Q_AUDIT_GENERATION_RECEIPT")
                resolved = resolve_answer(parse_generated(receipt.raw_output, receipt.stop_reason, recipe),
                    snapshot=view, packed=packed, binding=binding, recipe=recipe)
                changes = {"retrieval_backend": "collection_numpy_global_bm25", "answer_language": live["answer_language"]}
                if live["answer_language"] == "vi" and resolved.answer_status == "ABSTAINED":
                    changes["answer"] = "Tài liệu được tìm thấy chưa cung cấp đủ bằng chứng để trả lời câu hỏi này."
                require(resolved.model_copy(update=changes) == response, "Q_AUDIT_RESOLVED_RESPONSE")
                citation_count += len(response.citations)
                require(parsed.new_cost_micro_usd == receipt.original_cost_micro_usd + (
                    parsed.planning_trace.new_cost_micro_usd if parsed.planning_trace else 0), "Q_AUDIT_METHOD_COST_SPLIT")
                per_case.append(expectation(context, body(system, prompt, 800), {
                    "operation_id": receipt.operation_id, "cost_micro_usd": receipt.original_cost_micro_usd,
                    "usage": receipt.usage.model_dump(mode="json")}, case=number, method=mode, output=800))
            rewrite = live.get("rewrite_receipt")
            if rewrite is not None:
                context = CallContext(tenant_id="demo", run_id=RUN, request_id=identity, stage="REWRITE")
                intent = {"context": context.model_dump(mode="json"),
                    "question_sha256": sha256_bytes(live["input"]["original_question"].encode()),
                    "route": route_identity(routes), "recipe": RECIPE, "prompt_version": PROMPT_VERSION}
                binding_sha = sha256_bytes(canonical_json(intent))
                saved = cached_row(rewrites, "rewrites", stable_id("rewrite", binding_sha256=binding_sha), intent,
                    RewriteReceipt, stable_id("rewrite-context", **context.model_dump(mode="json")))
                require(saved.binding_sha256 == binding_sha and saved.provider_receipt.model_dump(mode="json") == rewrite
                        and not live["rewrite_cache_hit"], "Q_AUDIT_REWRITE_CACHE_BINDING")
                captured = []

                class SavedRewrite:
                    def generate(self, call_context, **kwargs):
                        require(call_context == context, "Q_AUDIT_REWRITE_CONTEXT")
                        captured.append(body(kwargs["system"], kwargs["prompt"], kwargs["max_output_tokens"]))
                        return ModelResult(operation_id=rewrite["operation_id"], text=rewrite["raw_output"],
                            stop_reason=rewrite["stop_reason"], usage=saved.provider_receipt.usage,
                            actual_micro_usd=rewrite["cost_micro_usd"])

                query, reproduced = rewrite_for_search(live["input"]["original_question"], gateway=SavedRewrite(),
                    tenant_id="demo", request_id=identity, run_id=RUN)
                require(query == live["search_query"] and reproduced == rewrite and len(captured) == 1
                        and live["shared_cost_micro_usd"] == rewrite["cost_micro_usd"], "Q_AUDIT_SHARED_REWRITE")
                per_case.append(expectation(context, captured[0], rewrite, case=number, method="shared", output=256))
            else:
                require(live["shared_cost_micro_usd"] == 0, "Q_AUDIT_UNEXPECTED_SHARED_COST")
            require(sum(row["cost_micro_usd"] for row in per_case) == live["new_known_cost_micro_usd"]
                    and len(per_case) == live["model_dispatch_attempts_this_invocation"], "Q_AUDIT_CASE_COST_OR_CALLS")
            pending.extend(per_case)
            cases.append({"case": number, "live_sha256": live_hash, "replay_sha256": replay_hash,
                "base_h_branch_matches": True, "actual_retrieval_from_cached_vectors_matches": True,
                "planner_subqueries": list(live["methods"][1]["planning_trace"]["subqueries"]),
                "final_candidates": len(live["methods"][1]["ranked_chunk_ids"]), "paid_replay_calls": 0,
                "aws_replay_calls": 0, "onnx_replay_calls": 0})
    total = sum(row["cost_micro_usd"] for row in pending)
    stages = Counter(row["context"]["stage"] for row in pending)
    require(stages == {"GENERATE": 6, "PLAN_QUERY": 3, "REWRITE": 1}
            and len(pending) == len({row["operation_id"] for row in pending}) == summary["model_dispatch_attempts"] == 10
            and total == summary["new_known_cost_micro_usd"], "Q_AUDIT_TOTAL_COST_OR_CALLS")
    require(old_r_hashes() == old_r, "Q_AUDIT_OLD_R_CHANGED_DURING_LOCAL_CHECKS")
    report.update(local_checks="PASS", cases=cases, source_appearances=source_count, citation_appearances=citation_count,
        plan_sha256=plan_hash, summary_sha256=summary_hash, execution_preflight_sha256=preflight_hash,
        previous_r_artifact_sha256=old_r, new_upper_bound_micro_usd=total, provider_invoice_micro_usd=None,
        price_sheet_version=prices.version, expected_stage_counts=dict(stages))
    return pending, prices, routes, preflight["counters"], marker, summary, old_r


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true", help="Check local evidence without AWS or writing an audit receipt")
    args = parser.parse_args(argv)
    target = DIRECTORY / "reconciliation.json"
    require(not target.exists(), "Q_AUDIT_RECEIPT_IMMUTABLE")
    report = {"status": "CHECKING", "checked_at": datetime.now(UTC).isoformat(), "model_calls_this_command": 0,
        "onnx_calls_this_command": 0, "encoder_calls_this_command": 0, "aws_resource_mutations_this_command": 0,
        "local_caches_opened_read_only": True, "aws_operations": [], "human_reviewed": False, "quality_metrics": None}

    def forbidden(*args, **kwargs):
        raise ValueError("Q_AUDIT_MODEL_OR_ENCODER_CALL_FORBIDDEN")

    OfficialHttpsTransport.request = forbidden
    ExternalModelGateway.generate = forbidden
    BgeEncoder.__init__ = forbidden
    OnnxCrossEncoder.__init__ = forbidden
    try:
        pending, prices, routes, baseline, marker, summary, old_r = audit_local(report)
        if args.local_only:
            print(json.dumps({"status": "LOCAL_CHECKS_PASSED_AWS_PENDING", "operations": len(pending),
                              "upper_bound_micro_usd": report["new_upper_bound_micro_usd"]}))
            return 0
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")

        def reads_only(model, **kwargs):
            service, operation = model.service_model.service_name, model.name
            require((service in {"sts", "dynamodb"} and operation in {"GetCallerIdentity", "GetItem"})
                    or (service == "signin" and operation == "CreateOAuth2Token"), "Q_AUDIT_AWS_MUTATION_FORBIDDEN")
            report["aws_operations"].append({"service": service, "operation": operation})

        session.events.register("before-call.*.*", reads_only)
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=30)
        require(session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"),
                "Q_AUDIT_AWS_IDENTITY_MISMATCH")
        client = session.client("dynamodb", config=cfg)
        require(client.meta.endpoint_url == "https://dynamodb.us-east-1.amazonaws.com", "Q_AUDIT_LEDGER_ENDPOINT")
        ledger = Ledger(DynamoLedgerStore(client, "kg-r72-query-state"))
        reconciled = []
        for expected in pending:
            op = ledger.operation(expected["operation_id"])
            rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id,
                                   routes.source_region, now=op.created_at, live=True)
            require(op.status == "SETTLED" and op.call_started and op.dispatch_id is not None
                    and op.spec.project_id == "kg-r72" and op.spec.provider == "DEEPSEEK"
                    and op.spec.billing_source == "DIRECT_API" and op.spec.funding_id == "external-r72-v1"
                    and op.spec.model_id == routes.generation_model_id and op.spec.invocation_id == routes.generation_invocation_id
                    and op.spec.source_region == routes.source_region and op.spec.price_sheet_version == prices.version
                    and all(getattr(op.spec, key) == value for key, value in expected["context"].items())
                    and op.spec.input_sha256 == expected["input_sha256"] and op.usage.model_dump(mode="json") == expected["usage"]
                    and op.actual_micro_usd == expected["cost_micro_usd"] == rate.quote(op.usage)
                    and 0 < op.usage.input_tokens <= routes.generation_context_ceiling
                    and 0 <= op.usage.output_tokens <= expected["output_bound"]
                    and op.spec.max_cost_micro_usd == rate.quote(TokenUsage(
                        input_tokens=routes.generation_context_ceiling, output_tokens=expected["output_bound"]))
                    and datetime.fromisoformat(marker["created_at"]) <= op.created_at <= datetime.fromisoformat(summary["created_at"]),
                    "Q_AUDIT_LEDGER_OPERATION_MISMATCH")
            reconciled.append({key: expected[key] for key in ("case", "method", "operation_id", "input_sha256",
                "cost_micro_usd", "usage")} | {"stage": op.spec.stage, "status": op.status})
        counters = {label: ledger.store.get((row["PK"], row["SK"])) for label, row in baseline.items()}
        for label, after in counters.items():
            require(after is not None and after["settled_micro_usd"] - baseline[label]["settled_micro_usd"]
                    == report["new_upper_bound_micro_usd"] and not after["halted"]
                    and after["version"] - baseline[label]["version"] == 2 * len(pending),
                    "Q_AUDIT_COUNTER_DELTA_OR_HALT")
            require(all(after[key] == baseline[label][key] for key in ("PK", "SK", "project_id", "tenant_id", "period",
                "cap_micro_usd", "reserved_micro_usd", "other_cost_estimate_micro_usd", "gross_stop_micro_usd")),
                "Q_AUDIT_COUNTER_CAP_RESERVATION_OR_SCOPE_CHANGED")
        require(old_r_hashes() == old_r, "Q_AUDIT_OLD_R_ARTIFACTS_CHANGED")
        report.update(status="PASS_SOURCE_CACHE_LEDGER_REPLAY", reconciled_operations=reconciled,
                      reconciled_operation_count=len(reconciled), counters=counters,
                      legacy_caps_preserved=True, legacy_reservations_preserved=True, previous_r_artifacts_unchanged=True)
    except Exception as exc:
        report.update(status="BLOCKED_AUDIT_NO_RETRY", error_category=safe_failure_code(exc))
        if isinstance(exc, ValueError) and str(exc).startswith("Q_AUDIT_"):
            report["failed_check"] = str(exc)
        if args.local_only:
            print(json.dumps({key: report[key] for key in ("status", "error_category", "failed_check") if key in report}))
            return 2
    write_once(target, report)
    print(json.dumps({key: report[key] for key in ("status", "error_category", "failed_check", "source_appearances",
        "citation_appearances", "new_upper_bound_micro_usd", "reconciled_operation_count") if key in report}))
    return 0 if report["status"] == "PASS_SOURCE_CACHE_LEDGER_REPLAY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
