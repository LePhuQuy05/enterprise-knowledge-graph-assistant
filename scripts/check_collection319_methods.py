"""Fresh bounded V3 preparation and eight-method local mechanism receipts.

Default freezes scope only; execute allows SIX preparation calls, no answers or
extraction. Replay is socket-blocked and requires exact recorded contexts.
"""

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.collection_answering import prepare_collection_answer
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.embeddings import query_embedding_input
from app.external_gateway import DEEPSEEK, ExternalRoutes
from app.generation import generation_request
from app.graph_collection_migration import collection_records
from app.graph_document_allocation import reallocate_graph_documents
from app.model_gateway import CallContext, safe_failure_code
from app.query_preparation_v3 import (
    PLANNER_SYSTEM_V3,
    REWRITE_SYSTEM_V3,
    QueryPlanCacheV3,
    QueryRewriteCacheV3,
)
from app.source_dependency_audit import SourceDependency, audit_source_dependencies
from app.source_message_runtime import (
    prepare_source_message_graph,
    repack_source_answer,
    repack_strict_query_answer,
)
from app.strict_collection_methods import prepare_strict_multi_query_answer
from scripts.check_reviewed_source_runtime import CONTROLS
from scripts.migrate_collection319_graph import OUTPUT as MIGRATION
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT
from scripts.run_collection_benchmark import OfflineRuntime

OUTPUT = ROOT / "results/collection319-methods-v1"
CACHE = ROOT / "data/cache/collection319-preparation-v1"
RUN = "collection319-native-preparation-v1"
STOP_MICRO_USD = 20000
QUESTIONS = {
    "s3-permission-layers": {
        "en": "How do S3 invocation permissions, the Lambda execution role, s3:GetObject and kms:Decrypt differ for reading an SSE-KMS object across accounts?",
        "vi": "Quyền S3 gọi Lambda, execution role của Lambda, s3:GetObject và kms:Decrypt khác nhau thế nào khi đọc object SSE-KMS giữa các tài khoản?"},
    "kms-cross-account": {
        "en": "Which policies must permit cross-account use of an AWS KMS key, and which operations support this permission model?",
        "vi": "Những policy nào phải cho phép sử dụng AWS KMS key giữa các tài khoản, và những thao tác nào hỗ trợ mô hình quyền này?"}}
METHODS = ("bm25", "vector", "hybrid", "hybrid_reranked", "graph_1hop", "graph_2hop",
    "hybrid_multi_query", "hybrid_multi_query_balanced")
CODE = ("scripts/check_collection319_methods.py", "scripts/migrate_collection319_graph.py",
    "app/graph_collection_migration.py", "app/reviewed_source_graph.py", "app/authorization_evidence.py",
    "app/authorization_retrieval.py", "app/strict_collection_methods.py", "app/method_execution.py",
    "app/query_preparation_v3.py", "app/query_planning_v2.py", "app/query_rewrite_v2.py",
    "app/collection_answering.py", "app/collection_lexical.py", "app/collections.py",
    "app/source_message_runtime.py", "app/source_message_packing.py", "app/graph_source_allocation.py",
    "app/graph_document_allocation.py", "app/source_dependency_audit.py", "app/generation.py",
    "app/context.py", "app/multi_query.py", "app/context_selection.py", "app/external_gateway.py")


def context_for(intent, language, stage, manifest_sha256):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=stable_id("collection319-native-preparation-v1",
        collection=manifest_sha256, intent=intent, language=language, question=QUESTIONS[intent][language]), stage=stage)


def preparation_payload(question, search, language):
    return canonical_json({"untrusted_original_question": question,
        "untrusted_english_search_query": search, "original_language": language}).decode()


def provider_body(system, prompt):
    return canonical_json({"model": DEEPSEEK, "messages": [{"role": "system", "content": system},
        {"role": "user", "content": prompt}], "thinking": {"type": "disabled"}, "temperature": 0,
        "max_tokens": 256, "stream": False, "response_format": {"type": "json_object"}})


class ScopedGateway:
    def __init__(self, base, manifest_sha256):
        self.base, self.provider, self.routes = base, base.provider, base.routes
        self.manifest_sha256, self.allowed, self.seen = manifest_sha256, {}, set()
        self.cost, self.halted = 0, False

    def allow(self, intent, language, stage, search):
        context = context_for(intent, language, stage, self.manifest_sha256)
        if stage not in {"REWRITE", "PLAN_QUERY"} or stage == "REWRITE" and language != "vi":
            raise ValueError("COLLECTION319_PREPARATION_STAGE")
        if (stage == "REWRITE" or language == "en") and search != QUESTIONS[intent][language]:
            raise ValueError("COLLECTION319_PREPARATION_ORIGINAL_QUERY")
        system = REWRITE_SYSTEM_V3 if stage == "REWRITE" else PLANNER_SYSTEM_V3
        request = (system, preparation_payload(QUESTIONS[intent][language], search, language), 256)
        key = context.model_dump_json()
        if key in self.allowed and self.allowed[key] != request:
            raise ValueError("COLLECTION319_PREPARATION_REBIND")
        self.allowed[key] = request
        return context

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or len(self.seen) >= 6 or self.cost >= STOP_MICRO_USD or key in self.seen
                or self.allowed.get(key) != (system, prompt, max_output_tokens)):
            raise ValueError("COLLECTION319_PREPARATION_DISPATCH_BOUND")
        self.seen.add(key)
        try:
            result = self.base.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
        except Exception:
            self.halted = True
            raise
        self.cost += result.actual_micro_usd
        return result


def freeze_scope(collection, graph):
    scope = {"version": RUN, "purpose": "EXPOSED_BILINGUAL_MECHANISM_CHECKS_NOT_INDEPENDENT_BENCHMARK",
        "collection_manifest_sha256": collection.manifest_sha256, "combined_graph_sha256": graph.release_sha256,
        "migration_plan_sha256": sha256_bytes((MIGRATION / "plan.json").read_bytes()),
        "migration_selection_sha256": sha256_bytes((MIGRATION / "selection.json").read_bytes()),
        "source_dependencies_sha256": sha256_bytes((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes()),
        "questions": QUESTIONS, "methods": METHODS, "extra_graph_controls": CONTROLS,
        "native_preparation_calls_max": 6, "generation_calls": 0, "extraction_calls": 0,
        "observed_stop_micro_usd": STOP_MICRO_USD, "run_cap_micro_usd": 500000,
        "max_chunks": 8, "user_message_proxy_tokens": 4096, "human_reviewed": 0, "serving_published": False,
        "routes_sha256": sha256_bytes((ROOT / "configs/models/external-direct-v1.json").read_bytes()),
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}}
    immutable(OUTPUT / "scope.json", scope)
    return scope


def managed_gateway(scope):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    config = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=config).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("COLLECTION319_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=config), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    if (funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 500000
            or not month or month["cap_micro_usd"] != 12000000 or month["halted"]):
        raise ValueError("COLLECTION319_FUNDING_OR_MONTH_CAP")
    raw = (ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(raw) != scope["price_sheet_sha256"]:
        raise ValueError("COLLECTION319_PRICE_PIN")
    prices = ExternalPriceSheet.model_validate_json(raw)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region,
        now=datetime.now(UTC), live=True)
    reserve = rate.quote(TokenUsage(input_tokens=routes.generation_context_ceiling, output_tokens=256))
    if reserve + STOP_MICRO_USD > 500000:
        raise ValueError("COLLECTION319_ADMISSION_RESERVATION")
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    provider = deepseek_preflight(transport)
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("COLLECTION319_PROVIDER_CHANGED")
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
    counter = ledger.store.get(ledger.run_key(RUN))
    if counter["halted"] or counter["reserved_micro_usd"] or counter["settled_micro_usd"]:
        raise ValueError("COLLECTION319_RUN_NOT_UNATTEMPTED_RECONCILE_FIRST")
    immutable(OUTPUT / "preflight.json", {"checked_at": datetime.now(UTC).isoformat(), "provider": provider,
        "funding": funding.model_dump(mode="json"), "worst_single_reservation_micro_usd": reserve,
        "observed_stop_micro_usd": STOP_MICRO_USD, "run_cap_micro_usd": 500000})
    base = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "COLLECTION319_SIX_PINNED_PREPARATION_CALLS_NO_ANSWERS"}), prices)
    return ScopedGateway(base, scope["collection_manifest_sha256"]), ledger


def prepare_queries(collection, routes, gateway):
    planner, rewriter = QueryPlanCacheV3(CACHE), QueryRewriteCacheV3(CACHE)
    records, tasks = [], []
    for intent, languages in QUESTIONS.items():
        for language, question in languages.items():
            search = question
            for stage in (("REWRITE", "PLAN_QUERY") if language == "vi" else ("PLAN_QUERY",)):
                context = context_for(intent, language, stage, collection.manifest_sha256)
                if gateway:
                    gateway.allow(intent, language, stage, search)
                system = REWRITE_SYSTEM_V3 if stage == "REWRITE" else PLANNER_SYSTEM_V3
                prompt = preparation_payload(question, search, language)
                lookup = (rewriter.resolve(context, question=question, language=language, routes=routes, gateway=gateway)
                    if stage == "REWRITE" else planner.resolve(context, question=question, search_query=search,
                        language=language, routes=routes, gateway=gateway))
                value = lookup.rewrite if stage == "REWRITE" else lookup.plan
                if bool(gateway) == lookup.cache_hit:
                    raise ValueError("COLLECTION319_NATIVE_REPLAY_IDENTITY")
                record = {"intent": intent, "language": language, "stage": stage,
                    "context": context.model_dump(mode="json"), "receipt": lookup.receipt,
                    "validated": value.model_dump(mode="json"), "system_sha256": sha256_bytes(system.encode()),
                    "prompt_sha256": sha256_bytes(prompt.encode()), "payload_sha256": sha256_bytes(provider_body(system, prompt)),
                    "semantic_review": "PENDING_NOT_INDEPENDENT_GOLD"}
                immutable(OUTPUT / "preparations" / f"{intent}-{language}-{stage}.json", record)
                records.append(record)
                print(json.dumps({"stage": stage, "intent": intent, "language": language, "cache_hit": lookup.cache_hit}), flush=True)
                if stage == "REWRITE":
                    search = value.search_query
                else:
                    if value.origin != "NATIVE_V3":
                        raise ValueError("COLLECTION319_NATIVE_PLANNER_REQUIRED")
                    tasks.append((intent, language, question, search, value))
    return records, tasks


def mechanism_cells(collection, runtime_paths, graph, tasks):
    docs, chunks, _ = collection_records(collection)
    controls = json.loads((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())["source_controls"]
    dependencies = tuple(SourceDependency.model_validate_json(canonical_json(row["dependency"])) for row in controls)
    rows = []
    with load_global_lexical_index(Path(runtime_paths["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
        vectors = {}

        def vector(text):
            if text not in vectors:
                lookup = runtime.cache.resolve_local("demo", text=query_embedding_input(text), recipe=runtime.query_recipe, encoder=runtime.encoder)
                vectors[text] = np.asarray(lookup.record.vector, dtype=np.float32)
            return vectors[text]

        def save(name, language, method, view, prepared, trace, outcome):
            system, prompt = generation_request(prepared.question, view, prepared.packed, "demo", prepared.recipe, answer_language=language)
            if (prepared.binding.mode_used != method or len(prepared.packed.chunk_ids) > 8 or prepared.packed.proxy_tokens > 4096
                    or outcome.get("effective_method") != method):
                raise ValueError("COLLECTION319_EFFECTIVE_METHOD_OR_BUDGET")
            if any(view.chunks[k] != chunks[k] for k in prepared.packed.chunk_ids):
                raise ValueError("COLLECTION319_CONTEXT_SOURCE")
            rows.append({"case": name, "language": language, "method": method,
                "binding": prepared.binding.model_dump(mode="json"), "outcome": outcome, "trace": trace,
                "context_chunk_ids": prepared.packed.chunk_ids, "context_assertion_ids": prepared.packed.assertion_ids,
                "source_message": prompt, "system_sha256": sha256_bytes(system.encode()), "prompt_sha256": sha256_bytes(prompt.encode()),
                "user_message_proxy_tokens": prepared.packed.proxy_tokens,
                "dependency_audit": audit_source_dependencies(docs, chunks, prepared.packed.chunk_ids, dependencies, tenant_id="demo"),
                "generation_executed": False, "hidden_fallbacks": 0, "human_reviewed": 0})

        def prepare_graph(question, search, language, hops):
            result = prepare_source_message_graph(collection, lexical, graph, question, vector(search), tenant_id="demo",
                owner_id="collection319-local", run_id=RUN, recipe=runtime.recipe, hops=hops, search_query=search,
                answer_language=language, allow_experimental=True)
            if result.prepared is None:
                raise ValueError("COLLECTION319_GRAPH_MECHANISM_FAILED_NO_FALLBACK")
            return reallocate_graph_documents(collection, lexical, graph, result, priority_source_cap=4)

        for intent, language, question, search, plan in tasks:
            paired = []
            for method in METHODS:
                if method.startswith("graph_"):
                    result = prepare_graph(question, search, language, int(method[6]))
                    save(intent, language, method, result.view, result.prepared, result.trace, result.outcome.model_dump(mode="json"))
                elif method.startswith("hybrid_multi_query"):
                    result = prepare_strict_multi_query_answer(collection, lexical, mode=method, question=question, search_query=search,
                        query_vector=vector(search), subquery_vectors=tuple(vector(q) for q in plan.subqueries), plan=plan,
                        tenant_id="demo", owner_id="collection319-local", run_id=RUN, recipe=runtime.recipe, language=language)
                    paired.append(result)
                    result = repack_strict_query_answer(result)
                    save(intent, language, method, result.view, result.prepared, result.trace, result.outcome.model_dump(mode="json"))
                else:
                    reranks = []
                    view, prepared = prepare_collection_answer(collection, lexical, question,
                        None if method == "bm25" else vector(search), tenant_id="demo", owner_id="collection319-local", run_id=RUN,
                        recipe=runtime.recipe, mode=method, search_query=search,
                        reranker=runtime.reranker if method == "hybrid_reranked" else None, rerank_trace=reranks)
                    prepared, packing = repack_source_answer(view, prepared, answer_language=language)
                    save(intent, language, method, view, prepared, {"source_message_packing": packing,
                        "reranking": [r.model_dump(mode="json", exclude={"duration_ms"}) for r in reranks]},
                        {"requested_method": method, "effective_method": method, "mechanism_status": "READY"})
            if (paired[0].trace["multi_query"] != paired[1].trace["multi_query"]
                    or not paired[1].outcome.context_selection_used):
                raise ValueError("COLLECTION319_Q_QB_POOL_OR_ALLOCATOR")
            print(json.dumps({"case": intent, "language": language, "paired_cells": len(rows)}), flush=True)
        for name, question in CONTROLS:
            for hops in (1, 2):
                result = prepare_graph(question, question, "en", hops)
                save("control-" + name, "en", f"graph_{hops}hop", result.view, result.prepared,
                    result.trace, result.outcome.model_dump(mode="json"))
    return rows


def run(*, execute=False, replay=False):
    collection, runtime_paths, graph = load_migrated()
    scope = freeze_scope(collection, graph)
    if not execute and not replay:
        print(json.dumps({"status": "SCOPE_FROZEN_NO_CALLS", "max_preparation_calls": 6, "generation_calls": 0}))
        return
    gateway, ledger = managed_gateway(scope) if execute else (None, None)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    try:
        records, tasks = prepare_queries(collection, routes, gateway)
        with ExitStack() as stack:
            for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
                stack.enter_context(patch(name, side_effect=AssertionError("COLLECTION319_RETRIEVAL_NETWORK_BLOCKED")))
            rows = mechanism_cells(collection, runtime_paths, graph, tasks)
        immutable(OUTPUT / "receipts.json", {"preparations": records, "cells": rows})
        cost = sum(r["receipt"]["cost_micro_usd"] for r in records)
        report = {"status": "LOCAL319_NATIVE_PREPARATION_AND_EIGHT_METHOD_MECHANISMS_VALIDATED", "documents": 319,
            "chunks": 3024, "paired_intents": 2, "paired_input_languages": ["en", "vi"], "paired_cells": 32,
            "extra_graph_control_cells": len(rows) - 32, "methods": METHODS, "native_preparation_receipts": len(records),
            "total_preparation_receipt_cost_micro_usd": cost, "receipts_sha256": sha256_bytes((OUTPUT / "receipts.json").read_bytes()),
            "graph_statuses": dict(Counter(r["outcome"]["mechanism_status"] for r in rows if r["method"].startswith("graph_"))),
            "dependency_incomplete_cells": sum(r["dependency_audit"]["incomplete_count"] > 0 for r in rows),
            "new_preparation_calls_this_invocation": len(gateway.seen) if gateway else 0,
            "generation_calls": 0, "extraction_calls": 0, "hidden_fallbacks": 0, "human_reviewed": 0,
            "answer_accuracy_measured": False, "serving_published": False}
        if ledger:
            for row in records:
                receipt = row["receipt"]
                op = ledger.operation(receipt["operation_id"])
                if (op.status != "SETTLED" or not op.call_started or op.bound_exceeded or op.spec.run_id != RUN
                        or op.spec.stage != row["stage"] or op.spec.input_sha256 != row["payload_sha256"]
                        or op.usage.model_dump(mode="json") != receipt["usage"] or op.actual_micro_usd != receipt["cost_micro_usd"]):
                    raise ValueError("COLLECTION319_LEDGER_PAYLOAD_USAGE_COST")
            counter = ledger.store.get(ledger.run_key(RUN))
            funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
            if counter["reserved_micro_usd"] or counter["settled_micro_usd"] != cost:
                raise ValueError("COLLECTION319_LEDGER_RECONCILIATION")
            immutable(OUTPUT / "ledger-audit.json", {"status": "SIX_SETTLED_PAYLOAD_USAGE_COST_MATCH",
                "checked_at": datetime.now(UTC).isoformat(), "run_counter": counter, "funding_counter": funding,
                "headroom_micro_usd": funding["cap_micro_usd"] - funding["settled_micro_usd"] - funding["reserved_micro_usd"]})
        if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
            raise ValueError("COLLECTION319_CODE_CHANGED_DURING_RUN")
        immutable(OUTPUT / ("replay.json" if replay else "report.json"), report)
        print(json.dumps(report))
    except Exception as exc:
        immutable(OUTPUT / "stopped.json", {"status": "STOPPED_NO_AUTOMATIC_RETRY_OR_FALLBACK", "error": safe_failure_code(exc),
            "provider_dispatch_attempts": len(gateway.seen) if gateway else 0,
            "observed_receipted_cost_micro_usd": gateway.cost if gateway else 0,
            "cache_raw_failures_retained": True, "serving_published": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--execute", action="store_true")
    choice.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        if args.replay:
            for name in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
                stack.enter_context(patch(name, side_effect=AssertionError("COLLECTION319_REPLAY_NETWORK_BLOCKED")))
        run(execute=args.execute, replay=args.replay)
