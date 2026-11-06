"""One exposed bilingual intent across eight methods, with receipted native stages.

Default freezes scope only. --execute permits at most 3 preparation +16 answer
calls under existing funding. --replay supplies no gateway and cannot infer.
This is a development answer canary, not a quality benchmark or serving release.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.answer_cache import AnswerCache
from app.collection_answering import prepare_collection_answer
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.embeddings import query_embedding_input
from app.external_gateway import ExternalRoutes
from app.generation import generation_request
from app.model_gateway import CallContext, safe_failure_code
from app.query_preparation_v3 import (
    PLANNER_SYSTEM_V3,
    REWRITE_SYSTEM_V3,
    QueryPlanCacheV3,
    QueryRewriteCacheV3,
)
from app.reviewed_source_graph import load_reviewed_source_graph
from app.strict_collection_methods import (
    prepare_strict_graph_answer,
    prepare_strict_multi_query_answer,
)
from scripts.run_collection_benchmark import ROOT, OfflineRuntime, load_inputs

RUN = "reviewed-source-eight-method-answers-v1"
OUTPUT = ROOT / "results/reviewed-source-answers-v1"
METHODS = ("bm25", "vector", "hybrid", "hybrid_reranked", "graph_1hop", "graph_2hop",
    "hybrid_multi_query", "hybrid_multi_query_balanced")
QUESTIONS = {
    "en": "Which native events can Secrets Manager publish directly to EventBridge without CloudTrail, and how does EventBridge encrypt events with KMS when a rule targets an event bus in another AWS Region?",
    "vi": "Secrets Manager có thể gửi trực tiếp những native events nào tới EventBridge mà không cần CloudTrail? Khi một rule nhắm tới event bus ở AWS Region khác, EventBridge mã hóa events với KMS như thế nào?",
}
CODE = ("scripts/pilot_reviewed_source_answers.py", "app/reviewed_source_graph.py", "app/reviewed_graph_runtime.py",
    "app/strict_collection_methods.py", "app/method_execution.py", "app/query_planning_v2.py", "app/query_preparation_v3.py",
    "app/query_rewrite_v2.py", "app/collection_answering.py", "app/context.py", "app/generation.py", "app/answer_cache.py")


def immutable(path, value):
    raw = canonical_json(value) + b"\n"
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("SOURCE_ANSWER_FROZEN_ARTIFACT_CHANGED")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(raw)


def code_hashes():
    return {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}


def context_for(language, stage):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=stable_id("source-answer-native-prep-v1",
        language=language, question=QUESTIONS[language]), stage=stage)


class ScopedGateway:
    def __init__(self, base):
        self.base, self.provider, self.routes = base, base.provider, base.routes
        self.allowed, self.seen, self.cost, self.halted = {}, set(), 0, False

    def allow(self, context, system, prompt, tokens):
        key = context.model_dump_json()
        value = (system, prompt, tokens)
        if context.run_id != RUN or context.tenant_id != "demo" or key in self.allowed and self.allowed[key] != value:
            raise ValueError("SOURCE_ANSWER_REGISTER_SCOPE")
        self.allowed[key] = value

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or self.cost >= 50000 or len(self.seen) >= 19 or key in self.seen
                or self.allowed.get(key) != (system, prompt, max_output_tokens)):
            raise ValueError("SOURCE_ANSWER_DISPATCH_BOUND")
        self.seen.add(key)
        try:
            response = self.base.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
        except Exception:
            self.halted = True
            raise
        self.cost += response.actual_micro_usd
        return response


def register_preparation(gateway, language, search, stage):
    if gateway is not None:
        gateway.allow(context_for(language, stage), REWRITE_SYSTEM_V3 if stage == "REWRITE" else PLANNER_SYSTEM_V3,
            canonical_json({"untrusted_original_question": QUESTIONS[language], "untrusted_english_search_query": search,
                "original_language": language}).decode(), 256)


def managed_gateway(plan):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("SOURCE_ANSWER_OPERATOR_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    if (funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 500000
            or not month or month["halted"] or month["cap_micro_usd"] != 12000000):
        raise ValueError("SOURCE_ANSWER_EXISTING_CAPS_REQUIRED")
    raw_prices = (ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(raw_prices) != plan["price_sheet_sha256"]:
        raise ValueError("SOURCE_ANSWER_PRICE_PIN")
    prices = ExternalPriceSheet.model_validate_json(raw_prices)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    preflight = deepseek_preflight(transport)
    if not preflight["available"] or preflight["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("SOURCE_ANSWER_PROVIDER_CHANGED")
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
    counter = ledger.store.get(ledger.run_key(RUN))
    if counter["halted"] or counter["reserved_micro_usd"] or counter["cap_micro_usd"] != 500000:
        raise ValueError("SOURCE_ANSWER_RUN_RECONCILIATION_REQUIRED")
    # Wallet metadata is local-only; no secrets are present in this projection.
    (OUTPUT / "preflight.json").write_bytes(canonical_json({"checked_at": datetime.now(UTC).isoformat(),
        "provider": preflight, "funding": funding.model_dump(mode="json"),
        "month": {k: month[k] for k in ("cap_micro_usd", "settled_micro_usd", "reserved_micro_usd")}}) + b"\n")
    base = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "FROZEN_REVIEWED_SOURCE_16_ANSWER_CANARY"}), prices)
    return ScopedGateway(base), ledger


def prepare_cases(collection, lexical, graph, runtime, gateway):
    cache_dir = ROOT / "data/cache/reviewed-source-answers-v1"
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    plan_cache, rewrite_cache = QueryPlanCacheV3(cache_dir), QueryRewriteCacheV3(cache_dir)
    cases, preparations, vectors = [], [], {}

    def vector(text):
        if text not in vectors:
            lookup = runtime.cache.resolve_local("demo", text=query_embedding_input(text), recipe=runtime.query_recipe, encoder=runtime.encoder)
            vectors[text] = np.asarray(lookup.record.vector, dtype=np.float32)
        return vectors[text]

    for language, question in QUESTIONS.items():
        search = question
        if language == "vi":
            register_preparation(gateway, language, search, "REWRITE")
            rewritten = rewrite_cache.resolve(context_for(language, "REWRITE"), question=question, language=language, routes=routes, gateway=gateway)
            search = rewritten.rewrite.search_query
            preparations.append({"language": language, "stage": "REWRITE", "receipt": rewritten.receipt,
                "validated": rewritten.rewrite.model_dump(mode="json")})
        register_preparation(gateway, language, search, "PLAN_QUERY")
        planned = plan_cache.resolve(context_for(language, "PLAN_QUERY"), question=question, search_query=search,
            language=language, routes=routes, gateway=gateway)
        if planned.plan.origin != "NATIVE_V3":
            raise ValueError("SOURCE_ANSWER_NATIVE_PLAN_REQUIRED")
        preparations.append({"language": language, "stage": "PLAN_QUERY", "receipt": planned.receipt,
            "validated": planned.plan.model_dump(mode="json")})
        paired = []
        for method in METHODS:
            params = dict(question=question, search_query=search, query_vector=vector(search), tenant_id="demo",
                owner_id="source-answer-local", run_id=RUN, recipe=runtime.recipe)
            if method.startswith("graph_"):
                strict = prepare_strict_graph_answer(collection, lexical, graph, **params,
                    hops=int(method[6]), allow_experimental=True, answer_language=language)
                if strict.prepared is None:
                    raise ValueError("SOURCE_ANSWER_GRAPH_MECHANISM_FAILED")
                view, prepared, trace = strict.view, strict.prepared, strict.trace
                outcome = strict.outcome.model_dump(mode="json")
            elif method.startswith("hybrid_multi_query"):
                strict = prepare_strict_multi_query_answer(collection, lexical, mode=method, **params,
                    plan=planned.plan, subquery_vectors=tuple(vector(q) for q in planned.plan.subqueries), language=language)
                paired.append(strict)
                view, prepared, trace = strict.view, strict.prepared, strict.trace
                outcome = strict.outcome.model_dump(mode="json")
            else:
                rerank_trace = []
                if method == "bm25":
                    params["query_vector"] = None
                identity = stable_id("source-answer-canary-v1", run=RUN, question=question, language=language,
                    method=method, collection=collection.manifest_sha256, recipe=runtime.recipe.fingerprint)
                view, prepared = prepare_collection_answer(collection, lexical, **params, mode=method, request_id=identity,
                    reranker=runtime.reranker if method == "hybrid_reranked" else None, rerank_trace=rerank_trace)
                trace = {"search_query": search, "retrieved_chunk_ids": prepared.retrieved_chunk_ids,
                    "context_chunk_ids": prepared.packed.chunk_ids,
                    "reranking": [r.model_dump(mode="json", exclude={"duration_ms"}) for r in rerank_trace]}
                outcome = {"requested_method": method, "effective_method": method, "mechanism_status": "READY",
                    "graph_path_used": False, "context_chunk_ids": prepared.packed.chunk_ids}
            system, prompt = generation_request(question, view, prepared.packed, "demo", runtime.recipe, answer_language=language)
            cases.append({"id": language + "-" + method, "language": language, "method": method,
                "view": view, "prepared": prepared, "trace": trace, "outcome": outcome, "system": system, "prompt": prompt})
        if len(paired) != 2 or paired[0].trace["multi_query"] != paired[1].trace["multi_query"]:
            raise ValueError("SOURCE_ANSWER_Q_QB_POOL_MISMATCH")
    if len(cases) != 16 or any(c["prepared"].binding.mode_used != c["method"] for c in cases):
        raise ValueError("SOURCE_ANSWER_COMPLETE_METHOD_MATRIX_REQUIRED")
    return cases, preparations


def run(*, execute=False, replay=False):
    _, _, collection, _, pipeline = load_inputs()
    selection = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    graph = load_reviewed_source_graph(ROOT / selection["release_path"], collection,
        trusted_sha256=selection["release_sha256"], allow_experimental=True)
    scope = {"version": "reviewed-source-bilingual-eight-method-answers-v1", "run_id": RUN,
        "purpose": "ONE_EXPOSED_INTENT_TWO_LANGUAGES_NOT_INDEPENDENT_QUALITY_BENCHMARK", "questions": QUESTIONS,
        "methods": METHODS, "source_manifest_sha256": collection.manifest_sha256, "graph_release_sha256": graph.release_sha256,
        "maximum_preparation_calls": 3, "maximum_answer_calls": 16, "run_cap_micro_usd": 500000,
        "observed_cost_stop_between_calls_micro_usd": 50000, "answer_output_tokens": 800,
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "code_sha256": code_hashes(), "deploy": False, "human_reviewed": 0}
    immutable(OUTPUT / "scope.json", scope)
    if not (execute or replay):
        print(json.dumps({"status": "FROZEN_SCOPE_ONLY_NOT_EXECUTED", "maximum_calls": 19, "new_model_calls": 0}))
        return 0
    gateway, ledger, results = None, None, []
    try:
        if execute:
            gateway, ledger = managed_gateway(scope)
        with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
            runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
            cases, preparations = prepare_cases(collection, lexical, graph, runtime, gateway)
            immutable(OUTPUT / "preparations.json", preparations)
            frozen = {"source_manifest_sha256": collection.manifest_sha256, "recipe": runtime.recipe.model_dump(mode="json"),
                "cases": [{"id": c["id"], "language": c["language"], "method": c["method"], "outcome": c["outcome"],
                    "trace": c["trace"], "context": c["prepared"].context.model_dump(mode="json"),
                    "system_sha256": sha256_bytes(c["system"].encode()), "prompt_sha256": sha256_bytes(c["prompt"].encode()),
                    "proxy_input_tokens_estimate": (len(c["system"]) + len(c["prompt"]) + 3) // 4} for c in cases]}
            frozen["forecast_generation_micro_usd"] = sum((r["proxy_input_tokens_estimate"] * 300000 + 800 * 1200000 + 999999) // 1000000 for r in frozen["cases"])
            frozen["forecast_note"] = "4chars/token input proxy; peak cache-miss rates; not exact tokens/admission/billing; excludes preparation"
            immutable(OUTPUT / "answer-plan.json", frozen)
            print(json.dumps({"status": "CONTEXTS_FROZEN", "cases": len(cases), "forecast_generation_micro_usd": frozen["forecast_generation_micro_usd"]}), flush=True)
            if gateway:
                for c in cases:
                    gateway.allow(c["prepared"].context, c["system"], c["prompt"], 800)
            cache = AnswerCache(ROOT / "data/cache/reviewed-source-answers-v1", namespace="LIVE")
            for c in cases:
                p = c["prepared"]
                lookup = cache.resolve(p.context, question=p.question, snapshot=c["view"], packed=p.packed,
                    binding=p.binding, recipe=p.recipe, gateway=gateway, answer_language=c["language"])
                if lookup.receipt is None:
                    raise ValueError("SOURCE_ANSWER_PROVIDER_RECEIPT_REQUIRED")
                record = {"id": c["id"], "method": c["method"], "language": c["language"], "outcome": c["outcome"],
                    "trace": c["trace"], "response": lookup.response.model_dump(mode="json"),
                    "receipt": lookup.receipt.model_dump(mode="json"), "semantic_review": "PENDING_NOT_GOLD"}
                immutable(OUTPUT / "answers" / (c["id"] + ".json"), record)
                results.append({"id": c["id"], "answer_status": lookup.response.answer_status,
                    "claims": len(lookup.response.claims), "citations": len(lookup.response.citations),
                    "cache_hit": lookup.cache_hit, "cost_micro_usd": lookup.receipt.original_cost_micro_usd})
                print(json.dumps(results[-1]), flush=True)
        if scope["code_sha256"] != code_hashes():
            raise ValueError("SOURCE_ANSWER_CODE_CHANGED_DURING_RUN")
        counter = ledger.store.get(ledger.run_key(RUN)) if ledger else None
        own_cost = sum(r["cost_micro_usd"] for r in results) + sum(p["receipt"]["cost_micro_usd"] for p in preparations)
        if counter and (counter["reserved_micro_usd"] or counter["settled_micro_usd"] != own_cost):
            raise ValueError("SOURCE_ANSWER_COST_RECONCILIATION")
        report = {"status": "EXACT_REPLAY_MATCH" if replay else "COMPLETE_SOURCE_REVIEW_PENDING", "answers": results,
            "provider_calls_this_invocation": len(gateway.seen) if gateway else 0, "new_cost_micro_usd": gateway.cost if gateway else 0,
            "run_counter": counter, "total_receipt_cost_micro_usd": own_cost, "human_reviewed": 0,
            "full_graph_coverage": False, "quality_accuracy_measured": False, "new_local_onnx_calls": runtime.encoder.inference_calls}
        if ledger:
            funding_key = ledger.external_funding_key("kg-r72")
            report["funding_counter"] = ledger.store.get(funding_key)
    except Exception as exc:
        report = {"status": "STOPPED_NO_AUTOMATIC_RETRY", "error": safe_failure_code(exc), "error_type": type(exc).__name__,
            "answers": results, "provider_calls_this_invocation": len(gateway.seen) if gateway else 0,
            "new_cost_micro_usd": gateway.cost if gateway else 0}
    name = "replay.json" if replay else f"attempt-{len(list(OUTPUT.glob('attempt-*.json'))) + 1:03d}.json"
    (OUTPUT / name).write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))
    return 2 if report["status"] == "STOPPED_NO_AUTOMATIC_RETRY" else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    raise SystemExit(run(execute=args.execute, replay=args.replay))
