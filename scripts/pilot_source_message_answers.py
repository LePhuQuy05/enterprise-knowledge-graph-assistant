"""Two exposed authorization intents, EN/VI, eight methods, native staged calls.

Default only freezes the scope. --execute is bounded by the existing 8 USD
funding ledger; --replay supplies no gateway. No repair, uncertain-call retry,
fallback, benchmark-label tuning, deployment or historical receipt mutation.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.answer_cache import AnswerCache
from app.authorization_evidence import load_authorization_facts
from app.authorization_retrieval import combine_evidence_graph
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
from app.source_message_runtime import (
    prepare_source_message_graph,
    repack_source_answer,
    repack_strict_query_answer,
)
from app.strict_collection_methods import prepare_strict_multi_query_answer
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT, load_sources
from scripts.run_collection_benchmark import OfflineRuntime, load_inputs

RUN = "source-message-authorization-answers-v1"
OUTPUT = ROOT / "results/source-message-answers-v1"
CACHE = ROOT / "data/cache/source-message-answers-v1"
METHODS = ("bm25", "vector", "hybrid", "hybrid_reranked", "graph_1hop", "graph_2hop",
    "hybrid_multi_query", "hybrid_multi_query_balanced")
QUESTIONS = {
    "s3-permission-layers": {
        "en": "For an Amazon S3 event that invokes Lambda, distinguish permission in the function resource-based policy from execution role permissions to read an SSE-KMS object. Which layers are needed, and what does successful invocation alone establish?",
        "vi": "Với sự kiện Amazon S3 gọi Lambda, hãy phân biệt quyền trong resource-based policy của function với quyền của execution role để đọc object SSE-KMS. Những lớp quyền nào cần có, và việc gọi function thành công tự nó chứng minh điều gì?",
    },
    "kms-cross-account": {
        "en": "For cross-account use of an AWS KMS key, what must the key policy and the external account IAM policy allow? Does allowing an operation in only one policy suffice, and are all AWS KMS operations covered?",
        "vi": "Khi dùng AWS KMS key khác tài khoản, key policy và IAM policy của tài khoản bên ngoài phải cho phép điều gì? Chỉ một trong hai policy cho phép đã đủ chưa, và có áp dụng cho mọi AWS KMS operation không?",
    },
}
CODE = ("scripts/pilot_source_message_answers.py", "app/source_message_packing.py", "app/source_message_runtime.py",
    "app/authorization_evidence.py", "app/authorization_retrieval.py", "app/reviewed_source_graph.py",
    "app/strict_collection_methods.py", "app/query_preparation_v3.py", "app/query_planning_v2.py", "app/query_rewrite_v2.py",
    "app/collection_answering.py", "app/answer_cache.py", "app/generation.py", "app/context.py", "app/method_execution.py")


def code_hashes():
    return {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}


def preparation_context(intent, language, stage):
    return CallContext(tenant_id="demo", run_id=RUN, request_id=stable_id("source-message-preparation-v1",
        intent=intent, language=language, question=QUESTIONS[intent][language]), stage=stage)


class ScopedGateway:
    def __init__(self, base):
        self.base, self.routes, self.provider = base, base.routes, base.provider
        self.allowed, self.seen, self.cost, self.halted = {}, set(), 0, False

    def allow(self, context, system, prompt, output_tokens):
        key = context.model_dump_json()
        request = (system, prompt, output_tokens)
        if (context.run_id != RUN or context.tenant_id != "demo" or context.attempt != 1
                or key in self.allowed and self.allowed[key] != request):
            raise ValueError("SOURCE_MESSAGE_NATIVE_SCOPE")
        self.allowed[key] = request

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or len(self.seen) >= 38 or self.cost >= 150000 or key in self.seen
                or self.allowed.get(key) != (system, prompt, max_output_tokens)):
            raise ValueError("SOURCE_MESSAGE_NATIVE_DISPATCH_BOUND")
        self.seen.add(key)
        try:
            reply = self.base.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
        except Exception:
            self.halted = True
            raise
        self.cost += reply.actual_micro_usd
        return reply


def managed_gateway(scope):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("SOURCE_MESSAGE_OPERATOR_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    if (funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 300000
            or not month or month["halted"] or month["cap_micro_usd"] != 12000000):
        raise ValueError("SOURCE_MESSAGE_EXISTING_CAPS_REQUIRED")
    prices_raw = (ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(prices_raw) != scope["price_sheet_sha256"]:
        raise ValueError("SOURCE_MESSAGE_PRICE_PIN")
    prices = ExternalPriceSheet.model_validate_json(prices_raw)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    provider = deepseek_preflight(transport)
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("SOURCE_MESSAGE_PROVIDER_CHANGED")
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=300000)
    counter = ledger.store.get(ledger.run_key(RUN))
    if counter["halted"] or counter["reserved_micro_usd"] or counter["cap_micro_usd"] != 300000:
        raise ValueError("SOURCE_MESSAGE_RUN_RECONCILIATION_REQUIRED")
    immutable(OUTPUT / "preflight.json", {"checked_at": datetime.now(UTC).isoformat(), "provider": provider,
        "funding": funding.model_dump(mode="json"), "month": {k: month[k] for k in ("cap_micro_usd", "settled_micro_usd", "reserved_micro_usd")}})
    base = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "FROZEN_SOURCE_MESSAGE_32_ANSWER_CANARY"}), prices)
    return ScopedGateway(base), ledger


def prepare_cases(collection, lexical, graph, runtime, gateway):
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    planner, rewriter = QueryPlanCacheV3(CACHE), QueryRewriteCacheV3(CACHE)
    cases, preparations, vectors = [], [], {}

    def vector(text):
        if text not in vectors:
            lookup = runtime.cache.resolve_local("demo", text=query_embedding_input(text), recipe=runtime.query_recipe, encoder=runtime.encoder)
            vectors[text] = np.asarray(lookup.record.vector, dtype=np.float32)
        return vectors[text]

    def register(context, system, question, search, language):
        if gateway:
            gateway.allow(context, system, canonical_json({"untrusted_original_question": question,
                "untrusted_english_search_query": search, "original_language": language}).decode(), 256)

    for intent, languages in QUESTIONS.items():
        for language, question in languages.items():
            search = question
            if language == "vi":
                context = preparation_context(intent, language, "REWRITE")
                register(context, REWRITE_SYSTEM_V3, question, search, language)
                rewrite = rewriter.resolve(context, question=question, language=language, routes=routes, gateway=gateway)
                search = rewrite.rewrite.search_query
                preparations.append({"intent": intent, "language": language, "stage": "REWRITE", "receipt": rewrite.receipt,
                    "validated": rewrite.rewrite.model_dump(mode="json")})
            context = preparation_context(intent, language, "PLAN_QUERY")
            register(context, PLANNER_SYSTEM_V3, question, search, language)
            plan = planner.resolve(context, question=question, search_query=search, language=language, routes=routes, gateway=gateway)
            if plan.plan.origin != "NATIVE_V3":
                raise ValueError("SOURCE_MESSAGE_NATIVE_PLANNER_REQUIRED")
            preparations.append({"intent": intent, "language": language, "stage": "PLAN_QUERY", "receipt": plan.receipt,
                "validated": plan.plan.model_dump(mode="json")})
            paired = []
            for method in METHODS:
                params = dict(question=question, search_query=search, query_vector=vector(search), tenant_id="demo",
                    owner_id="source-message-native", run_id=RUN, recipe=runtime.recipe)
                if method.startswith("graph_"):
                    strict = prepare_source_message_graph(collection, lexical, graph, **params,
                        hops=int(method[6]), allow_experimental=True, answer_language=language)
                    if strict.prepared is None:
                        raise ValueError("SOURCE_MESSAGE_GRAPH_MECHANISM_FAILED")
                    view, prepared, trace, outcome = strict.view, strict.prepared, strict.trace, strict.outcome.model_dump(mode="json")
                elif method.startswith("hybrid_multi_query"):
                    original = prepare_strict_multi_query_answer(collection, lexical, mode=method, **params,
                        plan=plan.plan, subquery_vectors=tuple(vector(s) for s in plan.plan.subqueries), language=language)
                    strict = repack_strict_query_answer(original)
                    paired.append(strict)
                    view, prepared, trace, outcome = strict.view, strict.prepared, strict.trace, strict.outcome.model_dump(mode="json")
                else:
                    if method == "bm25":
                        params["query_vector"] = None
                    reranking = []
                    view, before = prepare_collection_answer(collection, lexical, **params, mode=method,
                        reranker=runtime.reranker if method == "hybrid_reranked" else None, rerank_trace=reranking)
                    prepared, packing = repack_source_answer(view, before, answer_language=language)
                    trace = {"source_message_packing": packing, "reranking": [r.model_dump(mode="json") for r in reranking]}
                    outcome = {"requested_method": method, "effective_method": method, "mechanism_status": "READY",
                        "graph_path_used": False, "context_chunk_ids": prepared.packed.chunk_ids}
                system, prompt = generation_request(question, view, prepared.packed, "demo", runtime.recipe, answer_language=language)
                if prompt != prepared.packed.serialized or prepared.packed.proxy_tokens > 4096:
                    raise ValueError("SOURCE_MESSAGE_EXACT_PAYLOAD_BUDGET_REQUIRED")
                cases.append({"id": intent + "-" + language + "-" + method, "intent": intent, "language": language,
                    "method": method, "view": view, "prepared": prepared, "trace": trace, "outcome": outcome, "system": system, "prompt": prompt})
            if len(paired) != 2 or paired[0].trace["multi_query"] != paired[1].trace["multi_query"]:
                raise ValueError("SOURCE_MESSAGE_SHARED_Q_QB_POOL_REQUIRED")
            print(json.dumps({"prepared": intent + "-" + language, "cases": len(cases)}), flush=True)
    if len(cases) != 32:
        raise ValueError("SOURCE_MESSAGE_COMPLETE_MATRIX_REQUIRED")
    return cases, preparations


def run(*, execute=False, replay=False):
    collection, chunks, documents = load_sources()
    _, _, _, _, pipeline = load_inputs()
    selection = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    reviewed = load_reviewed_source_graph(ROOT / selection["release_path"], collection,
        trusted_sha256=selection["release_sha256"], allow_experimental=True)
    auth_selection = json.loads((ROOT / "configs/graph/authorization-evidence-local-v1.json").read_bytes())
    auth = load_authorization_facts(ROOT / auth_selection["release_path"], collection, chunks, documents,
        trusted_sha256=auth_selection["release_sha256"], allow_experimental=True)
    graph = combine_evidence_graph(collection, reviewed, auth)
    scope = {"version": "source-message-native-authorization-canary-v1", "run_id": RUN,
        "purpose": "TWO_EXPOSED_AUTHORIZATION_INTENTS_NOT_INDEPENDENT_QUALITY_BENCHMARK",
        "questions": QUESTIONS, "methods": METHODS, "collection_manifest_sha256": collection.manifest_sha256,
        "graph_release_sha256": graph.release_sha256, "maximum_preparation_calls": 6, "maximum_answer_calls": 32,
        "run_cap_micro_usd": 300000, "observed_cost_stop_between_calls_micro_usd": 150000,
        "user_message_proxy_token_cap": 4096, "max_chunks": 8, "output_token_cap": 800,
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "code_sha256": code_hashes(), "deploy": False, "human_reviewed": 0}
    immutable(OUTPUT / "scope.json", scope)
    if not (execute or replay):
        print(json.dumps({"status": "FROZEN_SCOPE_ONLY", "maximum_calls": 38, "new_inference_calls": 0}))
        return 0
    gateway, ledger, answers = None, None, []
    try:
        if execute:
            gateway, ledger = managed_gateway(scope)
        with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
            runtime = OfflineRuntime(collection, lexical, ("hybrid_reranked",))
            cases, preparations = prepare_cases(collection, lexical, graph, runtime, gateway)
            immutable(OUTPUT / "preparations.json", preparations)
            frozen = {"recipe": runtime.recipe.model_dump(mode="json"), "cases": [{"id": c["id"], "intent": c["intent"],
                "language": c["language"], "method": c["method"], "outcome": c["outcome"], "trace": c["trace"],
                "call_context": c["prepared"].context.model_dump(mode="json"), "system_sha256": sha256_bytes(c["system"].encode()),
                "prompt_sha256": sha256_bytes(c["prompt"].encode())} for c in cases]}
            immutable(OUTPUT / "answer-plan.json", frozen)
            if gateway:
                for c in cases:
                    gateway.allow(c["prepared"].context, c["system"], c["prompt"], 800)
            cache = AnswerCache(CACHE, namespace="LIVE")
            for c in cases:
                prepared = c["prepared"]
                lookup = cache.resolve(prepared.context, question=prepared.question, snapshot=c["view"], packed=prepared.packed,
                    binding=prepared.binding, recipe=prepared.recipe, gateway=gateway, answer_language=c["language"])
                if lookup.receipt is None:
                    raise ValueError("SOURCE_MESSAGE_NATIVE_ANSWER_RECEIPT_REQUIRED")
                immutable(OUTPUT / "answers" / (c["id"] + ".json"), {"id": c["id"], "intent": c["intent"],
                    "language": c["language"], "method": c["method"], "outcome": c["outcome"], "trace": c["trace"],
                    "response": lookup.response.model_dump(mode="json"), "receipt": lookup.receipt.model_dump(mode="json"),
                    "semantic_review": "PENDING_NOT_GOLD"})
                row = {"id": c["id"], "answer_status": lookup.response.answer_status, "claims": len(lookup.response.claims),
                    "citations": len(lookup.response.citations), "cache_hit": lookup.cache_hit,
                    "receipt_cost_micro_usd": lookup.receipt.original_cost_micro_usd}
                answers.append(row)
                print(json.dumps(row), flush=True)
        if scope["code_sha256"] != code_hashes():
            raise ValueError("SOURCE_MESSAGE_CODE_CHANGED_DURING_RUN")
        own_cost = sum(a["receipt_cost_micro_usd"] for a in answers) + sum(p["receipt"]["cost_micro_usd"] for p in preparations)
        counter = ledger.store.get(ledger.run_key(RUN)) if ledger else None
        if counter and (counter["reserved_micro_usd"] or counter["settled_micro_usd"] != own_cost):
            raise ValueError("SOURCE_MESSAGE_LEDGER_RECONCILIATION")
        report = {"status": "EXACT_REPLAY_MATCH" if replay else "NATIVE_ANSWERS_COMPLETE_SOURCE_REVIEW_PENDING",
            "answers": answers, "provider_calls_this_invocation": len(gateway.seen) if gateway else 0,
            "new_cost_micro_usd": gateway.cost if gateway else 0, "total_receipt_cost_micro_usd": own_cost,
            "run_counter": counter, "human_reviewed": 0, "quality_accuracy_measured": False,
            "serving_published": False, "native_extraction_of_new_graph_claimed": False}
        immutable(OUTPUT / "replay.json", report) if replay else immutable(OUTPUT / "attempt-001.json", report)
        print(json.dumps({k: v for k, v in report.items() if k != "answers"}))
        return 0
    except Exception as exc:
        report = {"status": "HALTED_NO_AUTOMATIC_REPAIR_OR_RETRY", "failure_class": type(exc).__name__,
            "safe_failure_code": safe_failure_code(exc), "completed_answers": answers,
            "provider_dispatches": len(gateway.seen) if gateway else 0, "new_cost_micro_usd": gateway.cost if gateway else 0}
        immutable(OUTPUT / "halt-001.json", report)
        print(json.dumps(report))
        return 2


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    if args.execute and args.replay:
        parser.error("Choose execute or replay")
    raise SystemExit(run(execute=args.execute, replay=args.replay))
