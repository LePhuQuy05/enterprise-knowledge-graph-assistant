"""Three source-inspection demos, with shared native V3 preparation for Q/QB.

Default freezes prompts without AWS/model calls. --execute dispatches up to
three answer calls; --replay forbids inference and validates saved responses.
Different questions across Graph and Q/QB are NOT a quality comparison.
"""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.answer_cache import AnswerCache
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.external_gateway import ExternalRoutes
from app.generation import generation_request
from app.method_execution import answer_strict_prepared
from app.query_preparation_v3 import QueryPlanCacheV3, QueryRewriteCacheV3
from app.strict_collection_methods import (
    prepare_strict_graph_answer,
    prepare_strict_multi_query_answer,
)
from scripts.prepare_strict_method_canary import build_plan, context_for
from scripts.run_collection_benchmark import ROOT, OfflineRuntime, load_inputs

RUN = "strict-method-answers-v1"
OUTPUT = ROOT / "results/method-correctness-v2/native-answers-v1"


def prepare():
    _, _, collection, _, pipeline = load_inputs()
    lexical = load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection)
    try:
        runtime = OfflineRuntime(collection, lexical, ("graph_1hop",))
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        tasks, _ = build_plan(3)
        task = tasks[0]
        cache_dir = ROOT / "data/cache/strict-method-preparation-v3"
        rewrite = QueryRewriteCacheV3(cache_dir).resolve(context_for(task, "REWRITE", 3),
            question=task.question, language="vi", routes=routes)
        search = rewrite.rewrite.search_query
        plan = QueryPlanCacheV3(cache_dir).resolve(context_for(task, "PLAN_QUERY", 3), question=task.question,
            search_query=search, language="vi", routes=routes).plan
        if not rewrite.cache_hit or plan.origin != "NATIVE_V3":
            raise ValueError("STRICT_ANSWERS_NATIVE_PREPARATION_REQUIRED")

        def vector(text):
            lookup = runtime.cache.resolve_local("demo", text=query_embedding_input(text), recipe=runtime.query_recipe, encoder=runtime.encoder)
            return np.asarray(lookup.record.vector, dtype=np.float32)

        question = "What protocols does a NAT gateway support, and what is NAT64 used for?"
        cases = [("nat-en-g1", prepare_strict_graph_answer(collection, lexical, runtime.graph, question, vector(question),
            tenant_id="demo", owner_id="strict-local", run_id=RUN, recipe=runtime.recipe, hops=1,
            allow_experimental=True, answer_language="en"))]
        args = dict(question=task.question, search_query=search, query_vector=vector(search),
            subquery_vectors=tuple(vector(q) for q in plan.subqueries), plan=plan, tenant_id="demo", owner_id="strict-local",
            run_id=RUN, recipe=runtime.recipe, language="vi")
        cases.extend((name, prepare_strict_multi_query_answer(collection, lexical, mode=mode, **args)) for name, mode in (
            ("step-functions-vi-q", "hybrid_multi_query"), ("step-functions-vi-qb", "hybrid_multi_query_balanced")))
        if any(c.outcome.answer_status != "READY" for _, c in cases):
            raise ValueError("STRICT_ANSWERS_METHOD_NOT_READY")
        if cases[1][1].trace["multi_query"] != cases[2][1].trace["multi_query"]:
            raise ValueError("STRICT_ANSWERS_Q_QB_SHARED_POOL")
        prompts = {name: generation_request(c.prepared.question, c.view, c.prepared.packed, "demo", c.prepared.recipe,
            answer_language=c.answer_language) for name, c in cases}
        frozen = {"run_id": RUN, "purpose": "DEVELOPMENT_SOURCE_INSPECTION_NOT_QUALITY_BENCHMARK",
            "collection_manifest_sha256": collection.manifest_sha256, "graph_file_sha256": runtime.graph.file_sha256,
            "maximum_new_calls": 3, "run_cap_micro_usd": 500000, "actual_cost_stop_between_calls_micro_usd": 10000,
            "max_output_tokens": 800, "generation_recipe": runtime.recipe.model_dump(mode="json"),
            "cases": [{"id": name, "question": c.prepared.question, "answer_language": c.answer_language,
                "outcome": c.outcome.model_dump(mode="json"), "context": c.prepared.context.model_dump(mode="json"),
                "system_sha256": sha256_bytes(prompts[name][0].encode()), "prompt_sha256": sha256_bytes(prompts[name][1].encode())}
                for name, c in cases],
            "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in (
                "app/method_execution.py", "app/strict_collection_methods.py", "app/query_planning_v2.py",
                "app/query_preparation_v3.py", "scripts/pilot_strict_method_answers.py")}}
        OUTPUT.mkdir(parents=True, exist_ok=True)
        raw = canonical_json(frozen) + b"\n"
        path = OUTPUT / "plan.json"
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("STRICT_ANSWERS_FROZEN_PLAN_CHANGED")
        if not path.exists():
            path.write_bytes(raw)
        return cases, prompts, routes, frozen
    finally:
        lexical.close()


class ScopedGateway:
    def __init__(self, gateway, cases, prompts):
        self.gateway, self.provider, self.routes = gateway, gateway.provider, gateway.routes
        self.allowed = {c.prepared.context.model_dump_json(): prompts[name] for name, c in cases}
        self.seen, self.halted, self.cost = set(), False, 0

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or self.cost >= 10000 or key in self.seen or self.allowed.get(key) != (system, prompt)
                or max_output_tokens != 800):
            raise ValueError("STRICT_ANSWERS_DISPATCH_BOUND")
        self.seen.add(key)
        try:
            result = self.gateway.generate(context, system=system, prompt=prompt, max_output_tokens=800)
            self.cost += result.actual_micro_usd
            return result
        except Exception:
            self.halted = True
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    cases, prompts, routes, plan = prepare()
    if not (args.execute or args.replay):
        print(json.dumps({"status": "PLANNED_ONLY", "ready_cases": len(cases), "provider_calls": 0}))
        return 0
    gateway, ledger = None, None
    if args.execute:
        import boto3
        from botocore.config import Config

        from app.external_gateway import ExternalModelGateway, ExternalPriceSheet
        from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
        from app.ledger import DynamoLedgerStore, Ledger

        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        config = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
        if not session.client("sts", config=config).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
            raise ValueError("STRICT_ANSWERS_OPERATOR_IDENTITY")
        ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=config), "kg-r72-query-state"))
        funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
        month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
        if not funding or funding["cap_micro_usd"] != 8000000 or funding["halted"] or not month or month["halted"] or month["cap_micro_usd"] != 12000000:
            raise ValueError("STRICT_ANSWERS_EXISTING_CAPS_REQUIRED")
        prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
        prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
        transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
        preflight = deepseek_preflight(transport)
        if not preflight["available"] or preflight["metadata_sha256"] != routes.account_model_metadata_sha256:
            raise ValueError("STRICT_ANSWERS_PROVIDER_MISMATCH")
        ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=plan["run_cap_micro_usd"])
        counter = ledger.store.get(ledger.run_key(RUN))
        if counter["halted"] or counter["reserved_micro_usd"]:
            raise ValueError("STRICT_ANSWERS_RECONCILIATION_REQUIRED")
        gateway = ScopedGateway(ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
            "live_inference_gate": "STRICT_THREE_ANSWER_DEVELOPMENT_DEMOS"}), prices), cases, prompts)
    cache = AnswerCache(ROOT / "data/cache/strict-method-answers-v1", namespace="LIVE")
    results = []
    try:
        for name, case in cases:
            lookup = answer_strict_prepared(case, cache, gateway=gateway)
            if lookup.receipt is None:
                raise ValueError("STRICT_ANSWERS_RECEIPT_REQUIRED")
            record = {"id": name, "outcome": case.outcome.model_dump(mode="json"), "trace": case.trace,
                "response": lookup.response.model_dump(mode="json"), "receipt": lookup.receipt.model_dump(mode="json"),
                "semantic_review": "PENDING"}
            raw = canonical_json(record) + b"\n"
            path = OUTPUT / (name + ".json")
            if path.exists() and path.read_bytes() != raw:
                raise ValueError("STRICT_ANSWERS_REPLAY_CHANGED")
            if not path.exists():
                path.write_bytes(raw)
            results.append({"id": name, "answer_status": lookup.response.answer_status,
                "claims": len(lookup.response.claims), "citations": len(lookup.response.citations),
                "cache_hit": lookup.cache_hit, "cost_micro_usd": lookup.receipt.original_cost_micro_usd})
        report = {"status": "REPLAY_MATCH" if args.replay else "COMPLETED_SOURCE_REVIEW_PENDING", "cases": results,
            "provider_calls_this_invocation": len(gateway.seen) if gateway else 0,
            "new_cost_micro_usd": gateway.cost if gateway else 0,
            "run_counter": ledger.store.get(ledger.run_key(RUN)) if ledger else None}
        if ledger and (report["run_counter"]["reserved_micro_usd"] or
                report["run_counter"]["settled_micro_usd"] != sum(r["cost_micro_usd"] for r in results)):
            raise ValueError("STRICT_ANSWERS_COST_RECONCILIATION")
    except Exception as exc:
        from app.model_gateway import safe_failure_code

        report = {"status": "STOPPED_NO_AUTOMATIC_RETRY", "error": safe_failure_code(exc), "cases": results,
            "provider_calls_this_invocation": len(gateway.seen) if gateway else 0, "new_cost_micro_usd": gateway.cost if gateway else 0}
    (OUTPUT / ("replay.json" if args.replay else "completion.json")).write_bytes(canonical_json(report) + b"\n")
    print(json.dumps(report))
    return 0 if report["status"] != "STOPPED_NO_AUTOMATIC_RETRY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
