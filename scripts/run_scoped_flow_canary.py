"""Bounded local native V6 -> collection319 -> V12 answer canary.

Default freezes inputs/code only. --execute uses the managed shared ledger;
--replay forbids network and dispatches no model. No deployment or promotion.
"""

import argparse
import json
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.model_gateway import safe_failure_code
from app.profiled_answer_runtime import ProfiledAnswerCache, ProfiledModelGateway
from app.profiled_generation import ProfiledGenerationRecipe
from app.scoped_method_flow import answer_scoped_flow, prepare_scoped_flow
from app.scoped_query_v6 import SYSTEM, ScopedPlanCacheV6, ScopeRecipeV6, preparation_context
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.run_collection_benchmark import OfflineRuntime

ROOT = Path(__file__).resolve().parents[1]
RUN = "scoped319-v6-v12-canary-20261105-v2"
OUT = ROOT / "results/scoped-flow-canary-2026-11-05-v2"
CACHE = ROOT / "data/cache/scoped-flow-canary-2026-11-05-v2"
LIVE_TIMEOUT_SECONDS = 25
PRICE = "configs/prices/external-direct-2026-11-05.json"
METHODS = ("hybrid", "graph_1hop", "graph_2hop", "hybrid_multi_query", "hybrid_multi_query_balanced")


def tasks():
    payload = json.loads((ROOT / "results/facet-recall-v5-canary-v1/receipts.json").read_bytes())
    rows = []
    for old in payload["native_preparations"]:
        if old["language"] == "vi" and not old["reused_v3_rewrite_receipt_sha256"]:
            raise ValueError("SCOPED_CANARY_NATIVE_REWRITE_LINEAGE")
        rows.append({k:old[k] for k in ("intent", "question", "search", "language")} | {
            "search_track":"ORIGINAL_EN" if old["language"] == "en" else "RECORDED_NATIVE_V3_REWRITE_SAME_INPUT",
            "rewrite_receipt_sha256":old["reused_v3_rewrite_receipt_sha256"]})
    if len(rows) != 4 or len({(r["intent"],r["language"]) for r in rows}) != 4:
        raise ValueError("SCOPED_CANARY_INPUT_MATRIX")
    return rows


def freeze():
    collection, _, graph = load_migrated()
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    code = sorted(str(p.relative_to(ROOT)).replace("\\", "/") for p in (ROOT / "app").rglob("*.py"))
    code += ["scripts/run_scoped_flow_canary.py", "scripts/migrate_collection319_graph.py",
        "scripts/run_collection_benchmark.py"]
    scope = {"version":RUN, "collection_sha256":collection.manifest_sha256,
        "graph_sha256":graph.release_sha256, "graph_fact_count":len(graph.facts),
        "inputs":tasks(), "methods":METHODS, "source_profile":12, "user_proxy_token_cap":4096,
        "scope_recipe":ScopeRecipeV6().model_dump(mode="json"),
        "generation_recipe":ProfiledGenerationRecipe(model_metadata_sha256=routes.account_model_metadata_sha256).model_dump(mode="json"),
        "price_sheet_sha256":sha256_bytes((ROOT / PRICE).read_bytes()),
        "routes_sha256":sha256_bytes((ROOT / "configs/models/external-direct-v1.json").read_bytes()),
        "prior_preparations_sha256":sha256_bytes((ROOT / "results/facet-recall-v5-canary-v1/receipts.json").read_bytes()),
        "max_native_preparations":4, "max_generated_answers":20, "max_calls":24,
        "run_cap_micro_usd":500000, "published_usage_stop_micro_usd":100000,
        "shared_funding_cap_micro_usd":8000000,
        "quality_scope":"TWO_PREVIOUSLY_EXPOSED_INTENTS_EN_VI_NOT_INDEPENDENT_BENCHMARK",
        "native_v6_not_v5_relabel":True, "serving_published":False,
        "code_sha256":{name:sha256_bytes((ROOT/name).read_bytes()) for name in code}}
    immutable(OUT / "scope.json", scope)
    return scope


def checked_scope():
    scope = json.loads((OUT / "scope.json").read_bytes())
    if (scope["price_sheet_sha256"] != sha256_bytes((ROOT/PRICE).read_bytes())
            or scope["routes_sha256"] != sha256_bytes((ROOT/"configs/models/external-direct-v1.json").read_bytes())
            or any(sha256_bytes((ROOT/name).read_bytes()) != value for name,value in scope["code_sha256"].items())):
        raise ValueError("SCOPED_CANARY_CODE_OR_INPUT_CHANGED")
    return scope


class BoundedGateway:
    def __init__(self, base, scope):
        self.base, self.profiled = base, ProfiledModelGateway(base)
        self.routes, self.provider, self.scope = base.routes, base.provider, scope
        self.allowed, self.calls, self.cost, self.halted = {}, 0, 0, False
        for task in scope["inputs"]:
            context = preparation_context(scope["collection_sha256"],task["question"],task["search"],task["language"],RUN)
            prompt = canonical_json({"untrusted_original_question":task["question"],
                "untrusted_english_search_query":task["search"],"original_language":task["language"]}).decode()
            self.allowed[context.request_id] = ("PLAN_QUERY",SYSTEM,prompt,256)

    def _before(self, context, spec):
        if (self.halted or context.run_id != RUN or context.attempt != 1
                or self.calls >= self.scope["max_calls"] or self.cost >= self.scope["published_usage_stop_micro_usd"]
                or self.allowed.pop(context.request_id,None) != spec):
            raise ValueError("SCOPED_CANARY_DISPATCH_NOT_ALLOWED")
        immutable(OUT / "dispatch-intents" / (context.request_id+".json"),
            {"context":context.model_dump(mode="json"), "scope_sha256":sha256_bytes((OUT/"scope.json").read_bytes()),
             "request_fingerprint":sha256_bytes(canonical_json(spec)),"attempts":1})
        self.calls += 1

    def _after(self, result):
        self.cost += result.actual_micro_usd
        immutable(OUT / "dispatch-receipts" / (result.operation_id+".json"),
            {"operation_id":result.operation_id,"usage":result.usage.model_dump(mode="json"),
             "published_cost_micro_usd":result.actual_micro_usd,"stop_reason":result.stop_reason,
             "raw_output":result.text,"raw_output_sha256":sha256_bytes((result.text or "").encode())})
        if self.cost > self.scope["published_usage_stop_micro_usd"]:
            self.halted = True
        return result

    def generate(self, context, *, system, prompt, max_output_tokens):
        self._before(context,("PLAN_QUERY",system,prompt,max_output_tokens))
        try:
            return self._after(self.base.generate(context,system=system,prompt=prompt,max_output_tokens=max_output_tokens))
        except Exception:
            self.halted = True
            raise

    def allow_flow(self, flow):
        if (not flow.ready or flow.method not in self.scope["methods"]
                or flow.context.collection_sha256 != self.scope["collection_sha256"]
                or flow.recipe.model_dump(mode="json") != self.scope["generation_recipe"]
                or not any(t["question"] == flow.context.question and t["language"] == flow.context.language
                    for t in self.scope["inputs"])):
            raise ValueError("SCOPED_CANARY_PROFILE_SCOPE")
        spec = ("GENERATE",flow.context.fingerprint,flow.recipe.fingerprint,1600)
        self.allowed[flow.call_context.request_id] = spec

    def generate_profiled(self, context, *, source_context, documents, chunks, recipe):
        self._before(context,("GENERATE",source_context.fingerprint,recipe.fingerprint,1600))
        try:
            return self._after(self.profiled.generate_profiled(context,source_context=source_context,
                documents=documents,chunks=chunks,recipe=recipe))
        except Exception:
            self.halted = True
            raise


def managed(scope, *, execute):
    import boto3
    from botocore.config import Config

    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    cfg = Config(retries={"total_max_attempts":1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts",config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("SCOPED_CANARY_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb",config=cfg),"kg-r72-query-state"))
    funding = validate_external_funding(ledger,profile=TRANSITION_PROFILE)
    if (funding.cap_micro_usd != 8000000
            or funding.cap_micro_usd-funding.settled_micro_usd-funding.reserved_micro_usd < scope["run_cap_micro_usd"]):
        raise ValueError("SCOPED_CANARY_FUNDING_HEADROOM")
    prices = ExternalPriceSheet.model_validate_json((ROOT/PRICE).read_bytes())
    routes = ExternalRoutes.model_validate_json((ROOT/"configs/models/external-direct-v1.json").read_bytes())
    prices.rate_for(routes.generation_model_id,routes.generation_invocation_id,routes.source_region,now=datetime.now(UTC),live=True)
    provider = OfficialHttpsTransport({"DEEPSEEK":load_private_key(ROOT,"DEEPSEEK")},timeout_seconds=LIVE_TIMEOUT_SECONDS)
    metadata = deepseek_preflight(provider)
    if not metadata["available"] or metadata["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("SCOPED_CANARY_MODEL_CHANGED")
    record = {"checked_at":datetime.now(UTC).isoformat(), "provider":metadata,
        "funding":funding.model_dump(mode="json"),"inference_dispatched":False,
        "run_cap_micro_usd":scope["run_cap_micro_usd"],"published_usage_stop_micro_usd":scope["published_usage_stop_micro_usd"]}
    if not execute:
        print(json.dumps(record))
        return None,None
    ledger.initialize_month("kg-r72",datetime.now(UTC).strftime("%Y-%m"))
    ledger.initialize_run("kg-r72","demo",RUN,cap_micro_usd=scope["run_cap_micro_usd"])
    counter = ledger.store.get(ledger.run_key(RUN))
    if counter["halted"] or counter["reserved_micro_usd"]:
        raise ValueError("SCOPED_CANARY_RECONCILIATION_REQUIRED")
    immutable(OUT/"preflight.json",record)
    base = ExternalModelGateway(ledger,provider,routes.model_copy(update={"live_inference_enabled":True,
        "live_inference_gate":"FROZEN_SCOPED319_V6_V12_CANARY"}),prices)
    return BoundedGateway(base,scope),ledger


def run(*, replay):
    scope = checked_scope()
    gateway,ledger = (None,None) if replay else managed(scope,execute=True)
    collection,paths,graph = load_migrated()
    if collection.manifest_sha256 != scope["collection_sha256"] or graph.release_sha256 != scope["graph_sha256"]:
        raise ValueError("SCOPED_CANARY_SOURCE_RELEASE_CHANGED")
    routes = ExternalRoutes.model_validate_json((ROOT/"configs/models/external-direct-v1.json").read_bytes())
    planner = ScopedPlanCacheV6(CACHE,collection.manifest_sha256)
    answers = ProfiledAnswerCache(CACHE)
    recipe = ProfiledGenerationRecipe.model_validate_json(canonical_json(scope["generation_recipe"]))
    cells = []
    try:
        with ExitStack() as stack:
            if replay:
                stack.enter_context(patch("socket.socket.connect",side_effect=AssertionError("REPLAY_NETWORK_FORBIDDEN")))
                stack.enter_context(patch("socket.socket.connect_ex",side_effect=AssertionError("REPLAY_NETWORK_FORBIDDEN")))
            lexical = stack.enter_context(load_global_lexical_index(Path(paths["lexical_manifest_path"]),collection))
            runtime = OfflineRuntime(collection,lexical,())
            for task in scope["inputs"]:
                context = preparation_context(collection.manifest_sha256,task["question"],task["search"],task["language"],RUN)
                lookup = planner.resolve(context,question=task["question"],search_query=task["search"],
                    routes=routes,gateway=gateway,language=task["language"])
                immutable(OUT/"preparations"/(task["intent"]+"-"+task["language"]+".json"),
                    {"task":task,"context":context.model_dump(mode="json"),"plan":lookup.plan.model_dump(mode="json"),
                     "receipt":lookup.receipt,"raw_native_v6":True})
                shared = None
                for method in METHODS:
                    flow = prepare_scoped_flow(collection,lexical,graph,runtime,question=task["question"],
                        search=task["search"],language=task["language"],method=method,plan=lookup.plan,
                        recipe=recipe,run_id=RUN,owner_id="scoped319-local")
                    if method.startswith("hybrid_multi_query"):
                        if shared is not None and shared != flow.trace["pool_fingerprint"]:
                            raise ValueError("SCOPED_CANARY_Q_QB_POOL_CHANGED")
                        shared = flow.trace["pool_fingerprint"]
                    row = {"task":task,"method":method,"status":flow.status,"pool_fingerprint":flow.trace["pool_fingerprint"],
                        "context_fingerprint":flow.context.fingerprint,"pool_ids":flow.trace["pool_ids"],
                        "context_chunk_ids":flow.context.chunk_ids,"trace":flow.trace,
                        "answer_origin":"NOT_GENERATED"}
                    if flow.ready:
                        if gateway is not None:
                            gateway.allow_flow(flow)
                        answer = answer_scoped_flow(flow,answers,gateway=gateway)
                        row |= {"response":answer.response.model_dump(mode="json"),
                            "answer_origin":"DETERMINISTIC_ABSTENTION" if answer.receipt is None else "NATIVE_V12_RECEIPT",
                            "generation_receipt_present":answer.receipt is not None,
                            "receipt":None if answer.receipt is None else answer.receipt.model_dump(mode="json")}
                    target = OUT/"cells"/(task["intent"]+"-"+task["language"]+"-"+method+".json")
                    if replay:
                        saved = json.loads(target.read_bytes())
                        for field in ("method","status","pool_fingerprint","context_fingerprint","context_chunk_ids","response","receipt"):
                            if canonical_json(saved.get(field)) != canonical_json(row.get(field)):
                                raise ValueError("SCOPED_CANARY_REPLAY_CHANGED_"+field.upper())
                    else:
                        immutable(target,row)
                    cells.append(row)
                    print(json.dumps({"intent":task["intent"],"language":task["language"],"method":method,
                        "mechanism_status":flow.status,"answer_status":row.get("response",{}).get("answer_status"),
                        "replay":replay}),flush=True)
        checked_scope()
        summary = {"status":"EXACT_NATIVE_CACHE_AND_CONTEXT_REPLAY_PASS" if replay else "NATIVE_CANARY_RECORDED",
            "cells":len(cells),"ready_cells":sum(r["status"] in {"READY","READY_NO_EVIDENCE_ABSTENTION"} for r in cells),
            "new_calls":0 if replay else gateway.calls,"new_published_cost_micro_usd":0 if replay else gateway.cost,
            "hidden_fallbacks":sum(r["trace"]["hidden_fallbacks"] for r in cells),
            "source_scope":"319_DOCUMENTS_RETRIEVAL_SPARSE_TYPED_GRAPH_NOT_FULL_EXTRACTION",
            "human_reviewers":0,"semantic_accuracy_measured":False,"serving_published":False}
        if not replay:
            operations = [json.loads(p.read_bytes()) for p in (OUT/"dispatch-receipts").glob("*.json")]
            states = [ledger.operation(r["operation_id"]) for r in operations]
            if any(s is None or s.status != "SETTLED" for s in states):
                raise ValueError("SCOPED_CANARY_LEDGER_NOT_SETTLED")
            summary["settled_operations"] = len(states)
            summary["ledger_published_cost_micro_usd"] = sum(r["published_cost_micro_usd"] for r in operations)
            if summary["ledger_published_cost_micro_usd"] != gateway.cost:
                raise ValueError("SCOPED_CANARY_COST_MISMATCH")
        immutable(OUT/("replay.json" if replay else "report.json"),summary)
        print(json.dumps(summary),flush=True)
    except Exception as exc:
        record = {"status":"HALTED_NO_AUTOMATIC_RETRY", "failure":safe_failure_code(exc),"completed_cells":len(cells),
            "new_calls":0 if gateway is None else gateway.calls,"published_cost_micro_usd":0 if gateway is None else gateway.cost}
        if not replay:
            immutable(OUT/"halt.json",record)
        print(json.dumps(record),flush=True)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--execute",action="store_true")
    action.add_argument("--replay",action="store_true")
    action.add_argument("--preflight",action="store_true")
    args = parser.parse_args()
    if args.execute or args.replay:
        run(replay=args.replay)
    elif args.preflight:
        managed(checked_scope(),execute=False)
    else:
        scope = freeze()
        print(json.dumps({"status":"FROZEN_NO_DISPATCH","inputs":len(scope["inputs"]),"max_calls":scope["max_calls"]}))


if __name__ == "__main__":
    main()
