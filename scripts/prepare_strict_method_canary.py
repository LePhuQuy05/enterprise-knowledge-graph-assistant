"""Small native V2 rewrite/planner canary under existing shared funding.

Default plans only. Execute: at most eight calls; no answer generation, repair,
transport retry, global cap raise, deployment or full graph extraction.
"""

import argparse
import json
from datetime import UTC, datetime

from app.benchmark_queries import QueryInput
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.model_gateway import CallContext, safe_failure_code
from app.query_planning_v2 import PlannerRecipeV2, QueryPlanCacheV2
from app.query_rewrite_v2 import QueryRewriteCacheV2, RewriteRecipeV2
from scripts.run_collection_benchmark import ROOT, load_inputs

RUN = "strict-method-preparation-v2"
SELECTED = (("aws100-019", "en"), ("aws100-066", "vi"), ("aws100-086", "en"),
    ("aws100-051", "vi"), ("aws100-007", "vi"))


def recipe_types(version):
    if version == 2:
        return PlannerRecipeV2, RewriteRecipeV2
    from app.query_preparation_v3 import PlannerRecipeV3, RewriteRecipeV3

    return PlannerRecipeV3, RewriteRecipeV3


def context_for(task, stage, version=2):
    planner, rewrite = recipe_types(version)
    return CallContext(tenant_id="demo", run_id=f"strict-method-preparation-v{version}", request_id=stable_id(f"strict-prep-v{version}",
        scenario=task.scenario_id, language=task.language, question_sha256=sha256_bytes(task.question.encode()),
        planner_recipe=planner().fingerprint, rewrite_recipe=rewrite().fingerprint), stage=stage)


class ScopedPreparationGateway:
    def __init__(self, gateway, tasks, version=2):
        self.gateway, self.provider, self.routes = gateway, gateway.provider, gateway.routes
        self.allowed = {context_for(t, s, version).model_dump_json() for t in tasks
            for s in (("REWRITE", "PLAN_QUERY") if t.language == "vi" else ("PLAN_QUERY",))}
        self.seen, self.halted, self.cost = set(), False, 0

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or self.cost >= 10000 or key not in self.allowed or key in self.seen
                or max_output_tokens != 256 or len(system.encode()) + len(prompt.encode()) > 12000):
            raise ValueError("STRICT_CANARY_DISPATCH_SCOPE_OR_BOUND")
        self.seen.add(key)
        try:
            result = self.gateway.generate(context, system=system, prompt=prompt, max_output_tokens=256)
            self.cost += result.actual_micro_usd
            return result
        except Exception:
            self.halted = True
            raise


def build_plan(version=2):
    _, dataset, _, _, _ = load_inputs()
    scenarios = {s.scenario_id: s for s in dataset.scenarios}
    selected = SELECTED if version == 2 else (("aws100-051", "vi"),)
    tasks = tuple(QueryInput(scenario_id=s, language=language,
        question=getattr(scenarios[s], "question_" + language)) for s, language in selected)
    planner, rewrite = recipe_types(version)
    return tasks, {"schema_version": 2, "run_id": f"strict-method-preparation-v{version}", "tasks": [t.model_dump(mode="json") for t in tasks],
        "planner_recipe": planner().model_dump(mode="json"), "rewrite_recipe": rewrite().model_dump(mode="json"),
        "maximum_calls": 8 if version == 2 else 2, "max_output_tokens": 256, "run_cap_micro_usd": 500000,
        "observed_stop_between_calls_micro_usd": 10000, "generate_answer_calls": 0, "global_caps_changed": False,
        "note": "0.50 USD admission cap permits one full 1M-context reservation. Actual-cost stop is 0.01 USD; no proxy-based reservation reduction."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--version", type=int, choices=(2, 3), default=2)
    args = parser.parse_args()
    tasks, plan = build_plan(args.version)
    run_id = plan["run_id"]
    output = ROOT / "results/method-correctness-v2" / ("native-preparation-v1" if args.version == 2 else "native-preparation-v3-semantic-v1")
    output.mkdir(parents=True, exist_ok=True)
    plan_raw = canonical_json(plan) + b"\n"
    plan_path = output / "plan.json"
    if plan_path.exists() and plan_path.read_bytes() != plan_raw:
        raise ValueError("STRICT_CANARY_FROZEN_PLAN_CHANGED")
    if not plan_path.exists():
        plan_path.write_bytes(plan_raw)
    if not args.execute:
        print(json.dumps({"status": "PLANNED_ONLY", "maximum_calls": plan["maximum_calls"], "aws_writes": 0}))
        return 0
    import boto3
    from botocore.config import Config

    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.ledger import DynamoLedgerStore, Ledger

    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region,
        now=datetime.now(UTC), live=True)
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("STRICT_CANARY_DEVELOPER_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))

    def counters():
        return {n: ledger.store.get(k) for n, k in (("run", ledger.run_key(run_id)),
            ("funding", ledger.external_funding_key("kg-r72")),
            ("month", ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m"))))}

    before = counters()
    for name, cap in (("funding", 2000000), ("month", 12000000)):
        row = before[name]
        if (not row or row["halted"] or row["cap_micro_usd"] != cap
                or row["cap_micro_usd"] - row["settled_micro_usd"] - row["reserved_micro_usd"] < 500000):
            raise ValueError("STRICT_CANARY_EXISTING_FUNDING_NOT_READY")
    if before["run"] and (before["run"]["reserved_micro_usd"] or before["run"]["halted"]
            or before["run"]["cap_micro_usd"] != 500000):
        raise ValueError("STRICT_CANARY_RUN_RECONCILIATION_REQUIRED")
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")})
    account = deepseek_preflight(transport)
    if not account["available"] or account["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("STRICT_CANARY_PROVIDER_MODEL_MISMATCH")
    ledger.initialize_run("kg-r72", "demo", run_id, cap_micro_usd=500000)
    raw_gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "STRICT_METHOD_V2_PLANNER_REWRITE_CANARY"}), prices)
    gateway = ScopedPreparationGateway(raw_gateway, tasks, args.version)
    cache_dir = ROOT / f"data/cache/strict-method-preparation-v{args.version}"
    if args.version == 2:
        planner, rewriter = QueryPlanCacheV2(cache_dir), QueryRewriteCacheV2(cache_dir)
    else:
        from app.query_preparation_v3 import QueryPlanCacheV3, QueryRewriteCacheV3

        planner, rewriter = QueryPlanCacheV3(cache_dir), QueryRewriteCacheV3(cache_dir)
    rows = []
    status, failure = "COMPLETE", None
    try:
        for task in tasks:
            search, rewrite_receipt, rewrite_proof = task.question, None, None
            if task.language == "vi":
                lookup = rewriter.resolve(context_for(task, "REWRITE", args.version), question=task.question,
                    language=task.language, routes=routes, gateway=gateway)
                search = lookup.rewrite.search_query
                rewrite_receipt, rewrite_proof = lookup.receipt, lookup.rewrite.model_dump(mode="json")
            lookup = planner.resolve(context_for(task, "PLAN_QUERY", args.version), question=task.question,
                search_query=search, language=task.language, routes=routes, gateway=gateway)
            row = {"scenario_id": task.scenario_id, "language": task.language, "question": task.question,
                "search_query": search, "plan": lookup.plan.model_dump(mode="json"), "planner_receipt": lookup.receipt,
                "rewrite": rewrite_proof, "rewrite_receipt": rewrite_receipt}
            path = output / (task.scenario_id + "." + task.language + ".json")
            raw = canonical_json(row) + b"\n"
            if path.exists() and path.read_bytes() != raw:
                raise ValueError("STRICT_CANARY_ROW_CHANGED")
            if not path.exists():
                path.write_bytes(raw)
            rows.append(row)
            print(json.dumps({"input": task.scenario_id + "." + task.language, "native_status": "READY",
                "calls_this_invocation": len(gateway.seen)}), flush=True)
    except Exception as exc:
        status, failure = "STOPPED_NO_AUTOMATIC_RETRY", safe_failure_code(exc)
    after = counters()
    if status == "COMPLETE":
        receipts = [r["planner_receipt"] for r in rows] + [r["rewrite_receipt"] for r in rows if r["rewrite_receipt"]]
        if (sum(r["cost_micro_usd"] for r in receipts) != after["run"]["settled_micro_usd"]
                or after["run"]["reserved_micro_usd"] != 0 or len(receipts) != plan["maximum_calls"]):
            status, failure = "RECONCILIATION_REQUIRED", "STRICT_CANARY_RECEIPT_COST_MISMATCH"
    report = {"status": status, "failure": failure, "before": before, "after": after,
        "plan_sha256": sha256_bytes(plan_raw), "rows": len(rows), "provider_calls_this_invocation": len(gateway.seen),
        "new_cost_micro_usd": gateway.cost, "generate_answer_calls": 0, "global_caps_changed": False,
        "balance_observation": account["balances"], "semantic_review": "UNREVIEWED", "full_graph_extraction": False}
    attempts = list(output.glob("attempt-*.json"))
    (output / f"attempt-{len(attempts) + 1:03d}.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "failure", "rows", "provider_calls_this_invocation", "new_cost_micro_usd")}))
    return 0 if status == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
