"""Frozen synthetic security controls and five-case V2 provider smoke; no automatic paid rerun."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.answer_cache import AnswerCache
from app.answer_pipeline import answer_prepared
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.generation import GenerationRecipeV2
from app.model_gateway import safe_failure_code
from app.security_evaluation import SecurityCatalog, inspect_candidate, offline_case, prepare_case
from scripts.answer_local import RUN, managed_gateway

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "data/experiments/generation-security-v1"


def build_plan(root=ROOT):
    catalog = SecurityCatalog.model_validate_json((root / "configs/security/generation-security-v1.json").read_bytes())
    routes = json.loads((root / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV2(model_metadata_sha256=routes["account_model_metadata_sha256"])
    code_paths = ("app/security_evaluation.py", "scripts/run_generation_security.py", "app/generation.py",
                  "app/answer_cache.py", "app/external_gateway.py", "app/external_http.py", "app/ledger.py")
    plan = {"version": catalog.version, "catalog_sha256": catalog.fingerprint, "scope": catalog.scope,
        "generation_recipe": recipe.model_dump(mode="json"), "run_id": RUN,
        "funding_cap_micro_usd": 2_000_000, "run_cap_micro_usd": 500_000,
        "maximum_model_dispatches": 5, "cloud_queries": 0, "human_review": "PENDING",
        "live_case_ids": [c.case_id for c in catalog.cases if c.live_smoke],
        "code_sha256": {p: sha256_bytes((root / p).read_bytes()) for p in code_paths},
        "price_sha256": sha256_bytes((root / "configs/prices/external-direct-2026-10-17.json").read_bytes()),
        "limitations": ["Synthetic fixed context, no retrieval or cloud auth test",
            "Scripted candidate output is not model behavior", "Five model calls are not a full benchmark",
            "Exact quotes and source IDs do not establish semantic support",
            "Marker and verbatim-fact flags require semantic review; no automatic attack success rate"]}
    plan["plan_sha256"] = sha256_bytes(canonical_json(plan))
    return catalog, recipe, plan


def freeze(directory, plan):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "plan.json"
    raw = canonical_json(plan)
    if path.exists() and path.read_bytes() != raw:
        raise ValueError("SECURITY_PLAN_CHANGED_NO_DISPATCH")
    if not path.exists():
        with path.open("xb") as handle:
            handle.write(raw)


def save(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def counters(gateway):
    return {"funding": gateway.ledger.store.get(gateway.ledger.external_funding_key("kg-r72")),
            "run": gateway.ledger.store.get(gateway.ledger.run_key(RUN))}


def execute_live(catalog, recipe, plan, directory, output, *, replay=False, gateway_factory=managed_gateway):
    """Execute once or replay offline. Existing output/failed cache is a stop, never a repair."""
    if output.exists():
        raise ValueError("SECURITY_REPORT_EXISTS_NO_RERUN")
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "STARTED", "plan_sha256": plan["plan_sha256"],
              "cases": [], "new_model_dispatches": 0, "cloud_queries": 0, "cloud_deployments": 0,
              "semantic_review": "PENDING", "provider_invoice_micro_usd": None, "replay": replay}
    # A persistent attempt marker precedes even the ledger preflight.
    with output.open("x", encoding="utf-8") as handle:
        json.dump(report, handle)
    gateway = None
    inference_start = 0
    inference_armed = False
    try:
        if not replay:
            gateway = gateway_factory()
            report["initial_counters"] = counters(gateway)
            # Read public account model metadata; not a model inference call.
            raw_models = gateway.provider.request("DEEPSEEK", "GET", "/models")
            items = raw_models.get("data", [])
            selected = [m for m in items if isinstance(m, dict) and m.get("id") == recipe.model_id]
            if len(selected) != 1:
                raise ValueError("SECURITY_MODEL_UNAVAILABLE")
            metadata = {k: selected[0][k] for k in ("id", "name", "context_window", "max_output_tokens")
                        if k in selected[0]}
            report["model_metadata"] = metadata
            if sha256_bytes(canonical_json(metadata)) != recipe.model_metadata_sha256:
                raise ValueError("SECURITY_MODEL_METADATA_CHANGED")
            inference_start = gateway.provider.dispatch_attempts
            inference_armed = True
            rate = gateway.prices.rate_for(recipe.model_id, recipe.invocation_id, recipe.source_region,
                                          now=gateway.ledger._now(), live=True)
            report["single_call_reservation_micro_usd"] = rate.quote(TokenUsage(
                input_tokens=gateway.routes.generation_context_ceiling, output_tokens=recipe.max_output_tokens))
            save(output, report)
        cache = AnswerCache(directory / "cache", namespace="LIVE")
        for case in (c for c in catalog.cases if c.live_smoke):
            snapshot, prepared = prepare_case(case, directory, recipe, RUN)
            operation_id = stable_id("op", **prepared.context.model_dump())
            row = {"case_id": case.case_id, "kind": case.kind, "objective": case.objective,
                "request_id": prepared.context.request_id, "operation_id": operation_id,
                "snapshot_sha256": snapshot.manifest_sha256, "semantic_review": "PENDING"}
            report["cases"].append(row)
            save(output, report)
            try:
                lookup = answer_prepared(prepared, snapshot, cache, gateway=gateway)
                receipt = lookup.receipt
                if receipt is None:
                    raise ValueError("SECURITY_MISSING_MODEL_RECEIPT")
                row.update(inspect_candidate(case, prepared, snapshot, receipt.raw_output, receipt.stop_reason),
                    cache_hit=lookup.cache_hit, new_budget_cost_micro_usd=lookup.new_cost_micro_usd,
                    original_budget_cost_micro_usd=receipt.original_cost_micro_usd,
                    usage=receipt.usage.model_dump(mode="json"),
                    raw_output_sha256=sha256_bytes(receipt.raw_output.encode()),
                    response_sha256=sha256_bytes(canonical_json(lookup.response.model_dump(mode="json"))))
                if gateway:
                    operation = gateway.ledger.operation(operation_id)
                    row["ledger_operation"] = operation.model_dump(mode="json")
                    if operation.status != "SETTLED" or operation.actual_micro_usd != receipt.original_cost_micro_usd:
                        raise ValueError("SECURITY_LEDGER_RECONCILIATION")
            except Exception as exc:
                row.update(status="STOPPED_NO_RETRY", error_category=safe_failure_code(exc))
                # AnswerCache already retained invalid settled output or uncertain claim.
                if gateway:
                    try:
                        operation = gateway.ledger.operation(operation_id)
                        row["ledger_operation"] = operation.model_dump(mode="json")
                        if operation.status == "SETTLED":
                            row["original_budget_cost_micro_usd"] = operation.actual_micro_usd
                            row["new_budget_cost_micro_usd"] = operation.actual_micro_usd
                    except Exception as read_error:
                        row["ledger_read_error"] = type(read_error).__name__
                raise
            finally:
                if gateway and inference_armed:
                    report["new_model_dispatches"] = gateway.provider.dispatch_attempts - inference_start
                save(output, report)
            print(json.dumps({"case_id": case.case_id, "structural_status": row["structural_status"],
                              "cache_hit": row["cache_hit"]}), flush=True)
        if report["new_model_dispatches"] > 5 or (replay and not all(r["cache_hit"] for r in report["cases"])):
            raise ValueError("SECURITY_DISPATCH_LIMIT")
        report["status"] = "COMPLETED_SEMANTIC_REVIEW_PENDING"
    except Exception as exc:
        report.update(status="STOPPED_NO_RETRY", error_category=safe_failure_code(exc))
    finally:
        if gateway:
            report["new_model_dispatches"] = (max(0, gateway.provider.dispatch_attempts - inference_start)
                                              if inference_armed else 0)
            try:
                report["final_counters"] = counters(gateway)
            except Exception as exc:
                report["counter_read_error"] = type(exc).__name__
        report["new_budget_cost_micro_usd"] = sum(r.get("new_budget_cost_micro_usd", 0) for r in report["cases"])
        report["settled_case_cost_micro_usd"] = sum(r.get("original_budget_cost_micro_usd", 0) for r in report["cases"])
        save(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    catalog, recipe, plan = build_plan()
    freeze(DIRECTORY, plan)
    if not args.execute and not args.replay:
        rows = []
        for case in catalog.cases:
            snapshot, prepared = prepare_case(case, DIRECTORY, recipe, RUN)
            rows.append(offline_case(case, prepared, snapshot))
        report = {"status": "OFFLINE_BOUNDARIES_MATCHED" if all(r["expected_boundary_observed"] for r in rows)
            else "OFFLINE_BOUNDARY_FAILURE", "plan_sha256": plan["plan_sha256"], "cases": rows,
            "attacks": 20, "benign": 8, "model_calls": 0,
            "scripted_rejected": sum(r["scripted_structural_status"] == "REJECTED" for r in rows),
            "known_semantic_limitations": sum(r["expected_boundary"] == "ACCEPT_NEEDS_SEMANTIC_REVIEW" for r in rows)}
        save(ROOT / "results/generation_security_offline.json", report)
    else:
        output = ROOT / ("results/generation_security_replay.json" if args.replay else "results/generation_security_live.json")
        report = execute_live(catalog, recipe, plan, DIRECTORY, output, replay=args.replay)
    print(json.dumps({k: report.get(k) for k in ("status", "new_model_dispatches", "new_budget_cost_micro_usd")}, indent=2))
    return 0 if report["status"] in {"OFFLINE_BOUNDARIES_MATCHED", "COMPLETED_SEMANTIC_REVIEW_PENDING"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
