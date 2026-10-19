"""Frozen quote-only pilot v2; paced Voyage, same allowance, no unknown-operation retry."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import boto3
from botocore.config import Config

from app.contracts import canonical_json
from app.corpus import load_corpus
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, load_private_key
from app.extraction import AliasDictionary
from app.ledger import DynamoLedgerStore, Ledger
from app.model_gateway import GatewayError, UncertainModelCall, safe_failure_code
from app.pilot import revised_quote_plan, run_selected
from scripts.check_external_smoke import _fresh_report

ROOT = Path(__file__).resolve().parents[1]
RUN = "external-pilot10-r72-v1"  # Reuse existing 0.50 USD allowance; never initialize/reset.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    output = ROOT / ("results/external_pilot10_v2_replay.json" if args.replay else "results/external_pilot10_v2.json")
    if args.execute and output.exists():
        print("Existing v2 receipt retained; no automatic rerun or new allowance.")
        return 1
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "BLOCKED", "run_id": RUN,
              "experiment_version": "external-pilot10-v2", "cloud_inference_enabled": False,
              "semantic_review": "PENDING", "new_deployments": 0}
    ledger, transport = None, None
    try:
        prior = _fresh_report(ROOT / "results/external_pilot10.json", datetime.now(UTC))
        if prior.get("status") != "UNKNOWN_STOPPED" or prior.get("run_id") != RUN or len(prior["embeddings"]) != 3:
            raise GatewayError("Expected original stopped pilot receipt required")
        old_plan = json.loads((ROOT / "data/experiments/external-pilot10-v1/plan.json").read_bytes())
        if old_plan["plan_sha256"] != prior["plan_sha256"]:
            raise GatewayError("Prior plan changed")
        receipt = json.loads((ROOT / "results/ingestion_pilot20_v2.json").read_bytes())
        corpus = load_corpus(Path(receipt["corpus_directory"]), tenant_id="demo")
        aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        plan = revised_quote_plan(corpus, old_plan, old_plan["selection"][3]["chunk_id"], aliases, routes)
        artifacts = ROOT / "data/experiments/external-pilot10-v2"
        artifacts.mkdir(parents=True, exist_ok=True)
        plan_file = artifacts / "plan.json"
        if plan_file.exists() and plan_file.read_bytes() != canonical_json(plan):
            raise GatewayError("V2 plan changed; inspect without dispatch")
        if not plan_file.exists():
            plan_file.write_bytes(canonical_json(plan))
        if not args.execute and not args.replay:
            print(json.dumps({"status": "PLANNED_NO_CALLS", "plan_sha256": plan["plan_sha256"],
                              "chunks": len(plan["selection"]), "same_run_cap_micro_usd": plan["run_cap_micro_usd"],
                              "voyage_pacing": plan["voyage_pacing"]}, indent=2))
            return 0
        report.update(plan_sha256=plan["plan_sha256"], voyage_pacing=plan["voyage_pacing"],
                      preserved_unknown_operation_id=prior["operation_id"])
        if args.replay:
            class OfflineGateway:
                provider = SimpleNamespace(live=True)

                def embed(self, *a, **kw):
                    raise GatewayError("Offline cache miss; no calls permitted")

                generate = embed

            gateway = OfflineGateway()
            gateway.routes = routes
        else:
            prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes())
            session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
            cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=10)
            if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
                raise GatewayError("Known developer identity required")
            ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
            unknown = ledger.operation(prior["operation_id"])
            if (unknown.status, unknown.spec.provider, unknown.spec.run_id) != ("UNKNOWN_SPEND", "VOYAGE", RUN):
                raise GatewayError("Original unknown operation must remain reserved")
            counter = ledger.store.get(ledger.run_key(RUN))
            if counter["cap_micro_usd"] != plan["run_cap_micro_usd"] or counter["halted"]:
                raise GatewayError("Existing pilot allowance changed or halted")
            transport = OfficialHttpsTransport({p: load_private_key(ROOT, p) for p in ("DEEPSEEK", "VOYAGE")},
                                               minimum_intervals={"VOYAGE": 21})
            gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={
                "live_inference_enabled": True, "live_inference_gate": "EXPLICIT_FROZEN_PILOT_V2_LOCAL_ONLY",
            }), prices)

        def checkpoint(progress):
            report.update(progress)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        run_selected(corpus, plan, aliases=aliases, gateway=gateway, run_id=RUN,
                     cache_directory=ROOT / "data/cache/external-models", artifact_directory=artifacts,
                     checkpoint=checkpoint)
    except Exception as exc:
        report.update(status="STOPPED", error_category=safe_failure_code(exc))
        if isinstance(exc, UncertainModelCall):
            report.update(status="UNKNOWN_STOPPED", operation_id=exc.operation_id)
    finally:
        if args.execute or args.replay:
            report["inference_dispatch_attempts_this_invocation"] = transport.dispatch_attempts if transport else 0
            if ledger is not None:
                try:
                    report["run_counter"] = ledger.store.get(ledger.run_key(RUN))
                    report["funding_counter"] = ledger.store.get(ledger.external_funding_key("kg-r72"))
                except Exception as exc:
                    report["counter_read_error_category"] = type(exc).__name__
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in ("status", "error_category", "chunks", "new_budget_cost_micro_usd",
                     "semantic_review", "inference_dispatch_attempts_this_invocation")}, indent=2))
    return 0 if report["status"] == "STRUCTURAL_PASS_SEMANTIC_PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
