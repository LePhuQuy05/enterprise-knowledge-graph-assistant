"""Continue only the independent DeepSeek branch; keep Voyage's unknown reservation."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config

from app.corpus import load_corpus
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, load_private_key
from app.extraction import AliasDictionary
from app.ledger import DynamoLedgerStore, Ledger
from app.model_gateway import GatewayError, UncertainModelCall, safe_failure_code
from app.pilot import run_selected
from scripts.check_external_smoke import _fresh_report

ROOT = Path(__file__).resolve().parents[1]
RUN = "external-pilot10-r72-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not args.execute:
        print("No calls; this phase never retries embeddings or initializes budgets.")
        return 0
    output = ROOT / "results/external_pilot10_extraction.json"
    if output.exists():
        print("Existing extraction-phase receipt retained; no automatic rerun.")
        return 1
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "BLOCKED", "run_id": RUN,
              "decision": "Continue independent extraction; preserve unknown embedding without retry or budget reset",
              "semantic_review": "PENDING", "cloud_inference_enabled": False}
    ledger, transport = None, None
    try:
        prior = _fresh_report(ROOT / "results/external_pilot10.json", datetime.now(UTC))
        if prior.get("status") != "UNKNOWN_STOPPED" or prior.get("run_id") != RUN:
            raise GatewayError("Expected stopped embedding-phase receipt required")
        artifacts = ROOT / "data/experiments/external-pilot10-v1"
        plan = json.loads((artifacts / "plan.json").read_bytes())
        if plan["plan_sha256"] != prior["plan_sha256"]:
            raise GatewayError("Frozen plan differs from the stopped pilot")
        receipt = json.loads((ROOT / "results/ingestion_pilot20_v2.json").read_bytes())
        corpus = load_corpus(Path(receipt["corpus_directory"]), tenant_id="demo")
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes())
        aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=10)
        if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
            raise GatewayError("Known developer identity required")
        ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
        unknown = ledger.operation(prior["operation_id"])
        if (unknown.status, unknown.spec.provider, unknown.spec.stage, unknown.spec.run_id) != (
            "UNKNOWN_SPEND", "VOYAGE", "EMBED", RUN,
        ):
            raise GatewayError("Only an independent Voyage embedding failure permits this phase")
        counter = ledger.store.get(ledger.run_key(RUN))
        if counter["cap_micro_usd"] != plan["run_cap_micro_usd"] or counter["halted"]:
            raise GatewayError("Existing pilot allowance differs or is halted")
        report["preserved_unknown_operation_id"] = unknown.operation_id
        transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")})
        gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={
            "live_inference_enabled": True, "live_inference_gate": "INDEPENDENT_DEEPSEEK_PILOT_BRANCH_ONLY",
        }), prices)

        def checkpoint(progress):
            report.update(progress)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        run_selected(corpus, plan, aliases=aliases, gateway=gateway, run_id=RUN,
                     cache_directory=ROOT / "data/cache/external-models", artifact_directory=artifacts,
                     checkpoint=checkpoint, stages=("EXTRACT",))
    except Exception as exc:
        report.update(status="STOPPED", error_category=safe_failure_code(exc))
        if isinstance(exc, UncertainModelCall):
            report.update(status="UNKNOWN_STOPPED", operation_id=exc.operation_id)
    finally:
        report["inference_dispatch_attempts"] = transport.dispatch_attempts if transport else 0
        if ledger is not None:
            try:
                report["run_counter"] = ledger.store.get(ledger.run_key(RUN))
                report["funding_counter"] = ledger.store.get(ledger.external_funding_key("kg-r72"))
            except Exception as exc:
                report["counter_read_error"] = type(exc).__name__
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report.get(k) for k in (
        "status", "error_category", "chunks", "new_budget_cost_micro_usd", "inference_dispatch_attempts",
    )}, indent=2))
    return 0 if report["status"] == "EXTRACTION_STRUCTURAL_PASS_EMBEDDINGS_PENDING" else 1


if __name__ == "__main__":
    raise SystemExit(main())
