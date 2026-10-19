"""Three tiny synthetic calls under one durable allowance; never enables cloud serving."""

import argparse
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config

from app.external_gateway import (
    ExternalCallUncertain,
    ExternalModelGateway,
    ExternalPriceSheet,
    ExternalRoutes,
)
from app.external_http import OfficialHttpsTransport, load_private_key, strict_json
from app.ledger import DynamoLedgerStore, Ledger
from app.model_gateway import CallContext, GatewayError

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "Fixture Alpha uses Fixture Beta."


def _fresh_report(path: Path, now: datetime) -> dict:
    result = json.loads(path.read_bytes())
    checked = datetime.fromisoformat(result["checked_at"])
    if checked.utcoffset() != timedelta(0) or not timedelta(0) <= now - checked < timedelta(days=1):
        raise GatewayError("Preflight receipt is stale or has an invalid timestamp")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="Explicit bounded paid smoke; no deployment")
    args = parser.parse_args()
    if not args.execute:
        print("No calls. Use --execute only after account/rates/managed ledger gates.")
        return 0
    existing_path = ROOT / "results/external_model_smoke.json"
    if existing_path.exists():
        prior = json.loads(existing_path.read_bytes())
        if prior.get("run_id"):
            print(json.dumps({"status": prior.get("status"), "new_calls": 0,
                              "note": "Existing durable smoke receipt retained; no automatic rerun."}))
            return 0 if prior.get("status") == "PASS" else 1
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "BLOCKED", "steps": [],
              "cloud_inference_enabled": False, "new_deployments": 0, "provider_invoice_micro_usd": None}
    transport = None
    try:
        now = datetime.now(UTC)
        probe = _fresh_report(ROOT / "results/external_ledger_managed.json", now)
        if probe.get("status") != "PASS" or probe.get("backend") != "MANAGED_DYNAMODB":
            raise GatewayError("Managed external ledger probe has not passed")
        accounts = _fresh_report(ROOT / "results/external_account_preflight.json", now)
        deepseek = accounts["providers"]["DEEPSEEK"]
        if deepseek.get("available") is not True:
            raise GatewayError("DeepSeek account has not passed read-only preflight")
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        if routes.account_model_metadata_sha256 != deepseek["metadata_sha256"] or (
            deepseek["model"]["context_window"] != routes.generation_context_ceiling
        ):
            raise GatewayError("Verified model metadata differs from pinned routes")
        prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes())
        funding = json.loads((ROOT / "configs/external_funding.json").read_bytes())
        if funding["external_cap_micro_usd"] > 2_000_000 or funding["initial_smoke_cap_micro_usd"] > 500_000:
            raise GatewayError("Smoke exceeds the selected conservative funding limits")
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        sdk = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=10)
        if not session.client("sts", config=sdk).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
            raise GatewayError("Known developer identity required")
        ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=sdk), "kg-r72-query-state"))
        run = funding["initial_smoke_run_id"]
        if ledger.store.get(ledger.run_key(run)) is not None:
            raise GatewayError("Initial smoke already initialized; inspect existing receipts, never reset/retry automatically")
        ledger.initialize_month("kg-r72", now.strftime("%Y-%m"))
        ledger.initialize_external_funding("kg-r72", cap_micro_usd=funding["external_cap_micro_usd"])
        ledger.initialize_run("kg-r72", "demo", run, cap_micro_usd=funding["initial_smoke_cap_micro_usd"])
        transport = OfficialHttpsTransport({p: load_private_key(ROOT, p) for p in ("DEEPSEEK", "VOYAGE")})
        gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={
            "live_inference_enabled": True, "live_inference_gate": "EXPLICIT_BOUNDED_LOCAL_SMOKE_ONLY",
        }), prices)
        private = ROOT / "results/private/external-smoke"
        private.mkdir(parents=True, exist_ok=True)
        report.update(status="STARTED", run_id=run, cost_basis=prices.cost_basis,
                      external_cap_micro_usd=funding["external_cap_micro_usd"],
                      initial_smoke_cap_micro_usd=funding["initial_smoke_cap_micro_usd"])

        def save_result(name, result):
            safe = {"operation_id": result.operation_id, "usage": result.usage.model_dump(),
                    "budget_settled_micro_usd": result.actual_micro_usd, "cost_basis": result.cost_basis,
                    "provider": result.provider, "observed_model": result.observed_model,
                    "text": result.text, "stop_reason": result.stop_reason,
                    "input_role": result.input_role, "embedding": result.embedding}
            (private / (name + ".json")).write_text(json.dumps(safe) + "\n", encoding="utf-8")
            report["steps"].append({k: safe[k] for k in (
                "operation_id", "usage", "budget_settled_micro_usd", "provider", "observed_model", "input_role",
            )} | {"name": name})
            (ROOT / "results/external_model_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        generated = gateway.generate(
            CallContext(tenant_id="demo", run_id=run, request_id="generation-v1", stage="SMOKE"),
            system='Return JSON only with exactly one field named "quote". Copy the supplied sentence exactly.',
            prompt=SOURCE, max_output_tokens=64,
        )
        save_result("generation", generated)
        if generated.stop_reason != "end_turn" or strict_json(generated.text.encode()) != {"quote": SOURCE}:
            raise GatewayError("Generation JSON/exact quote smoke failed after settlement")
        vectors = []
        for role, text in (("document", SOURCE), ("query", "What does Fixture Alpha use?")):
            result = gateway.embed(CallContext(tenant_id="demo", run_id=run, request_id="embedding-" + role + "-v1", stage="EMBED"),
                                   text=text, input_role=role)
            save_result("embedding-" + role, result)
            vectors.append(np.asarray(result.embedding))
        report.update(status="PASS", json_exact_quote=True, document_query_roles_verified=True,
                      dimension=1024, document_query_cosine=float(vectors[0] @ vectors[1]),
                      semantic_quality_verified=False,
                      budget_settled_micro_usd=sum(s["budget_settled_micro_usd"] for s in report["steps"]))
        report["funding_counter"] = ledger.store.get(ledger.external_funding_key("kg-r72"))
    except ExternalCallUncertain as exc:
        report.update(status="UNKNOWN_STOPPED", operation_id=exc.operation_id, error_category=exc.category)
    except Exception as exc:
        report["error_category"] = type(exc).__name__
        if isinstance(exc, GatewayError):
            report["safe_message"] = str(exc)
    finally:
        report["inference_dispatch_attempts"] = transport.dispatch_attempts if transport else 0
        (ROOT / "results/external_model_smoke.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
