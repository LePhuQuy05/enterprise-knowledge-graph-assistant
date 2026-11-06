"""Read-only release facts, provider balance and existing allowances. No inference or AWS writes."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.contracts import canonical_json
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
from app.ledger import Ledger
from app.run_work_index import CloudDynamoRunStore

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("COLLECTION_ANSWER_PREFLIGHT_EXISTS")
    report = {"at": datetime.now(UTC).isoformat(), "writes": False, "model_calls": 0}
    try:
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=10, read_timeout=15)
        identity = session.client("sts", config=cfg).get_caller_identity()
        if identity["Account"] != "357433444130" or not identity["Arn"].endswith(":user/kg-dev"):
            raise ValueError("KNOWN_DEVELOPER_REQUIRED")
        store = CloudDynamoRunStore(session.client("dynamodb", config=cfg), "kg-r72-query-state")
        ledger = Ledger(store)
        counters = {name: store.get(key) for name, key in (
            ("existing_run", ledger.run_key("external-pilot10-r72-v1")),
            ("proposed_run", ledger.run_key("collection316-serving-v1")),
            ("funding", ledger.external_funding_key("kg-r72")),
            ("month", ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m"))))}
        report["counters"] = counters
        function = session.client("lambda", config=cfg).get_function_configuration(FunctionName="kg-r72-query")
        report["existing_secret_arn"] = function["Environment"]["Variables"]["KG_DEEPSEEK_SECRET_ARN"]
        report["existing_web_code_sha256_b64"] = function["CodeSha256"]
        for stack in ("kg-r72-owned-runs", "kg-r72-tracer"):
            value = session.client("cloudformation", config=cfg).get_template(StackName=stack, TemplateStage="Original")["TemplateBody"]
            report[stack + "_template"] = json.loads(value) if isinstance(value, str) else value
        prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, "external-default",
                        now=datetime.now(UTC), live=True)
        report["provider"] = deepseek_preflight(OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}))
        if report["provider"]["metadata_sha256"] != routes.account_model_metadata_sha256:
            raise ValueError("PROVIDER_MODEL_METADATA_CHANGED")
        report["status"] = "PASS_READ_ONLY_NO_NEW_ALLOWANCE_OR_MODEL_CALL"
    except Exception as exc:
        report["status"] = "BLOCKED_READ_ONLY"
        report["error_category"] = exc.response["Error"]["Code"] if isinstance(exc, ClientError) else type(exc).__name__
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        output.write(canonical_json(report) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "writes", "model_calls") } | {
        "error_category": report.get("error_category"), "provider": report.get("provider")}))


if __name__ == "__main__":
    main()
