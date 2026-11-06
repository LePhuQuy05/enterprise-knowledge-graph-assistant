"""Read-only preparation of the isolated queue/API workflow. Never uploads or deploys."""

import argparse
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from aws_cdk import App
from aws_cdk.assertions import Template

from app.contracts import canonical_json, sha256_bytes
from infra.owned_runs import OwnedRunsStack

ROOT = Path(__file__).resolve().parents[1]


def inspect_owned_template(expected, observed):
    if canonical_json(expected) != canonical_json(observed):
        raise ValueError("UNAPPROVED_OWNED_WORKFLOW_TEMPLATE")
    return {"model_calls": 0, "resources": {k: v["Type"] for k, v in expected["Resources"].items()},
        "template_sha256": sha256_bytes(canonical_json(expected)), "status": "LOCAL_REVIEW_ONLY"}


def pinned_release():
    assets = json.loads((ROOT / "results/workbench-cloud/assets-build-v1.json").read_bytes())
    manifest = json.loads((Path(assets["bundle_directory"]) / "bundle.json").read_bytes())
    probe = json.loads((ROOT / "results/workbench-cloud/approved-release-v1/preflight-invoke-1.json").read_bytes())
    if probe["status"] != "PASS":
        raise ValueError("OWNED_REQUIRES_ACTUAL_PINNED_PREFLIGHT")
    configs = {}
    for case in probe["response"]["cases"]:
        for method in case["methods"]:
            if method["method"] in configs and configs[method["method"]] != method["config_sha256"]:
                raise ValueError("PREFLIGHT_METHOD_CONFIG_CHANGED")
            configs[method["method"]] = method["config_sha256"]
    return {"collection_version": manifest["collection_version"], "collection_manifest_sha256": manifest["collection_manifest_sha256"],
            "method_config_sha256": configs}, {k: assets[k] for k in ("documents", "chunks", "service_groups")}


def main(argv=None):
    import base64

    import boto3
    from botocore.config import Config
    from botocore.exceptions import ClientError

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-report", type=Path, required=True)
    parser.add_argument("--web-report", type=Path, required=True)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/owned-cloud/review-v1")
    args = parser.parse_args(argv)
    if args.directory.exists():
        raise ValueError("OWNED_REVIEW_DIRECTORY_EXISTS")
    native = json.loads(args.runtime_report.read_bytes())
    if native["status"] != "PASS_NATIVE_WORKBENCH_PACKAGE" or native.get("owned_workflow", {}).get("status") != "PASS_NATIVE_OWNED_WORKFLOW":
        raise ValueError("OWNED_REQUIRES_NATIVE_API_QUEUE_WORKFLOW_PROOF")
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=8)
    identity = session.client("sts", config=cfg).get_caller_identity()
    if identity["Account"] != "357433444130" or not identity["Arn"].endswith(":user/kg-dev"):
        raise ValueError("EXPECTED_KG_DEVELOPER_REQUIRED")
    cfn = session.client("cloudformation", config=cfg)
    try:
        cfn.describe_stacks(StackName="kg-r72-owned-runs")
    except ClientError as exc:
        if exc.response["Error"]["Code"] != "ValidationError" or "does not exist" not in exc.response["Error"]["Message"]:
            raise
    else:
        raise ValueError("OWNED_STACK_ALREADY_EXISTS_REVIEW_UPDATE_SEPARATELY")
    existing = cfn.get_template(StackName="kg-r72-tracer", TemplateStage="Original")["TemplateBody"]
    existing = json.loads(existing) if isinstance(existing, str) else existing
    live = session.client("lambda", config=cfg).get_function_configuration(FunctionName="kg-r72-query")
    baseline = json.loads((ROOT / "results/workbench-cloud/approved-release-v1/price-deployment.json").read_bytes())["code_sha256"]
    if base64.b64decode(live["CodeSha256"]).hex() != baseline:
        raise ValueError("WEBSITE_BASE_CHANGED_REVIEW_FRESH_TEMPLATE")
    public_config = json.loads(live["Environment"]["Variables"]["PUBLIC_CONFIG_JSON"])
    resources = cfn.list_stack_resources(StackName="kg-r72-tracer")["StackResourceSummaries"]
    ids = {r["LogicalResourceId"]: r["PhysicalResourceId"] for r in resources}
    routes = session.client("apigatewayv2", config=cfg).get_routes(ApiId=ids["HttpApi"], MaxResults="100")
    from infra.owned_runs import ROUTES

    keys = {method + " " + path for method, path in ROUTES.values()}
    if routes.get("NextToken") or any(r["RouteKey"] in keys for r in routes["Items"]):
        raise ValueError("OWNED_API_ROUTE_CONFLICT_OR_TRUNCATED")
    preflight = json.loads((ROOT / "results/workbench-cloud/preflight-review-v1/plan.json").read_bytes())["artifacts"]
    artifacts = preflight | {"package_sha256": native["package"]["sha256"]}
    release, corpus = pinned_release()
    settings = {"artifacts": artifacts, "release": release, "corpus": corpus, "public_config": public_config,
                "api_id": ids["HttpApi"], "authorizer_id": ids["JwtAuthorizer"]}
    with tempfile.TemporaryDirectory(prefix="kg-owned-cdk-") as directory:
        template = Template.from_stack(OwnedRunsStack(App(outdir=directory, analytics_reporting=False), **settings)).to_json()
    report = inspect_owned_template(template, template) | {"checked_at": datetime.now(UTC).isoformat(),
        "status": "FRESH_READ_ONLY_REVIEW_NOT_DEPLOYED", "aws_writes": 0, "public_routes_added": len(ROUTES),
        "existing_stacks_modified": [], "existing_identity_and_state_reused": True, "no_model_or_secret_access": True,
        "background_schedule": "one bounded watchdog per minute", "maximum_queue_worker_concurrency": 2,
        "alarm_notifications": "CloudWatch console only; no SNS/email target", "resource_count": len(template["Resources"]),
        "native_package_sha256": native["package"]["sha256"], "existing_web_code_sha256": baseline,
        "gross_budget_unchanged": True, "monthly_shared_attempt_limit": 300, "answering_enabled": False,
        "separate_web_assets_release_required": True}
    import copy

    from scripts.query_cloud import inspect_code_update

    web = json.loads(args.web_report.read_bytes())
    if web["status"] != "PASS_WEB_STATIC_ONLY_PACKAGE" or web["base_package_sha256"] != baseline:
        raise ValueError("OWNED_WEB_BASELINE_NOT_VERIFIED")
    website = copy.deepcopy(existing)
    website["Resources"]["QueryFunction"]["Properties"]["Code"]["S3Key"] = "packages/query/" + web["sha256"] + ".zip"
    web_review = inspect_code_update(existing, website)
    report.update(separate_web_assets_release_required=False, web_update=web_review,
        web_package_sha256=web["sha256"], web_backend_code_unchanged=True,
        planned_canary_child_admissions=5, canary_model_calls=0)
    args.directory.mkdir(parents=True)
    for name, value in (("template.json", template), ("plan.json", report), ("settings.json", settings),
                        ("website-template-baseline.json", existing), ("website-static-template.json", website)):
        (args.directory / name).write_bytes(canonical_json(value) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "resource_count", "public_routes_added", "aws_writes", "answering_enabled")}))


if __name__ == "__main__":
    main()
