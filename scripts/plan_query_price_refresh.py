"""Read-only review of the exact price ZIP against the current live template; no uploads or execution."""

import argparse
import base64
import copy
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config

from app.contracts import canonical_json, sha256_bytes
from scripts.query_cloud import inspect_code_update

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/workbench-cloud/price-refresh-review-v1")
    args = parser.parse_args(argv)
    if (args.directory / "plan.json").exists():
        raise ValueError("PRICE_REFRESH_PLAN_EXISTS")
    build = json.loads((ROOT / "results/workbench-cloud/query-price-refresh-v1.json").read_bytes())
    path = Path(build["package_path"])
    with path.open("rb") as data:
        if build["status"] != "PASS_EXACT_PRICE_REFRESH_NATIVE" or hashlib.file_digest(data, "sha256").hexdigest() != build["sha256"]:
            raise ValueError("PRICE_REFRESH_NATIVE_BUILD_NOT_VERIFIED")
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=8)
    identity = session.client("sts", config=cfg).get_caller_identity()
    if identity["Account"] != "357433444130" or not identity["Arn"].endswith(":user/kg-dev"):
        raise ValueError("PRICE_REFRESH_KNOWN_DEVELOPER_REQUIRED")
    cfn = session.client("cloudformation", config=cfg)
    stack = cfn.describe_stacks(StackName="kg-r72-tracer")["Stacks"][0]
    if stack["StackStatus"] != "UPDATE_COMPLETE":
        raise ValueError("PRICE_REFRESH_STACK_NOT_STABLE")
    before = cfn.get_template(StackName="kg-r72-tracer", TemplateStage="Original")["TemplateBody"]
    before = json.loads(before) if isinstance(before, str) else before
    live = session.client("lambda", config=cfg).get_function_configuration(FunctionName="kg-r72-query")
    if base64.b64decode(live["CodeSha256"]).hex() != build["base_package_sha256"]:
        raise ValueError("PRICE_REFRESH_LIVE_CODE_CHANGED_REBUILD_REVIEW")
    after = copy.deepcopy(before)
    after["Resources"]["QueryFunction"]["Properties"]["Code"]["S3Key"] = "packages/query/" + build["sha256"] + ".zip"
    review = inspect_code_update(before, after)
    cfn.validate_template(TemplateBody=json.dumps(after))
    plan = review | {"status": "FRESH_LIVE_READ_REVIEW_NOT_STAGED_OR_DEPLOYED", "checked_at": datetime.now(UTC).isoformat(),
        "new_package_sha256": build["sha256"], "base_package_sha256": build["base_package_sha256"],
        "price_sheet_expires_at": build["price_sheet_expires_at"], "zip_delta": build["delta"],
        "rollback_code": before["Resources"]["QueryFunction"]["Properties"]["Code"],
        "unmodified": ["Cognito accounts", "API URL/routes", "IAM", "Lambda environment/timeout/memory", "quotas/caps", "snapshot/history", "static UI"],
        "model_calls": 0, "aws_writes": 0, "cloud_deployed": False}
    args.directory.mkdir(parents=True, exist_ok=True)
    for name, data in (("previous-template.json", before), ("template.json", after), ("plan.json", plan)):
        with (args.directory / name).open("xb") as output:
            output.write(canonical_json(data) + b"\n")
    print(json.dumps({"status": plan["status"], "modified": review["modified"], "added": [], "removed": [],
        "template_sha256": sha256_bytes(canonical_json(after)), "model_calls": 0, "aws_writes": 0}))


if __name__ == "__main__":
    main()
