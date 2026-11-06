"""Read-only account/bucket/stack readiness; never stage assets or execute CloudFormation."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from app.contracts import canonical_json
from infra.workbench_preflight import STACK_NAME

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("WORKBENCH_AWS_PREFLIGHT_RECEIPT_EXISTS")
    plan = json.loads((ROOT / "results/workbench-cloud/preflight-review-v1/plan.json").read_bytes())
    report = {"status": "NOT_CHECKED", "checked_at": datetime.now(UTC).isoformat(), "model_calls": 0,
        "writes": False, "cloud_deployed": False, "checks": {}}
    try:
        session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
        cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=8)
        identity = session.client("sts", config=cfg).get_caller_identity()
        if identity["Account"] != "357433444130" or not identity["Arn"].endswith(":user/kg-dev"):
            raise ValueError("KNOWN_DEVELOPER_ACCOUNT_REQUIRED")
        report["checks"]["developer_identity"] = "KNOWN_NON_ROOT_KG_DEV"
        cfn = session.client("cloudformation", config=cfg)
        foundation = cfn.describe_stacks(StackName="kg-r72-foundation")["Stacks"][0]
        outputs = {row["OutputKey"]: row["OutputValue"] for row in foundation["Outputs"]}
        if outputs["ArtifactBucketName"] != plan["artifacts"]["bucket"]:
            raise ValueError("FOUNDATION_BUCKET_NO_LONGER_MATCHES_PLAN")
        report["checks"]["foundation_bucket"] = "MATCHES_EXISTING_STACK_OUTPUT"
        try:
            cfn.describe_stacks(StackName=STACK_NAME)
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "ValidationError" or "does not exist" not in exc.response["Error"].get("Message", ""):
                raise
            report["checks"]["new_stack"] = "ABSENT_CREATE_ONLY_REVIEW"
        else:
            raise ValueError("PREFLIGHT_STACK_ALREADY_EXISTS_REVIEW_UPDATE_SEPARATELY")
        # Head an already approved public-model object; no object contents or credentials are exported.
        staged = json.loads((ROOT / "results/cloud_asset_staging.json").read_bytes())
        obj = next(row for row in staged["objects"] if row["kind"] == "MODEL")
        head = session.client("s3", config=cfg).head_object(Bucket=plan["artifacts"]["bucket"], Key=obj["key"])
        if head["ContentLength"] != obj["bytes"]:
            raise ValueError("EXISTING_BUCKET_OBJECT_CHANGED")
        report["checks"]["s3_existing_object"] = "HEAD_SIZE_MATCH"
        report["status"] = "PASS_READ_ONLY_CREATE_PREFLIGHT"
    except Exception as exc:
        code = exc.response["Error"]["Code"] if isinstance(exc, ClientError) else type(exc).__name__
        report.update(status="BLOCKED_READ_ONLY", error_category=code)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        output.write(canonical_json(report) + b"\n")
    print(json.dumps(report))
    return 0 if report["status"] == "PASS_READ_ONLY_CREATE_PREFLIGHT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
