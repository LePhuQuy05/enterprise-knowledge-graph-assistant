"""Prepare a local-only, native-verified preflight review. No upload/deploy/SDK calls."""

import argparse
import hashlib
import json
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from aws_cdk import App
from aws_cdk.assertions import Template

from app.contracts import canonical_json, sha256_bytes
from infra.workbench_preflight import WorkbenchPreflightStack, validate_artifacts

ROOT = Path(__file__).resolve().parents[1]


def synthesize(artifacts):
    with tempfile.TemporaryDirectory(prefix="kg-workbench-cdk-") as directory:
        return Template.from_stack(WorkbenchPreflightStack(App(outdir=directory, analytics_reporting=False), artifacts)).to_json()


def inspect_preflight_template(template, artifacts):
    validate_artifacts(artifacts)
    if canonical_json(template) != canonical_json(synthesize(artifacts)):
        raise ValueError("WORKBENCH_PREFLIGHT_TEMPLATE_DIFFERS_FROM_APPROVED_RESOURCE_CONTRACT")
    return {"status": "LOCAL_REVIEW_ONLY_NOT_STAGED", "resources": {k: v["Type"] for k, v in template["Resources"].items()},
        "public_routes": 0, "model_enabled": False, "shared_state_writes": False,
        "new_secret_or_bucket": False, "modified_existing_stacks": [], "cloud_deployed": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=ROOT / "results/workbench-cloud/preflight-review-v1")
    args = parser.parse_args(argv)
    directory = args.directory
    if (directory / "plan.json").exists():
        raise ValueError("WORKBENCH_PREFLIGHT_PLAN_EXISTS")
    assets = json.loads((ROOT / "results/workbench-cloud/assets-build-v1.json").read_bytes())
    native = json.loads((ROOT / "results/workbench-cloud/native-package-v1.json").read_bytes())
    staged = json.loads((ROOT / "results/cloud_asset_staging.json").read_bytes())
    if assets["status"] != "LOCAL_BUNDLE_VERIFIED" or native["status"] != "PASS_NATIVE_WORKBENCH_PACKAGE":
        raise ValueError("WORKBENCH_PREFLIGHT_REQUIRES_LOCAL_NATIVE_PROOF")
    package = Path(native["package_path"])
    with package.open("rb") as source:
        if hashlib.file_digest(source, "sha256").hexdigest() != native["package"]["sha256"]:
            raise ValueError("WORKBENCH_PREFLIGHT_PACKAGE_CHANGED")
    artifacts = {"bucket": staged["runtime_config"]["bucket"], "bundle_version": assets["bundle_version"],
        "bundle_manifest_sha256": assets["bundle_manifest_sha256"], "bundle_manifest_bytes": assets["bundle_manifest_bytes"],
        "package_sha256": native["package"]["sha256"]}
    template = synthesize(artifacts)
    review = inspect_preflight_template(template, artifacts)
    raw = canonical_json(template) + b"\n"
    plan = review | {"created_at": datetime.now(UTC).isoformat(), "template_sha256": sha256_bytes(raw), "artifacts": artifacts,
        "planned_upload_objects": assets["files"] + 2, "planned_upload_bytes": assets["total_bytes"] + assets["bundle_manifest_bytes"] + package.stat().st_size,
        "aws_calls_this_command": 0, "model_calls_this_command": 0, "maximum_future_smoke_invocations": 2,
        "actual_aws_identity_and_iam_verified": False, "existing_bucket_observation_date": "2026-10-19-or-earlier-local-receipt",
        "rollback_scope": "Delete only new preflight stack; explicitly reviewed cleanup for its content-addressed S3 objects; existing web/state unchanged",
        "next_gates": ["Fresh AWS identity/bucket/stack/capacity and regional cost check", "Approval of this concrete private preflight deployment",
            "Conditional S3 upload/readback and reviewed CloudFormation change set", "At most two model-free native AWS invocations + logs/hash audit"]}
    directory.mkdir(parents=True, exist_ok=True)
    for name, value in (("template.json", raw), ("plan.json", canonical_json(plan) + b"\n")):
        with (directory / name).open("xb") as output:
            output.write(value)
    print(json.dumps(review | {"directory": str(directory), "planned_upload_bytes": plan["planned_upload_bytes"]}))


if __name__ == "__main__":
    main()
