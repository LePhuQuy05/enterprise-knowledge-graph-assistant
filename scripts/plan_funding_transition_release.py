"""Read-only CloudFormation review of the exact two-file compatibility patch."""

import base64
import copy
import json
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.funding_policy import COMPATIBLE_AUTHORIZATION_ID, TRANSITION_PROFILE

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "results/funding-transition/review-v1"


def main():
    import boto3
    from botocore.config import Config

    build = json.loads((DIRECTORY / "plan.json").read_bytes())
    path = DIRECTORY / "query-funding-transition.zip"
    if sha256_bytes(path.read_bytes()) != build["proposed_zip_sha256"]:
        raise ValueError("FUNDING_RELEASE_PACKAGE_CHANGED")
    consumers = json.loads((DIRECTORY / "active-consumers.json").read_bytes())
    if any(r["owned_answer_policy_present"] for r in consumers["functions"]):
        raise ValueError("FUNDING_RELEASE_OTHER_ACTIVE_ANSWER_CONSUMER")
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    config = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    cfn = session.client("cloudformation", config=config)
    live = session.client("lambda", config=config).get_function_configuration(FunctionName="kg-r72-query")
    if live["CodeSha256"] != build["base_code_sha256_base64"] or live["RevisionId"] != build["revision_id"]:
        raise ValueError("FUNDING_RELEASE_LIVE_REVISION_CHANGED")
    if cfn.describe_stacks(StackName="kg-r72-tracer")["Stacks"][0]["StackStatus"] != "UPDATE_COMPLETE":
        raise ValueError("FUNDING_RELEASE_STACK_NOT_STABLE")
    before = cfn.get_template(StackName="kg-r72-tracer", TemplateStage="Original")["TemplateBody"]
    before = json.loads(before) if isinstance(before, str) else before
    after = copy.deepcopy(before)
    props = after["Resources"]["QueryFunction"]["Properties"]
    props["Code"]["S3Key"] = "packages/query/" + build["proposed_zip_sha256"] + ".zip"
    props.setdefault("Environment", {}).setdefault("Variables", {})["KG_FUNDING_POLICY"] = TRANSITION_PROFILE
    changed = [name for name in before["Resources"] if before["Resources"][name] != after["Resources"][name]]
    changed_props = [name for name in props if props[name] != before["Resources"]["QueryFunction"]["Properties"].get(name)]
    if changed != ["QueryFunction"] or sorted(changed_props) != ["Code", "Environment"]:
        raise ValueError("FUNDING_RELEASE_TEMPLATE_SCOPE")
    cfn.validate_template(TemplateBody=json.dumps(after))
    report = {"status": "FRESH_READ_ONLY_RELEASE_REVIEW_NOT_STAGED_OR_DEPLOYED", "checked_at": datetime.now(UTC).isoformat(),
        "stack": "kg-r72-tracer", "changed_resources": changed, "changed_properties": changed_props,
        "new_resources": [], "removed_resources": [], "changed_zip_entries": build["changed_zip_entries"],
        "package_sha256": build["proposed_zip_sha256"], "package_code_sha256_base64": base64.b64encode(bytes.fromhex(build["proposed_zip_sha256"])).decode(),
        "base_code_sha256_base64": live["CodeSha256"], "base_revision_id": live["RevisionId"],
        "proposed_s3_key": props["Code"]["S3Key"], "proposed_profile": TRANSITION_PROFILE,
        "template_sha256": sha256_bytes(canonical_json(after)), "active_cap_before_deploy_micro_usd": 2000000,
        "authorized_cap_after_verified_reader_micro_usd": 8000000, "allocation_audit_id": COMPATIBLE_AUTHORIZATION_ID,
        "month_cap_changes": False, "account_or_iam_changes": False, "ui_or_route_changes": False,
        "paid_model_calls_in_release_smoke": 0, "aws_writes": 0, "deployed": False,
        "rollback_rule": "Keep compatible reader while cap=8USD. Only lower cap if committed+reserved<=2USD; never reset spend to enable rollback.",
        "user_release_approval_required": True}
    for name, value in (("previous-template.json", before), ("proposed-template.json", after), ("release-plan.json", report)):
        output = DIRECTORY / name
        if output.exists():
            raise ValueError("FUNDING_RELEASE_REVIEW_EXISTS")
        output.write_bytes(canonical_json(value) + b"\n")
    print(json.dumps({k: report[k] for k in ("status", "changed_resources", "changed_properties", "aws_writes")}))


if __name__ == "__main__":
    main()
