"""Stage code, review a constrained update, publish a validated release; no inference."""

import argparse
import base64
import copy
import hashlib
import json
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.ledger import Change, DynamoLedgerStore
from scripts.foundation import _cfn, _session, _wait_change_set
from scripts.stage_cloud_assets import upload_and_verify
from scripts.tracer import assert_no_cycles

ROOT = Path(__file__).resolve().parents[1]
STACK_NAME = "kg-r72-tracer"
OUTPUT = ROOT / "results/query-cloud"
MODIFIED = {"QueryFunction", "QueryRole", "DefaultStage"}
ADDED = {"PrivateQueryCreate", "PrivateQueryList", "PrivateQueryDetail", "PrivateSource"}


def save(name, data):
    OUTPUT.mkdir(parents=True, exist_ok=True)
    previous = OUTPUT / name
    if name in {"plan.json", "deployment.json"} and previous.exists():
        old = json.loads(previous.read_bytes())
        if old.get("template_sha256"):
            history = OUTPUT / "history"
            history.mkdir(parents=True, exist_ok=True)
            (history / (Path(name).stem + "_" + old["template_sha256"][:12] + ".json")).write_text(
                json.dumps(old, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / name).write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def stage(session):
    source = json.loads((ROOT / "results/cloud_asset_staging.json").read_bytes())
    build = json.loads((ROOT / "results/query_package_build.json").read_bytes())
    probe = json.loads((ROOT / "results/runtime-preflight/probe.json").read_bytes())
    if source["status"] != "STAGED_READBACK_PASS" or probe["status"] != "PASS" or build["status"] != "PASS":
        raise ValueError("VERIFIED_NATIVE_RUNTIME_AND_ASSETS_REQUIRED")
    path = Path(build["package_path"])
    with path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != build["sha256"]:
            raise ValueError("PACKAGE_CHANGED")
    row = {"kind": "PACKAGE", "key": "packages/query/" + build["sha256"] + ".zip", "path": str(path),
           "bytes": path.stat().st_size, "sha256": build["sha256"]}
    state = upload_and_verify(session.client("s3"), row)
    data = copy.deepcopy(source)
    data.update(checked_at=datetime.now(UTC).isoformat(), package_key=row["key"],
                objects=[r for r in source["objects"] if r["kind"] != "PACKAGE"] + [row])
    data["uploads"] = [r for r in source["uploads"] if not r["key"].startswith("packages/")] + [
        {"key": row["key"], "sha256": row["sha256"], "status": state}]
    data["total_bytes"] = sum(r["bytes"] for r in data["objects"])
    return save("assets.json", data)


def synthesize():
    from aws_cdk import App

    from infra.tracer import TracerStack

    assets = json.loads((OUTPUT / "assets.json").read_bytes())
    secret = json.loads((ROOT / "results/runtime-preflight/secret_setup.json").read_bytes())
    if assets["status"] != "STAGED_READBACK_PASS" or secret["status"] not in {
        "INITIALIZED_AND_READBACK_VERIFIED", "EXISTING_MATCH_VERIFIED"
    }:
        raise ValueError("VERIFIED_STAGED_CODE_AND_SECRET_REQUIRED")
    app = App(outdir=str(OUTPUT / "cdk"), analytics_reporting=False)
    TracerStack(app, runtime={"assets": assets, "secret_arn": secret["arn"]})
    assembly = app.synth()
    path = Path(assembly.directory) / (STACK_NAME + ".template.json")
    return path, json.loads(path.read_bytes())


def inspect_update(before, after):
    old, new = before["Resources"], after["Resources"]
    if set(old) - set(new) or set(new) - set(old) != ADDED:
        raise ValueError("ONLY_FOUR_APPROVED_ROUTES_MAY_BE_ADDED")
    changed = {name for name in old if canonical_json(old[name]) != canonical_json(new[name])}
    if not changed <= MODIFIED or "QueryFunction" not in changed:
        raise ValueError("UPDATE_MUST_PRESERVE_COGNITO_API_AND_UNRELATED_RESOURCES")
    if canonical_json(before.get("Outputs")) != canonical_json(after.get("Outputs")):
        raise ValueError("PUBLIC_IDENTITIES_AND_OUTPUTS_MUST_NOT_CHANGE")
    assert_no_cycles(new)
    for name in ADDED:
        props = new[name]["Properties"]
        if new[name]["Type"] != "AWS::ApiGatewayV2::Route" or props["AuthorizationType"] != "JWT" or (
            props["AuthorizationScopes"] != ["kg/query" if name == "PrivateQueryCreate" else "kg/read"]
        ):
            raise ValueError("NEW_ROUTES_MUST_REQUIRE_CORRECT_JWT_SCOPES")
    return {"modified": sorted(changed), "added": sorted(ADDED), "removed": [],
            "template_sha256": sha256_bytes(canonical_json(after)), "paid_model_calls": 0}


def inspect_code_update(before, after):
    old, new = before["Resources"], after["Resources"]
    if set(old) != set(new) or canonical_json(before.get("Outputs")) != canonical_json(after.get("Outputs")):
        raise ValueError("CODE_UPDATE_CANNOT_CHANGE_RESOURCES_OR_IDENTITIES")
    changed = {name for name in old if canonical_json(old[name]) != canonical_json(new[name])}
    if changed != {"QueryFunction"}:
        raise ValueError("CODE_UPDATE_MAY_ONLY_CHANGE_QUERY_FUNCTION_CODE")
    left, right = copy.deepcopy(old["QueryFunction"]), copy.deepcopy(new["QueryFunction"])
    old_code = left["Properties"].pop("Code")
    new_code = right["Properties"].pop("Code")
    if canonical_json(left) != canonical_json(right) or old_code.get("S3Bucket") != new_code.get("S3Bucket") or (
        set(new_code) != {"S3Bucket", "S3Key"}
    ):
        raise ValueError("CODE_UPDATE_CANNOT_CHANGE_ENVIRONMENT_ROLE_OR_BUCKET")
    return {"modified": ["QueryFunction"], "added": [], "removed": [],
            "template_sha256": sha256_bytes(canonical_json(after)), "paid_model_calls": 0}


def inspect_changes(response, *, code_only=False):
    if response.get("StackName") != STACK_NAME or response.get("Status") != "CREATE_COMPLETE" or (
        response.get("ExecutionStatus") != "AVAILABLE" or response.get("NextToken")
    ):
        raise ValueError("CHANGE_SET_NOT_READY")
    changes = [c["ResourceChange"] for c in response["Changes"]]
    actual_add = {c["LogicalResourceId"] for c in changes if c["Action"] == "Add"}
    actual_modify = {c["LogicalResourceId"] for c in changes if c["Action"] == "Modify"}
    expected_add = set() if code_only else ADDED
    allowed_modify = {"QueryFunction", "LambdaIntegration"} if code_only else MODIFIED | {"LambdaIntegration"}
    if actual_add != expected_add or not actual_modify <= allowed_modify or any(
        c["Action"] not in {"Add", "Modify"} or (c["Action"] == "Modify" and c.get("Replacement") != "False") for c in changes
    ):
        raise ValueError("UNAPPROVED_CHANGE_OR_REPLACEMENT")
    for change in changes:
        if change["LogicalResourceId"] == "LambdaIntegration":
            if change.get("ResourceType") != "AWS::ApiGatewayV2::Integration" or not change.get("Details") or any(
                d.get("CausingEntity") != "QueryFunction.Arn" or d.get("Target", {}).get("Name") != "IntegrationUri"
                or d.get("Target", {}).get("RequiresRecreation") != "Never" for d in change["Details"]
            ):
                raise ValueError("UNAPPROVED_INTEGRATION_REEVALUATION")


def deployment_plan(session, *, code_only=False):
    path, template = synthesize()
    client = _cfn(session)
    stack = client.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
    if stack["StackStatus"] not in {"CREATE_COMPLETE", "UPDATE_COMPLETE"}:
        raise ValueError("EXISTING_STACK_NOT_STABLE")
    before = client.get_template(StackName=STACK_NAME, TemplateStage="Original")["TemplateBody"]
    if isinstance(before, str):
        before = json.loads(before)
    save("previous_template.json", before)
    report = (inspect_code_update if code_only else inspect_update)(before, template)
    client.validate_template(TemplateBody=json.dumps(template))
    result = client.create_change_set(StackName=STACK_NAME, ChangeSetName="query-serving-" + uuid.uuid4().hex,
        ChangeSetType="UPDATE", TemplateBody=json.dumps(template), Capabilities=["CAPABILITY_NAMED_IAM"],
        Description="Four authenticated query routes and constrained serving runtime; no identity replacement")
    ready = _wait_change_set(client, result["Id"], stack_name=STACK_NAME)
    inspect_changes(ready, code_only=code_only)
    report.update(status="REVIEWED_NOT_EXECUTED", change_set_id=result["Id"], template_path=str(path),
                  checked_at=datetime.now(UTC).isoformat(), update_kind="CODE_ONLY" if code_only else "QUERY_INITIAL")
    return save("plan.json", report)


def deployment_apply(session):
    report = json.loads((OUTPUT / "plan.json").read_bytes())
    client = _cfn(session)
    change = client.describe_change_set(StackName=STACK_NAME, ChangeSetName=report["change_set_id"])
    code_only = report.get("update_kind") == "CODE_ONLY"
    inspect_changes(change, code_only=code_only)
    remote = client.get_template(StackName=STACK_NAME, ChangeSetName=report["change_set_id"], TemplateStage="Original")["TemplateBody"]
    if isinstance(remote, str):
        remote = json.loads(remote)
    if sha256_bytes(canonical_json(remote)) != report["template_sha256"]:
        raise ValueError("REVIEWED_TEMPLATE_CHANGED")
    before = json.loads((OUTPUT / "previous_template.json").read_bytes())
    (inspect_code_update if code_only else inspect_update)(before, remote)
    client.execute_change_set(StackName=STACK_NAME, ChangeSetName=report["change_set_id"], ClientRequestToken=uuid.uuid4().hex)
    for _ in range(100):
        stack = client.describe_stacks(StackName=STACK_NAME)["Stacks"][0]
        status = stack["StackStatus"]
        if status == "UPDATE_COMPLETE":
            return save("deployment.json", {"status": status, "stack_name": STACK_NAME,
                "template_sha256": report["template_sha256"], "checked_at": datetime.now(UTC).isoformat(),
                "outputs": {r["OutputKey"]: r["OutputValue"] for r in stack["Outputs"]}, "paid_model_calls": 0})
        if status not in {"UPDATE_IN_PROGRESS", "UPDATE_COMPLETE_CLEANUP_IN_PROGRESS"}:
            raise ValueError("UPDATE_NOT_COMPLETE_CHECK_EVENTS_BEFORE_ANY_RETRY")
        time.sleep(2)
    raise ValueError("UPDATE_PENDING_DO_NOT_REPEAT_EXECUTION")


def publish(session):
    deployment = json.loads((OUTPUT / "deployment.json").read_bytes())
    assets = json.loads((OUTPUT / "assets.json").read_bytes())
    if deployment["status"] != "UPDATE_COMPLETE" or assets["status"] != "STAGED_READBACK_PASS":
        raise ValueError("DEPLOYMENT_AND_STAGING_REQUIRED")
    config = assets["runtime_config"]
    function = session.client("lambda").get_function_configuration(FunctionName="kg-r72-query")
    expected = assets["package_key"].rsplit("/", 1)[-1].removesuffix(".zip")
    if base64.b64decode(function["CodeSha256"], validate=True).hex() != expected or (
        function.get("State") != "Active" or function.get("LastUpdateStatus") != "Successful"
        or function.get("Handler") != "app.query_handler.handler"
        or function.get("Environment", {}).get("Variables", {}).get("MODEL_ENABLED") != "true"
    ):
        raise ValueError("DEPLOYED_QUERY_CODE_OR_CONFIGURATION_CHANGED")
    if config["tenant_id"] != "demo":
        raise ValueError("ONLY_APPROVED_DEMO_TENANT_ALLOWED")
    store = DynamoLedgerStore(session.client("dynamodb"), "kg-r72-metadata")
    pk, now = "TENANT#demo", datetime.now(UTC).isoformat()
    pointer_key, metadata_key = (pk, "RELEASE"), (pk, "SNAPSHOT#" + config["snapshot_version"])
    pointer, metadata = store.get(pointer_key), store.get(metadata_key)
    if pointer or metadata:
        if not pointer or not metadata or pointer.get("snapshot_version") != config["snapshot_version"] or (
            pointer.get("manifest_sha256") != config["snapshot_manifest_sha256"]
            or metadata.get("manifest_sha256") != config["snapshot_manifest_sha256"]
            or metadata.get("publication_status") != "PUBLISHED" or metadata.get("revoked") is not False
            or metadata.get("valid_for_history") is not True or metadata.get("asset_config") != config
        ):
            raise ValueError("EXISTING_RELEASE_CONFLICT_DO_NOT_OVERWRITE_OR_REVIVE")
        return save("publication.json", {"status": "EXISTING_RELEASE_VERIFIED", "snapshot_version": config["snapshot_version"],
                                        "generation": pointer["generation"], "paid_model_calls": 0})
    common = {"PK": pk, "version": 1, "schema_version": 1, "tenant_id": "demo",
              "snapshot_version": config["snapshot_version"], "manifest_sha256": config["snapshot_manifest_sha256"],
              "published_at": now}
    metadata = common | {"SK": metadata_key[1], "record_type": "SNAPSHOT_RELEASE", "publication_status": "PUBLISHED",
                          "valid_for_history": True, "revoked": False, "asset_config": config}
    pointer = common | {"SK": pointer_key[1], "record_type": "RELEASE_POINTER", "generation": 1}
    store.commit([Change(metadata_key, None, metadata), Change(pointer_key, None, pointer)])
    if store.get(pointer_key) != pointer or store.get(metadata_key) != metadata:
        raise ValueError("PUBLICATION_READBACK_MISMATCH")
    return save("publication.json", {"status": "PUBLISHED_READBACK_PASS", "snapshot_version": config["snapshot_version"],
                                    "generation": 1, "checked_at": now, "paid_model_calls": 0})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["stage", "plan", "code-plan", "apply", "publish"])
    args = parser.parse_args()
    try:
        action = {"stage": stage, "plan": deployment_plan, "code-plan": lambda s: deployment_plan(s, code_only=True),
                  "apply": deployment_apply, "publish": publish}[args.command]
        result = action(_session("kg-dev"))
        print(json.dumps({k: v for k, v in result.items() if k not in {"objects", "uploads", "runtime_config"}}, indent=2))
        return 0
    except Exception as exc:
        result = {"status": "INCOMPLETE", "error_category": type(exc).__name__, "paid_model_calls": 0}
        if isinstance(exc, ValueError) and str(exc).isupper() and len(str(exc)) < 110:
            result["safe_error"] = str(exc)
        if hasattr(exc, "response"):
            result["service_error_code"] = exc.response.get("Error", {}).get("Code")
        print(json.dumps(result))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
