import copy
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from app import workbench_preflight as runtime
from infra.workbench_preflight import validate_artifacts
from scripts.plan_workbench_preflight import inspect_preflight_template, synthesize


@pytest.fixture(scope="module")
def planned():
    pins = {"bucket": "fixture-private-bucket", "bundle_version": "wb_" + "a" * 64,
        "bundle_manifest_sha256": "b" * 64, "bundle_manifest_bytes": 8441, "package_sha256": "c" * 64}
    return pins, synthesize(pins)


def test_private_preflight_has_only_three_resources_and_no_inference_or_shared_write_permissions(planned):
    pins, template = planned
    review = inspect_preflight_template(template, pins)
    assert review["public_routes"] == 0 and not review["cloud_deployed"]
    assert set(template["Resources"]) == {"Logs", "Role", "Function"}
    function = template["Resources"]["Function"]["Properties"]
    assert function["Handler"] == "app.workbench_preflight.handler"
    assert function["Environment"]["Variables"]["MODEL_ENABLED"] == "false"
    assert function["MemorySize"] == 1024 and function["Timeout"] == 90
    assert function["EphemeralStorage"]["Size"] == 512
    assert "VpcConfig" not in function and "ReservedConcurrentExecutions" not in function
    statements = template["Resources"]["Role"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"]
    assert {action for row in statements for action in row["Action"]} == {"s3:GetObject", "logs:CreateLogStream", "logs:PutLogEvents"}


@pytest.mark.parametrize("corruption", ["iam", "model", "handler", "memory", "route", "code", "log"])
def test_review_rejects_unapproved_resource_and_permission_changes(planned, corruption, monkeypatch):
    pins, expected = planned
    changed = copy.deepcopy(expected)
    props = changed["Resources"]["Function"]["Properties"]
    if corruption == "iam":
        changed["Resources"]["Role"]["Properties"]["Policies"][0]["PolicyDocument"]["Statement"][0]["Action"] = ["s3:*"]
    elif corruption == "model":
        props["Environment"]["Variables"]["MODEL_ENABLED"] = "true"
    elif corruption == "handler":
        props["Handler"] = "app.query_handler.handler"
    elif corruption == "memory":
        props["MemorySize"] = 10240
    elif corruption == "route":
        changed["Resources"]["Public"] = {"Type": "AWS::Lambda::Url", "Properties": {"AuthType": "NONE"}}
    elif corruption == "code":
        props["Code"]["S3Key"] = "packages/unreviewed.zip"
    else:
        changed["Resources"]["Logs"]["Properties"]["RetentionInDays"] = 365
    monkeypatch.setattr("scripts.plan_workbench_preflight.synthesize", lambda _: expected)
    with pytest.raises(ValueError, match="RESOURCE_CONTRACT"):
        inspect_preflight_template(changed, pins)


@pytest.mark.parametrize("change", [{"bundle_version": "../escape"}, {"package_sha256": "wrong"},
    {"bundle_manifest_bytes": True}, {"secret_arn": "not-needed"}, {"bucket": "https://bad-host"}])
def test_artifact_pins_must_be_explicit_and_bounded(planned, change):
    pins, _ = planned
    with pytest.raises(ValueError):
        validate_artifacts(pins | change)


@pytest.mark.parametrize("event", [{}, {"operation": "GENERATE"}, {"operation": "RETRIEVAL_PREFLIGHT", "question": "arbitrary"}])
def test_function_is_not_a_general_query_or_worker_endpoint(event, monkeypatch):
    monkeypatch.setattr("boto3.client", lambda *a, **k: pytest.fail("Rejected event must not construct clients"))
    assert runtime.handler(event, None)["status"] == "REJECTED"


def test_model_disabled_and_remaining_time_are_checked_before_s3(monkeypatch):
    monkeypatch.setenv("MODEL_ENABLED", "true")
    monkeypatch.setattr("boto3.client", lambda *a, **k: pytest.fail("No model or S3 calls"))
    assert runtime.handler({"operation": "RETRIEVAL_PREFLIGHT"}, None)["status"] == "FAIL"
    monkeypatch.setenv("MODEL_ENABLED", "false")
    context = SimpleNamespace(get_remaining_time_in_millis=lambda: 64000)
    assert runtime.handler({"operation": "RETRIEVAL_PREFLIGHT"}, context)["status"] == "FAIL"


def test_function_composition_uses_only_s3_and_reports_safe_failure(monkeypatch):
    import json

    release = {"bucket": "fixture-private", "version": "wb_" + "a" * 64, "manifest_sha256": "b" * 64,
        "manifest_bytes": 123, "tenant_id": "demo"}
    monkeypatch.setenv("MODEL_ENABLED", "false")
    monkeypatch.setenv("KG_WORKBENCH_RELEASE_JSON", json.dumps(release))
    observed = []
    def client(name, **kwargs):
        observed.append(name)
        return "synthetic-s3-client"
    monkeypatch.setattr("boto3.client", client)
    @contextmanager
    def assets(candidate, **pins):
        assert candidate == "synthetic-s3-client" and pins["version"] == release["version"]
        yield "synthetic-loaded-assets"
    monkeypatch.setattr(runtime, "load_s3_workbench_assets", assets)
    monkeypatch.setattr(runtime, "exercise_retrieval", lambda *args: {"model_calls": 0, "quality_metrics": None})
    context = SimpleNamespace(get_remaining_time_in_millis=lambda: 90000)
    assert runtime.handler({"operation": "RETRIEVAL_PREFLIGHT"}, context)["status"] == "PASS"
    assert observed == ["s3"]
    def fail(*args):
        raise RuntimeError("SYNTHETIC_SECRET_THAT_MUST_NOT_LEAK")
    monkeypatch.setattr(runtime, "exercise_retrieval", fail)
    result = runtime.handler({"operation": "RETRIEVAL_PREFLIGHT"}, context)
    assert result["status"] == "FAIL" and "SYNTHETIC_SECRET" not in json.dumps(result)
