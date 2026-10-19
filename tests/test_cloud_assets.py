import io
import json
from pathlib import Path

import pytest
from aws_cdk import App
from aws_cdk.assertions import Template

from app.cloud_assets import download_checked, load_s3_snapshot, read_deepseek_secret
from app.contracts import sha256_bytes
from infra.runtime_preflight import RuntimePreflightStack


class Store:
    def __init__(self, values):
        self.values, self.calls = values, []

    def get_object(self, **kwargs):
        self.calls.append(kwargs["Key"])
        raw = self.values[kwargs["Key"]]
        return {"Body": io.BytesIO(raw), "ContentLength": len(raw)}


def test_checked_download_reuses_verified_cache_without_network(tmp_path):
    store = Store({"key": b"payload"})
    args = dict(bucket="fixture", key="key", target=tmp_path / "data", expected_sha256=sha256_bytes(b"payload"), expected_bytes=7)
    download_checked(store, **args)
    download_checked(store, **args)
    assert store.calls == ["key"]
    args["target"].write_bytes(b"changed")
    with pytest.raises(ValueError, match="CACHED_CHECKSUM"):
        download_checked(store, **args)
    assert store.calls == ["key"]


@pytest.mark.parametrize("kind", ["length", "checksum", "overrun"])
def test_partial_or_corrupt_assets_fail_closed_and_remove_partial_file(tmp_path, kind):
    store = Store({"key": b"wrong!!"})
    args = dict(bucket="fixture", key="key", target=tmp_path / "data", expected_sha256=sha256_bytes(b"payload"), expected_bytes=7)
    if kind == "length":
        args["expected_bytes"] = 8
    elif kind == "overrun":
        store.get_object = lambda **kw: {"Body": io.BytesIO(b"wrong!!!"), "ContentLength": 7}
    with pytest.raises(ValueError):
        download_checked(store, **args)
    assert not args["target"].exists() and not (tmp_path / "data.partial").exists()


@pytest.mark.parametrize("bad", ["prefix", "hash", "version", "tenant"])
def test_untrusted_snapshot_paths_rejected_before_io(tmp_path, bad):
    store = Store({})
    args = dict(bucket="fixture", prefix="tenants/demo/snapshots/snap_" + "0" * 64 + "/",
        manifest_sha256="0" * 64, manifest_bytes=100, snapshot_version="snap_" + "0" * 64,
        tenant_id="demo", cache_root=tmp_path)
    args[{"prefix": "prefix", "hash": "manifest_sha256", "version": "snapshot_version", "tenant": "tenant_id"}[bad]] = "../../outside"
    with pytest.raises(ValueError):
        load_s3_snapshot(store, **args)
    assert not store.calls


def test_synthetic_snapshot_cannot_be_loaded_as_live_demo(loaded, snapshot_dir, tmp_path):
    values = {p.name: p.read_bytes() for p in Path(snapshot_dir).iterdir() if p.is_file()}
    prefix = f"tenants/{loaded.manifest.tenant_id}/snapshots/{loaded.manifest.snapshot_version}/"
    store = Store({prefix + name: raw for name, raw in values.items()})
    with pytest.raises(ValueError, match="PURPOSE"):
        load_s3_snapshot(store, bucket="fixture", prefix=prefix,
            manifest_sha256=sha256_bytes(values["manifest.json"]), manifest_bytes=len(values["manifest.json"]),
            snapshot_version=loaded.manifest.snapshot_version, tenant_id=loaded.manifest.tenant_id, cache_root=tmp_path)


class SecretStore:
    def __init__(self, value, arn="test-arn"):
        self.value, self.arn = value, arn

    def get_secret_value(self, **kwargs):
        return {"ARN": self.arn, "SecretString": json.dumps(self.value)}


def test_secret_is_hidden_and_wrong_provider_shape_or_arn_rejected():
    fixture = {"schema_version": 1, "provider": "DEEPSEEK", "api_key": "synthetic-key-for-unit-test"}
    key = read_deepseek_secret(SecretStore(fixture), secret_arn="test-arn")
    assert key.get_secret_value() == fixture["api_key"] and fixture["api_key"] not in str(key)
    for value in (fixture | {"provider": "VOYAGE"}, fixture | {"schema_version": True}, fixture | {"extra": "x"}):
        with pytest.raises(ValueError):
            read_deepseek_secret(SecretStore(value), secret_arn="test-arn")
    with pytest.raises(ValueError):
        read_deepseek_secret(SecretStore(fixture, "wrong"), secret_arn="test-arn")


def test_preflight_iac_has_read_only_exact_assets_secret_and_no_public_or_model_permissions():
    artifacts = {"runtime_config": {"bucket": "approved-fixture-bucket"}, "package_key": "packages/test.zip",
                 "objects": [{"kind": "MODEL", "key": "models/fixed/model.onnx"},
                             {"kind": "SNAPSHOT", "key": "tenants/demo/snapshots/fixed/manifest.json"}]}
    stack = RuntimePreflightStack(App(), artifacts)
    raw = Template.from_stack(stack).to_json()
    resources = raw["Resources"]
    assert {r["Type"] for r in resources.values()} == {"AWS::IAM::Role", "AWS::Lambda::Function",
        "AWS::SecretsManager::Secret", "AWS::Logs::LogGroup"}
    secret = next(r for r in resources.values() if r["Type"] == "AWS::SecretsManager::Secret")
    assert secret["DeletionPolicy"] == "Retain" and "SecretString" not in secret["Properties"]
    role = next(r["Properties"] for r in resources.values() if r["Type"] == "AWS::IAM::Role")
    statements = role["Policies"][0]["PolicyDocument"]["Statement"]
    assert {action for s in statements for action in s["Action"]} == {
        "s3:GetObject", "secretsmanager:GetSecretValue", "logs:CreateLogStream", "logs:PutLogEvents"}
    assert not role.get("ManagedPolicyArns")
    fn = next(r["Properties"] for r in resources.values() if r["Type"] == "AWS::Lambda::Function")
    assert fn["Environment"]["Variables"]["MODEL_ENABLED"] == "false"
    assert fn["Timeout"] == 28 and fn["MemorySize"] == 1024
    assert "VpcConfig" not in fn and "ReservedConcurrentExecutions" not in fn
