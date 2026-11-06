import base64
import copy

import pytest

from scripts import release_funding_transition as release


@pytest.fixture(autouse=True)
def synthetic_release_plan(monkeypatch):
    # Scope validation must not require private templates or an operator's funding state.
    monkeypatch.setattr(release, "read", lambda name: {"proposed_s3_key": "packages/query/fixture.zip"}
                        if name == "release-plan.json" else (_ for _ in ()).throw(AssertionError(name)))


def templates():
    before = {"Resources": {
        "QueryFunction": {"Properties": {
            "Code": {"S3Bucket": release.BUCKET, "S3Key": "packages/query/previous.zip"},
            "Environment": {"Variables": {"MODEL_ENABLED": "true"}},
        }},
        "QueryRole": {"Properties": {"RoleName": "fixture-role"}},
    }, "Outputs": {}}
    after = copy.deepcopy(before)
    props = after["Resources"]["QueryFunction"]["Properties"]
    props["Code"]["S3Key"] = "packages/query/fixture.zip"
    props["Environment"]["Variables"]["KG_FUNDING_POLICY"] = release.TRANSITION_PROFILE
    return copy.deepcopy(before), copy.deepcopy(after)


def test_approved_code_and_flag_scope():
    release.inspect_templates(*templates())


@pytest.mark.parametrize("mutation", ["iam", "remove_flag", "other_environment", "outputs"])
def test_funding_patch_rejects_unapproved_scope(mutation):
    before, after = templates()
    if mutation == "iam":
        after["Resources"]["QueryRole"]["Properties"]["RoleName"] = "unexpected-role"
    elif mutation == "remove_flag":
        del after["Resources"]["QueryFunction"]["Properties"]["Environment"]["Variables"]["KG_FUNDING_POLICY"]
    elif mutation == "other_environment":
        after["Resources"]["QueryFunction"]["Properties"]["Environment"]["Variables"]["MODEL_ENABLED"] = "false"
    else:
        after["Outputs"]["Unexpected"] = {"Value": "changed"}
    with pytest.raises(ValueError, match="UNAPPROVED_TEMPLATE_DELTA"):
        release.inspect_templates(before, after)


@pytest.mark.parametrize("mutation", [None, "checksum", "length", "encryption", "metadata"])
def test_existing_s3_object_requires_server_checksum(tmp_path, mutation):
    package = tmp_path / "package.zip"
    package.write_bytes(b"sample-package")
    digest = release.sha256_bytes(package.read_bytes())
    plan = {"package_sha256": digest, "proposed_s3_key": "packages/query/" + digest + ".zip"}
    head = {"ChecksumSHA256": base64.b64encode(bytes.fromhex(digest)).decode(),
            "ContentLength": package.stat().st_size, "ServerSideEncryption": "AES256",
            "Metadata": {"sha256": digest}}
    if mutation == "checksum":
        head["ChecksumSHA256"] = "incorrect"
    elif mutation == "length":
        head["ContentLength"] += 1
    elif mutation == "encryption":
        del head["ServerSideEncryption"]
    elif mutation == "metadata":
        head["Metadata"] = {}
    class HeadOnly:
        def head_object(self, **kwargs):
            assert kwargs["ChecksumMode"] == "ENABLED"
            return head
    if mutation:
        with pytest.raises(ValueError, match="S3_CHECKSUM_LENGTH_METADATA_OR_ENCRYPTION"):
            release.stage_checksum_verified(HeadOnly(), package, plan)
    else:
        assert release.stage_checksum_verified(HeadOnly(), package, plan) == "EXISTING_S3_CHECKSUM_VERIFIED"
