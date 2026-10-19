import json
from datetime import UTC, datetime

import pytest
from botocore.exceptions import ClientError, ProfileNotFound

from app.preflight import CANDIDATES, probe_aws


class FakeSession:
    def __init__(self, *, root=False, denied=False):
        self.root, self.denied = root, denied
        self.calls = []

    def client(self, service, **kwargs):
        def invoke(operation, **arguments):
            self.calls.append((service, operation, arguments))
            if operation == "get_caller_identity":
                suffix = ":root" if self.root else ":user/kg-dev"
                return {"Arn": "arn:aws:iam::123456789012" + suffix}
            if self.denied:
                raise ClientError({"Error": {"Code": "AccessDeniedException",
                                              "Message": "private-credential-do-not-print"}}, operation)
            if operation == "get_account_plan_state":
                return {"accountId": "123456789012", "accountPlanType": "PAID",
                        "accountPlanStatus": "ACTIVE", "accountPlanRemainingCredits": {"amount": 100}}
            if operation == "get_foundation_model":
                return {"modelDetails": {"modelId": arguments["modelIdentifier"],
                        "modelArn": "arn:aws:bedrock:us-east-1:123456789012:private",
                        "modelLifecycle": {"status": "ACTIVE", "endOfLifeTime": datetime.now(UTC)}}}
            if operation == "get_foundation_model_availability":
                return {"modelId": arguments["modelId"], "authorizationStatus": "NOT_AUTHORIZED"}
            if operation == "list_inference_profiles":
                return {"inferenceProfileSummaries": [{"inferenceProfileId": "us.example",
                        "status": "ACTIVE", "models": [{"modelArn":
                        "arn:aws:bedrock:us-west-2::foundation-model/" + CANDIDATES[0]}]}]}
            if operation == "get_account_settings":
                return {"AccountLimit": {"ConcurrentExecutions": 10}}
            pytest.fail("Unexpected service operation")

        class Client:
            def __getattr__(self, operation):
                return lambda **arguments: invoke(operation, **arguments)

        return Client()


def test_read_only_preflight_redacts_identifiers_and_sdk_datetimes(monkeypatch):
    session = FakeSession()
    monkeypatch.setattr("app.preflight.boto3.Session", lambda **kwargs: session)
    result = probe_aws("kg-dev", "us-east-1")
    encoded = json.dumps(result)
    assert result["metadata_calls_attempted"] == len(session.calls) == 10
    assert result["status"] == "METADATA_READABLE"
    assert result["ready_for_billable_work"] is False
    assert "123456789012" not in encoded and "private" not in encoded
    assert result["inference_calls"] == result["writes_or_subscriptions"] == 0
    assert all(operation.startswith(("get_", "list_")) for _, operation, _ in session.calls)
    assert all(service != "bedrock-runtime" for service, _, _ in session.calls)


def test_root_identity_stops_other_probes(monkeypatch):
    session = FakeSession(root=True)
    monkeypatch.setattr("app.preflight.boto3.Session", lambda **kwargs: session)
    result = probe_aws("kg-dev", "us-east-1")
    assert result["status"] == "DEVELOPMENT_IDENTITY_REQUIRED"
    assert result["metadata_calls_attempted"] == len(session.calls) == 1


def test_access_denied_preserves_only_error_code(monkeypatch):
    session = FakeSession(denied=True)
    monkeypatch.setattr("app.preflight.boto3.Session", lambda **kwargs: session)
    result = probe_aws("kg-dev", "us-east-1")
    assert result["status"] == "INCOMPLETE"
    assert "private-credential" not in json.dumps(result)
    assert result["probes"]["account_plan"]["error_code"] == "AccessDeniedException"


def test_missing_profile_and_region_validation(monkeypatch):
    def missing(**kwargs):
        raise ProfileNotFound(profile="kg-dev-private")

    monkeypatch.setattr("app.preflight.boto3.Session", missing)
    assert probe_aws("kg-dev", "us-east-1")["probes"]["profile"]["error_code"] == "ProfileNotFound"
    with pytest.raises(ValueError):
        probe_aws("kg-dev", "other")
