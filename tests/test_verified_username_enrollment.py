"""Reproduce the confirmed-user login failure through the real authentication seam."""

from types import SimpleNamespace

import pytest
from botocore.exceptions import ClientError

from app.account_registration import VerifiedEmailEnrollment
from app.ledger import MemoryLedgerStore
from app.query_auth import ApiError, authenticate

SUB = "11111111-1111-4111-8111-111111111111"
USERNAME = "new.reviewer"
CONFIG = {"issuer": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_Fixture", "client_id": "fixture"}


def verified_profile(sub=SUB):
    return {"Username": USERNAME, "Enabled": True, "UserStatus": "CONFIRMED", "UserAttributes": [
        {"Name": "sub", "Value": sub}, {"Name": "email", "Value": "reviewer@example.invalid"},
        {"Name": "email_verified", "Value": "true"}]}


def event(username=USERNAME):
    return {"requestContext": {"authorizer": {"jwt": {"claims": {
        "token_use": "access", "iss": CONFIG["issuer"], "client_id": CONFIG["client_id"],
        "sub": SUB, "username": username, "exp": "2000", "scope": "kg/read"}}}}}


def test_confirmed_verified_username_account_can_complete_first_login():
    calls = []

    def get(**kwargs):
        calls.append(kwargs)
        # Matches the observed Cognito pool: a subject UUID is not the username lookup key.
        if kwargs["Username"] != USERNAME:
            raise ClientError({"Error": {"Code": "UserNotFoundException", "Message": "fixture"}}, "AdminGetUser")
        return verified_profile()

    store = MemoryLedgerStore()
    provision = VerifiedEmailEnrollment(SimpleNamespace(admin_get_user=get), store, CONFIG)
    auth = authenticate(event(), public_config=CONFIG, metadata_store=store,
                        required_scope="kg/read", now=1000, enroll_member=provision)
    assert auth.user_id == SUB and auth.tenant_id == "demo"
    assert calls == [{"UserPoolId": "us-east-1_Fixture", "Username": USERNAME}]
    row = store.get(("USER#" + SUB, "MEMBERSHIP"))
    assert row["status"] == "ACTIVE"
    assert not {"username", "email", "password", "token"} & row.keys()


@pytest.mark.parametrize("username", [None, "", " ", 17, "x" * 129])
def test_missing_or_invalid_verified_username_never_uses_request_input(username):
    calls = []
    store = MemoryLedgerStore()
    provision = VerifiedEmailEnrollment(SimpleNamespace(admin_get_user=lambda **kw: calls.append(kw)), store, CONFIG)
    request = event(username)
    request["headers"] = {"username": USERNAME}
    request["body"] = '{"username":"new.reviewer"}'
    with pytest.raises(ApiError, match="ACCESS_TOKEN_REQUIRED"):
        authenticate(request, public_config=CONFIG, metadata_store=store,
                     required_scope="kg/read", now=1000, enroll_member=provision)
    assert calls == [] and store.get(("USER#" + SUB, "MEMBERSHIP")) is None


def test_valid_username_resolving_another_subject_never_enrolls():
    store = MemoryLedgerStore()
    other = "22222222-2222-4222-8222-222222222222"
    provision = VerifiedEmailEnrollment(SimpleNamespace(admin_get_user=lambda **kw: verified_profile(other)), store, CONFIG)
    with pytest.raises(ApiError, match="ACCOUNT_VERIFICATION_REQUIRED"):
        authenticate(event(), public_config=CONFIG, metadata_store=store,
                     required_scope="kg/read", now=1000, enroll_member=provision)
    assert store.get(("USER#" + SUB, "MEMBERSHIP")) is None
    assert store.get(("USER#" + other, "MEMBERSHIP")) is None


def test_existing_email_username_identity_still_enrolls_with_same_subject():
    calls = []
    store = MemoryLedgerStore()
    def get(**kwargs):
        calls.append(kwargs)
        return verified_profile()
    provision = VerifiedEmailEnrollment(SimpleNamespace(admin_get_user=get), store, CONFIG)
    auth = authenticate(event("old@example.invalid"), public_config=CONFIG, metadata_store=store,
                        required_scope="kg/read", now=1000, enroll_member=provision)
    assert calls[0]["Username"] == "old@example.invalid" and auth.user_id == SUB
