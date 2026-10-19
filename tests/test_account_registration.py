import copy
import threading
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.account_registration import VerifiedEmailEnrollment
from app.ledger import Change, MemoryLedgerStore
from app.query_auth import ApiError, authenticate, membership_context

SUB = "11111111-1111-4111-8111-111111111111"
OTHER = "22222222-2222-4222-8222-222222222222"
CONFIG = {"issuer": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_Fixture", "client_id": "fixture"}


def profile():
    return {"Enabled": True, "UserStatus": "CONFIRMED", "UserAttributes": [
        {"Name": "sub", "Value": SUB}, {"Name": "email", "Value": "reviewer@example.invalid"},
        {"Name": "email_verified", "Value": "true"}]}


def enroll(store, response=None):
    calls = []
    def get(**kw):
        calls.append(kw)
        return copy.deepcopy(profile() if response is None else response)
    return VerifiedEmailEnrollment(SimpleNamespace(admin_get_user=get), store, CONFIG), calls


def claims(**overrides):
    return {"requestContext": {"authorizer": {"jwt": {"claims": {
        "token_use": "access", "iss": CONFIG["issuer"], "client_id": "fixture", "sub": SUB,
        "exp": "2000", "scope": "kg/read", "username": "fixture.user", **overrides}}}}}


def test_verified_identity_enrolls_once_without_persisting_email_or_token():
    store = MemoryLedgerStore()
    provision, calls = enroll(store)
    for _ in range(2):
        auth = authenticate(claims(), public_config=CONFIG, metadata_store=store,
                            required_scope="kg/read", now=1000, enroll_member=provision)
        assert auth.user_id == SUB and auth.tenant_id == "demo"
    assert calls == [{"UserPoolId": "us-east-1_Fixture", "Username": "fixture.user"}]
    row = store.get(("USER#" + SUB, "MEMBERSHIP"))
    assert row["version"] == 1 and row["roles"] == ["reviewer"]
    assert not ({"email", "password", "token", "access_token"} & row.keys())
    assert store.get(("USER#" + OTHER, "MEMBERSHIP")) is None


@pytest.mark.parametrize("changed", [{"Enabled": False}, {"UserStatus": "UNCONFIRMED"},
    {"UserStatus": "FORCE_CHANGE_PASSWORD"}, {"UserAttributes": []},
    {"UserAttributes": [{"Name": "sub", "Value": OTHER}, {"Name": "email_verified", "Value": "true"},
                        {"Name": "email", "Value": "x@example.invalid"}]},
    {"UserAttributes": [{"Name": "sub", "Value": SUB}, {"Name": "email_verified", "Value": "false"},
                        {"Name": "email", "Value": "x@example.invalid"}]}])
def test_unverified_disabled_or_mismatched_profile_never_writes(changed):
    store = MemoryLedgerStore()
    provision, _ = enroll(store, profile() | changed)
    with pytest.raises(ApiError, match="ACCOUNT_VERIFICATION_REQUIRED"):
        membership_context(store, SUB, enroll_member=provision, cognito_username="fixture.user")
    assert store.get(("USER#" + SUB, "MEMBERSHIP")) is None


@pytest.mark.parametrize("changed", [{"sub": "../../other"}, {"token_use": "id"}, {"iss": "https://evil.invalid"},
    {"client_id": "other"}, {"exp": "900"}, {"scope": "openid"}])
def test_bad_authorizer_claims_cannot_trigger_enrollment(changed):
    store = MemoryLedgerStore()
    provision, calls = enroll(store)
    with pytest.raises(ApiError):
        authenticate(claims(**changed), public_config=CONFIG, metadata_store=store,
                     required_scope="kg/read", now=1000, enroll_member=provision)
    assert calls == []


@pytest.mark.parametrize("status", ["DISABLED", "ACTIVE"])
def test_existing_invalid_membership_never_reactivated(status):
    store = MemoryLedgerStore()
    key = ("USER#" + SUB, "MEMBERSHIP")
    row = {"PK": key[0], "SK": key[1], "version": 1, "record_type": "MEMBERSHIP", "schema_version": 1,
           "status": status, "tenant_id": "demo", "roles": ["blocked"]}
    store.commit([Change(key, None, row)])
    provision, calls = enroll(store)
    with pytest.raises(ApiError, match="ACTIVE_MEMBERSHIP_REQUIRED"):
        membership_context(store, SUB, enroll_member=provision, cognito_username="fixture.user")
    assert calls == [] and store.get(key) == row


def test_concurrent_first_login_creates_one_identity_without_overwrite():
    barrier = threading.Barrier(2)
    store = MemoryLedgerStore()
    def get(**kw):
        barrier.wait(timeout=5)
        return profile()
    provision = VerifiedEmailEnrollment(SimpleNamespace(admin_get_user=get), store, CONFIG)
    with ThreadPoolExecutor(max_workers=2) as pool:
        users = list(pool.map(lambda _: membership_context(store, SUB, enroll_member=provision,
                                                          cognito_username="fixture.user"), range(2)))
    assert all(u.user_id == SUB for u in users)
    assert store.get(("USER#" + SUB, "MEMBERSHIP"))["version"] == 1


def test_enrollment_failure_is_retryable_status_setup_without_creating_membership():
    store = MemoryLedgerStore()
    def fail(**kw):
        raise TimeoutError("sensitive provider message")
    provision = VerifiedEmailEnrollment(SimpleNamespace(admin_get_user=fail), store, CONFIG)
    with pytest.raises(ApiError, match="ACCOUNT_SETUP_UNAVAILABLE") as error:
        membership_context(store, SUB, enroll_member=provision, cognito_username="fixture.user")
    assert error.value.status == 503 and "sensitive" not in str(error.value)
    assert store.get(("USER#" + SUB, "MEMBERSHIP")) is None


def test_admin_disable_wins_race_with_enrollment():
    store = MemoryLedgerStore()
    original = store.commit
    def race(changes):
        disabled = changes[0].item | {"status": "DISABLED"}
        original([Change(changes[0].key, None, disabled)])
        original(changes)
    store.commit = race
    provision, _ = enroll(store)
    with pytest.raises(ApiError, match="ACTIVE_MEMBERSHIP_REQUIRED"):
        membership_context(store, SUB, enroll_member=provision, cognito_username="fixture.user")


def test_enrollment_requires_pinned_cognito_issuer():
    with pytest.raises(ValueError, match="POOL_CONFIGURATION"):
        VerifiedEmailEnrollment(None, None, {"issuer": "https://attacker.invalid/pool"})
