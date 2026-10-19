"""First-login enrollment for a verified Cognito identity; never store credentials."""

import re
from datetime import UTC, datetime

from botocore.exceptions import ClientError

from app.ledger import Change, ConcurrentWrite
from app.query_auth import SUB, ApiError


class VerifiedEmailEnrollment:
    def __init__(self, cognito, metadata_store, public_config):
        self.cognito, self.store = cognito, metadata_store
        match = re.fullmatch(r"https://cognito-idp\.us-east-1\.amazonaws\.com/(us-east-1_[a-zA-Z0-9]+)",
                             public_config.get("issuer", ""))
        if not match:
            raise ValueError("ENROLLMENT_POOL_CONFIGURATION_INVALID")
        self.pool_id = match[1]

    def __call__(self, sub, *, cognito_username=None):
        # Called only after API Gateway JWT and application scope/identity validation.
        if type(sub) is not str or not re.fullmatch(SUB, sub):
            raise ApiError(401, "ACCESS_TOKEN_REQUIRED")
        # This claim comes only from the already-verified access token, never request input.
        # A pool with separate usernames need not resolve a subject UUID as a username.
        if type(cognito_username) is not str or not re.fullmatch(r"\S{1,128}", cognito_username):
            raise ApiError(401, "ACCESS_TOKEN_REQUIRED")
        try:
            profile = self.cognito.admin_get_user(UserPoolId=self.pool_id, Username=cognito_username)
        except ClientError as exc:
            if exc.response["Error"]["Code"] == "UserNotFoundException":
                raise ApiError(403, "ACCOUNT_VERIFICATION_REQUIRED") from None
            raise ApiError(503, "ACCOUNT_SETUP_UNAVAILABLE") from None
        except Exception:
            raise ApiError(503, "ACCOUNT_SETUP_UNAVAILABLE") from None
        attrs = {a["Name"]: a["Value"] for a in profile.get("UserAttributes", [])}
        if profile.get("Enabled") is not True or profile.get("UserStatus") != "CONFIRMED" or (
            attrs.get("sub") != sub or attrs.get("email_verified") != "true" or not attrs.get("email")
        ):
            raise ApiError(403, "ACCOUNT_VERIFICATION_REQUIRED")
        key = ("USER#" + sub, "MEMBERSHIP")
        stamp = datetime.now(UTC).isoformat()
        row = {"PK": key[0], "SK": key[1], "record_type": "MEMBERSHIP", "schema_version": 1,
               "version": 1, "tenant_id": "demo", "status": "ACTIVE", "roles": ["reviewer"],
               "created_at": stamp, "updated_at": stamp, "enrollment": "COGNITO_VERIFIED_EMAIL_V1"}
        try:
            self.store.commit([Change(key, None, row)])
        except ConcurrentWrite:
            # Concurrent login or an administrative disable wins; caller reads and validates again.
            pass
        except Exception:
            raise ApiError(503, "ACCOUNT_SETUP_UNAVAILABLE") from None
