"""API errors and server-derived identity after API Gateway signature verification."""

import re
import time
from dataclasses import dataclass

SUB = r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}"
TENANT = r"[a-z][a-z0-9_-]{0,63}"


class ApiError(ValueError):
    def __init__(self, status, code, *, request_id=None):
        self.status, self.code, self.request_id = status, code, request_id
        super().__init__(code)

    def body(self):
        return {"error": {"code": self.code, "message": self.code.replace("_", " ").lower(),
                          "retryable": False}, "request_id": self.request_id}


@dataclass(frozen=True)
class AuthContext:
    tenant_id: str
    user_id: str

    def __post_init__(self):
        if not re.fullmatch(TENANT, self.tenant_id) or not re.fullmatch(SUB, self.user_id):
            raise ApiError(403, "ACCESS_DENIED")

    @property
    def partition(self):
        return f"TENANT#{self.tenant_id}#USER#{self.user_id}"


def authenticate(event, *, public_config, metadata_store, required_scope, now=None, enroll_member=None):
    """Never decode a bearer token here. Trust only the authorizer's verified claims."""
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    sub, exp = claims.get("sub", ""), str(claims.get("exp", ""))
    if type(sub) is not str or not re.fullmatch(SUB, sub) or claims.get("token_use") != "access" or (
        claims.get("iss") != public_config["issuer"] or claims.get("client_id") != public_config["client_id"]
        or not exp.isdigit() or int(exp) <= (time.time() if now is None else now)
    ):
        raise ApiError(401, "ACCESS_TOKEN_REQUIRED")
    if required_scope not in str(claims.get("scope", "")).split():
        raise ApiError(403, "SCOPE_REQUIRED")
    return membership_context(metadata_store, sub, enroll_member=enroll_member,
                              cognito_username=claims.get("username"))


def membership_context(metadata_store, sub, *, enroll_member=None, cognito_username=None):
    """Validate a trusted membership record, including DynamoDB SS or JSON list roles."""
    try:
        item = metadata_store.get(("USER#" + sub, "MEMBERSHIP"))
    except Exception:
        raise ApiError(503, "AUTHORIZATION_UNAVAILABLE") from None
    if item is None and enroll_member is not None:
        enroll_member(sub, cognito_username=cognito_username)
        try:
            item = metadata_store.get(("USER#" + sub, "MEMBERSHIP"))
        except Exception:
            raise ApiError(503, "AUTHORIZATION_UNAVAILABLE") from None
    roles = item.get("roles") if item else None
    if not item or item.get("record_type") != "MEMBERSHIP" or type(item.get("schema_version")) is not int or (
        item["schema_version"] != 1 or item.get("status") != "ACTIVE" or type(roles) not in {list, set, frozenset}
        or not all(type(role) is str for role in roles) or "reviewer" not in roles
    ):
        raise ApiError(403, "ACTIVE_MEMBERSHIP_REQUIRED")
    return AuthContext(item.get("tenant_id", ""), sub)


def require_current_membership(auth, metadata_store):
    if membership_context(metadata_store, auth.user_id) != auth:
        raise ApiError(403, "ACCESS_DENIED")


def reauthorize(auth, event, *, public_config, metadata_store, required_scope):
    if authenticate(event, public_config=public_config, metadata_store=metadata_store,
                    required_scope=required_scope) != auth:
        raise ApiError(403, "ACCESS_DENIED")
