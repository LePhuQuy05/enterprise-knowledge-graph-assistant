"""Small stdlib Lambda adapter for the first authenticated deployment.

No snapshot/model initialization and no paid-model permission in this tracer.
API Gateway verifies JWT signatures; this handler adds access-token/membership checks.
The final query package will replace inline deployment with pinned Linux dependencies.
"""

import json
import os
import re
import time

PUBLIC_CONFIG = json.loads(os.environ.get("PUBLIC_CONFIG_JSON", "{}"))
STATIC_ASSETS = {}  # Filled from an explicit frontend allowlist at synth time.
HEADERS = {
    "cache-control": "no-store", "x-content-type-options": "nosniff",
    "referrer-policy": "no-referrer", "x-frame-options": "DENY",
    "strict-transport-security": "max-age=31536000",
    "content-security-policy": "default-src 'none'; script-src 'self'; style-src 'self'; "
    "connect-src 'self' https://*.auth.us-east-1.amazoncognito.com https://cognito-idp.us-east-1.amazonaws.com; "
    "base-uri 'none'; frame-ancestors 'none'; form-action 'none'",
}


def response(status, body, content_type="application/json"):
    return {"statusCode": status, "headers": HEADERS | {"content-type": content_type},
            "isBase64Encoded": False,
            "body": json.dumps(body) if content_type == "application/json" else body}


def membership(sub):
    import boto3  # Static requests never construct a client or read state.
    from botocore.config import Config

    result = boto3.client("dynamodb", region_name="us-east-1", config=Config(
        retries={"total_max_attempts": 1}, connect_timeout=2, read_timeout=3,
    )).get_item(
        TableName="kg-r72-metadata", Key={"PK": {"S": "USER#" + sub}, "SK": {"S": "MEMBERSHIP"}},
        ConsistentRead=True,
    )
    return result.get("Item")


def handler(event, context):
    method = event.get("requestContext", {}).get("http", {}).get("method")
    path = event.get("rawPath", "")
    if method != "GET":
        return response(405, {"error": {"code": "METHOD_NOT_ALLOWED"}})
    if path == "/config.json":
        return response(200, PUBLIC_CONFIG)
    asset_path = "/" if path == "/auth/callback" else path
    if asset_path in STATIC_ASSETS:
        asset = STATIC_ASSETS[asset_path]
        return response(200, asset["body"], asset["content_type"])
    if path != "/v1/status":
        return response(404, {"error": {"code": "NOT_FOUND"}})
    claims = event.get("requestContext", {}).get("authorizer", {}).get("jwt", {}).get("claims", {})
    sub = claims.get("sub", "")
    expires = str(claims.get("exp", ""))
    if claims.get("token_use") != "access" or claims.get("iss") != PUBLIC_CONFIG.get("issuer") or (
        claims.get("client_id") != PUBLIC_CONFIG.get("client_id")
        or not re.fullmatch(r"[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", sub)
        or not expires.isdigit() or int(expires) <= time.time()
    ):
        return response(401, {"error": {"code": "ACCESS_TOKEN_REQUIRED"}})
    if "kg/read" not in str(claims.get("scope", "")).split():
        return response(403, {"error": {"code": "READ_SCOPE_REQUIRED"}})
    try:
        item = membership(sub)
        if not item or item.get("record_type") != {"S": "MEMBERSHIP"} or (
            item.get("schema_version") != {"N": "1"} or item.get("status") != {"S": "ACTIVE"}
            or "reviewer" not in item.get("roles", {}).get("SS", [])
        ):
            return response(403, {"error": {"code": "ACTIVE_MEMBERSHIP_REQUIRED"}})
        tenant = item.get("tenant_id", {}).get("S", "")
        if not re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", tenant):
            raise ValueError("Invalid server-side tenant")
    except Exception:
        return response(503, {"error": {"code": "AUTHORIZATION_UNAVAILABLE"}})
    return response(200, {
        "status": "AUTHENTICATED_FIXTURE", "tenant_id": tenant, "user_id": sub,
        "fixture_only": True, "model_enabled": False,
        "message": "Sign-in is working. Document answers are not enabled in this setup preview.",
    })
