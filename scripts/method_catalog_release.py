"""Pure review of a catalog opt-in template; no AWS calls or deployment executor."""

import copy

from app.contracts import canonical_json, sha256_bytes
from scripts.query_cloud import inspect_code_update
from scripts.tracer import assert_no_cycles


def inspect_catalog_update(before, after):
    """Allow the read route, its stage dependency/throttle, and optional code-only update."""
    old, new = before["Resources"], after["Resources"]
    route_id = "PrivateRagMethods"
    if set(new) - set(old) != {route_id} or set(old) - set(new):
        raise ValueError("CATALOG_ONLY_ONE_READ_ROUTE_MAY_BE_ADDED")
    expected_route = copy.deepcopy(old["PrivateStatus"])
    expected_route["Properties"]["RouteKey"] = "GET /v1/rag-methods"
    if canonical_json(new[route_id]) != canonical_json(expected_route) or (
        expected_route["Properties"].get("AuthorizationType") != "JWT"
        or expected_route["Properties"].get("AuthorizationScopes") != ["kg/read"]
    ):
        raise ValueError("CATALOG_ROUTE_MUST_PRESERVE_STATUS_JWT_AND_INTEGRATION")
    expected_stage = copy.deepcopy(old["DefaultStage"])
    expected_stage["DependsOn"] = sorted(set(expected_stage.get("DependsOn", [])) | {route_id})
    expected_stage["Properties"]["RouteSettings"]["GET /v1/rag-methods"] = {
        "ThrottlingBurstLimit": 10, "ThrottlingRateLimit": 5}
    if canonical_json(new["DefaultStage"]) != canonical_json(expected_stage):
        raise ValueError("CATALOG_STAGE_ONLY_DEPENDENCY_AND_READ_THROTTLE_MAY_CHANGE")
    normalized = copy.deepcopy(after)
    del normalized["Resources"][route_id]
    normalized["Resources"]["DefaultStage"] = copy.deepcopy(old["DefaultStage"])
    modified = ["DefaultStage"]
    if canonical_json(old["QueryFunction"]) != canonical_json(new["QueryFunction"]):
        inspect_code_update(before, normalized)
        # The code-only inspector protects resources/outputs; also protect other template fields.
        normalized["Resources"]["QueryFunction"] = copy.deepcopy(old["QueryFunction"])
        modified.append("QueryFunction")
    if canonical_json(before) != canonical_json(normalized):
        raise ValueError("CATALOG_MUST_PRESERVE_IAM_IDENTITIES_CONFIG_AND_UNRELATED_FIELDS")
    assert_no_cycles(new)
    return {"status": "LOCAL_TEMPLATE_REVIEW_ONLY", "added": [route_id], "modified": sorted(modified),
        "removed": [], "template_sha256": sha256_bytes(canonical_json(after)),
        "model_calls": 0, "aws_calls": 0, "cloud_deployed": False}
