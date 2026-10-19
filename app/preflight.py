"""Explicit, bounded read-only AWS probes; never invoke/subscribe/create resources."""

from datetime import UTC, datetime

import boto3
from botocore.config import Config
from botocore.exceptions import BotoCoreError, ClientError

CANDIDATES = (
    "amazon.titan-embed-text-v2:0",
    "anthropic.claude-haiku-4-5-20251001-v1:0",
    "amazon.nova-lite-v1:0",
)


def probe_aws(profile: str, region: str) -> dict:
    if region != "us-east-1":
        raise ValueError("Only the approved preflight region is supported")
    config = Config(connect_timeout=5, read_timeout=10, retries={"total_max_attempts": 1})
    report = dict(
        schema_version=1, checked_at=datetime.now(UTC).isoformat(), profile=profile, region=region,
        status="INCOMPLETE", metadata_calls_attempted=0, inference_calls=0,
        writes_or_subscriptions=0, ready_for_billable_work=False, probes={},
        limitation="Metadata availability is not an inference test; rates, ledger and deploy remain",
    )
    try:
        session = boto3.Session(profile_name=profile, region_name=region)
    except BotoCoreError as exc:
        report["probes"]["profile"] = {"status": "FAIL", "error_code": type(exc).__name__}
        return report

    def probe(label, service, operation, selector, **kwargs):
        report["metadata_calls_attempted"] += 1
        try:
            client = session.client(service, config=config)
            response = getattr(client, operation)(**kwargs)
            selected = selector(response)
            report["probes"][label] = {"status": "PASS", "data": selected}
            return selected
        except (BotoCoreError, ClientError) as exc:
            error_code = exc.response["Error"]["Code"] if isinstance(exc, ClientError) else type(exc).__name__
            report["probes"][label] = {"status": "FAIL", "error_code": error_code}
            return None

    identity = probe("identity", "sts", "get_caller_identity", lambda value: {
        "principal_type": "IAM_USER" if ":user/" in value["Arn"] else "OTHER",
        "expected_dev_user": value["Arn"].endswith(":user/kg-dev"),
        "is_root": value["Arn"].endswith(":root"),
    })
    if not identity or not identity["expected_dev_user"] or identity["is_root"]:
        report["status"] = "DEVELOPMENT_IDENTITY_REQUIRED"
        return report
    probe("account_plan", "freetier", "get_account_plan_state", lambda value: {
        "type": value.get("accountPlanType"), "status": value.get("accountPlanStatus"),
        "remaining_credits": value.get("accountPlanRemainingCredits"),
    })
    for model_id in CANDIDATES:
        probe("model:" + model_id, "bedrock", "get_foundation_model", lambda value: {
            key: value["modelDetails"].get(key) for key in (
                "modelId", "providerName", "modelName", "inputModalities", "outputModalities",
                "inferenceTypesSupported",
            )
        }, modelIdentifier=model_id)
        probe("availability:" + model_id, "bedrock", "get_foundation_model_availability", lambda value: {
            "model_id": value.get("modelId"), "authorization": value.get("authorizationStatus"),
            "entitlement": value.get("entitlementAvailability"), "region": value.get("regionAvailability"),
            "agreement": value.get("agreementAvailability", {}).get("status"),
        }, modelId=model_id)

    def select_profiles(value):
        profiles = []
        for item in value.get("inferenceProfileSummaries", []):
            models = [{"model_id": model["modelArn"].split("foundation-model/")[-1],
                       "region": model["modelArn"].split(":")[3]}
                      for model in item.get("models", [])]
            if any(model["model_id"] in CANDIDATES for model in models):
                profiles.append({"profile_id": item["inferenceProfileId"], "status": item["status"],
                                 "models": models})
        return {"candidate_profiles": profiles, "more_pages": bool(value.get("nextToken"))}

    probe("inference_profiles", "bedrock", "list_inference_profiles", select_profiles,
          typeEquals="SYSTEM_DEFINED", maxResults=100)
    probe("lambda_account_limits", "lambda", "get_account_settings", lambda value: {
        key: value.get("AccountLimit", {}).get(key)
        for key in ("ConcurrentExecutions", "UnreservedConcurrentExecutions", "CodeSizeUnzipped")
    })
    if all(item["status"] == "PASS" for item in report["probes"].values()):
        report["status"] = "METADATA_READABLE"
    return report
