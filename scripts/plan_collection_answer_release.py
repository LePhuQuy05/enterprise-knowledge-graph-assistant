"""Compare explicit answer opt-in with the captured actual stacks. No AWS writes."""

import copy
import json
import tempfile
from pathlib import Path

from aws_cdk import App
from aws_cdk.assertions import Template

from app.cloud_owned_answers import CollectionAnswerPolicy
from app.contracts import canonical_json, sha256_bytes
from infra.owned_runs import OwnedRunsStack

ROOT = Path(__file__).resolve().parents[1]


def main():
    directory = ROOT / "results/collection316-serving/review-v3"
    if directory.exists():
        raise ValueError("COLLECTION_ANSWER_REVIEW_EXISTS")
    preflight = json.loads((ROOT / "results/collection316-serving/preflight-v1.json").read_bytes())
    if preflight["status"] != "PASS_READ_ONLY_NO_NEW_ALLOWANCE_OR_MODEL_CALL":
        raise ValueError("COLLECTION_ANSWER_READ_ONLY_PREFLIGHT_REQUIRED")
    native = json.loads((ROOT / "results/collection316-serving/native-package-v3.json").read_bytes())
    if native["collection_answers"]["status"] != "PASS_NATIVE_COLLECTION_ANSWER_FIXTURE":
        raise ValueError("COLLECTION_ANSWER_NATIVE_PROOF_REQUIRED")
    settings = json.loads((ROOT / "results/owned-cloud/review-v1/settings.json").read_bytes())
    settings["artifacts"]["package_sha256"] = native["package"]["sha256"]
    settings.update(answer_policy=CollectionAnswerPolicy().model_dump(mode="json"),
                    deepseek_secret_arn=preflight["existing_secret_arn"])
    with tempfile.TemporaryDirectory(prefix="kg-collection-answer-review-") as temp:
        expected = Template.from_stack(OwnedRunsStack(App(outdir=temp, analytics_reporting=False), **settings)).to_json()
    before = preflight["kg-r72-owned-runs_template"]
    if set(before["Resources"]) != set(expected["Resources"]):
        raise ValueError("COLLECTION_ANSWER_CANNOT_ADD_OR_REMOVE_RESOURCES")
    # Preserve all historic resource payloads; take only six reviewed opt-in resources.
    candidate = copy.deepcopy(before)
    allowed = {"Api", "Worker", "Watchdog", "ApiRole", "WorkerRole", "WatchdogRole"}
    for name in allowed:
        candidate["Resources"][name] = expected["Resources"][name]
    changed = sorted(name for name in before["Resources"] if before["Resources"][name] != candidate["Resources"][name])
    assert set(changed) <= allowed
    for kind in ("Api", "Worker", "Watchdog"):
        a, b = before["Resources"][kind]["Properties"], candidate["Resources"][kind]["Properties"]
        assert {k: v for k, v in a.items() if k not in {"Code", "Environment"}} == {
            k: v for k, v in b.items() if k not in {"Code", "Environment"}}
    web = json.loads((ROOT / "results/collection316-serving/web-package-v4.json").read_bytes())
    website_before = preflight["kg-r72-tracer_template"]
    website = copy.deepcopy(website_before)
    website["Resources"]["QueryFunction"]["Properties"]["Code"]["S3Key"] = "packages/query/" + web["sha256"] + ".zip"
    report = {"status": "LOCAL_REVIEW_READY_NOT_DEPLOYED", "aws_writes": 0, "new_resources": 0,
        "changed_owned_resources": changed, "changed_web_resources": ["QueryFunction"],
        "existing_accounts_urls_history_preserved": True, "web_backend_unchanged": True,
        "owned_routes_unchanged": True, "monthly_child_limit": 300,
        "answer_methods": ["hybrid", "vector"], "max_answer_methods": 1,
        "new_run_allowance_proposed_micro_usd": 750000, "shared_funding_cap_unchanged_micro_usd": 2000000,
        "funding_initialization_or_cap_raise": False, "new_run_not_yet_initialized": True,
        "provider_balance_observed": preflight["provider"]["balances"],
        "canary_proposed": {"child_admissions": 2, "maximum_model_calls": 3, "en_hybrid_then_vi_vector": True,
                            "observed_usage_stop_between_runs_usd": 0.05, "hard_run_allowance_usd": 0.75, "no_transport_retry": True},
        "native_package_sha256": native["package"]["sha256"], "web_package_sha256": web["sha256"],
        "template_sha256": sha256_bytes(canonical_json(candidate)),
        "actual_cloud_generation_and_browser_acceptance": "PENDING_APPROVAL_AND_RELEASE"}
    directory.mkdir(parents=True)
    for name, value in (("plan.json", report), ("template-baseline.json", before), ("template.json", candidate),
                        ("website-baseline.json", website_before), ("website-template.json", website), ("settings.json", settings)):
        (directory / name).write_bytes(canonical_json(value) + b"\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
