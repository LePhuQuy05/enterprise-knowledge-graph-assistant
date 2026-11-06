"""Prepare a concrete static-only Methods release and rollback. No AWS writes."""

import base64
import copy
import json
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from app.contracts import canonical_json, sha256_bytes
from scripts.answer_local import ROOT


def require(value, code):
    if not value:
        raise ValueError(code)


def main():
    directory = ROOT / "results/collection316-serving/method-pages-review-v2"
    require(not directory.exists(), "METHOD_RELEASE_REVIEW_EXISTS")
    preflight = json.loads((ROOT / "results/collection316-serving/preflight-v3.json").read_bytes())
    package = json.loads((ROOT / "results/collection316-serving/web-package-v7.json").read_bytes())
    baseline_package = json.loads((ROOT / "results/owned-cloud/approved-release-v1/scope-package.json").read_bytes())
    require(preflight["status"] == "PASS_READ_ONLY_NO_NEW_ALLOWANCE_OR_MODEL_CALL", "METHOD_RELEASE_PREFLIGHT")
    require(base64.b64decode(preflight["existing_web_code_sha256_b64"]).hex()
        == package["base_package_sha256"] == baseline_package["sha256"], "METHOD_RELEASE_ACTUAL_BASELINE")
    require(package["changed_zip_entries"] == ["configs/frontend-static.json"]
        and package["backend_unchanged"] and len(package["assets"]) == 16, "METHOD_RELEASE_STATIC_ONLY")
    from pathlib import Path

    with Path(package["package_path"]).open("rb") as handle:
        import hashlib

        require(hashlib.file_digest(handle, "sha256").hexdigest() == package["sha256"], "METHOD_RELEASE_ZIP_HASH")
    with ZipFile(package["package_path"]) as archive:
        assets = json.loads(archive.read("configs/frontend-static.json"))
    for route, value in assets.items():
        require(sha256_bytes(value["body"].encode()) == package["assets"][route]["sha256"], "METHOD_RELEASE_ASSET_HASH")
    data = json.loads(assets["/assets/methods-data.json"]["body"])
    require(data["graph"]["documents"] == 316 and len(data["graph"]["edges"]) == 17
        and data["graph"]["legacy_graph_reused"] is False and data["graph"]["human_reviewed"] is False,
        "METHOD_RELEASE_GRAPH_SCOPE")
    suite_path = ROOT / "results/collection-graph/full-suite-reviewed-graph-v1.xml"
    suites = ET.parse(suite_path).getroot().iter("testsuite")
    counts = {field: 0 for field in ("tests", "failures", "errors", "skipped")}
    for suite in suites:
        for field in counts:
            counts[field] += int(suite.get(field, "0"))
    require(counts["tests"] > 2000 and counts["failures"] == counts["errors"] == 0, "METHOD_RELEASE_FULL_TESTS")
    browser = json.loads((ROOT / "results/collection-graph/reviewed-methods-browser-v5.json").read_bytes())
    gating = json.loads((ROOT / "results/collection-graph/method-production-capabilities-browser-v2.json").read_bytes())
    replay = json.loads((ROOT / "results/collection-graph/current-code-replay-v2.json").read_bytes())
    scroll = json.loads((ROOT / "results/collection-graph/scroll-regression-browser-v1.json").read_bytes())
    keyboard = json.loads((ROOT / "results/collection-graph/method-main-scroll-keyboard-v1.json").read_bytes())
    require(browser["passed"] == 70 and all(row["pass"] for row in browser["checks"])
        and gating["passed"] == 25 and all(row["pass"] for row in gating["checks"])
        and replay["status"] == "PASS_CURRENT_CODE_NETWORK_FORBIDDEN_REPLAY", "METHOD_RELEASE_LOCAL_EVIDENCE")
    require(scroll["passed"] == 5 and all(row["pass"] for row in scroll["checks"])
        and keyboard["passed"] == 12 and all(row["pass"] for row in keyboard["cases"]), "METHOD_RELEASE_SCROLL_REGRESSION")
    before = preflight["kg-r72-tracer_template"]
    candidate = copy.deepcopy(before)
    code = candidate["Resources"]["QueryFunction"]["Properties"]["Code"]
    require(set(code) == {"S3Bucket", "S3Key"}, "METHOD_RELEASE_CODE_POINTER")
    code["S3Key"] = "packages/query/" + package["sha256"] + ".zip"
    require(set(before["Resources"]) == set(candidate["Resources"]), "METHOD_RELEASE_RESOURCES")
    changed = [name for name in before["Resources"] if before["Resources"][name] != candidate["Resources"][name]]
    require(changed == ["QueryFunction"], "METHOD_RELEASE_CHANGE_SCOPE")
    props = candidate["Resources"]["QueryFunction"]["Properties"]
    require({k: v for k, v in props.items() if k != "Code"}
        == {k: v for k, v in before["Resources"]["QueryFunction"]["Properties"].items() if k != "Code"},
        "METHOD_RELEASE_NO_OTHER_PROPERTIES")
    report = {"status": "READY_STATIC_ONLY_NOT_DEPLOYED", "aws_writes": 0, "model_calls": 0,
        "changed_resources": changed, "new_or_removed_resources": 0, "changed_properties": ["QueryFunction.Code"],
        "changed_zip_entries": package["changed_zip_entries"], "unchanged_zip_entries": package["unchanged_entries"],
        "package_sha256": package["sha256"], "package_path": package["package_path"],
        "template_sha256": sha256_bytes(canonical_json(candidate)), "assets": package["assets"],
        "method_pages": ["bm25", "vector", "hybrid", "hybrid_reranked", "hybrid_multi_query",
            "hybrid_multi_query_balanced", "graph_1hop", "graph_2hop"],
        "collection_documents": 316, "collection_chunks": 2978, "graph_experimental_assertions": 17,
        "graph_relation_source_documents": 9, "graph_human_reviewed": False,
        "new_question_retrieval_methods": ["bm25", "vector", "hybrid", "hybrid_reranked"],
        "new_question_answer_methods": [], "live_graph_or_planner_enabled": False,
        "current_accounts_history_auth_quotas_iam_environment_preserved": True,
        "new_funding_or_allowance": False, "full_test_suite": counts,
        "local_browser_checks": browser["passed"], "current_capability_fixture_checks": gating["passed"],
        "scroll_lifecycle_browser_checks": scroll["passed"], "real_keyboard_page_bottom_checks": keyboard["passed"],
        "fix": "Native tour close and deactivation release orphaned body scroll locks; preserve a reopened dialog lock.",
        "previous_review": "method-pages-review-v1: no approval; user reported local main scroll bug; superseded by this candidate",
        "managed_acceptance": "PENDING_EXPLICIT_APPROVAL_AND_DEPLOYMENT",
        "proposed_release_checks": {"static_asset_hashes": 16, "negative_auth": True,
            "model_free_legacy_smoke": True, "no_new_paid_question": True, "authenticated_browser": "operator session required"},
        "rollback": {"package_sha256": baseline_package["sha256"], "package_path": baseline_package["package_path"],
            "code": before["Resources"]["QueryFunction"]["Properties"]["Code"], "baseline_template": "baseline-template.json"}}
    directory.mkdir(parents=True)
    for name, value in (("plan.json", report), ("baseline-template.json", before), ("template.json", candidate)):
        with (directory / name).open("xb") as handle:
            handle.write(canonical_json(value) + b"\n")
    print(json.dumps({key: report[key] for key in ("status", "changed_resources", "package_sha256", "aws_writes", "model_calls")}))


if __name__ == "__main__":
    main()
