"""Export selected numeric evidence and authored notes; never copy raw results or runtime artifacts."""

import argparse
import io
import json
import math
import re
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime
from pathlib import Path

from app.contracts import canonical_json, sha256_bytes
from app.release_artifacts import read_source

ROOT = Path(__file__).resolve().parents[1]
DOCS = {"README.md": "README.md", "ARCHITECTURE.md": "docs/architecture.md",
        "DEVELOPMENT.md": "docs/development.md", "EVALUATION.md": "docs/evaluation.md",
        "SECURITY.md": "docs/security-overview.md",
        "LICENSE": "LICENSE", "NOTICE.md": "NOTICE.md"}
INPUTS = {"baseline": "results/bge_baseline.json", "deployment": "results/release_deadline_deployment_status.json",
    "browser": "results/query-cloud/browser_post_confirmation.json", "security": "results/generation_security_verification.json",
    "audit": "results/release_audit_latest.json",
    "dev": "results/evaluation-dev/20261017T214800Z-6197bdfe/summary.json",
    "mentions": "results/service-mention-evaluation/20261017T220835Z-d76cb9a8/summary.json"}


def integer(value, *, maximum=1_000_000):
    if type(value) is not int or not 0 <= value <= maximum:
        raise ValueError("INVALID_PUBLIC_EVIDENCE_INTEGER")
    return value


def fraction(value):
    if type(value) not in {float, int} or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("INVALID_PUBLIC_EVIDENCE_FRACTION")
    return value


def digest(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("INVALID_PUBLIC_EVIDENCE_DIGEST")
    return value


def timestamp(value):
    try:
        if type(value) is not str or len(value) > 50:
            raise ValueError
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        raise ValueError("INVALID_PUBLIC_EVIDENCE_TIMESTAMP") from None
    if parsed.tzinfo is None:
        raise ValueError("EVIDENCE_TIMESTAMP_MUST_HAVE_TIMEZONE")
    return parsed.isoformat()


def project_evidence(records):
    """Closed projections: arbitrary input keys/strings/URLs/identities are never exported."""
    baseline, deployment, security, audit = (records[k] for k in ("baseline", "deployment", "security", "audit"))
    systems = {}
    for key in ("B1_VECTOR", "B2_HYBRID", "G1_DRAFT", "G2_DRAFT"):
        raw = records["dev"]["systems"][key]
        provisional, approved = raw["provisional_ai_draft"], raw["human_approved_only"]
        if approved["answerable_denominator"] != 0 or approved["complete_evidence_at_k"] is not None:
            raise ValueError("REVIEW_CHANGED_REVISE_RELEASE_PROTOCOL")
        systems[key] = {"answerable_denominator": integer(provisional["answerable_denominator"]),
            "complete_evidence_at_5": fraction(provisional["complete_evidence_at_k"]),
            "context_complete": fraction(provisional["context_complete"]), "annotation_status": "AI_DRAFT",
            "human_approved_denominator": 0, "human_approved_quality": None}
    scans = {key: {"raw_findings": integer(audit["secret_scans"][key]["finding_count"]),
        "reviewed_false_positives": integer(len(audit["secret_scans"][key]["reviewed_false_positives"])),
        "unreviewed_findings": integer(audit["secret_scans"][key]["unreviewed_count"])}
        for key in ("workspace", "index", "ref_trees")}
    if (any(s["unreviewed_findings"] for s in scans.values()) or audit["dependency_audit"]["findings"]
            or audit["npm_audit"]["counts"]["total"] != 0):
        raise ValueError("UNRESOLVED_AUDIT_FINDINGS_BLOCK_BUNDLE")
    if audit["status"] != "NO_KNOWN_ADVISORIES_OR_UNRESOLVED_SECRET_FINDINGS_IN_SCANNED_SCOPE":
        raise ValueError("INCOMPLETE_AUDIT_BLOCKS_BUNDLE")
    snapshot = baseline["snapshot_version"]
    if not isinstance(snapshot, str) or not re.fullmatch(r"snap_[0-9a-f]{64}", snapshot):
        raise ValueError("INVALID_SNAPSHOT_ID")
    return {"schema_version": 1, "release_status": "EVIDENCE_CANDIDATE_NOT_FINAL_ACCEPTANCE",
        "live_demo_url": "https://cyg0uvml94.execute-api.us-east-1.amazonaws.com/",
        "source_license": "MIT_FOR_ORIGINAL_CODE_ONLY", "git_release_commit": None,
        "application": {"official_documents": integer(baseline["documents"]), "chunks": integer(baseline["chunks"]),
            "embedding_dimensions": integer(baseline["embedding_recipe"]["dimension"]),
            "embedding_model": "BAAI/bge-small-en-v1.5", "generation_alias": "deepseek-flash",
            "production_modes": ["vector", "hybrid"], "graph_serving_approved": False,
            "snapshot_version": snapshot, "manifest_sha256": digest(baseline["manifest_sha256"])},
        "last_deployment_verification": {"checked_at": timestamp(deployment["checked_at"]),
            "package_sha256": digest(deployment["ready_package_sha256"]),
            "managed_direct_invocations": integer(deployment["new_direct_lambda_invocations"]),
            "checks": integer(deployment["managed_checks"]), "model_calls": integer(deployment["new_model_calls"]),
            "positive_jwt_test": False, "scope": "READS_AND_REJECTED_INPUTS_SYNTHETIC_CLAIMS"},
        "browser_new_submission": {"evidence_kind": "USER_REPORTED",
            "confirmed": records["browser"].get("status") == "CONFIRMED_NEW_BROWSER_SUBMISSION",
            "automated_browser_e2e": False},
        "development_retrieval": {"questions": 15, "independent_holdout": False, "systems": systems,
            "meaning": "SOURCE_SPAN_COVERAGE_NOT_ANSWER_ACCURACY"},
        "mention_ablation": {"ce_at_5": fraction(records["mentions"]["systems"]["M_SERVICE_MENTIONS"]
            ["provisional_ai_draft"]["complete_evidence_at_k"]), "denominator": 12, "selected_for_production": False},
        "generation_security": {"offline_cases": integer(security["offline_cases"]),
            "offline_model_trials": integer(security["offline_model_trials"]),
            "known_semantic_limitations": integer(security["known_semantic_limitations"]),
            "live_synthetic_model_calls": integer(security["live_synthetic_model_calls"]),
            "budget_cost_micro_usd": integer(security["new_budget_cost_micro_usd"]),
            "replay_calls": integer(security["replay_model_calls"]), "semantic_review": "AI_PRELIMINARY_HUMAN_PENDING"},
        "audit": {"checked_at": timestamp(audit["checked_at"]),
            "pypi_versions_queried": integer(audit["dependency_audit"]["pypi_packages"]),
            "npm_versions_queried": integer(audit["dependency_audit"]["npm_packages"]),
            "known_advisories_returned": 0, "secret_scans": scans,
            "reachable_commits": integer(audit["reachable_commits"]),
            "unique_ref_roots": integer(audit["unique_ref_roots"]), "native_and_os_audit_complete": False},
        "pending": ["Human source/claim review and independent quality evaluation", "Full cloud adversarial suite",
            "Broader load/recovery/rollback checks", "Video recording and reviewer delivery",
            "Final source commit/tag, fresh-checkout verification and post-freeze audit"],
        "paid_model_calls_for_bundle_build": 0, "aws_calls_for_bundle_build": 0}


def archive_bytes(files):
    expected = set(DOCS) | {"evidence.json"}
    if set(files) != expected:
        raise ValueError("UNAPPROVED_BUNDLE_MEMBER")
    inventory = [{"path": name, "bytes": len(raw), "sha256": sha256_bytes(raw)} for name, raw in sorted(files.items())]
    manifest = canonical_json({"schema_version": 1, "scope": "EVIDENCE_ONLY_NO_SOURCE_OR_RAW_CORPUS",
                               "files": inventory})
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for name, raw in sorted((files | {"manifest.json": manifest}).items()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            bundle.writestr(info, raw)
    return stream.getvalue(), inventory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if not args.build:
        print("Use --build to write a local, evidence-only candidate. No upload, AWS or model calls.")
        return 0
    records = {key: json.loads((ROOT / path).read_bytes()) for key, path in INPUTS.items()}
    evidence = project_evidence(records)
    test_path = ROOT / "results/release_candidate_full_tests.xml"
    suites = list(ET.parse(test_path).getroot().iter("testsuite"))
    failures = sum(int(s.get("failures", 0)) + int(s.get("errors", 0)) for s in suites)
    if not suites or failures:
        raise ValueError("FULL_TEST_RECEIPT_REQUIRED")
    evidence["local_tests"] = {"python_cases": sum(int(s.get("tests", 0)) for s in suites),
        "failures": failures, "source_xml_sha256": sha256_bytes(test_path.read_bytes()),
        "scope": "LOCAL_WORKSPACE_NOT_AWS_PACKAGE_OR_FINAL_GIT_CHECKOUT"}
    if json.loads((ROOT / "results/release_candidate_ruff.json").read_bytes()) != []:
        raise ValueError("RUFF_RECEIPT_REQUIRED")
    evidence["source_receipt_hashes"] = {path: sha256_bytes((ROOT / path).read_bytes()) for path in INPUTS.values()}
    files = {name: read_source(ROOT, path) for name, path in DOCS.items()}
    files["evidence.json"] = canonical_json(evidence)
    raw, inventory = archive_bytes(files)
    checksum = sha256_bytes(raw)
    output = ROOT / "artifacts" / ("recruiter-evidence-" + checksum[:16] + ".zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and output.read_bytes() != raw:
        raise ValueError("BUNDLE_CHECKSUM_COLLISION")
    if not output.exists():
        with output.open("xb") as handle:
            handle.write(raw)
    (ROOT / "docs/release/evidence-v1.json").write_bytes(canonical_json(evidence))
    report = {"status": "LOCAL_EVIDENCE_CANDIDATE_BUILT_NOT_FINAL_RELEASE", "path": output.relative_to(ROOT).as_posix(),
        "sha256": checksum, "bytes": len(raw), "files": inventory, "members_including_manifest": len(inventory) + 1,
        "aws_calls": 0, "model_calls": 0, "public_upload": False, "git_release_commit": None}
    (ROOT / "results/recruiter_bundle.json").write_bytes(canonical_json(report))
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
