import io
import json
import zipfile
from pathlib import Path

import pytest

from app.contracts import sha256_bytes
from app.release_artifacts import (
    approved_source,
    index_sources,
    materialize,
    osv_summary,
    read_source,
    ref_tree_sources,
    relative_path,
    triage_findings,
)
from scripts.build_recruiter_bundle import (
    DOCS,
    archive_bytes,
    fraction,
    integer,
    project_evidence,
    timestamp,
)


@pytest.mark.parametrize("name", ["../secret", "/etc/passwd", "C:/secret", "app/../data/key", "app\\key", "app//key", "./README.md"])
def test_public_path_cannot_escape_root_or_change_meaning(name):
    with pytest.raises(ValueError, match="UNSAFE_RELEASE_PATH"):
        relative_path(name)


@pytest.mark.parametrize("name", ["results/private/reviewer_credentials.json", "data/keys/deepseek.key",
    ".env", "docs/private/key.json", "configs/credentials.json", "app/__pycache__/x.pyc", "node_modules/a.js"])
def test_private_and_generated_paths_are_never_approved(name):
    assert not approved_source(name)


def test_materialization_is_new_directory_only_and_inventory_covers_actual_bytes(tmp_path):
    files = {"app/example.py": b"print('synthetic')\n", "README.md": b"fixture"}
    target = tmp_path / "audit"
    inventory = materialize(files, target)
    assert len(inventory["files"]) == 2
    for row in inventory["files"]:
        assert sha256_bytes((target / row["path"]).read_bytes()) == row["sha256"]
    with pytest.raises(FileExistsError):
        materialize(files, target)


def test_symlink_and_large_source_cannot_enter_export(tmp_path, monkeypatch):
    source = tmp_path / "README.md"
    source.write_bytes(b"fixture")
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda self: self == source or original(self))
    with pytest.raises(ValueError, match="SYMLINK"):
        read_source(tmp_path, "README.md")
    monkeypatch.setattr(Path, "is_symlink", original)
    source.write_bytes(b"x" * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="UNSAFE_RELEASE_SOURCE"):
        read_source(tmp_path, "README.md")


@pytest.mark.parametrize("response", [{}, {"results": []}, {"results": [{"next_page_token": "remaining"}]},
    {"results": [{"error": "unavailable"}]}, {"results": [{"vulns": [None]}]}])
def test_missing_partial_or_malformed_advisory_response_never_means_clean(response):
    with pytest.raises(ValueError):
        osv_summary([{"ecosystem": "PyPI", "name": "example", "version": "1"}], response)


def test_advisory_findings_keep_package_identity_and_do_not_drop_order():
    packages = [{"ecosystem": "PyPI", "name": "a", "version": "1"}, {"ecosystem": "npm", "name": "b", "version": "2"}]
    result = osv_summary(packages, {"results": [{}, {"vulns": [{"id": "TEST-001"}]}]})
    assert result["status"] == "KNOWN_ADVISORIES_FOUND"
    assert result["findings"] == [packages[1] | {"advisory_ids": ["TEST-001"]}]


def test_triage_only_matches_exact_reviewed_line_not_future_value_or_rule():
    files = {"app/x.py": b"PUBLIC_DIGEST = 'known digest'\n"}
    finding = {"File": "app/x.py", "StartLine": 1, "RuleID": "generic-api-key"}
    entries = [{"path": "app/x.py", "line_number": 1, "rule_id": "generic-api-key",
        "line_sha256": sha256_bytes(files["app/x.py"].decode().strip().encode()), "classification": "PUBLIC_DIGEST"}]
    assert triage_findings([finding], files, entries)["unreviewed_count"] == 0
    assert triage_findings([finding], {"app/x.py": b"a different value"}, entries)["unreviewed_count"] == 1
    assert triage_findings([finding | {"RuleID": "other"}], files, entries)["unreviewed_count"] == 1
    assert triage_findings([finding | {"EndLine": 2}], files, entries)["unreviewed_count"] == 1


def test_tree_refs_without_commit_history_are_scanned_by_actual_blobs(tmp_path, monkeypatch):
    oid, blob = "a" * 40, "b" * 40

    def fake_git(root, *args):
        if args[0] == "for-each-ref":
            return (oid + " tree\n" + oid + " tree\n").encode()
        if args[0] == "ls-tree":
            return f"100644 blob {blob}\tREADME.md\0".encode()
        assert args == ("cat-file", "blob", blob)
        return b"public fixture"

    monkeypatch.setattr("app.release_artifacts.git", fake_git)
    files, refs, roots = ref_tree_sources(tmp_path)
    assert (refs, roots) == (2, 1) and files == {oid + "/README.md": b"public fixture"}


def test_staged_private_path_is_rejected_before_reading_its_blob(tmp_path, monkeypatch):
    called = []

    def fake_git(root, *args):
        called.append(args[0])
        return ("100644 " + "b" * 40 + " 0\tresults/private/reviewer_credentials.json\0").encode()

    monkeypatch.setattr("app.release_artifacts.git", fake_git)
    with pytest.raises(ValueError, match="UNAPPROVED_INDEX_PATH"):
        index_sources(tmp_path)
    assert called == ["ls-files"]


def test_evidence_zip_rejects_unknown_members_and_has_deterministic_verified_manifest():
    files = {name: b"authored fixture" for name in DOCS} | {"evidence.json": b"{}"}
    first, inventory = archive_bytes(files)
    assert first == archive_bytes(files)[0]
    with zipfile.ZipFile(io.BytesIO(first)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["files"] == inventory
        for row in inventory:
            assert sha256_bytes(bundle.read(row["path"])) == row["sha256"]
    with pytest.raises(ValueError, match="UNAPPROVED_BUNDLE_MEMBER"):
        archive_bytes(files | {"results/private/key.json": b"not exported"})


@pytest.mark.parametrize("value", [True, -1, float("nan"), float("inf"), "secret"])
def test_numeric_evidence_cannot_smuggle_arbitrary_strings_or_nonfinite_values(value):
    with pytest.raises(ValueError):
        integer(value)
    with pytest.raises(ValueError):
        fraction(value)


def public_input_fixture():
    system = {"provisional_ai_draft": {"answerable_denominator": 12, "complete_evidence_at_k": 10 / 12,
                                      "context_complete": 10 / 12},
              "human_approved_only": {"answerable_denominator": 0, "complete_evidence_at_k": None}}
    scan = {"finding_count": 10, "reviewed_false_positives": [{}] * 10, "unreviewed_count": 0}
    return {"baseline": {"snapshot_version": "snap_" + "a" * 64, "documents": 20, "chunks": 236,
                         "embedding_recipe": {"dimension": 384}, "manifest_sha256": "b" * 64},
        "deployment": {"checked_at": "2026-10-18T00:00:00+00:00", "ready_package_sha256": "c" * 64,
                       "new_direct_lambda_invocations": 25, "managed_checks": 29, "new_model_calls": 0},
        "security": {"offline_cases": 28, "offline_model_trials": 0, "known_semantic_limitations": 10,
                     "live_synthetic_model_calls": 5, "new_budget_cost_micro_usd": 1850, "replay_model_calls": 0},
        "audit": {"status": "NO_KNOWN_ADVISORIES_OR_UNRESOLVED_SECRET_FINDINGS_IN_SCANNED_SCOPE",
                  "checked_at": "2026-10-18T00:00:00+00:00", "dependency_audit": {"findings": [],
                      "pypi_packages": 59, "npm_packages": 1}, "npm_audit": {"counts": {"total": 0}},
                  "secret_scans": {k: dict(scan) for k in ("workspace", "index", "ref_trees")},
                  "reachable_commits": 0, "unique_ref_roots": 1},
        "dev": {"systems": {k: system for k in ("B1_VECTOR", "B2_HYBRID", "G1_DRAFT", "G2_DRAFT")}},
        "mentions": {"systems": {"M_SERVICE_MENTIONS": system}},
        "browser": {"status": "CONFIRMED_NEW_BROWSER_SUBMISSION"}}


def test_closed_projection_drops_private_fields_at_every_level():
    records = public_input_fixture()
    marker = "SYNTHETIC_PRIVATE_EXPORT_CANARY"
    for record in records.values():
        record["private_key"] = marker
    records["baseline"]["corpus_directory"] = "C:/Users/private/" + marker
    records["audit"]["secret_scans"]["workspace"]["raw_match"] = marker
    records["deployment"]["outputs"] = {"SecretArn": marker}
    result = project_evidence(records)
    assert marker not in json.dumps(result)
    assert result["development_retrieval"]["systems"]["B2_HYBRID"]["human_approved_quality"] is None
    assert result["git_release_commit"] is None


def test_unresolved_audit_result_blocks_candidate_instead_of_becoming_clean():
    records = public_input_fixture()
    records["audit"]["secret_scans"]["workspace"]["unreviewed_count"] = 1
    with pytest.raises(ValueError, match="UNRESOLVED_AUDIT"):
        project_evidence(records)


def test_timestamp_rejection_does_not_echo_untrusted_input():
    with pytest.raises(ValueError) as error:
        timestamp("SYNTHETIC_PRIVATE_EXPORT_CANARY")
    assert "CANARY" not in str(error.value)
