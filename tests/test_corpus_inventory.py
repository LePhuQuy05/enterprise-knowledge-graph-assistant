import gzip
import json
import sqlite3

import pytest

from app.archive import Archive, ArchivePolicy, location, resource_format
from app.contracts import canonical_json, stable_id, text_hash
from app.corpus_inventory import audit_archive, guide_key, page_stem
from scripts.audit_corpus_coverage import main

BASE = "https://docs.aws.amazon.com/IAM/latest/UserGuide/"


@pytest.fixture
def archive_fixture(tmp_path):
    root = tmp_path / "archive"
    policy = ArchivePolicy(version="inventory-fixture-v1", min_free_disk_bytes=0)
    with Archive(root, policy):
        pass
    return root, policy


def add_document(root, policy, url, text="The role must allow reading the object.", *,
                 state="DONE", alter=None):
    """Write fixtures through the real archive byte/DB format; no HTTP transport."""
    with Archive(root, policy) as archive:
        archive._add(url, "PAGE", "fixture", None)
        if state != "DONE":
            archive.db.execute("UPDATE resources SET state=? WHERE url=?", (state, url))
            archive.db.commit()
            return
        raw = ("synthetic fixture for " + url + "\n" + text).encode()
        raw_sha = archive._put(raw)
        language, family = location(url)
        version = "fixture-normalizer-v1"
        doc = {
            "schema_version": 1,
            "archive_document_version_id": stable_id("archdoc", source_uri=url,
                                                     raw_sha256=raw_sha, normalizer=version),
            "source_uri": url, "effective_uri": url,
            "family": family, "language": language, "format": resource_format(url),
            "version_hint": "latest" if "/latest/" in url else "versioned_or_unspecified",
            "fetched_at": "2026-10-23T00:00:00+00:00", "raw_sha256": raw_sha,
            "normalized_sha256": text_hash(text), "normalizer_version": version,
            "title": "Synthetic IAM fixture", "text": text,
            "sections": [{"start_char": 0, "end_char": len(text),
                          "section_path": ["Role permissions"], "anchor": "permissions"}],
            "warnings": [], "semantic_review": "NOT_REVIEWED",
        }
        if alter:
            alter(doc)
        document_sha = archive._put(canonical_json(doc))
        archive.db.execute("INSERT INTO versions VALUES (?,?,?)", (
            doc["archive_document_version_id"], url, document_sha))
        archive.db.execute("UPDATE resources SET state='DONE',raw_sha256=?,document_sha256=?,"
                           "effective_url=?,fetched_at=? WHERE url=?",
                           (raw_sha, document_sha, url, "2026-10-23T00:00:00+00:00", url))
        archive.db.commit()
        return doc, document_sha


def test_states_usable_is_not_done_indexed_or_semantically_reviewed(archive_fixture, tmp_path):
    root, policy = archive_fixture
    add_document(root, policy, BASE + "roles.html")
    add_document(root, policy, BASE + "queue.html", state="QUEUED")
    add_document(root, policy, BASE + "raw.pdf", state="RAW_ONLY")
    add_document(root, policy, BASE + "redirect.html", state="REDIRECTED")
    add_document(root, policy, BASE + "failed.html", state="FAILED")
    output = tmp_path / "candidates.jsonl"
    result = audit_archive(root, policy, candidate_output=output)
    assert result["status"] == "AUDITED"
    assert result["counts"] == {
        "page_done_records": 1, "structurally_usable_records": 1,
        "invalid_done_records": 0, "records_with_warnings": 0,
    }
    assert result["archive_resource_states"]["PAGE"] == {
        "DONE": 1, "FAILED": 1, "QUEUED": 1, "RAW_ONLY": 1, "REDIRECTED": 1}
    assert result["serving_indexed_records"] is None
    assert result["semantic_reviewed_records"] == 0
    assert not result["absolute_aws_completeness_claimed"]
    [candidate] = [json.loads(line) for line in output.read_text().splitlines()]
    assert candidate["source_uri"] == BASE + "roles.html"
    assert candidate["raw_blob_verified"] is True
    assert candidate["semantic_review"] == "NOT_REVIEWED"
    assert "text" not in candidate and "sections" not in candidate
    assert (result["source_http_requests"], result["inference_calls"]) == (0, 0)


def test_same_stem_is_only_candidate_unless_text_hash_agrees(archive_fixture):
    root, policy = archive_fixture
    add_document(root, policy, BASE + "same.html", "A policy allows access only under condition X.")
    add_document(root, policy, BASE + "same.md", "A policy allows access only under condition X.")
    add_document(root, policy, BASE + "different.html", "A role allows access.")
    add_document(root, policy, BASE + "different.md", "A role denies access.")
    add_document(root, policy, BASE.replace("latest", "2020") + "same.md",
                 "An earlier, different version.")
    result = audit_archive(root, policy)
    dedup = result["deduplication"]
    assert dedup["exact_normalized_text_groups"] == 1
    assert dedup["exact_duplicate_records_beyond_first"] == 1
    assert dedup["unique_normalized_text_hashes"] == 4
    assert dedup["html_markdown_same_stem_candidate_groups"] == 2
    assert dedup["html_markdown_same_stem_exact_text_groups"] == 1
    assert dedup["html_markdown_same_stem_unverified_equivalence_groups"] == 1
    assert result["guide_counts"] == {"IAM/2020/UserGuide": 1, "IAM/latest/UserGuide": 4}
    assert guide_key(BASE + "a.html") != guide_key(BASE.replace("latest", "2020") + "a.html")


def test_candidates_deterministic_and_provenance_retained(archive_fixture, tmp_path):
    root, policy = archive_fixture
    doc, digest = add_document(root, policy, BASE + "z.html")
    add_document(root, policy, BASE + "a.md")
    output = tmp_path / "candidates.jsonl"
    audit_archive(root, policy, candidate_output=output)
    first = output.read_bytes()
    audit_archive(root, policy, candidate_output=output)
    assert output.read_bytes() == first
    values = [json.loads(line) for line in first.splitlines()]
    assert [value["source_uri"] for value in values] == [BASE + "a.md", BASE + "z.html"]
    assert values[1]["document_sha256"] == digest
    assert values[1]["archive_document_version_id"] == doc["archive_document_version_id"]
    assert values[1]["normalized_sha256"] == doc["normalized_sha256"]
    # Exact content dedup reports groups but emits every source reference.
    assert len(values) == 2


@pytest.mark.parametrize(("alter", "reason"), [
    (lambda d: d.update(schema_version=True), "DOCUMENT_SCHEMA_VERSION_INVALID"),
    (lambda d: d.update(normalized_sha256="0" * 64), "DOCUMENT_TEXT_HASH_MISMATCH"),
    (lambda d: d.update(archive_document_version_id="forged"), "DOCUMENT_IDENTITY_MISMATCH"),
    (lambda d: d.update(fetched_at="2026-10-23T00:00:00"), "DOCUMENT_FETCH_TIME_INVALID"),
    (lambda d: d.update(source_uri="https://evil.invalid/doc.html"), None),
    (lambda d: d["sections"][0].update(end_char=10_000), "DOCUMENT_SECTION_OUT_OF_RANGE"),
    (lambda d: d.update(semantic_review="APPROVED"), "DOCUMENT_REVIEW_STATUS_INVALID"),
    (lambda d: d.update(extra_unexpected=True), "DOCUMENT_SCHEMA_INVALID"),
])
def test_invalid_metadata_excluded_not_counted_as_usable(archive_fixture, tmp_path, alter, reason):
    root, policy = archive_fixture
    add_document(root, policy, BASE + "invalid.html", alter=alter)
    output = tmp_path / "out.jsonl"
    result = audit_archive(root, policy, candidate_output=output)
    assert result["status"] == "AUDITED_WITH_INVALID_RECORDS"
    assert result["counts"]["page_done_records"] == 1
    assert result["counts"]["structurally_usable_records"] == 0
    assert result["counts"]["invalid_done_records"] == 1
    if reason:
        assert result["invalid_reason_counts"] == {reason: 1}
    assert output.read_bytes() == b""
    assert "evil.invalid" not in json.dumps(result["invalid_samples"])


@pytest.mark.parametrize("damage", ["missing_normalized", "raw_tampered", "traversal", "version_link"])
def test_missing_or_tampered_blob_path_and_version_link_rejected(archive_fixture, damage):
    root, policy = archive_fixture
    doc, digest = add_document(root, policy, BASE + "damaged.html")
    if damage == "missing_normalized":
        (root / "blobs" / digest[:2] / (digest + ".gz")).unlink()
    elif damage == "raw_tampered":
        raw_sha = doc["raw_sha256"]
        (root / "blobs" / raw_sha[:2] / (raw_sha + ".gz")).write_bytes(gzip.compress(b"changed"))
    else:
        with sqlite3.connect(root / "archive.sqlite3") as db:
            if damage == "traversal":
                db.execute("UPDATE resources SET document_sha256='../outside' WHERE kind='PAGE'")
            else:
                db.execute("DELETE FROM versions")
    result = audit_archive(root, policy)
    assert result["counts"]["invalid_done_records"] == 1
    assert result["counts"]["structurally_usable_records"] == 0


@pytest.mark.parametrize("url", [
    "https://docs.aws.amazon.com.evil.invalid/IAM/latest/a.html",
    "https://user@docs.aws.amazon.com/IAM/latest/a.html",
    "https://docs.aws.amazon.com/IAM/latest/%2e%2e/a.html",
])
def test_unsafe_resource_url_rejected(archive_fixture, url):
    root, policy = archive_fixture
    add_document(root, policy, BASE + "safe.html")
    with sqlite3.connect(root / "archive.sqlite3") as db:
        db.execute("UPDATE resources SET url=? WHERE kind='PAGE'", (url,))
    result = audit_archive(root, policy)
    assert result["counts"]["invalid_done_records"] == 1
    assert result["counts"]["structurally_usable_records"] == 0


def test_audit_does_not_create_or_mutate_archive_files(archive_fixture, tmp_path):
    root, policy = archive_fixture
    add_document(root, policy, BASE + "role.html")
    def snapshot():
        return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mtime_ns)
                for path in root.rglob("*") if path.is_file()}
    before = snapshot()
    audit_archive(root, policy, candidate_output=tmp_path / "result.jsonl")
    assert snapshot() == before
    with pytest.raises(ValueError, match="OUTPUT_MUST_BE_OUTSIDE_ARCHIVE"):
        audit_archive(root, policy, candidate_output=root / "never-create.jsonl")
    assert snapshot() == before


def test_refetch_same_bytes_preserves_original_version_provenance(archive_fixture, tmp_path):
    root, policy = archive_fixture
    doc, digest = add_document(root, policy, BASE + "role.html")
    with sqlite3.connect(root / "archive.sqlite3") as db:
        db.execute("UPDATE resources SET fetched_at=?,effective_url=? WHERE kind='PAGE'", (
            "2026-10-24T00:00:00+00:00", BASE + "new-redirect.html"))
    output = tmp_path / "candidates.jsonl"
    result = audit_archive(root, policy, candidate_output=output)
    assert result["counts"]["structurally_usable_records"] == 1
    candidate = json.loads(output.read_bytes())
    assert candidate["fetched_at"] == doc["fetched_at"]
    assert candidate["effective_uri"] == doc["effective_uri"]
    assert candidate["document_sha256"] == digest
    assert candidate["archive_resource_fetched_at"] == "2026-10-24T00:00:00+00:00"
    assert candidate["archive_resource_effective_uri"] == BASE + "new-redirect.html"


def test_cli_reports_invalid_count_and_refuses_archive_outputs(archive_fixture, tmp_path, capsys):
    root, policy = archive_fixture
    add_document(root, policy, BASE + "role.html")
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.model_dump_json())
    args = ["--root", str(root), "--policy", str(policy_path),
            "--output", str(tmp_path / "report.json"), "--candidates", str(tmp_path / "manifest.jsonl")]
    assert main(args) == 0
    assert json.loads((tmp_path / "report.json").read_text())["counts"]["structurally_usable_records"] == 1
    assert json.loads(capsys.readouterr().out)["inference_calls"] == 0
    with pytest.raises(SystemExit):
        main(args[:-2] + ["--candidates", str(root / "out.jsonl")])
    assert not (root / "out.jsonl").exists()


def test_stem_does_not_merge_versions_and_rejects_unsafe_uri():
    assert page_stem(BASE + "role.md") == page_stem(BASE + "role.html")
    assert page_stem(BASE + "role.md") != page_stem(BASE.replace("latest", "2010") + "role.html")
    with pytest.raises(ValueError):
        page_stem("https://evil.invalid/role.html")


def test_filtered_audit_reads_only_selected_done_blobs_and_marks_scope(archive_fixture, tmp_path, monkeypatch):
    root, policy = archive_fixture
    selected, selected_digest = add_document(root, policy, BASE + "selected.html")
    _, unselected_digest = add_document(root, policy, BASE + "outside.html")
    (root / "blobs" / unselected_digest[:2] / (unselected_digest + ".gz")).unlink()
    add_document(root, policy, BASE + "queued.html", state="QUEUED")
    add_document(root, policy, BASE + "failed.html", state="FAILED")
    reads = []
    original = Archive.read_blob
    def track(archive, digest, *args, **kwargs):
        reads.append(digest)
        return original(archive, digest, *args, **kwargs)
    monkeypatch.setattr(Archive, "read_blob", track)
    output = tmp_path / "filtered.jsonl"
    result = audit_archive(root, policy, candidate_output=output, selected_urls=(
        BASE + "selected.html", BASE + "queued.html", BASE + "failed.html", BASE + "missing.html"))
    assert result["audit_scope"] == result["resource_state_counts_scope"] == "FILTERED"
    assert result["selected_url_count"] == 4
    assert result["selected_page_not_done_count"] == 2
    assert result["selected_missing_resource_count"] == 1
    assert result["selected_not_page_count"] == 0
    assert result["counts"]["page_done_records"] == result["counts"]["structurally_usable_records"] == 1
    assert result["archive_resource_states"] == {"PAGE": {"DONE": 1, "FAILED": 1, "QUEUED": 1}}
    assert set(reads) == {selected_digest, selected["raw_sha256"]}
    assert json.loads(output.read_bytes())["source_uri"] == BASE + "selected.html"
    assert result["counts"]["invalid_done_records"] == 0


def test_full_audit_default_includes_unselected_records(archive_fixture):
    root, policy = archive_fixture
    add_document(root, policy, BASE + "a.html")
    add_document(root, policy, BASE + "b.html")
    result = audit_archive(root, policy)
    assert result["audit_scope"] == "FULL"
    assert result["selected_url_count"] is None
    assert result["selected_page_not_done_count"] is None
    assert result["counts"]["structurally_usable_records"] == 2


def test_filtered_selection_deterministic_deduplicated_and_empty_valid(archive_fixture, tmp_path):
    root, policy = archive_fixture
    add_document(root, policy, BASE + "a.html")
    add_document(root, policy, BASE + "b.html")
    output = tmp_path / "selected.jsonl"
    first = audit_archive(root, policy, candidate_output=output,
                          selected_urls=(BASE + "b.html", BASE + "a.html", BASE + "b.html"))
    contents = output.read_bytes()
    second = audit_archive(root, policy, candidate_output=output,
                           selected_urls=(BASE + "a.html", BASE + "b.html"))
    assert first["selected_url_count"] == 2
    assert first["selected_urls_sha256"] == second["selected_urls_sha256"]
    assert output.read_bytes() == contents
    empty = audit_archive(root, policy, candidate_output=output, selected_urls=())
    assert empty["audit_scope"] == "FILTERED" and empty["selected_url_count"] == 0
    assert empty["counts"]["page_done_records"] == 0 and output.read_bytes() == b""


def test_filtered_selection_rejects_unsafe_urls_before_output(archive_fixture, tmp_path):
    root, policy = archive_fixture
    output = tmp_path / "never-create.jsonl"
    with pytest.raises(ValueError):
        audit_archive(root, policy, candidate_output=output, selected_urls=("https://evil.invalid/a.html",))
    assert not output.exists()


def test_cli_filtered_selection_file(archive_fixture, tmp_path, capsys):
    root, policy = archive_fixture
    add_document(root, policy, BASE + "a.html")
    add_document(root, policy, BASE + "b.html")
    policy_path, selection_path = tmp_path / "policy.json", tmp_path / "selection.json"
    policy_path.write_text(policy.model_dump_json())
    selection_path.write_text(json.dumps({"sources": [{"url": BASE + "b.html"}]}))
    assert main(["--root", str(root), "--policy", str(policy_path),
        "--output", str(tmp_path / "report.json"), "--candidates", str(tmp_path / "candidates.jsonl"),
        "--selection", str(selection_path)]) == 0
    result = json.loads((tmp_path / "report.json").read_bytes())
    assert result["audit_scope"] == "FILTERED" and result["selected_url_count"] == 1
    assert result["counts"]["page_done_records"] == 1
    assert json.loads(capsys.readouterr().out)["audit_scope"] == "FILTERED"
