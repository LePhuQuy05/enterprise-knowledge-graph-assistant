import json

import pytest

from scripts.build_coverage_pilot import merge_references


def write(path, rows):
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    return path


def row(uri, version):
    return {"source_uri": uri, "document_sha256": version, "structural_validation": "PASS"}


def test_merge_is_sorted_idempotent_and_does_not_change_inputs(tmp_path):
    a = write(tmp_path / "a.jsonl", [row("b", "v2"), row("a", "v1")])
    b = write(tmp_path / "b.jsonl", [row("a", "v1"), row("c", "v3")])
    before = a.read_bytes(), b.read_bytes()
    output = tmp_path / "merged.jsonl"
    report = merge_references([a, b], output)
    assert report["records"] == 3 and report["duplicate_observations"] == 1
    assert [json.loads(line)["source_uri"] for line in output.read_bytes().splitlines()] == ["a", "b", "c"]
    raw = output.read_bytes()
    merge_references([b, a], output)
    assert output.read_bytes() == raw
    assert (a.read_bytes(), b.read_bytes()) == before


def test_conflicting_source_versions_fail_before_output(tmp_path):
    a = write(tmp_path / "a.jsonl", [row("a", "v1")])
    b = write(tmp_path / "b.jsonl", [row("a", "v2")])
    output = tmp_path / "merged.jsonl"
    with pytest.raises(ValueError, match="SOURCE_VERSION_CONFLICT"):
        merge_references([a, b], output)
    assert not output.exists()


def test_unverified_reference_cannot_enter_merged_manifest(tmp_path):
    path = write(tmp_path / "a.jsonl", [{"source_uri": "a", "structural_validation": "FAILED"}])
    with pytest.raises(ValueError, match="REFERENCE_INVALID"):
        merge_references([path], tmp_path / "merged.jsonl")


def test_merge_refuses_to_overwrite_source_inventory(tmp_path):
    path = write(tmp_path / "source.jsonl", [row("a", "v1")])
    before = path.read_bytes()
    with pytest.raises(ValueError, match="OUTPUT_OVERWRITES_INPUT"):
        merge_references([path], path)
    assert path.read_bytes() == before
