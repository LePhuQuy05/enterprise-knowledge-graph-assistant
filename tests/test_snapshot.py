import io
import json
import zipfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from app.contracts import SnapshotManifest, canonical_json, sha256_bytes
from app.fixtures import (
    FIXTURE_EMBEDDING_MODEL,
    FIXTURE_TENANT,
    build_fixture,
    fixture_content,
    fixture_recipes,
    fixture_vector,
)
from app.snapshot import SnapshotError, build_snapshot, load_snapshot


def read(directory, **changes):
    return load_snapshot(directory, **(dict(
        tenant_id=FIXTURE_TENANT, embedding_model_id=FIXTURE_EMBEDDING_MODEL, allow_fixture=True,
    ) | changes))


def reseal(directory, name=None, raw=None, uncompressed=None):
    """Malicious republisher updates checksums too, to exercise deeper invariants."""
    manifest_path = directory / "manifest.json"
    data = json.loads(manifest_path.read_bytes())
    if name:
        (directory / name).write_bytes(raw)
        item = next(f for f in data["files"] if f["name"] == name)
        item.update(sha256=sha256_bytes(raw), byte_size=len(raw),
                    uncompressed_size=len(raw) if uncompressed is None else uncompressed)
    data["snapshot_version"] = SnapshotManifest.version_for(data)
    manifest_path.write_bytes(canonical_json(data))


def test_fixture_roundtrip_preserves_source_provenance_and_negation(loaded):
    assert loaded.manifest.counts.model_dump() == dict(
        documents=2, chunks=5, entities=5, mentions=9, assertions=4,
    )
    assert loaded.vectors.shape == (5, 1024)
    assert loaded.vectors.dtype == np.float32
    assert not loaded.vectors.flags.writeable
    assert sum(a.polarity == "NEGATIVE" for a in loaded.content.assertions) == 1
    entity = next(e for e in loaded.content.entities if e.canonical_name == "Service Z")
    assert any(m.entity_id == entity.entity_id for m in loaded.content.mentions)
    assert not any(entity.entity_id in (a.subject_id, a.object_id) for a in loaded.content.assertions)
    assert any("\u2028" in c.text for c in loaded.content.chunks)


def test_idempotent_build_keeps_published_manifest(snapshot_dir):
    before = (snapshot_dir / "manifest.json").read_bytes()
    assert build_fixture(snapshot_dir.parent) == snapshot_dir
    assert (snapshot_dir / "manifest.json").read_bytes() == before
    assert not list(snapshot_dir.parent.glob(".staging-*"))


def test_zip_metadata_does_not_depend_on_host_os(snapshot_dir):
    with zipfile.ZipFile(snapshot_dir / "vectors.npz") as archive:
        info = archive.infolist()[0]
        assert info.date_time == (1980, 1, 1, 0, 0, 0)
        assert info.create_system == 3
        assert info.external_attr == 0o600 << 16


def test_transient_windows_rename_lock_retries_without_rebuilding(tmp_path, monkeypatch):
    original = Path.rename
    calls = []

    def flaky(path, destination):
        calls.append(path)
        if len(calls) == 1:
            error = PermissionError("Synthetic Windows sharing/access lock")
            error.winerror = 5
            raise error
        return original(path, destination)

    monkeypatch.setattr(Path, "rename", flaky)
    monkeypatch.setattr("app.snapshot.time.sleep", lambda _: None)
    directory = build_fixture(tmp_path / "snapshots")
    assert read(directory)
    assert len(calls) == 2
    assert calls[0] == calls[1]


@pytest.mark.parametrize("windows_error,expected_calls", [(5, 5), (None, 1)])
def test_persistent_rename_failure_still_fails_closed(tmp_path, monkeypatch, windows_error, expected_calls):
    calls = []

    def fail(path, destination):
        calls.append(path)
        error = PermissionError("Synthetic persistent access denial")
        if windows_error is not None:
            error.winerror = windows_error
        raise error

    monkeypatch.setattr(Path, "rename", fail)
    monkeypatch.setattr("app.snapshot.time.sleep", lambda _: None)
    with pytest.raises(PermissionError):
        build_fixture(tmp_path / "snapshots")
    assert len(calls) == expected_calls
    assert not list((tmp_path / "snapshots").glob("snap_*"))


def test_input_order_does_not_change_snapshot_id(tmp_path):
    content = fixture_content()
    extraction, embedding = fixture_recipes()
    vectors = {c.chunk_id: fixture_vector(c.text) for c in content.chunks}
    params = dict(tenant_id=FIXTURE_TENANT, purpose="FIXTURE", extraction_recipe=extraction,
                  embedding_recipe=embedding, corpus_version="test", normalizer_version="normalize-v1",
                  chunking_version="chunk-v1", tokenizer_version="fixture-words-v1")
    a = build_snapshot(tmp_path / "a", content, vectors, **params)
    reversed_content = replace(content, **{
        field: tuple(reversed(getattr(content, field)))
        for field in ("documents", "chunks", "entities", "mentions", "assertions")
    })
    b = build_snapshot(tmp_path / "b", reversed_content, vectors, **params)
    assert a.name == b.name
    for name in ("documents.jsonl", "chunks.jsonl", "mentions.jsonl", "vectors.npz"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


@pytest.mark.parametrize("changes", [
    {"tenant_id": "other"}, {"embedding_model_id": "other"}, {"allow_fixture": False},
    {"trusted_manifest_sha256": "0" * 64},
])
def test_approved_context_required(snapshot_dir, changes):
    with pytest.raises(SnapshotError):
        read(snapshot_dir, **changes)


def test_trusted_manifest(loaded, snapshot_dir):
    assert read(snapshot_dir, trusted_manifest_sha256=loaded.manifest_sha256)


@pytest.mark.parametrize("change", ["missing", "extra", "tampered", "duplicate_json_key", "old_schema"])
def test_corrupted_snapshot_fails_closed(snapshot_dir, change):
    if change == "missing":
        (snapshot_dir / "mentions.jsonl").unlink()
    elif change == "extra":
        (snapshot_dir / "unexpected.txt").write_text("x")
    elif change == "tampered":
        with (snapshot_dir / "chunks.jsonl").open("ab") as stream:
            stream.write(b" ")
    elif change == "duplicate_json_key":
        path = snapshot_dir / "manifest.json"
        path.write_bytes(b'{"schema_version":2,' + path.read_bytes()[1:])
    else:
        path = snapshot_dir / "manifest.json"
        data = json.loads(path.read_bytes()) | {"schema_version": 1}
        data["snapshot_version"] = SnapshotManifest.version_for(data)
        path.write_bytes(canonical_json(data))
    with pytest.raises(SnapshotError):
        read(snapshot_dir)


def test_resealed_dangling_mention_is_rejected(snapshot_dir):
    path = snapshot_dir / "mentions.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().split(b"\n") if line]
    from app.contracts import Mention

    rows[0]["entity_id"] = "ent_missing"
    rows[0]["mention_id"] = Mention.expected_id(rows[0])
    rows.sort(key=lambda row: row["mention_id"])
    reseal(snapshot_dir, "mentions.jsonl", b"".join(canonical_json(r) + b"\n" for r in rows))
    with pytest.raises(SnapshotError, match="dangling"):
        read(snapshot_dir)


def test_valid_hash_does_not_hide_wrong_chunk_positions(snapshot_dir):
    path = snapshot_dir / "chunks.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().split(b"\n") if line]
    # Position is not part of identity, but per-document positions must still be valid.
    rows[0]["position"] = 100
    reseal(snapshot_dir, "chunks.jsonl", b"".join(canonical_json(r) + b"\n" for r in rows))
    with pytest.raises(SnapshotError, match="positions"):
        read(snapshot_dir)


def test_valid_hashes_cannot_hide_changed_normalized_source(snapshot_dir):
    from app.contracts import text_hash

    path = snapshot_dir / "documents.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().split(b"\n") if line]
    rows[0]["normalized_text"] = "x" + rows[0]["normalized_text"][1:]
    rows[0]["normalized_sha256"] = text_hash(rows[0]["normalized_text"])
    reseal(snapshot_dir, "documents.jsonl", b"".join(canonical_json(r) + b"\n" for r in rows))
    with pytest.raises(SnapshotError, match="slice"):
        read(snapshot_dir)


def test_unsupported_qualifier_rejected_even_with_new_assertion_id(snapshot_dir):
    from app.contracts import RelationAssertion

    path = snapshot_dir / "assertions.jsonl"
    rows = [json.loads(line) for line in path.read_bytes().split(b"\n") if line]
    rows[0]["qualifiers"] = [{"kind": "REGION", "text": "an invented region"}]
    rows[0]["assertion_id"] = RelationAssertion.expected_id(rows[0])
    rows.sort(key=lambda row: row["assertion_id"])
    reseal(snapshot_dir, "assertions.jsonl", b"".join(canonical_json(r) + b"\n" for r in rows))
    with pytest.raises(SnapshotError, match="qualifier"):
        read(snapshot_dir)


@pytest.mark.parametrize("kind", ["dimension", "dtype", "nan", "zero", "object", "extra_member"])
def test_resealed_dangerous_vectors_rejected_before_serving(snapshot_dir, loaded, kind):
    matrix = loaded.vectors.copy()
    if kind == "dimension":
        matrix = matrix[:, :512]
    elif kind == "dtype":
        matrix = matrix.astype(np.float64)
    elif kind == "nan":
        matrix[0, 0] = np.nan
    elif kind == "zero":
        matrix[0] = 0
    elif kind == "object":
        matrix = np.empty((5, 1024), dtype=object)
        matrix[:] = "not executed"
    npy = io.BytesIO()
    np.lib.format.write_array(npy, matrix, allow_pickle=kind == "object", version=(1, 0))
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("embeddings.npy", npy.getvalue())
        if kind == "extra_member":
            archive.writestr("../unexpected.npy", b"never extract archives")
    reseal(snapshot_dir, "vectors.npz", output.getvalue(), len(npy.getvalue()))
    with pytest.raises(SnapshotError):
        read(snapshot_dir)


def test_decompression_size_bound(snapshot_dir):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("embeddings.npy", bytes(1_000_000))
    reseal(snapshot_dir, "vectors.npz", output.getvalue(), 1_000_000)
    with pytest.raises(SnapshotError, match="expansion"):
        read(snapshot_dir)
