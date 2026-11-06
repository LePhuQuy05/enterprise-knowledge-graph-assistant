import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.collection_lexical import build_global_lexical_index
from app.collections import build_collection
from app.contracts import LocalBgeRecipe, canonical_json, sha256_bytes
from app.fixtures import FIXTURE_TENANT, fixture_content, fixture_recipes
from app.snapshot import SnapshotContent, build_snapshot
from app.workbench_assets import (
    BundleFile,
    WorkbenchAssetBundle,
    load_s3_workbench_assets,
    materialize_workbench_assets,
    open_workbench_assets,
    plan_workbench_assets,
)

ROOT = Path(__file__).resolve().parents[1]


class Objects:
    def __init__(self, bundle, directory):
        self.prefix = "workbench/" + bundle.version + "/"
        self.values = {self.prefix + row.path: (directory / row.path).read_bytes() for row in bundle.files}
        self.values[self.prefix + "bundle.json"] = (directory / "bundle.json").read_bytes()
        self.calls = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        raw = self.values[kwargs["Key"]]
        return {"Body": io.BytesIO(raw), "ContentLength": len(raw)}


@pytest.fixture
def bundle_case(tmp_path):
    manifests, dirs = {}, {}
    for name, original in (("bge", "bge-small-en-v1.5-onnx.json"), ("reranker", "ms-marco-MiniLM-L2-v2-onnx.json")):
        config = json.loads((ROOT / "configs/models" / original).read_bytes())
        directory = tmp_path / name
        for row in config["files"]:
            raw = ("SYNTHETIC_MODEL_NOT_EXECUTABLE " + row["path"]).encode()
            target = directory / row["path"]
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            row.update(bytes=len(raw), sha256=sha256_bytes(raw))
        manifest = directory / "manifest.json"
        manifest.write_bytes(canonical_json(config))
        manifests[name], dirs[name] = manifest, directory
    recipe = LocalBgeRecipe(artifact_manifest_sha256=sha256_bytes(manifests["bge"].read_bytes()),
        onnxruntime_version="fixture", tokenizers_version="fixture", input_role="document",
        input_recipe_version="title-section-text-v1")
    fixture = fixture_content()
    content = SnapshotContent(documents=fixture.documents, chunks=fixture.chunks, entities=(), mentions=(), assertions=())
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    index = tmp_path / "index"
    shard = build_snapshot(index, content, {c.chunk_id: vector for c in content.chunks},
        tenant_id=FIXTURE_TENANT, purpose="FIXTURE", extraction_recipe=fixture_recipes()[0], embedding_recipe=recipe,
        corpus_version="bundle-fixture-v1", normalizer_version="normalize-v1", chunking_version="chunk-v1", tokenizer_version="fixture-words-v1")
    collection_path = build_collection(index, [shard], tenant_id=FIXTURE_TENANT,
        embedding_model_id=recipe.model_id, allow_fixture=True)
    from app.collections import load_collection

    collection = load_collection(collection_path, tenant_id=FIXTURE_TENANT, embedding_model_id=recipe.model_id, allow_fixture=True)
    lexical = build_global_lexical_index(collection, tmp_path / "lexical")
    bundle, sources = plan_workbench_assets(collection_path, lexical, dirs["bge"], manifests["bge"], dirs["reranker"],
        manifests["reranker"], tenant_id=FIXTURE_TENANT, allow_fixture=True)
    directory = tmp_path / "portable"
    materialize_workbench_assets(directory, bundle, sources, allow_fixture=True)
    objects = Objects(bundle, directory)
    pins = dict(bucket="fixture-private", version=bundle.version, manifest_sha256=sha256_bytes((directory / "bundle.json").read_bytes()),
        manifest_bytes=(directory / "bundle.json").stat().st_size, tenant_id=FIXTURE_TENANT, cache_root=tmp_path / "cache",
        allow_fixture=True)
    return SimpleNamespace(bundle=bundle, sources=sources, directory=directory, objects=objects, pins=pins)


def test_portable_bundle_preserves_source_and_global_lexical_identity(bundle_case):
    s = bundle_case
    with open_workbench_assets(s.directory, s.bundle, tenant_id=FIXTURE_TENANT, allow_fixture=True) as loaded:
        assert loaded.collection.manifest_sha256 == s.bundle.collection_manifest_sha256
        assert loaded.lexical.search("Service B")
        for ref in loaded.collection.manifest.shards:
            assert loaded.collection.get_shard(ref.shard_id).manifest_sha256 == ref.manifest_sha256
    assert materialize_workbench_assets(s.directory, s.bundle, s.sources, allow_fixture=True) == s.directory / "bundle.json"


def test_cloud_bundle_load_and_warm_read_do_not_refetch_verified_objects(bundle_case):
    s = bundle_case
    with load_s3_workbench_assets(s.objects, **s.pins) as cold:
        first = cold.lexical.search("Service B")
    calls = len(s.objects.calls)
    assert calls == len(s.bundle.files) + 1
    with load_s3_workbench_assets(s.objects, **s.pins) as warm:
        assert warm.lexical.search("Service B") == first
    assert len(s.objects.calls) == calls


def test_fixture_cannot_load_in_demo_even_with_valid_release_hash(bundle_case):
    s = bundle_case
    with pytest.raises(ValueError, match="PURPOSE"):
        load_s3_workbench_assets(s.objects, **(s.pins | {"allow_fixture": False}))
    assert len(s.objects.calls) == 1


@pytest.mark.parametrize("field,value", [("version", "../outside"), ("manifest_sha256", "x"),
    ("manifest_bytes", True), ("manifest_bytes", 1024 * 1024 + 1)])
def test_invalid_release_pins_fail_before_s3_or_filesystem(bundle_case, field, value):
    s = bundle_case
    with pytest.raises(ValueError, match="PIN_INVALID"):
        load_s3_workbench_assets(s.objects, **(s.pins | {field: value}))
    assert not s.objects.calls and not s.pins["cache_root"].exists()


@pytest.mark.parametrize("path", ["../secret", "collection/../../secret", "C:/keys", "collection\\manifest.json", "bundle.json", "models/bge/CON"])
def test_bundle_paths_cannot_escape_or_overwrite_control_manifest(path):
    with pytest.raises(ValueError):
        BundleFile(path=path, sha256="a" * 64, bytes=1)


def test_tampered_cache_is_rejected_without_redownload_or_silent_repair(bundle_case):
    s = bundle_case
    with load_s3_workbench_assets(s.objects, **s.pins):
        pass
    target = s.pins["cache_root"] / s.bundle.version / "lexical/index.sqlite3"
    raw = target.read_bytes()
    target.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    calls = len(s.objects.calls)
    with pytest.raises(ValueError, match="CACHED_CHECKSUM"):
        load_s3_workbench_assets(s.objects, **s.pins)
    assert len(s.objects.calls) == calls


def test_disk_guard_stops_after_small_manifest_before_large_downloads(bundle_case, monkeypatch):
    s = bundle_case
    monkeypatch.setattr("app.workbench_assets.shutil.disk_usage", lambda _: SimpleNamespace(free=1))
    with pytest.raises(ValueError, match="INSUFFICIENT_CACHE_DISK"):
        load_s3_workbench_assets(s.objects, **s.pins)
    assert len(s.objects.calls) == 1


def test_changed_source_prevents_publication_marker(bundle_case, tmp_path):
    s = bundle_case
    s.sources["lexical/index.sqlite3"].write_bytes(b"changed after plan")
    target = tmp_path / "new-portable"
    with pytest.raises(ValueError, match="SOURCE_CHANGED_AFTER_PLAN"):
        materialize_workbench_assets(target, s.bundle, s.sources, allow_fixture=True)
    assert not (target / "bundle.json").exists()


def test_bundle_inventory_and_zero_byte_snapshot_files_are_preserved(bundle_case):
    s = bundle_case
    assert any(row.bytes == 0 for row in s.bundle.files)
    payload = s.bundle.model_dump(mode="json")
    payload["files"] += [{"path": "lexical/EXTRA", "bytes": 1, "sha256": sha256_bytes(b"x")}]
    payload["files"].sort(key=lambda row: row["path"])
    payload["version"] = WorkbenchAssetBundle.version_for(payload)
    extra = WorkbenchAssetBundle.model_validate_json(canonical_json(payload))
    (s.directory / "lexical/EXTRA").write_bytes(b"x")
    with pytest.raises(ValueError, match="INVENTORY"):
        open_workbench_assets(s.directory, extra, tenant_id=FIXTURE_TENANT, allow_fixture=True)


def test_exact_executable_model_manifest_is_bound_to_document_recipe(bundle_case):
    s = bundle_case
    manifest = s.sources["models/bge/manifest.json"]
    config = json.loads(manifest.read_bytes())
    config["notice"] = "changed recipe identity"
    manifest.write_bytes(canonical_json(config))
    with pytest.raises(ValueError, match="MODEL_RECIPE_MISMATCH"):
        plan_workbench_assets(s.sources["collection/manifest.json"], s.sources["lexical/manifest.json"],
            manifest.parent, manifest, s.sources["models/reranker/manifest.json"].parent,
            s.sources["models/reranker/manifest.json"], tenant_id=FIXTURE_TENANT, allow_fixture=True)
