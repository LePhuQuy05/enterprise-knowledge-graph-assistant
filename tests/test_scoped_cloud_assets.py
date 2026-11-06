"""Actual319 source validation with an in-memory S3 stand-in; no network."""

import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.contracts import canonical_json, sha256_bytes
from app.scoped_cloud_assets import (
    GRAPH_FILES,
    ScopedGraphBundle,
    load_s3_scoped_graph,
    materialize_scoped_graph,
    open_scoped_graph,
    plan_scoped_graph,
)
from scripts.migrate_collection319_graph import OUTPUT, load_migrated


@pytest.fixture(scope="module")
def actual_graph():
    if not (OUTPUT/"selection.json").is_file() or not (Path(__file__).resolve().parents[1]/"results/collection319-local/runtime.json").is_file():
        pytest.skip("Actual319 local artifacts are not distributed in the public source checkout")
    collection, _, graph = load_migrated()
    bundle = plan_scoped_graph(OUTPUT, collection, graph_release_sha256=graph.release_sha256)
    return SimpleNamespace(collection=collection, graph=graph, bundle=bundle)


class Objects:
    def __init__(self, bundle, directory):
        self.prefix = f"scoped-graph/{bundle.version}/"
        self.values = {self.prefix+row.path: (directory/row.path).read_bytes() for row in bundle.files}
        self.values[self.prefix+"bundle.json"] = (directory/"bundle.json").read_bytes()
        self.calls = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        raw = self.values[kwargs["Key"]]
        return {"Body": io.BytesIO(raw), "ContentLength": len(raw)}


@pytest.fixture
def portable(actual_graph, tmp_path):
    a = actual_graph
    path = materialize_scoped_graph(tmp_path/"portable", OUTPUT, a.bundle, a.collection)
    objects = Objects(a.bundle, path.parent)
    pins = dict(bucket="fixture-private", version=a.bundle.version, manifest_bytes=path.stat().st_size,
        manifest_sha256=sha256_bytes(path.read_bytes()), cache_root=tmp_path/"cache")
    return SimpleNamespace(**vars(a), directory=path.parent, objects=objects, pins=pins)


def test_exact_actual_sources_download_and_warm_reuse_without_requery(portable):
    p = portable
    _, bundle, graph = load_s3_scoped_graph(p.objects, p.collection, **p.pins)
    assert bundle == p.bundle and graph.release_sha256 == p.graph.release_sha256
    assert len(p.objects.calls) == 7 and len(graph.facts) == 27
    assert tuple(row.path for row in bundle.files) == GRAPH_FILES and bundle.human_reviewed is False
    _, _, replay = load_s3_scoped_graph(p.objects, p.collection, **p.pins)
    assert replay == graph and len(p.objects.calls) == 7


@pytest.mark.parametrize("kind", ["bytes", "missing", "combined"])
def test_graph_integrity_and_combined_identity_rejected(portable, kind):
    p = portable
    bundle = p.bundle
    path = p.directory/"authorization/facts.json"
    if kind == "bytes":
        raw = path.read_bytes()
        path.write_bytes(raw.replace(b"records", b"recordX", 1))
    elif kind == "missing":
        path.unlink()
    else:
        value = bundle.model_dump(mode="json") | {"graph_release_sha256": "0"*64}
        value["version"] = ScopedGraphBundle.version_for(value)
        bundle = ScopedGraphBundle.model_validate_json(canonical_json(value))
    with pytest.raises(ValueError):
        open_scoped_graph(p.directory, bundle, p.collection)


@pytest.mark.parametrize("alteration", ["path", "extra", "order", "version", "tenant", "collection"])
def test_untrusted_manifest_cannot_choose_paths_or_change_collection(portable, alteration):
    p = portable
    payload = p.bundle.model_dump(mode="json")
    if alteration == "path":
        payload["files"][0]["path"] = "../../credentials"
    elif alteration == "extra":
        payload["files"].append(payload["files"][0])
    elif alteration == "order":
        payload["files"].reverse()
    elif alteration == "version":
        payload["version"] = "sg_"+"0"*64
    elif alteration == "tenant":
        payload["tenant_id"] = "another"
    else:
        payload["collection_manifest_sha256"] = "0"*64
    if alteration in {"tenant", "collection"}:
        payload["version"] = ScopedGraphBundle.version_for(payload)
    with pytest.raises(ValueError):
        bundle = ScopedGraphBundle.model_validate_json(canonical_json(payload))
        open_scoped_graph(p.directory, bundle, p.collection)


def test_bad_s3_checksum_and_partial_body_never_create_accepted_bundle(portable):
    p = portable
    key = p.objects.prefix+"reviewed/actions.json"
    original = p.objects.values[key]
    p.objects.values[key] = b"X"*len(original)
    with pytest.raises(ValueError, match="CHECKSUM_MISMATCH"):
        load_s3_scoped_graph(p.objects, p.collection, **p.pins)
    assert not list(p.pins["cache_root"].rglob("*.partial"))
    p.objects.values[key] = original[:-1]
    with pytest.raises(ValueError, match="SIZE_MISMATCH"):
        load_s3_scoped_graph(p.objects, p.collection, **p.pins)


def test_target_materialization_is_immutable(portable):
    p = portable
    assert materialize_scoped_graph(p.directory, OUTPUT, p.bundle, p.collection).is_file()
    target = p.directory/"bundle.json"
    target.write_text(json.dumps({"fake": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="TARGET_CONFLICT"):
        materialize_scoped_graph(p.directory, OUTPUT, p.bundle, p.collection)
