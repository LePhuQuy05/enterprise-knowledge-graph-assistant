"""Pin the source-proven experimental graph alongside the V1 portable corpus.

The existing workbench inventory is unchanged. This small second bundle never
loads Python/pickle or follows graph-provided paths/URLs. Graph coverage is
reported separately from corpus size and is not a statement of correctness.
"""

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, model_validator

from app.authorization_evidence import load_authorization_facts
from app.authorization_retrieval import combine_evidence_graph
from app.cloud_assets import download_checked
from app.contracts import Record, Sha256, TenantId, canonical_json, sha256_bytes
from app.reviewed_source_graph import load_reviewed_source_graph
from app.workbench_assets import checked_path, load_s3_workbench_assets

GRAPH_FILES = tuple(sorted(("reviewed/release.json", "reviewed/relations-v2.json", "reviewed/actions.json",
    "reviewed/sidecar.json", "authorization/release.json", "authorization/facts.json")))
MAX_GRAPH_BYTES = 16*1024*1024
MAX_MANIFEST_BYTES = 16384


class GraphFile(Record):
    path: Literal[GRAPH_FILES]
    sha256: Sha256
    bytes: Annotated[int, Field(gt=0, le=MAX_GRAPH_BYTES)]


class ScopedGraphBundle(Record):
    schema_version: Literal[2] = 2
    kind: Literal["SCOPED_SOURCE_GRAPH_ASSETS"] = "SCOPED_SOURCE_GRAPH_ASSETS"
    version: Annotated[str, Field(pattern=r"^sg_[0-9a-f]{64}$")]
    tenant_id: TenantId
    collection_manifest_sha256: Sha256
    graph_release_sha256: Sha256
    files: Annotated[tuple[GraphFile, ...], Field(min_length=6, max_length=6)]
    experimental: Literal[True] = True
    human_reviewed: Literal[False] = False

    @staticmethod
    def version_for(payload):
        return "sg_"+sha256_bytes(canonical_json({k: v for k, v in payload.items() if k != "version"}))

    @model_validator(mode="after")
    def integrity(self):
        if tuple(row.path for row in self.files) != GRAPH_FILES or sum(row.bytes for row in self.files) > MAX_GRAPH_BYTES:
            raise ValueError("SCOPED_GRAPH_INVENTORY_OR_BOUND")
        if self.version != self.version_for(self.model_dump(mode="json")):
            raise ValueError("SCOPED_GRAPH_IDENTITY")
        return self


def open_scoped_graph(root, bundle, collection):
    bundle = ScopedGraphBundle.model_validate_json(canonical_json(bundle.model_dump(mode="json")))
    if collection.manifest_sha256 != bundle.collection_manifest_sha256 or collection.manifest.tenant_id != bundle.tenant_id:
        raise ValueError("SCOPED_GRAPH_COLLECTION_BINDING")
    for row in bundle.files:
        path = checked_path(root, row.path)
        if not path.is_file() or path.stat().st_size != row.bytes or sha256_bytes(path.read_bytes()) != row.sha256:
            raise ValueError("SCOPED_GRAPH_FILE_INTEGRITY")
    pins = {row.path: row.sha256 for row in bundle.files}
    reviewed = load_reviewed_source_graph(checked_path(root, "reviewed/release.json"), collection,
        trusted_sha256=pins["reviewed/release.json"], allow_experimental=True)
    docs, chunks = {}, {}
    for shard in collection.manifest.shards:
        loaded = collection.get_shard(shard.shard_id)
        docs.update(loaded.documents)
        chunks.update(loaded.chunks)
    authorization = load_authorization_facts(checked_path(root, "authorization/release.json"), collection, chunks, docs,
        trusted_sha256=pins["authorization/release.json"], allow_experimental=True)
    graph = combine_evidence_graph(collection, reviewed, authorization)
    if graph.release_sha256 != bundle.graph_release_sha256:
        raise ValueError("SCOPED_GRAPH_COMBINED_BINDING")
    return graph


def plan_scoped_graph(root, collection, *, graph_release_sha256):
    refs = []
    for name in GRAPH_FILES:
        path = checked_path(root, name)
        if not path.is_file() or not 0 < path.stat().st_size <= MAX_GRAPH_BYTES:
            raise ValueError("SCOPED_GRAPH_SOURCE_BOUND")
        refs.append(GraphFile(path=name, sha256=sha256_bytes(path.read_bytes()), bytes=path.stat().st_size))
    payload = {"schema_version": 2, "kind": "SCOPED_SOURCE_GRAPH_ASSETS", "tenant_id": collection.manifest.tenant_id,
        "collection_manifest_sha256": collection.manifest_sha256, "graph_release_sha256": graph_release_sha256,
        "files": [row.model_dump(mode="json") for row in refs], "experimental": True, "human_reviewed": False}
    bundle = ScopedGraphBundle.model_validate_json(canonical_json(payload | {"version": ScopedGraphBundle.version_for(payload)}))
    open_scoped_graph(root, bundle, collection)
    return bundle


def materialize_scoped_graph(destination, source, bundle, collection):
    open_scoped_graph(source, bundle, collection)
    for row in bundle.files:
        target = checked_path(destination, row.path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.stat().st_size != row.bytes or sha256_bytes(target.read_bytes()) != row.sha256:
                raise ValueError("SCOPED_GRAPH_TARGET_CONFLICT")
        else:
            with checked_path(source, row.path).open("rb") as src, target.open("xb") as dst:
                shutil.copyfileobj(src, dst)
    open_scoped_graph(destination, bundle, collection)
    target = checked_path(destination, "bundle.json")
    raw = canonical_json(bundle.model_dump(mode="json"))
    if target.exists():
        if target.read_bytes() != raw:
            raise ValueError("SCOPED_GRAPH_TARGET_CONFLICT")
    else:
        with target.open("xb") as output:
            output.write(raw)
    return target


def load_s3_scoped_graph(client, collection, *, bucket, version, manifest_sha256, manifest_bytes, cache_root):
    import re

    if not re.fullmatch(r"sg_[0-9a-f]{64}", version) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) or (
            type(manifest_bytes) is not int or not 0 < manifest_bytes <= MAX_MANIFEST_BYTES):
        raise ValueError("SCOPED_GRAPH_DOWNLOAD_PINS")
    root = checked_path(cache_root, version)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"scoped-graph/{version}/"
    manifest = checked_path(root, "bundle.json")
    download_checked(client, bucket=bucket, key=prefix+"bundle.json", target=manifest,
        expected_sha256=manifest_sha256, expected_bytes=manifest_bytes)
    bundle = ScopedGraphBundle.model_validate_json(manifest.read_bytes())
    if bundle.version != version or bundle.tenant_id != collection.manifest.tenant_id or (
            bundle.collection_manifest_sha256 != collection.manifest_sha256):
        raise ValueError("SCOPED_GRAPH_DOWNLOAD_BINDING")
    # The fixed inventory is small, but reserve space before any data downloads.
    if shutil.disk_usage(root).free < sum(row.bytes for row in bundle.files)+64*1024*1024:
        raise ValueError("SCOPED_GRAPH_DISK_MARGIN")
    for row in bundle.files:
        download_checked(client, bucket=bucket, key=prefix+row.path, target=checked_path(root, row.path),
            expected_sha256=row.sha256, expected_bytes=row.bytes)
    return root, bundle, open_scoped_graph(root, bundle, collection)


@dataclass
class LoadedScopedAssets:
    base: object
    graph_root: Path
    graph_bundle: ScopedGraphBundle
    graph: object

    def close(self):
        self.base.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def load_s3_scoped_assets(client, *, assets, graph_assets, cache_root):
    base = load_s3_workbench_assets(client, **assets, cache_root=Path(cache_root)/"corpus")
    try:
        root, bundle, graph = load_s3_scoped_graph(client, base.collection, **graph_assets, cache_root=Path(cache_root)/"graph")
        return LoadedScopedAssets(base, root, bundle, graph)
    except Exception:
        base.close()
        raise
