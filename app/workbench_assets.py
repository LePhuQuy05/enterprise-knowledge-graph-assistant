"""Portable, pinned collection/model assets for an unreleased cloud workbench runtime."""

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from app.cloud_assets import download_checked
from app.collection_lexical import load_global_lexical_index
from app.collections import _relative_path, load_collection
from app.contracts import Record, Sha256, TenantId, canonical_json, sha256_bytes
from app.local_assets import verify_bundle
from app.reranking import verify_reranker_bundle

MAX_BUNDLE_BYTES = 384 * 1024 * 1024
MAX_FILE_BYTES = 140 * 1024 * 1024
MAX_BUNDLE_MANIFEST_BYTES = 1024 * 1024
DISK_MARGIN_BYTES = 64 * 1024 * 1024


class BundleFile(Record):
    path: str
    sha256: Sha256
    bytes: Annotated[int, Field(ge=0, le=MAX_FILE_BYTES)]

    @field_validator("path")
    @classmethod
    def safe_path(cls, value):
        _relative_path(value)
        if not value.startswith(("collection/", "lexical/", "models/bge/", "models/reranker/")):
            raise ValueError("WORKBENCH_FILE_PREFIX_INVALID")
        return value


class WorkbenchAssetBundle(Record):
    schema_version: Literal[1] = 1
    kind: Literal["WORKBENCH_COLLECTION_ASSETS"] = "WORKBENCH_COLLECTION_ASSETS"
    version: Annotated[str, Field(pattern=r"^wb_[0-9a-f]{64}$")]
    tenant_id: TenantId
    purpose: Literal["DEMO", "FIXTURE"]
    collection_version: Annotated[str, Field(pattern=r"^collection_[0-9a-f]{64}$")]
    collection_manifest_sha256: Sha256
    files: Annotated[tuple[BundleFile, ...], Field(min_length=1, max_length=1500)]

    @staticmethod
    def version_for(payload):
        return "wb_" + sha256_bytes(canonical_json({k: v for k, v in payload.items() if k != "version"}))

    @model_validator(mode="after")
    def integrity(self):
        names = tuple(row.path for row in self.files)
        if names != tuple(sorted(set(names))) or len({name.casefold() for name in names}) != len(names):
            raise ValueError("WORKBENCH_FILE_PATHS_UNIQUE_SORTED_CASE_SAFE")
        if sum(row.bytes for row in self.files) > MAX_BUNDLE_BYTES:
            raise ValueError("WORKBENCH_BUNDLE_SIZE_LIMIT")
        if self.version != self.version_for(self.model_dump(mode="json")):
            raise ValueError("WORKBENCH_BUNDLE_IDENTITY")
        return self

    @property
    def total_bytes(self):
        return sum(row.bytes for row in self.files)


def checked_path(root, relative):
    _relative_path(relative)
    root = Path(root).absolute()
    if root.is_symlink() or root.is_junction():
        raise ValueError("WORKBENCH_LINKED_ROOT")
    path = root
    for part in relative.split("/"):
        path = path / part
        if path.is_symlink() or path.is_junction():
            raise ValueError("WORKBENCH_LINKED_PATH")
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("WORKBENCH_PATH_OUTSIDE_ROOT")
    return path


def file_reference(path, relative):
    if path.is_symlink() or not path.is_file() or not 0 <= path.stat().st_size <= MAX_FILE_BYTES:
        raise ValueError("WORKBENCH_SOURCE_SIZE_OR_TYPE")
    with path.open("rb") as source:
        digest = hashlib.file_digest(source, "sha256").hexdigest()
    return BundleFile(path=relative, sha256=digest, bytes=path.stat().st_size)


@dataclass
class LoadedWorkbenchAssets:
    root: Path
    bundle: WorkbenchAssetBundle
    collection: object
    lexical: object

    def close(self):
        self.lexical.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def open_workbench_assets(root, bundle, *, tenant_id, allow_fixture=False):
    """Validate exact inventory and all shards, then return a thread-owned lexical connection."""
    bundle = WorkbenchAssetBundle.model_validate_json(bundle.model_dump_json())
    if bundle.tenant_id != tenant_id or (bundle.purpose != "DEMO" and not allow_fixture):
        raise ValueError("WORKBENCH_TENANT_OR_PURPOSE")
    root = Path(root)
    for row in bundle.files:
        if file_reference(checked_path(root, row.path), row.path) != row:
            raise ValueError("WORKBENCH_FILE_CHECKSUM_OR_SIZE")
    collection = load_collection(checked_path(root, "collection/manifest.json"), tenant_id=tenant_id,
        embedding_model_id="BAAI/bge-small-en-v1.5", allow_fixture=allow_fixture, cache_shards=4,
        trusted_manifest_sha256=bundle.collection_manifest_sha256)
    if collection.manifest.collection_version != bundle.collection_version or collection.manifest.purpose != bundle.purpose:
        raise ValueError("WORKBENCH_COLLECTION_BINDING")
    expected = {"collection/manifest.json", "lexical/manifest.json", "lexical/index.sqlite3",
                "models/bge/manifest.json", "models/reranker/manifest.json"}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        expected.add(f"collection/{ref.relative_path}/manifest.json")
        expected.update(f"collection/{ref.relative_path}/{row.name}" for row in shard.manifest.files)
    configs = {}
    for name, verifier in (("bge", verify_bundle), ("reranker", verify_reranker_bundle)):
        path = checked_path(root, f"models/{name}/manifest.json")
        if path.stat().st_size > 64 * 1024:
            raise ValueError("WORKBENCH_MODEL_MANIFEST_LIMIT")
        configs[name] = json.loads(path.read_bytes())
        verifier(path.parent, configs[name])
        expected.update(f"models/{name}/{row['path']}" for row in configs[name]["files"])
    if {row.path for row in bundle.files} != expected or collection.manifest.embedding_recipe.artifact_manifest_sha256 != (
        sha256_bytes(checked_path(root, "models/bge/manifest.json").read_bytes())
    ):
        raise ValueError("WORKBENCH_INVENTORY_OR_MODEL_RECIPE")
    # The lexical loader checks collection/version/recipe/database identity and read-only SQLite metadata.
    lexical_ref = next(row for row in bundle.files if row.path == "lexical/manifest.json")
    lexical = load_global_lexical_index(checked_path(root, "lexical/manifest.json"), collection,
        trusted_manifest_sha256=lexical_ref.sha256)
    return LoadedWorkbenchAssets(root, bundle, collection, lexical)


def plan_workbench_assets(collection_path, lexical_path, bge_dir, bge_manifest, reranker_dir, reranker_manifest,
                          *, tenant_id="demo", allow_fixture=False):
    collection = load_collection(Path(collection_path), tenant_id=tenant_id,
        embedding_model_id="BAAI/bge-small-en-v1.5", allow_fixture=allow_fixture, cache_shards=4)
    sources = {"collection/manifest.json": Path(collection_path), "lexical/manifest.json": Path(lexical_path),
        "lexical/index.sqlite3": Path(lexical_path).parent / "index.sqlite3",
        "models/bge/manifest.json": Path(bge_manifest), "models/reranker/manifest.json": Path(reranker_manifest)}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        for name in ("manifest.json", *(row.name for row in shard.manifest.files)):
            sources[f"collection/{ref.relative_path}/{name}"] = checked_path(collection.path.parent, ref.relative_path + "/" + name)
    for name, directory, manifest, verifier in (("bge", bge_dir, bge_manifest, verify_bundle),
        ("reranker", reranker_dir, reranker_manifest, verify_reranker_bundle)):
        config = json.loads(Path(manifest).read_bytes())
        verifier(Path(directory), config)
        sources.update({f"models/{name}/{row['path']}": checked_path(directory, row["path"]) for row in config["files"]})
    with load_global_lexical_index(Path(lexical_path), collection):
        pass
    rows = tuple(file_reference(path, name) for name, path in sorted(sources.items()))
    payload = {"tenant_id": tenant_id, "purpose": collection.manifest.purpose,
        "schema_version": 1, "kind": "WORKBENCH_COLLECTION_ASSETS",
        "collection_version": collection.manifest.collection_version, "collection_manifest_sha256": collection.manifest_sha256,
        "files": [r.model_dump(mode="json") for r in rows]}
    if collection.manifest.embedding_recipe.artifact_manifest_sha256 != sha256_bytes(Path(bge_manifest).read_bytes()):
        raise ValueError("WORKBENCH_MODEL_RECIPE_MISMATCH")
    bundle = WorkbenchAssetBundle.model_validate_json(canonical_json(payload | {"version": WorkbenchAssetBundle.version_for(payload)}))
    return bundle, sources


def materialize_workbench_assets(destination, bundle, sources, *, allow_fixture=False):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for row in bundle.files:
        source, target = Path(sources[row.path]), checked_path(destination, row.path)
        if file_reference(source, row.path) != row:
            raise ValueError("WORKBENCH_SOURCE_CHANGED_AFTER_PLAN")
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if file_reference(target, row.path) != row:
                raise ValueError("WORKBENCH_IMMUTABLE_TARGET_CONFLICT")
        else:
            with source.open("rb") as incoming, target.open("xb") as outgoing:
                shutil.copyfileobj(incoming, outgoing)
    with open_workbench_assets(destination, bundle, tenant_id=bundle.tenant_id, allow_fixture=allow_fixture):
        pass
    manifest = checked_path(destination, "bundle.json")
    raw = canonical_json(bundle.model_dump(mode="json")) + b"\n"
    if manifest.exists():
        if manifest.read_bytes() != raw:
            raise ValueError("WORKBENCH_IMMUTABLE_MANIFEST_CONFLICT")
    else:
        with manifest.open("xb") as output:
            output.write(raw)
    return manifest


def load_s3_workbench_assets(client, *, bucket, version, manifest_sha256, manifest_bytes, tenant_id, cache_root,
                             allow_fixture=False):
    """Only trusted release pins select a bundle; no listing, dynamic URLs or archive extraction."""
    import re

    if not re.fullmatch(r"wb_[0-9a-f]{64}", version) or not re.fullmatch(r"[0-9a-f]{64}", manifest_sha256) or (
        type(manifest_bytes) is not int or not 0 < manifest_bytes <= MAX_BUNDLE_MANIFEST_BYTES
    ):
        raise ValueError("WORKBENCH_RELEASE_PIN_INVALID")
    root = checked_path(cache_root, version)
    root.mkdir(parents=True, exist_ok=True)
    prefix = f"workbench/{version}/"
    download_checked(client, bucket=bucket, key=prefix + "bundle.json", target=root / "bundle.json",
        expected_sha256=manifest_sha256, expected_bytes=manifest_bytes)
    bundle = WorkbenchAssetBundle.model_validate_json((root / "bundle.json").read_bytes())
    if bundle.version != version or bundle.tenant_id != tenant_id or (bundle.purpose != "DEMO" and not allow_fixture):
        raise ValueError("WORKBENCH_TENANT_OR_PURPOSE")
    missing = sum(row.bytes for row in bundle.files if not checked_path(root, row.path).is_file())
    if shutil.disk_usage(root).free < missing + DISK_MARGIN_BYTES:
        raise ValueError("WORKBENCH_INSUFFICIENT_CACHE_DISK")
    for row in bundle.files:
        download_checked(client, bucket=bucket, key=prefix + row.path, target=checked_path(root, row.path),
            expected_sha256=row.sha256, expected_bytes=row.bytes)
    return open_workbench_assets(root, bundle, tenant_id=tenant_id, allow_fixture=allow_fixture)
