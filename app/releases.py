"""Read-only serving release policy, separately committed by the publisher."""

import re

from app.query_auth import ApiError


class ReleaseRegistry:
    def __init__(self, store):
        self.store = store

    def get(self, tenant, snapshot_version, *, for_history=True):
        if not re.fullmatch(r"snap_[0-9a-f]{64}", snapshot_version):
            raise ApiError(503, "SNAPSHOT_UNAVAILABLE")
        row = self.store.get(("TENANT#" + tenant, "SNAPSHOT#" + snapshot_version))
        if not row or row.get("record_type") != "SNAPSHOT_RELEASE" or row.get("schema_version") != 1 or (
            row.get("tenant_id") != tenant or row.get("snapshot_version") != snapshot_version
            or row.get("publication_status") != "PUBLISHED" or row.get("revoked") is not False
            or (for_history and row.get("valid_for_history") is not True)
        ):
            raise ApiError(503, "SNAPSHOT_UNAVAILABLE")
        config = row.get("asset_config", {})
        if config.get("tenant_id") != tenant or config.get("snapshot_version") != snapshot_version or (
            config.get("snapshot_manifest_sha256") != row.get("manifest_sha256")
            or not re.fullmatch(r"[0-9a-f]{64}", str(row.get("manifest_sha256", "")))
        ):
            raise ApiError(503, "SNAPSHOT_UNAVAILABLE")
        return row

    def current(self, tenant):
        pointer = self.store.get(("TENANT#" + tenant, "RELEASE"))
        if not pointer or pointer.get("record_type") != "RELEASE_POINTER" or pointer.get("schema_version") != 1 or (
            pointer.get("tenant_id") != tenant or type(pointer.get("generation")) is not int
            or pointer["generation"] < 1
        ):
            raise ApiError(503, "SNAPSHOT_UNAVAILABLE")
        row = self.get(tenant, pointer.get("snapshot_version", ""), for_history=False)
        if row["manifest_sha256"] != pointer.get("manifest_sha256"):
            raise ApiError(503, "SNAPSHOT_UNAVAILABLE")
        return row
