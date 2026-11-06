"""Private content-addressed results; authorized services derive every object key."""

import re
from pathlib import Path

from botocore.exceptions import ClientError

from app.contracts import canonical_json, sha256_bytes
from app.owned_runs import RUN_ID

MAX_RESULT_BYTES = 4 * 1024 * 1024


def result_key(auth, run_id, method, digest):
    if not re.fullmatch(RUN_ID, run_id) or method not in {"bm25", "vector", "hybrid", "hybrid_reranked"} or (
        not re.fullmatch(r"[0-9a-f]{64}", digest)
    ):
        raise ValueError("RUN_ARTIFACT_ID_INVALID")
    return f"tenants/{auth.tenant_id}/users/{auth.user_id}/runs/{run_id}/methods/{method}/{digest}.json"


class LocalRunBlobs:
    def __init__(self, root, *, read_only=False):
        self.root, self.read_only = Path(root).resolve(), read_only
        if self.read_only:
            if not self.root.is_dir():
                raise ValueError("RUN_READ_ONLY_ARTIFACT_ROOT_MISSING")
        else:
            self.root.mkdir(parents=True, exist_ok=True)

    def path(self, key):
        if type(key) is not str or "\\" in key:
            raise ValueError("RUN_ARTIFACT_PATH_INVALID")
        path = (self.root / key).resolve()
        if not path.is_relative_to(self.root) or path == self.root:
            raise ValueError("RUN_ARTIFACT_PATH_INVALID")
        return path

    def put(self, key, raw):
        if self.read_only:
            raise ValueError("RUN_ARTIFACTS_READ_ONLY")
        path = self.path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with path.open("xb") as output:
                output.write(raw)
                output.flush()
        except FileExistsError:
            if self.get(key) != raw:
                raise ValueError("RUN_IMMUTABLE_ARTIFACT_CONFLICT") from None

    def get(self, key):
        with self.path(key).open("rb") as stream:
            raw = stream.read(MAX_RESULT_BYTES + 1)
        if len(raw) > MAX_RESULT_BYTES:
            raise ValueError("RUN_RESULT_SIZE_LIMIT")
        return raw


class S3RunBlobs:
    """Requires a private deployment bucket and a separately reviewed IAM policy."""

    def __init__(self, client, bucket):
        self.client, self.bucket = client, bucket

    def put(self, key, raw):
        try:
            self.client.put_object(Bucket=self.bucket, Key=key, Body=raw, ContentType="application/json",
                IfNoneMatch="*", Metadata={"sha256": sha256_bytes(raw)})
        except ClientError as exc:
            if exc.response["Error"]["Code"] != "PreconditionFailed":
                raise
            if self.get(key) != raw:
                raise ValueError("RUN_IMMUTABLE_ARTIFACT_CONFLICT") from None

    def get(self, key):
        response = self.client.get_object(Bucket=self.bucket, Key=key)
        try:
            if response.get("ContentLength", 0) > MAX_RESULT_BYTES:
                raise ValueError("RUN_RESULT_SIZE_LIMIT")
            raw = response["Body"].read(MAX_RESULT_BYTES + 1)
            if len(raw) > MAX_RESULT_BYTES:
                raise ValueError("RUN_RESULT_SIZE_LIMIT")
            return raw
        finally:
            response["Body"].close()


class RunArtifactRepository:
    def __init__(self, blobs):
        self.blobs = blobs

    def save(self, auth, run_id, method, result):
        raw = canonical_json(result)
        if len(raw) > MAX_RESULT_BYTES:
            raise ValueError("RUN_RESULT_SIZE_LIMIT")
        digest = sha256_bytes(raw)
        key = result_key(auth, run_id, method, digest)
        self.blobs.put(key, raw)
        if self.blobs.get(key) != raw:
            raise ValueError("RUN_RESULT_READBACK_MISMATCH")
        return {"sha256": digest, "bytes": len(raw)}

    def read(self, auth, run_id, method, reference):
        if set(reference) != {"sha256", "bytes"} or type(reference["bytes"]) is not int or (
            not 0 < reference["bytes"] <= MAX_RESULT_BYTES
        ):
            raise ValueError("RUN_RESULT_REFERENCE_INVALID")
        raw = self.blobs.get(result_key(auth, run_id, method, reference["sha256"]))
        if len(raw) != reference["bytes"] or sha256_bytes(raw) != reference["sha256"]:
            raise ValueError("RUN_RESULT_CHECKSUM_MISMATCH")
        return raw
