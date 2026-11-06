"""V2 worker, private artifacts and API projections; no route/deployment changes."""

import re

from app.contracts import canonical_json, sha256_bytes
from app.external_http import strict_json
from app.owned_runs import RUN_ID, TERMINAL
from app.query_auth import ApiError, AuthContext
from app.run_artifacts import MAX_RESULT_BYTES
from app.run_budget import RunStageBusy
from app.run_execution import OwnedRunWorker
from app.run_service import OwnedRunService
from app.scoped_method_flow import METHODS
from app.scoped_run_contracts import (
    ScopedRunMessageV2,
    ScopedRunReleaseV2,
    ScopedRunRequestV2,
    validate_scoped_result,
)


def scoped_artifact_key(auth, run_id, method, digest):
    if not re.fullmatch(RUN_ID, run_id) or method not in METHODS or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError("SCOPED_ARTIFACT_ID")
    return f"tenants/{auth.tenant_id}/users/{auth.user_id}/runs/{run_id}/qualified-v2/{method}/{digest}.json"


class ScopedArtifactRepository:
    def __init__(self, blobs):
        self.blobs = blobs

    def save(self, auth, run_id, method, result):
        raw = canonical_json(result)
        if not 0 < len(raw) <= MAX_RESULT_BYTES:
            raise ValueError("SCOPED_ARTIFACT_SIZE")
        digest = sha256_bytes(raw)
        key = scoped_artifact_key(auth, run_id, method, digest)
        self.blobs.put(key, raw)
        if self.blobs.get(key) != raw:
            raise ValueError("SCOPED_ARTIFACT_READBACK")
        return {"sha256": digest, "bytes": len(raw)}

    def read(self, auth, run_id, method, reference):
        if set(reference) != {"sha256", "bytes"} or type(reference["bytes"]) is not int or not 0 < reference["bytes"] <= MAX_RESULT_BYTES:
            raise ValueError("SCOPED_ARTIFACT_REFERENCE")
        raw = self.blobs.get(scoped_artifact_key(auth, run_id, method, reference["sha256"]))
        if len(raw) != reference["bytes"] or sha256_bytes(raw) != reference["sha256"]:
            raise ValueError("SCOPED_ARTIFACT_CHECKSUM")
        return raw


class ScopedRunWorker(OwnedRunWorker):
    def process(self, raw_message):
        message = ScopedRunMessageV2.model_validate_json(canonical_json(raw_message))
        auth = AuthContext(message.tenant_id, message.owner_id)
        parent = self.repo.get(auth, message.run_id)
        if parent["admission_token"] != message.admission_token:
            raise ApiError(404, "RUN_NOT_FOUND")
        try:
            self.authorize(auth)
            self.release_check(parent["release"])
        except ApiError as exc:
            if exc.status not in {401, 403}:
                raise
            if self.repo.child(auth, message.run_id, message.method_id)["status"] in {"QUEUED", "CLAIMED"}:
                self.repo.finish(auth, message.run_id, message.method_id, "CANCELLED", error="ACCESS_DENIED")
            return "ACK"
        if self.repo.child(auth, message.run_id, message.method_id)["status"] in TERMINAL:
            self.repo.cleanup_budget(auth, message.run_id, message.method_id)
            return "ACK"
        try:
            self.executor.prepare(auth, parent)
        except RunStageBusy:
            return "RETRY"
        except Exception:
            safe = self.executor.safe_failure(auth, parent, message.method_id)
            self.repo.finish(auth, message.run_id, message.method_id, "FAILED" if safe else "UNKNOWN",
                error="RUN_PREPARATION_FAILED" if safe else "RUN_RECONCILIATION_REQUIRED")
            return "ACK"
        outcome, token = self.repo.claim(message)
        if outcome != "CLAIMED":
            return outcome
        started = False
        try:
            self.authorize(auth)
            self.release_check(parent["release"])
            self.repo.start(auth, message.run_id, message.method_id, token)
            started = True

            def guard():
                self.authorize(auth)
                self.release_check(parent["release"])
                self.repo.guard(auth, message.run_id, message.method_id, token)

            guard()
            result = validate_scoped_result(parent, message.method_id,
                self.executor(auth, parent, message.method_id, guard))
            self.authorize(auth)
            self.release_check(parent["release"])
            self.repo.guard(auth, message.run_id, message.method_id, token, allow_cancelled=True)
            ref = self.artifacts.save(auth, message.run_id, message.method_id, result.model_dump(mode="json"))
            self.authorize(auth)
            self.repo.finish(auth, message.run_id, message.method_id, "COMPLETED", token=token, result_ref=ref)
            return "ACK"
        except Exception:
            if started:
                safe = self.executor.safe_failure(auth, parent, message.method_id)
                self.repo.finish(auth, message.run_id, message.method_id, "FAILED" if safe else "UNKNOWN", token=token,
                    error="GENERATION_INVALID" if safe else "RUN_RECONCILIATION_REQUIRED")
                return "ACK"
            return "RETRY"


class ScopedRunService(OwnedRunService):
    def submit(self, auth, payload, key):
        try:
            request = ScopedRunRequestV2.model_validate_json(canonical_json(payload))
        except Exception:
            raise ApiError(400, "INVALID_REQUEST") from None
        self.authorize(auth)
        row, fresh = self.repo.admit(auth, key, request, self.release_provider())
        return self.detail(auth, row["run_id"]) | {"idempotent_replay": not fresh}

    def detail(self, auth, run_id):
        value = super().detail(auth, run_id)
        value["schema_version"] = 2
        value["workflow"] = "SCOPED_V14_QUALIFIED_ANSWER"
        release = ScopedRunReleaseV2.model_validate_json(canonical_json(self.repo.get(auth, run_id)["release"]))
        value["corpus"] = {"documents": release.document_count, "chunks": release.chunk_count,
            "graph_facts": release.graph_fact_count, "graph_source_documents": release.graph_source_document_count}
        return value

    def result(self, auth, run_id, method):
        self.detail(auth, run_id)
        child = self.repo.child(auth, run_id, method)
        if child["status"] != "COMPLETED":
            raise ApiError(409, "RUN_RESULT_UNAVAILABLE")
        raw = self.artifacts.read(auth, run_id, method, child["result_ref"])
        result = validate_scoped_result(self.repo.get(auth, run_id), method, strict_json(raw))
        self.authorize(auth)
        return result

    def public_result(self, auth, run_id, method):
        result = self.result(auth, run_id, method)
        public = result.model_dump(mode="json", exclude={"owner_id", "tenant_id"})
        public["context_sources"] = [source.model_dump(mode="json", exclude={"text"}) for source in result.context_sources]
        public |= {"method_id": method, "quality_metrics": None, "semantic_review": "NOT_REVIEWED"}
        if len(canonical_json(public)) > 96*1024:
            raise ApiError(503, "RESPONSE_TOO_LARGE")
        return public

    def source(self, auth, run_id, method, chunk_id):
        result = self.result(auth, run_id, method)
        source = next((item for item in result.context_sources if item.chunk_id == chunk_id), None)
        if source is None:
            raise ApiError(404, "SOURCE_NOT_FOUND")
        return source.model_dump(mode="json")

    def trace(self, auth, run_id):
        detail = self.detail(auth, run_id)
        methods = []
        for child in detail["methods"]:
            if child["result_available"]:
                result = self.result(auth, run_id, child["method_id"])
                methods.append({"method_id": result.effective_method, "qualification": result.qualification,
                    "events": [event.model_dump(mode="json") for event in result.trace],
                    "retrieval": result.retrieval.model_dump(mode="json"), "config_sha256": result.config_sha256,
                    "graph_evidence": [fact.model_dump(mode="json") for fact in result.graph_evidence]})
        return {"schema_version": 2, "run_id": run_id, "methods": methods, "quality_metrics": None}
