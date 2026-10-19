"""Authenticated query use cases; state is durable before any provider dispatch."""

import math

from app.answer_pipeline import PreparedAnswer, prepare_answer
from app.aws_knowledge import KnowledgeError, create_bundle, load_bundle, pack_bundle
from app.contracts import QueryRequest, QueryResponse, canonical_json, sha256_bytes
from app.generation import (
    AnswerBinding,
    empty_answer,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.model_gateway import CallContext
from app.query_auth import ApiError
from app.query_language import RewriteError, answer_language, rewrite_for_search
from app.query_telemetry import NullTelemetry

RUN = "external-pilot10-r72-v1"


class QueryService:
    def __init__(self, repository, releases, runtime_loader, gateway_factory, recipe, *, authorization_check, telemetry=None,
                 knowledge_factory=None):
        self.repo, self.releases, self.runtime_loader = repository, releases, runtime_loader
        self.gateway_factory, self.recipe = gateway_factory, recipe
        self.telemetry = telemetry or NullTelemetry()
        self.authorization_check = authorization_check
        self.knowledge_factory = knowledge_factory

    def evidence_for(self, auth, row):
        return load_bundle(row["source_bundle"], row["source_bundle_sha256"], tenant_id=auth.tenant_id,
            owner_id=auth.user_id, request_id=row["request_id"], snapshot_version=row["snapshot_version"])

    def authorize(self, auth):
        with self.telemetry.stage("authorization_refresh"):
            self.authorization_check(auth)

    def release_for(self, auth, row):
        release = self.releases.get(auth.tenant_id, row["snapshot_version"])
        if release["manifest_sha256"] != row["manifest_sha256"]:
            raise ApiError(503, "SNAPSHOT_UNAVAILABLE", request_id=row["request_id"])
        return release

    def public_record(self, auth, row):
        self.release_for(auth, row)
        result = {key: row[key] for key in ("request_id", "question", "mode", "status", "created_at", "snapshot_version")}
        if row["status"] == "COMPLETED":
            raw = canonical_json(row["response"])
            if sha256_bytes(raw) != row.get("response_sha256"):
                raise ApiError(503, "STATE_UNAVAILABLE")
            result["response"] = QueryResponse.model_validate_json(raw).model_dump(mode="json")
            if row.get("corpus_scope") == "aws_official":
                bundle = self.evidence_for(auth, row)
                if result["response"].get("evidence_bundle_sha256") != bundle.sha256:
                    raise ApiError(503, "STATE_UNAVAILABLE")
        if row["status"] == "FAILED":
            result["error"] = {"code": row.get("error_code", "RECONCILIATION_REQUIRED"), "retryable": False}
        return result

    def submit(self, auth, payload, key, *, remaining_seconds):
        try:
            request = QueryRequest.model_validate(payload)
        except Exception:
            raise ApiError(400, "INVALID_REQUEST") from None
        mode = request.mode or "hybrid"
        if mode not in {"vector", "hybrid"}:
            raise ApiError(400, "MODE_UNAVAILABLE")
        if request.corpus_scope == "aws_official" and (mode != "hybrid" or self.knowledge_factory is None):
            raise ApiError(400, "MODE_UNAVAILABLE")
        with self.telemetry.stage("release_read"):
            release = self.releases.current(auth.tenant_id)
        with self.telemetry.stage("admission"):
            payload = {"question": request.question, "mode": mode}
            # Preserve replay of old default payloads across this additive release.
            if request.corpus_scope != "snapshot" or request.answer_language != "auto":
                payload.update(corpus_scope=request.corpus_scope, answer_language=request.answer_language)
            row, fresh = self.repo.admit(auth, key, payload, release, self.recipe.fingerprint)
        request_id = row["request_id"]
        self.telemetry.bind(request_id=request_id, snapshot_version=row["snapshot_version"], mode=row["mode"],
                            recipe_sha256=row["recipe_sha256"])
        if not fresh:
            record = self.public_record(auth, row)
            if row["status"] == "PROCESSING":
                raise ApiError(409, "REQUEST_IN_PROGRESS", request_id=request_id)
            return record | {"idempotent_replay": True}
        try:
            if remaining_seconds() < 5:
                raise ApiError(504, "QUERY_TIMEOUT", request_id=request_id)
            language = answer_language(request.question, request.answer_language)
            search_phrase, model_gateway = request.question, None
            if answer_language(request.question) == "vi" or request.answer_language == "vi":
                self.release_for(auth, row)
                self.authorize(auth)
                with self.telemetry.stage("gateway_setup"):
                    model_gateway = self.gateway_factory(min(14, max(2, math.floor(remaining_seconds() - 5))))
                with self.telemetry.stage("query_rewrite"):
                    search_phrase, rewrite_receipt = rewrite_for_search(request.question, gateway=model_gateway,
                        tenant_id=auth.tenant_id, request_id=request_id, run_id=RUN,
                        receipt_sink=lambda receipt: self.repo.update(auth, request_id, rewrite_receipt=receipt))
                self.release_for(auth, row)
                self.authorize(auth)
            if request.corpus_scope == "aws_official":
                if remaining_seconds() < 9:
                    raise ApiError(504, "QUERY_TIMEOUT", request_id=request_id)
                with self.telemetry.stage("official_search"):
                    items = self.knowledge_factory(max(0.2, min(6, remaining_seconds() - 8))).search(search_phrase)
                self.release_for(auth, row)
                self.authorize(auth)
                snapshot = create_bundle(items, tenant_id=auth.tenant_id, owner_id=auth.user_id,
                    request_id=request_id, snapshot_version=row["snapshot_version"], search_phrase=search_phrase)
                self.repo.update(auth, request_id, source_bundle=snapshot.record, source_bundle_sha256=snapshot.sha256)
                packed = pack_bundle(snapshot)
                binding = AnswerBinding(request_id=request_id, tenant_id=auth.tenant_id, owner_id=auth.user_id,
                    mode_requested=mode, mode_used=mode, used_graph=False, warnings=("LIVE_SEARCH_NOT_PILOT_BENCHMARK",))
                prepared = PreparedAnswer(CallContext(tenant_id=auth.tenant_id, run_id=RUN,
                    request_id=request_id, stage="GENERATE"), binding, packed, request.question, self.recipe, packed.chunk_ids)
            else:
                with self.telemetry.stage("runtime_load"):
                    encoder, snapshot = self.runtime_loader(release)
                if snapshot.manifest_sha256 != row["manifest_sha256"] or snapshot.manifest.tenant_id != auth.tenant_id:
                    raise ApiError(503, "SNAPSHOT_UNAVAILABLE")
                try:
                    with self.telemetry.stage("query_embedding"):
                        vector = encoder.encode([search_phrase], role="query")[0]
                except ValueError as exc:
                    if str(exc) == "BGE_INPUT_EXCEEDS_512_NO_TRUNCATION":
                        raise ApiError(400, "QUESTION_TOKEN_LIMIT") from None
                    raise
                with self.telemetry.stage("retrieval_context"):
                    prepared = prepare_answer(snapshot, search_phrase, vector, tenant_id=auth.tenant_id,
                        owner_id=auth.user_id, run_id=RUN, recipe=self.recipe, mode=mode, request_id=request_id)
            self.release_for(auth, row)  # Recheck revocation after potentially slow cold load.
            self.authorize(auth)
            if not prepared.packed.chunk_ids:
                response = empty_answer(snapshot=snapshot, packed=prepared.packed, binding=prepared.binding)
            else:
                # Preserve time for reserve/settle, receipt persistence, validation and final state.
                timeout = min(18, math.floor(remaining_seconds() - 5))
                if timeout < 2:
                    raise ApiError(504, "QUERY_TIMEOUT", request_id=request_id)
                system, prompt = generation_request(request.question, snapshot, prepared.packed, auth.tenant_id, self.recipe,
                                                    answer_language=language)
                with self.telemetry.stage("gateway_setup"):
                    gateway = model_gateway or self.gateway_factory(timeout)
                if gateway.routes.account_model_metadata_sha256 != self.recipe.model_metadata_sha256 or (
                    gateway.routes.generation_model_id != self.recipe.model_id
                ):
                    raise ApiError(503, "MODEL_UNAVAILABLE")
                self.release_for(auth, row)
                self.authorize(auth)  # Secret/gateway setup may also span an authorization change.
                if remaining_seconds() < 7:
                    raise ApiError(504, "QUERY_TIMEOUT", request_id=request_id)
                with self.telemetry.stage("model_call"):
                    result = gateway.generate(prepared.context, system=system, prompt=prompt,
                                               max_output_tokens=self.recipe.max_output_tokens)
                receipt = {"operation_id": result.operation_id, "raw_output": result.text or "",
                    "stop_reason": result.stop_reason or "MISSING", "usage": result.usage.model_dump(mode="json"),
                    "original_cost_micro_usd": result.actual_micro_usd, "recipe_sha256": self.recipe.fingerprint,
                    "context_chunk_ids": list(prepared.packed.chunk_ids),
                    "request_sha256": sha256_bytes(canonical_json([system, prompt]))}
                if len(canonical_json(receipt)) > 64 * 1024:
                    raise ApiError(502, "GENERATION_INVALID")
                with self.telemetry.stage("receipt_write"):
                    self.repo.update(auth, request_id, receipt=receipt)  # Durable even if parse/quotes fail.
                self.telemetry.event("query_model_receipt", operation_id=result.operation_id,
                    input_tokens=result.usage.input_tokens, output_tokens=result.usage.output_tokens,
                    cost_micro_usd=result.actual_micro_usd, context_chunks=len(prepared.packed.chunk_ids))
                with self.telemetry.stage("answer_validation"):
                    response = resolve_answer(parse_generated(receipt["raw_output"], receipt["stop_reason"], self.recipe),
                        snapshot=snapshot, packed=prepared.packed, binding=prepared.binding, recipe=self.recipe)
            if language == "vi" and response.answer_status == "ABSTAINED":
                response = response.model_copy(update={"answer": "Tài liệu được tìm thấy chưa cung cấp đủ bằng chứng để trả lời câu hỏi này."})
            response = response.model_copy(update={"answer_language": language})
            if request.corpus_scope == "aws_official":
                response = response.model_copy(update={"retrieval_backend": "aws_knowledge_mcp",
                    "evidence_bundle_sha256": snapshot.sha256})
            self.release_for(auth, row)
            self.authorize(auth)
            data = response.model_dump(mode="json")
            with self.telemetry.stage("final_write"):
                final = self.repo.update(auth, request_id, status="COMPLETED", response=data,
                    response_sha256=sha256_bytes(canonical_json(data)),
                    allowed_chunk_ids=sorted({c.chunk_id for c in response.citations}))
            return self.public_record(auth, final) | {"idempotent_replay": False}
        except Exception as exc:
            from app.generation import GenerationError
            from app.ledger import BudgetDenied

            error = exc if isinstance(exc, ApiError) else ApiError(
                502 if isinstance(exc, GenerationError) else 429 if isinstance(exc, BudgetDenied) else 503,
                "GENERATION_INVALID" if isinstance(exc, GenerationError) else (
                    "BUDGET_LIMIT" if isinstance(exc, BudgetDenied) else "MODEL_OR_STATE_UNAVAILABLE"), request_id=request_id)
            if isinstance(exc, KnowledgeError):
                error = ApiError(503, "OFFICIAL_SEARCH_UNAVAILABLE", request_id=request_id)
            elif isinstance(exc, RewriteError):
                error = ApiError(502, "QUERY_REWRITE_INVALID", request_id=request_id)
            try:
                self.repo.update(auth, request_id, status="FAILED", error_code=error.code)
            except Exception:
                pass  # A durable PROCESSING record cannot be dispatched a second time.
            error.request_id = request_id
            raise error from None

    def detail(self, auth, request_id):
        with self.telemetry.stage("detail_read"):
            row = self.repo.get(auth, request_id)
            self.telemetry.bind(request_id=request_id, snapshot_version=row["snapshot_version"], mode=row["mode"])
            return self.public_record(auth, row)

    def history(self, auth, *, limit=10, cursor=None):
        with self.telemetry.stage("history_read"):
            return self._history(auth, limit=limit, cursor=cursor)

    def _history(self, auth, *, limit, cursor):
        rows, next_cursor = self.repo.history(auth, limit=limit, cursor=cursor)
        results = []
        for row in rows:
            try:
                record = self.public_record(auth, row)
            except ApiError as exc:
                if exc.code == "SNAPSHOT_UNAVAILABLE":
                    continue
                raise
            results.append({k: v for k, v in record.items() if k != "response"})
        return {"items": results, "next_cursor": next_cursor}

    def source(self, auth, request_id, chunk_id):
        with self.telemetry.stage("source_read"):
            return self._source(auth, request_id, chunk_id)

    def _source(self, auth, request_id, chunk_id):
        row = self.repo.get(auth, request_id)
        self.telemetry.bind(request_id=request_id, snapshot_version=row["snapshot_version"], mode=row["mode"])
        release = self.release_for(auth, row)
        if row["status"] != "COMPLETED" or chunk_id not in row.get("allowed_chunk_ids", []):
            raise ApiError(404, "SOURCE_NOT_FOUND")
        self.public_record(auth, row)  # Response checksum and release policy precede source lookup.
        if row.get("corpus_scope") == "aws_official":
            snapshot = self.evidence_for(auth, row)
        else:
            with self.telemetry.stage("runtime_load"):
                _, snapshot = self.runtime_loader(release)
        chunk = snapshot.chunks.get(chunk_id)
        if chunk is None or chunk.tenant_id != auth.tenant_id or (
            row.get("corpus_scope") != "aws_official" and snapshot.manifest_sha256 != row["manifest_sha256"]
        ):
            raise ApiError(404, "SOURCE_NOT_FOUND")
        document = snapshot.documents[chunk.document_version_id]
        self.release_for(auth, row)
        result = {"request_id": request_id, "chunk_id": chunk_id, "snapshot_version": row["snapshot_version"],
                "document_version_id": chunk.document_version_id, "title": document.title,
                "source_uri": document.source_uri, "section_path": list(chunk.section_path), "text": chunk.text}
        result.update(source_kind="OFFICIAL_SEARCH_EXCERPT" if row.get("corpus_scope") == "aws_official" else "SNAPSHOT_CHUNK",
            retrieved_at=document.fetched_at.isoformat(), source_last_modified=None,
            normalizer_version=document.normalizer_version, raw_sha256=document.raw_sha256,
            normalized_sha256=document.normalized_sha256)
        if row.get("corpus_scope") == "aws_official":
            result["evidence_bundle_sha256"] = snapshot.sha256
        return result
