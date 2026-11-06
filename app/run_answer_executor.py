"""Owned L/D/H/R answers using the common evidence packer and receipted model stages."""

from time import perf_counter

import numpy as np

from app.collection_answering import prepare_collection_answer
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.generation import (
    GenerationRecipeV9,
    empty_answer,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.query_auth import ApiError
from app.query_language import answer_language, rewrite_for_search
from app.rag_methods import MethodResultV1, StageEventV1
from app.run_budget import ReceiptedStageGateway
from app.run_collection_adapter import CollectionRetrievalExecutor


class CollectionOwnedAnswerExecutor(CollectionRetrievalExecutor):
    def __init__(self, collection, lexical, *, controller, repository, authorize, **runtime):
        super().__init__(collection, lexical, **runtime)
        if type(runtime["recipe"]) is not GenerationRecipeV9:
            raise ValueError("OWNED_ANSWER_RECIPE_V9_ONLY")
        self.controller, self.repo, self.authorize = controller, repository, authorize
        if repository.answer_budget is not controller or runtime["recipe"].model_metadata_sha256 != (
            controller.gateway.routes.account_model_metadata_sha256
        ) or canonical_json(runtime["routes"].model_dump(mode="json")) != canonical_json(
            controller.gateway.routes.model_dump(mode="json")
        ):
            raise ValueError("OWNED_ANSWER_RUNTIME_BUDGET_BINDING")

    def parent_guard(self, auth, parent):
        self.authorize(auth)
        current = self.repo.get(auth, parent["run_id"])
        self.check_release(current["release"])
        if current["cancel_requested"] or current["deadline_at"] <= int(self.repo.now().timestamp()):
            raise ApiError(409, "RUN_STOP_REQUESTED")

    def prepare(self, auth, parent):
        self.parent_guard(auth, parent)
        payload = parent["request"]
        language = answer_language(payload["original_question"], payload["language"])
        if language == "en":
            return payload["original_question"], language
        slot = self.controller.slot_for()
        gateway = ReceiptedStageGateway(self.controller, auth, parent, slot, lambda: self.parent_guard(auth, parent))
        context = self.controller.context(parent, slot)
        query, _ = rewrite_for_search(payload["original_question"], gateway=gateway, tenant_id=auth.tenant_id,
            request_id=context.request_id, run_id=context.run_id)
        self.parent_guard(auth, parent)
        return query, language

    def safe_failure(self, auth, parent, method):
        return self.controller.safe_failure(auth, parent, method)

    def __call__(self, auth, parent, method, guard):
        if parent["request"]["run_kind"] != "answer" or auth.tenant_id != self.collection.manifest.tenant_id:
            raise ValueError("OWNED_ANSWER_SCOPE_MISMATCH")
        search_query, language = self.prepare(auth, parent)
        recipe, encoder = self.runtime["recipe"], self.runtime.get("encoder")
        events, vector = [], None
        if method != "bm25":
            guard()
            started = perf_counter()
            value = self.runtime["embedding_cache"].resolve_local(auth.tenant_id,
                text=query_embedding_input(search_query), recipe=encoder.recipe("query"), encoder=encoder)
            vector = np.asarray(value.record.vector, dtype=np.float32)
            events.append(StageEventV1(stage="query_embedding", duration_ms=(perf_counter() - started) * 1000,
                status="COMPLETED", input_sha256=sha256_bytes(search_query.encode("utf-8")), output_count=1))
        guard()
        started, rerank = perf_counter(), []
        slot = self.controller.slot_for(method)
        context = self.controller.context(parent, slot)
        view, prepared = prepare_collection_answer(self.collection, self.lexical,
            parent["request"]["original_question"], vector, tenant_id=auth.tenant_id, owner_id=auth.user_id,
            run_id=context.run_id, request_id=context.request_id, recipe=recipe, mode=method, search_query=search_query,
            reranker=self.runtime.get("reranker") if method == "hybrid_reranked" else None, rerank_trace=rerank)
        events.append(StageEventV1(stage="retrieval_and_pack", duration_ms=(perf_counter() - started) * 1000,
            status="COMPLETED", input_sha256=sha256_bytes(search_query.encode("utf-8")),
            output_count=len(prepared.packed.chunk_ids)))
        sources = []
        for source_id, key in zip(prepared.packed.source_ids, prepared.packed.chunk_ids, strict=True):
            chunk, ref = view.chunks[key], view.source_refs[key]
            document = view.documents[chunk.document_version_id]
            sources.append({"source_id": source_id, "chunk_id": key, "document_version_id": document.document_version_id,
                "shard_id": ref.shard_id, "shard_manifest_sha256": ref.manifest_sha256,
                "source_uri": document.source_uri, "title": document.title, "section_path": list(chunk.section_path),
                "start_char": chunk.start_char, "end_char": chunk.end_char,
                "text_sha256": sha256_bytes(chunk.text.encode("utf-8")), "text": chunk.text})
        guard()
        started, result, cache_hit = perf_counter(), None, False
        if not prepared.packed.chunk_ids:
            response = empty_answer(snapshot=view, packed=prepared.packed, binding=prepared.binding,
                recipe=recipe, answer_language=language)
        else:
            system, prompt = generation_request(prepared.question, view, prepared.packed, auth.tenant_id,
                recipe, answer_language=language)
            gateway = ReceiptedStageGateway(self.controller, auth, parent, slot, guard)
            result = gateway.generate(context, system=system, prompt=prompt, max_output_tokens=recipe.max_output_tokens)
            cache_hit = gateway.cache_hit
            response = resolve_answer(parse_generated(result.text, result.stop_reason, recipe), snapshot=view,
                packed=prepared.packed, binding=prepared.binding, recipe=recipe, answer_language=language)
        localized = {"answer_language": language, "retrieval_backend": "collection_numpy_global_bm25"}
        if language == "vi" and response.answer_status == "ABSTAINED":
            localized["answer"] = "Tài liệu được tìm thấy chưa cung cấp đủ bằng chứng để trả lời câu hỏi này."
        response = response.model_copy(update=localized)
        events.append(StageEventV1(stage="generation_and_validation", duration_ms=(perf_counter() - started) * 1000,
            status="COMPLETED", input_sha256=self.release.method_config_sha256[method], output_count=len(response.claims)))
        from app.rag_methods import MethodConfigV1, RerankedMethodConfigV1

        extra = {"reranker_recipe": self.runtime["reranker"].recipe,
                 "reranker_recipe_sha256": self.runtime["reranker"].recipe.fingerprint} if rerank else {}
        config_class = RerankedMethodConfigV1 if rerank else MethodConfigV1
        config = config_class(method_id=method, fusion="RRF60" if method in {"hybrid", "hybrid_reranked"} else "NONE",
            collection_manifest_sha256=self.collection.manifest_sha256,
            lexical_index_version=self.lexical.manifest.index_version,
            document_recipe_sha256=self.collection.manifest.embedding_recipe.fingerprint,
            query_recipe_sha256=encoder.recipe("query").fingerprint if method != "bm25" else None,
            generation_recipe_sha256=recipe.fingerprint, **extra)
        # The worker checks ownership/lease again before publishing. A cancellation
        # arriving during the final call must not discard an already-produced answer.
        return MethodResultV1(requested_method=method, effective_method=method, config=config,
            config_sha256=config.fingerprint, collection_version=self.release.collection_version,
            status="COMPLETED", ranked_chunk_ids=prepared.retrieved_chunk_ids,
            context_chunk_ids=prepared.packed.chunk_ids, context_proxy_tokens=prepared.packed.proxy_tokens,
            context_sources=tuple(sources), response=response, cache_hit=cache_hit,
            new_cost_micro_usd=0 if cache_hit or result is None else result.actual_micro_usd,
            original_receipt_cost_micro_usd=result.actual_micro_usd if result else None,
            usage=result.usage.model_dump(mode="json") if result else None, trace=tuple(events),
            rerank_trace=rerank[0] if rerank else None).model_dump(mode="json")
