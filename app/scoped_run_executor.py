"""Owned V14 answers over the actual scoped pool/graph/Q/QB mechanisms."""

import json
from dataclasses import dataclass
from time import perf_counter

from app.answer_qualification import policy_fingerprint
from app.collection_answering import (
    validate_collection_lexical_binding,
    validate_collection_reranker,
)
from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalModelGateway
from app.graph_collection_migration import collection_records
from app.profiled_generation import ProfiledGenerationRecipe
from app.profiled_source_context import SourceContextProfile
from app.qualified_answer_service import resolve_qualified_answer
from app.query_auth import ApiError
from app.query_language import answer_language, rewrite_for_search
from app.rag_runner import _validate_encoder
from app.run_budget import EnvelopeLedger, ReceiptedStageGateway
from app.scoped_method_flow import METHODS, Q_METHODS, prepare_scoped_flow
from app.scoped_query_v6 import SYSTEM, ScopedPlanV6, ScopeRecipeV6, validate_source_facets
from app.scoped_run_budget import PLAN_SLOT, ScopedRunBudgetController
from app.scoped_run_contracts import (
    ScopedMethodConfigV2,
    ScopedMethodResultV2,
    ScopedRunReleaseV2,
    ScopedStageEventV2,
    validate_scoped_result,
)
from app.task_scoped_generation_v14 import TaskScopedRecipeV14
from app.transactional_answer_cache import TransactionalAnswerCache


@dataclass(frozen=True)
class OwnedScopedTask:
    search: str
    language: str
    plan: ScopedPlanV6 | None
    events: tuple[ScopedStageEventV2, ...]


def event(stage, started, text, count):
    return ScopedStageEventV2(stage=stage, duration_ms=(perf_counter()-started)*1000, status="COMPLETED",
        input_sha256=sha256_bytes(text.encode()), output_count=count)


class ScopedOwnedAnswerExecutor:
    def __init__(self, collection, lexical, graph, runtime, *, controller, repository, authorize):
        if not isinstance(controller, ScopedRunBudgetController) or controller.gateway is None or (
                repository.answer_budget is not controller or repository.store is not controller.store):
            raise ValueError("SCOPED_EXECUTOR_BUDGET_BINDING")
        validate_collection_lexical_binding(collection, lexical)
        self.collection, self.lexical, self.graph, self.runtime = collection, lexical, graph, runtime
        self.controller, self.repo, self.authorize = controller, repository, authorize
        self.generation_recipe = TaskScopedRecipeV14(model_metadata_sha256=controller.routes.account_model_metadata_sha256)
        self.profile = SourceContextProfile(max_chunks=12)
        if runtime.recipe.model_metadata_sha256 != controller.routes.account_model_metadata_sha256:
            raise ValueError("SCOPED_EXECUTOR_RECIPE_METADATA")
        encoder = getattr(runtime, "encoder", None)
        query_recipe = _validate_encoder(collection, encoder) if encoder is not None else None
        if query_recipe is not None and runtime.query_recipe != query_recipe:
            raise ValueError("SCOPED_EXECUTOR_QUERY_RECIPE")
        if graph is not None and graph.collection_manifest_sha256 != collection.manifest_sha256:
            raise ValueError("SCOPED_EXECUTOR_GRAPH_COLLECTION")
        reranker = getattr(runtime, "reranker", None)
        if reranker is not None:
            validate_collection_reranker(collection, reranker)
        self.documents, self.chunks, _ = collection_records(collection)
        self.source_refs = {key: ref for ref in collection.manifest.shards
            for key in collection.get_shard(ref.shard_id).chunks}
        configs = {}
        for method in METHODS:
            if method != "bm25" and encoder is None or method.startswith("graph_") and graph is None or (
                    method == "hybrid_reranked" and reranker is None):
                continue
            configs[method] = ScopedMethodConfigV2(method_id=method,
                collection_manifest_sha256=collection.manifest_sha256, lexical_index_version=lexical.manifest.index_version,
                document_recipe_sha256=collection.manifest.embedding_recipe.fingerprint,
                query_recipe_sha256=query_recipe.fingerprint if method != "bm25" else None,
                reranker_recipe_sha256=reranker.recipe.fingerprint if method == "hybrid_reranked" else None,
                graph_release_sha256=graph.release_sha256 if method.startswith("graph_") else None,
                planner_recipe_sha256=ScopeRecipeV6().fingerprint if method in Q_METHODS else None,
                generation_recipe=self.generation_recipe, source_profile=self.profile,
                fixture_only=collection.manifest.purpose == "FIXTURE" or not controller.live)
        self.release = ScopedRunReleaseV2(collection_version=collection.manifest.collection_version,
            collection_manifest_sha256=collection.manifest_sha256, budget_run_id=controller.budget_run_id,
            document_count=len(self.documents), chunk_count=len(self.chunks),
            graph_fact_count=len(graph.facts) if graph is not None else 0,
            graph_source_document_count=len({self.chunks[f.chunk_id].document_version_id for f in graph.facts}) if graph else 0,
            configs=configs, method_config_sha256={key: value.fingerprint for key, value in configs.items()},
            qualification_policy_sha256=policy_fingerprint())
        self._tasks = {}

    def check_release(self, value):
        release = ScopedRunReleaseV2.model_validate_json(canonical_json(value))
        if release != self.release or release.qualification_policy_sha256 != policy_fingerprint():
            raise ValueError("SCOPED_EXECUTOR_RELEASE_CHANGED")

    def parent_guard(self, auth, parent):
        self.authorize(auth)
        current = self.repo.get(auth, parent["run_id"])
        self.check_release(current["release"])
        if current["cancel_requested"] or current["deadline_at"] <= int(self.repo.now().timestamp()):
            raise ApiError(409, "RUN_STOP_REQUESTED")

    def safe_failure(self, auth, parent, method):
        return self.controller.safe_failure(auth, parent, method)

    def prepare(self, auth, parent):
        self.parent_guard(auth, parent)
        key = (auth.user_id, parent["run_id"], parent["payload_sha256"])
        if key in self._tasks:
            return self._tasks[key]
        request = parent["request"]
        question = request["original_question"]
        language, events = answer_language(question, request["language"]), []
        search, plan = question, None
        if language == "vi":
            started = perf_counter()
            slot = self.controller.slot_for()
            context = self.controller.context(parent, slot)
            gateway = ReceiptedStageGateway(self.controller, auth, parent, slot, lambda: self.parent_guard(auth, parent))
            search, _ = rewrite_for_search(question, gateway=gateway, tenant_id=auth.tenant_id,
                request_id=context.request_id, run_id=context.run_id)
            events.append(event("shared_search_translation", started, question, 1))
        if any(method in Q_METHODS for method in request["method_ids"]):
            started = perf_counter()
            gateway = ReceiptedStageGateway(self.controller, auth, parent, PLAN_SLOT, lambda: self.parent_guard(auth, parent))
            result = gateway.generate(self.controller.context(parent, PLAN_SLOT), system=SYSTEM, max_output_tokens=256,
                prompt=canonical_json({"untrusted_original_question": question,
                    "untrusted_english_search_query": search, "original_language": language}).decode())
            if result.stop_reason != "end_turn" or not result.text or len(result.text.encode()) > 6000:
                raise ValueError("SCOPED_OWNED_FACETS_OUTPUT_BOUND")
            receipt = {"version": "owned-source-task-v6-receipt-v1", "operation_id": result.operation_id,
                "raw_output": result.text, "stop_reason": result.stop_reason,
                "usage": result.usage.model_dump(mode="json"), "cost_micro_usd": result.actual_micro_usd,
                "recipe_sha256": ScopeRecipeV6().fingerprint, "collection_sha256": self.collection.manifest_sha256,
                "question_sha256": sha256_bytes(question.encode()), "search_sha256": sha256_bytes(search.encode())}
            plan = ScopedPlanV6(origin="NATIVE_V6" if self.controller.live else "FIXTURE",
                collection_manifest_sha256=self.collection.manifest_sha256,
                question_sha256=sha256_bytes(question.encode()), search_query_sha256=sha256_bytes(search.encode()),
                language=language, facets=validate_source_facets(result.text, question, search),
                recipe_sha256=ScopeRecipeV6().fingerprint, operation_id=result.operation_id,
                receipt_sha256=sha256_bytes(canonical_json(receipt)))
            events.append(event("shared_native_task_facets", started, search, len(plan.facets)))
        self.parent_guard(auth, parent)
        task = OwnedScopedTask(search, language, plan, tuple(events))
        # A Lambda invocation executes one queue item. Bound warm-instance state;
        # durable stage receipts remain authoritative after an eviction.
        if len(self._tasks) >= 8:
            self._tasks.clear()
        self._tasks[key] = task
        return task

    def __call__(self, auth, parent, method, guard):
        guard()
        self.check_release(parent["release"])
        if method not in parent["request"]["method_ids"] or auth.tenant_id != self.collection.manifest.tenant_id:
            raise ValueError("SCOPED_EXECUTOR_TASK_SCOPE")
        task = self.prepare(auth, parent)
        started = perf_counter()
        flow = prepare_scoped_flow(self.collection, self.lexical, self.graph, self.runtime,
            question=parent["request"]["original_question"], search=task.search, language=task.language, method=method,
            run_id=parent["run_id"], owner_id=auth.user_id,
            recipe=ProfiledGenerationRecipe(model_metadata_sha256=self.generation_recipe.model_metadata_sha256),
            profile=self.profile, plan=task.plan)
        events = [*task.events, event("retrieval_and_atomic_source_packing", started, task.search, len(flow.context.chunk_ids))]
        guard()
        context = self.controller.context(parent, self.controller.slot_for(method))
        binding = flow.binding.model_copy(update={"request_id": context.request_id})
        payload, receipt, cache_hit, new_cost = None, None, False, 0
        started = perf_counter()
        if flow.ready:
            base = self.controller.gateway
            gateway = ExternalModelGateway(EnvelopeLedger(self.controller, auth, parent,
                self.controller.slot_for(method), guard), base.provider, base.routes, base.prices)
            cache = TransactionalAnswerCache(self.controller.store, tenant_id=auth.tenant_id, owner_id=auth.user_id,
                namespace="LIVE" if self.controller.live else "FIXTURE")
            resolved = resolve_qualified_answer(context, source_context=flow.context, documents=flow.documents,
                chunks=flow.chunks, binding=binding, recipe=self.generation_recipe, search=task.search,
                raw_cache=cache, gateway=gateway)
            payload, receipt, cache_hit, new_cost = resolved.public_payload(), resolved.receipt, resolved.cache_hit, resolved.new_cost_micro_usd
        else:
            payload = {"qualification": "NOT_READY", "response": None, "native_answer_status": None,
                "review_flags": [], "answer_origin": "NOT_READY"}
        events.append(event("generation_and_display_qualification" if flow.ready else "mechanism_not_ready_no_dispatch",
            started, flow.context.fingerprint, len(payload["response"]["claims"]) if payload["response"] else 0))
        sources = []
        for index, key in enumerate(flow.context.chunk_ids, 1):
            chunk, ref = self.chunks[key], self.source_refs[key]
            doc = self.documents[chunk.document_version_id]
            sources.append({"source_id": f"s{index}", "chunk_id": key, "document_version_id": doc.document_version_id,
                "shard_id": ref.shard_id, "shard_manifest_sha256": ref.manifest_sha256, "source_uri": doc.source_uri,
                "title": doc.title, "section_path": list(chunk.section_path), "start_char": chunk.start_char,
                "end_char": chunk.end_char, "text_sha256": chunk.text_sha256, "text": chunk.text})
        trace = flow.trace
        retrieval = {"mechanism_status": flow.status, "queries": trace["queries"],
            "query_rankings": trace.get("query_rankings", ()), "pool_fingerprint": trace["pool_fingerprint"],
            "bundles": [{"bundle_id": b["bundle_id"], "chunk_ids": b["chunk_ids"],
                "assertion_ids": b["assertion_ids"], "selection": b["reason"]} for b in trace["packing"]["decisions"]],
            "balanced_allocator_used": trace["balanced_allocator_used"]}
        if task.plan is not None and method in Q_METHODS:
            retrieval |= {"planner_origin": task.plan.origin, "planner_operation_id": task.plan.operation_id,
                "planner_receipt_sha256": task.plan.receipt_sha256}
        graph_evidence = []
        facts = {fact.assertion_id: fact for fact in self.graph.facts} if self.graph else {}
        for key in trace["packing"]["context_assertion_ids"]:
            fact = facts[key]
            record = json.loads(fact.context_json)
            projection = record.get("node_id_projection", {})
            nodes = {}
            for node in record["nodes"]:
                if node["node_id"]:
                    projected = projection.get(node["node_id"], node["node_id"])
                    nodes[projected] = {"node_id": projected, "name": node["name"],
                        "identity_scope": node["identity_scope"], "role": node.get("role")}
            graph_evidence.append({"assertion_id": key, "chunk_id": fact.chunk_id, "kind": fact.kind,
                "clause": fact.clause, "nodes": list(nodes.values()), "links": fact.links,
                "unresolved_references_present": any(not node["node_id"] for node in record["nodes"])})
        result = ScopedMethodResultV2.model_validate_json(canonical_json(dict(
            tenant_id=auth.tenant_id, owner_id=auth.user_id, run_id=parent["run_id"],
            request_id=context.request_id, original_question_sha256=sha256_bytes(parent["request"]["original_question"].encode()),
            answer_language=task.language, requested_method=method, effective_method=method,
            config=self.release.configs[method].model_dump(mode="json"), config_sha256=self.release.method_config_sha256[method],
            collection_version=self.release.collection_version, ranked_chunk_ids=tuple(trace["pool_ids"]),
            context_chunk_ids=flow.context.chunk_ids, context_proxy_tokens=flow.context.proxy_tokens,
            context_sources=sources, source_context_sha256=flow.context.fingerprint,
            qualification_policy_sha256=policy_fingerprint(), qualification=payload["qualification"],
            response=payload["response"], native_answer_status=payload["native_answer_status"],
            review_flags=payload["review_flags"], used_graph=binding.used_graph, answer_origin=payload["answer_origin"],
            cache_hit=cache_hit, new_cost_micro_usd=new_cost,
            original_receipt_cost_micro_usd=receipt.original_cost_micro_usd if receipt else None,
            usage=receipt.usage.model_dump(mode="json") if receipt else None,
            trace=[item.model_dump(mode="json") for item in events], retrieval=retrieval, graph_evidence=graph_evidence)))
        return validate_scoped_result(parent, method, result.model_dump(mode="json")).model_dump(mode="json")
