"""V14 raw generation/cache -> independent display qualification.

The raw cache identity excludes post-processing policy: rechecking an unchanged
native receipt never requires another paid call or a rewritten raw response.
"""

from dataclasses import dataclass

from app.answer_qualification import QualifiedAnswer, qualify_answer
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.model_gateway import CallContext
from app.profiled_generation import (
    ProfiledAnswer,
    ProfiledGenerationRecipe,
    resolve_profiled_answer,
)
from app.scoped_method_flow import PreparedScopedFlow
from app.task_scoped_generation_v14 import (
    TaskScopedGatewayV14,
    TaskScopedRecipeV14,
    parse_v14_answer,
    task_scoped_request,
)


@dataclass(frozen=True)
class QualifiedAnswerLookup:
    qualification: QualifiedAnswer
    receipt: object
    cache_hit: bool
    deterministic_no_evidence: bool = False

    @property
    def new_cost_micro_usd(self):
        return 0 if self.cache_hit or self.receipt is None else self.receipt.original_cost_micro_usd

    def public_payload(self):
        return self.qualification.public_payload() | {
            "answer_origin": "SERVER_DETERMINISTIC_ABSTENTION" if self.deterministic_no_evidence else "NATIVE_RECEIPT",
            "model_cache_hit": self.cache_hit}


@dataclass(frozen=True)
class MethodNotReady:
    method: str
    reason: str

    def public_payload(self):
        return {"schema_version": 1, "qualification": "NOT_READY", "response": None,
            "mode_requested": self.method, "mode_used": self.method, "reason": self.reason,
            "model_dispatched": False, "hidden_fallbacks": 0}


def resolve_qualified_answer(call, *, source_context, documents, chunks, binding, recipe,
    search, raw_cache, gateway=None):
    if (type(call) is not CallContext or call.stage != "GENERATE" or call.attempt != 1
            or call.request_id != binding.request_id or call.tenant_id != binding.tenant_id
            or source_context.tenant_id != binding.tenant_id or type(recipe) is not TaskScopedRecipeV14):
        raise GenerationError("QUALIFIED_SERVICE_TRUSTED_CONTEXT_BINDING")
    if binding.mode_used.startswith("graph_") and not binding.used_graph:
        raise GenerationError("QUALIFIED_SERVICE_GRAPH_NOT_READY_NO_FALLBACK")
    if raw_cache is not None and hasattr(raw_cache, "owner_id") and (
            raw_cache.owner_id != binding.owner_id or raw_cache.tenant_id != binding.tenant_id):
        raise GenerationError("QUALIFIED_SERVICE_CACHE_OWNER_OR_TENANT_BINDING")
    system, prompt = task_scoped_request(source_context, documents, chunks, recipe, search=search)
    parent = ProfiledGenerationRecipe(model_metadata_sha256=recipe.model_metadata_sha256)

    def resolve_output(output):
        return resolve_profiled_answer(output, source_context, documents, chunks, binding=binding, recipe=parent)

    if not source_context.chunk_ids:
        output = ProfiledAnswer(answer_status="ABSTAINED", claims=(), reason="INSUFFICIENT_EVIDENCE",
            limitations=({"code": "MISSING_SUPPLIED_EVIDENCE", "question_part": source_context.question[:200]},))
        response = resolve_output(output)
        return QualifiedAnswerLookup(qualify_answer(response, source_context, documents, chunks, binding=binding),
            None, False, True)

    def parse(raw, stop):
        return resolve_output(parse_v14_answer(raw, stop, question=source_context.question, search=search))

    def dispatch(actual_call, **request):
        if actual_call != call or request != {"system": system, "prompt": prompt, "max_output_tokens": 1600}:
            raise GenerationError("QUALIFIED_SERVICE_CANONICAL_DISPATCH_REQUIRED")
        return TaskScopedGatewayV14(gateway).generate_task_scoped(call, source_context=source_context,
            documents=documents, chunks=chunks, recipe=recipe, search=search)

    raw = raw_cache.resolve_stage(call, system=system, prompt=prompt, max_output_tokens=1600,
        binding_sha256=sha256_bytes(canonical_json(binding.model_dump(mode="json"))), recipe_sha256=recipe.fingerprint,
        parser=parse, gateway=gateway, dispatch=None if gateway is None else dispatch)
    # REVIEW_REQUIRED is a completed model operation, not a failed API call.
    # Keep its raw READY cache receipt and independently withhold public output.
    result = qualify_answer(raw.value, source_context, documents, chunks, binding=binding)
    return QualifiedAnswerLookup(result, raw.receipt, raw.cache_hit)


def resolve_qualified_flow(flow, *, raw_cache, gateway=None):
    if type(flow) is not PreparedScopedFlow:
        raise GenerationError("QUALIFIED_SERVICE_TYPED_FLOW_REQUIRED")
    if (flow.method != flow.binding.mode_requested or flow.method != flow.binding.mode_used
            or flow.context.question != flow.trace["question"] or flow.context.language != flow.trace["language"]
            or flow.context.collection_sha256 != flow.trace["collection_sha256"]
            or flow.binding.used_graph != flow.trace["packing"]["used_graph"] or flow.trace["hidden_fallbacks"]):
        raise GenerationError("QUALIFIED_SERVICE_FLOW_TASK_OR_METHOD_CHANGED")
    if not flow.ready:
        return MethodNotReady(flow.method, flow.status)
    recipe = TaskScopedRecipeV14(model_metadata_sha256=flow.recipe.model_metadata_sha256)
    request_id = stable_id("qualified-native-v14", parent=flow.call_context.request_id, recipe=recipe.fingerprint,
        search=flow.trace["english_search_query"])
    call = flow.call_context.model_copy(update={"request_id": request_id})
    binding = flow.binding.model_copy(update={"request_id": request_id})
    return resolve_qualified_answer(call, source_context=flow.context, documents=flow.documents, chunks=flow.chunks,
        binding=binding, recipe=recipe, search=flow.trace["english_search_query"], raw_cache=raw_cache, gateway=gateway)
