"""Task granularity guard with the unchanged V12 user message/token budget.

V13 added a search envelope that can push a nearly-full source context above
4096 proxy tokens. V14 puts only constant policy in the system message.
"""

from typing import Literal

from app.contracts import QueryRequest, Record, Sha256, canonical_json, sha256_bytes
from app.external_gateway import DEEPSEEK, ExternalModelGateway
from app.generation import GenerationError
from app.ingestion.text import ProxyTokenizer
from app.model_gateway import CallContext
from app.profiled_generation import ProfiledGenerationRecipe, profiled_generation_request
from app.task_scoped_generation_v13 import SCOPE_RULES, parse_task_scoped_answer

RULES = SCOPE_RULES.replace("from untrusted_question or untrusted_english_search", "from untrusted_question")


class TaskScopedRecipeV14(Record):
    version: Literal["task-granularity-budget-neutral-v14"] = "task-granularity-budget-neutral-v14"
    model_metadata_sha256: Sha256
    source_protocol: Literal["V12_EXACT_USER_MESSAGE"] = "V12_EXACT_USER_MESSAGE"
    rules_sha256: Sha256 = sha256_bytes(RULES.encode())
    output_tokens: Literal[1600] = 1600
    user_message_proxy_tokens: Literal[4096] = 4096

    @property
    def fingerprint(self):
        if self.rules_sha256 != sha256_bytes(RULES.encode()):
            raise GenerationError("TASK_V14_RULES_PIN")
        return sha256_bytes(canonical_json(self.model_dump(mode="json")))


def task_scoped_request(context, documents, chunks, recipe, *, search):
    if (type(recipe) is not TaskScopedRecipeV14 or QueryRequest(question=search).question != search
            or context.language == "en" and context.question != search):
        raise GenerationError("TASK_V14_RECIPE_OR_SEARCH")
    recipe.fingerprint
    system, prompt = profiled_generation_request(context, documents, chunks,
        ProfiledGenerationRecipe(model_metadata_sha256=recipe.model_metadata_sha256))
    if ProxyTokenizer().count(prompt) > 4096:
        raise GenerationError("TASK_V14_USER_MESSAGE_BUDGET_NO_TRUNCATION")
    return system + "\n" + RULES, prompt


class TaskScopedGatewayV14:
    def __init__(self, base):
        if base.provider.live and type(base) is not ExternalModelGateway:
            raise GenerationError("TASK_V14_APPROVED_BASE_REQUIRED")
        self.base, self.provider, self.routes = base, base.provider, base.routes

    def generate_task_scoped(self, call, *, source_context, documents, chunks, recipe, search):
        if (type(call) is not CallContext or call.stage != "GENERATE" or call.attempt != 1
                or call.tenant_id != source_context.tenant_id
                or recipe.model_metadata_sha256 != self.routes.account_model_metadata_sha256):
            raise GenerationError("TASK_V14_GATEWAY_BINDING")
        system, prompt = task_scoped_request(source_context, documents, chunks, recipe, search=search)
        body = canonical_json({"model": DEEPSEEK,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"}, "temperature": 0, "max_tokens": 1600,
            "stream": False, "response_format": {"type": "json_object"}})
        return self.base._call(call, "DEEPSEEK", body, self.routes.generation_context_ceiling, 1600, None)


def parse_v14_answer(raw, stop, *, question, search):
    result = parse_task_scoped_answer(raw, stop, question=question, search=search)
    if any(g.question_part not in question for g in result.limitations):
        raise GenerationError("TASK_V14_LIMITATION_MUST_QUOTE_VISIBLE_ORIGINAL_QUESTION")
    return result
