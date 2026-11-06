"""Opt-in V12 model/cache boundary; legacy generation caps and caches are frozen.

Only a validated ProfiledSourceContext can enter the 1600-token path. Actual
dispatch uses the existing managed-ledger ExternalModelGateway._call boundary.
Failed/unknown requests remain durable and are never automatically repaired.
"""

from pathlib import Path
from uuid import uuid4

from app.answer_cache import AnswerCache, AnswerCacheBlocked, AnswerLookup, AnswerReceipt
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.external_gateway import DEEPSEEK, ExternalModelGateway
from app.generation import GenerationError
from app.model_gateway import CallContext, safe_failure_code
from app.profiled_generation import (
    ProfiledAnswer,
    ProfiledGenerationRecipe,
    parse_profiled_answer,
    profiled_generation_request,
    resolve_profiled_answer,
)
from app.profiled_source_context import validate_profiled_context


class ProfiledModelGateway:
    """No arbitrary prompt or arbitrary higher-cap generation entry point."""

    def __init__(self, base):
        if base.provider.live and type(base) is not ExternalModelGateway:
            raise GenerationError("PROFILE_GATEWAY_APPROVED_BASE_REQUIRED")
        self.base, self.provider, self.routes = base, base.provider, base.routes

    def generate_profiled(self, context, *, source_context, documents, chunks, recipe):
        if (type(context) is not CallContext or context.stage != "GENERATE" or context.attempt != 1
                or context.tenant_id != source_context.tenant_id or type(recipe) is not ProfiledGenerationRecipe
                or recipe.model_id != self.routes.generation_model_id
                or recipe.model_metadata_sha256 != self.routes.account_model_metadata_sha256):
            raise GenerationError("PROFILE_GATEWAY_CONTEXT_ROUTE_OR_RECIPE")
        system, prompt = profiled_generation_request(source_context, documents, chunks, recipe)
        body = canonical_json({"model": DEEPSEEK,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
            "thinking": {"type": "disabled"}, "temperature": 0, "max_tokens": recipe.max_output_tokens,
            "stream": False, "response_format": {"type": "json_object"}})
        # _call checks fresh prices, reserves/settles the shared funding and
        # run/month budgets, and retains uncertainty. No proxy count is called exact.
        return self.base._call(context, "DEEPSEEK", body, self.routes.generation_context_ceiling,
            recipe.max_output_tokens, None)


class ProfiledAnswerCache(AnswerCache):
    def __init__(self, directory, *, namespace="LIVE"):
        directory = Path(directory)
        if namespace not in {"LIVE", "FIXTURE"} or directory.is_symlink():
            raise GenerationError("PROFILE_ANSWER_CACHE_NAMESPACE_OR_PATH")
        directory.mkdir(parents=True, exist_ok=True)
        self.namespace, self.path = namespace, directory / "profiled_answers_v12.sqlite3"
        if self.path.is_symlink():
            raise GenerationError("PROFILE_ANSWER_CACHE_LINK")
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS answers (namespace TEXT NOT NULL,cache_key TEXT NOT NULL,"
                "state TEXT NOT NULL CHECK(state IN ('IN_PROGRESS','FAILED','READY')),owner TEXT NOT NULL,"
                "intent BLOB NOT NULL,payload BLOB,payload_sha256 TEXT,error_category TEXT,"
                "PRIMARY KEY(namespace,cache_key))")

    def resolve_profiled(self, context, *, source_context, documents, chunks, binding, recipe, gateway=None):
        if (type(context) is not CallContext or context.stage != "GENERATE" or context.attempt != 1
                or context.tenant_id != binding.tenant_id or context.request_id != binding.request_id
                or source_context.tenant_id != binding.tenant_id):
            raise GenerationError("PROFILE_ANSWER_TRUSTED_CONTEXT_BINDING")
        validate_profiled_context(source_context, documents, chunks)
        system, prompt = profiled_generation_request(source_context, documents, chunks, recipe)
        binding_sha = sha256_bytes(canonical_json({"binding": binding.model_dump(mode="json"),
            "context_fingerprint": source_context.fingerprint, "recipe": recipe.fingerprint,
            "system_and_prompt": sha256_bytes(canonical_json([system, prompt]))}))
        key = stable_id("profiled-answer-cache-v12", context=context.model_dump(mode="json"))

        def validate(receipt):
            if (receipt.recipe_sha256 != recipe.fingerprint or receipt.binding_sha256 != binding_sha
                    or receipt.operation_id != stable_id("op", **context.model_dump(mode="json"))):
                raise GenerationError("PROFILE_ANSWER_RECEIPT_TASK_OWNER_OR_RECIPE")
            return resolve_profiled_answer(parse_profiled_answer(receipt.raw_output, receipt.stop_reason),
                source_context, documents, chunks, binding=binding, recipe=recipe)

        receipt = self._get(key)
        if receipt is not None:
            return AnswerLookup(validate(receipt), receipt, True)
        if not source_context.chunk_ids:
            output = ProfiledAnswer(answer_status="ABSTAINED", claims=(), reason="INSUFFICIENT_EVIDENCE",
                limitations=({"code":"MISSING_SUPPLIED_EVIDENCE", "question_part":source_context.question[:200]},))
            return AnswerLookup(resolve_profiled_answer(output, source_context, documents, chunks,
                binding=binding, recipe=recipe), None, False, True)
        if gateway is None:
            raise GenerationError("PROFILE_ANSWER_CACHE_MISS_NO_GATEWAY")
        if (self.namespace != ("LIVE" if gateway.provider.live else "FIXTURE")
                or gateway.routes.generation_model_id != recipe.model_id
                or gateway.routes.account_model_metadata_sha256 != recipe.model_metadata_sha256):
            raise GenerationError("PROFILE_ANSWER_GATEWAY_NAMESPACE_OR_MODEL")
        owner = uuid4().hex
        intent = canonical_json({"context":context.model_dump(mode="json"), "binding_sha256":binding_sha,
            "recipe_sha256":recipe.fingerprint, "source_context_fingerprint":source_context.fingerprint})
        with self._connect() as db:
            changed = db.execute("INSERT OR IGNORE INTO answers(namespace,cache_key,state,owner,intent) "
                "VALUES (?,?,'IN_PROGRESS',?,?)", (self.namespace, key, owner, intent)).rowcount
            if changed != 1:
                raise AnswerCacheBlocked("PROFILE_ANSWER_CACHE_CLAIM_EXISTS")
        try:
            result = gateway.generate_profiled(context, source_context=source_context,
                documents=documents, chunks=chunks, recipe=recipe)
            receipt = AnswerReceipt(namespace=self.namespace, cache_key=key, operation_id=result.operation_id,
                recipe_sha256=recipe.fingerprint, binding_sha256=binding_sha, raw_output=result.text or "",
                stop_reason=result.stop_reason or "MISSING", usage=result.usage,
                original_cost_micro_usd=result.actual_micro_usd)
            raw = receipt.model_dump_json().encode()
            if len(raw) > 128*1024:
                raise GenerationError("PROFILE_ANSWER_RECEIPT_SIZE")
            with self._connect() as db:
                changed = db.execute("UPDATE answers SET payload=?,payload_sha256=? WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'", (raw, sha256_bytes(raw), self.namespace, key, owner)).rowcount
                if changed != 1:
                    raise AnswerCacheBlocked("PROFILE_ANSWER_CACHE_OWNER_LOST")
            response = validate(receipt)
            with self._connect() as db:
                changed = db.execute("UPDATE answers SET state='READY' WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'", (self.namespace, key, owner)).rowcount
                if changed != 1:
                    raise AnswerCacheBlocked("PROFILE_ANSWER_CACHE_OWNER_LOST")
            return AnswerLookup(response, receipt, False)
        except Exception as exc:
            with self._connect() as db:
                db.execute("UPDATE answers SET state='FAILED',error_category=? WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'", (safe_failure_code(exc), self.namespace, key, owner))
            raise
