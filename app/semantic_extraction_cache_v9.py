"""V9 native raw receipts with source/recipe keys, durable claims and no paid retries."""

import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.external_http import strict_json
from app.extraction import ExtractionError
from app.model_gateway import CallContext, safe_failure_code
from app.semantic_extraction_v9 import (
    semantic_raw_recipe,
    semantic_request,
    validate_semantic_output,
)


@dataclass(frozen=True)
class SemanticExtractionLookup:
    validation: object
    receipt: dict
    cache_hit: bool
    new_cost_micro_usd: int


class SemanticExtractionCache:
    def __init__(self, directory: Path, *, namespace="LIVE"):
        if namespace not in {"LIVE", "FIXTURE"} or Path(directory).is_symlink():
            raise ExtractionError("SEMANTIC_CACHE_NAMESPACE_OR_PATH")
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path, self.namespace = directory / "semantic_extractions_v9.sqlite3", namespace
        if self.path.is_symlink():
            raise ExtractionError("SEMANTIC_CACHE_UNSAFE_PATH")
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS extractions(namespace TEXT, cache_key TEXT, binding BLOB, "
                "owner TEXT, state TEXT CHECK(state IN ('IN_PROGRESS','READY','FAILED')), payload BLOB, payload_sha TEXT, "
                "error TEXT, PRIMARY KEY(namespace,cache_key))")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            with db:
                yield db
        finally:
            db.close()

    def resolve(self, context, *, text, recipe, gateway=None):
        context = CallContext.model_validate_json(context.model_dump_json())
        if context.stage != "EXTRACT" or context.attempt != 1:
            raise ExtractionError("SEMANTIC_CACHE_CONTEXT")
        system, prompt = semantic_request(text)
        if (recipe.version != "deepseek-clause-head-context-v9" or recipe.parameters.get("max_output_tokens") != 2048
                or recipe.parameters.get("system_prompt_sha256") != sha256_bytes(system.encode())):
            raise ExtractionError("SEMANTIC_CACHE_RECIPE")
        raw_recipe = semantic_raw_recipe(recipe)
        identity = {"tenant": context.tenant_id, "source": sha256_bytes(text.encode()), "recipe": raw_recipe,
            "system_sha256": sha256_bytes(system.encode()), "prompt_sha256": sha256_bytes(prompt.encode())}
        binding = canonical_json(identity)
        key = stable_id("semantic-v9-raw", **identity)
        with self._db() as db:
            saved = db.execute("SELECT binding,state,payload,payload_sha FROM extractions WHERE namespace=? AND cache_key=?",
                (self.namespace, key)).fetchone()
        if saved:
            if saved[0] != binding or saved[1] != "READY":
                raise ExtractionError("SEMANTIC_CACHE_RECONCILIATION_REQUIRED")
            if not isinstance(saved[2], bytes) or len(saved[2]) > 65536 or sha256_bytes(saved[2]) != saved[3]:
                raise ExtractionError("SEMANTIC_CACHE_CHECKSUM")
            receipt = strict_json(saved[2])
            validation = self._validate(receipt, identity, text)
            return SemanticExtractionLookup(validation, receipt, True, 0)
        if gateway is None:
            raise ExtractionError("SEMANTIC_CACHE_MISS_NO_GATEWAY")
        if (getattr(gateway.provider, "live", None) is not (self.namespace == "LIVE")
                or gateway.routes.generation_model_id != recipe.model_id
                or gateway.routes.generation_invocation_id != recipe.invocation_id
                or gateway.routes.source_region != recipe.source_region
                or gateway.routes.account_model_metadata_sha256 != recipe.parameters.get("model_metadata_sha256")):
            raise ExtractionError("SEMANTIC_CACHE_ROUTE_OR_NAMESPACE")
        owner = uuid4().hex
        with self._db() as db:
            if db.execute("INSERT OR IGNORE INTO extractions(namespace,cache_key,binding,owner,state) VALUES(?,?,?,?,'IN_PROGRESS')",
                    (self.namespace, key, binding, owner)).rowcount != 1:
                raise ExtractionError("SEMANTIC_CACHE_ALREADY_CLAIMED")
        try:
            response = gateway.generate(context, system=system, prompt=prompt, max_output_tokens=2048)
            receipt = {"context": context.model_dump(mode="json"), "operation_id": response.operation_id,
                "identity": identity, "raw_output": response.text, "stop_reason": response.stop_reason,
                "usage": response.usage.model_dump(mode="json"), "cost_micro_usd": response.actual_micro_usd}
            payload = canonical_json(receipt)
            if len(payload) > 65536:
                raise ExtractionError("SEMANTIC_CACHE_RECEIPT_BOUND")
            with self._db() as db:
                if db.execute("UPDATE extractions SET payload=?,payload_sha=? WHERE namespace=? AND cache_key=? "
                        "AND owner=? AND state='IN_PROGRESS'", (payload, sha256_bytes(payload), self.namespace, key, owner)).rowcount != 1:
                    raise ExtractionError("SEMANTIC_CACHE_OWNER_LOST")
            validation = self._validate(receipt, identity, text)
            with self._db() as db:
                if db.execute("UPDATE extractions SET state='READY' WHERE namespace=? AND cache_key=? "
                        "AND owner=? AND state='IN_PROGRESS'", (self.namespace, key, owner)).rowcount != 1:
                    raise ExtractionError("SEMANTIC_CACHE_OWNER_LOST")
            return SemanticExtractionLookup(validation, receipt, False, response.actual_micro_usd)
        except Exception as exc:
            with self._db() as db:
                db.execute("UPDATE extractions SET state='FAILED',error=? WHERE namespace=? AND cache_key=? "
                    "AND owner=? AND state='IN_PROGRESS'", (safe_failure_code(exc), self.namespace, key, owner))
            raise

    @staticmethod
    def _validate(receipt, identity, text):
        if receipt["identity"] != identity:
            raise ExtractionError("SEMANTIC_CACHE_RECEIPT_IDENTITY")
        context = CallContext.model_validate(receipt["context"])
        if (receipt["operation_id"] != stable_id("op", **context.model_dump()) or context.stage != "EXTRACT"
                or context.tenant_id != identity["tenant"] or context.attempt != 1
                or type(receipt["cost_micro_usd"]) is not int or receipt["cost_micro_usd"] < 0):
            raise ExtractionError("SEMANTIC_CACHE_RECEIPT_OPERATION")
        TokenUsage.model_validate(receipt["usage"])
        return validate_semantic_output(receipt["raw_output"], text, stop_reason=receipt["stop_reason"])
