"""Owner-partitioned durable raw receipts for the qualified answer service.

Uses the existing transactional store protocol. Memory/SQLite are fixtures;
live dispatch requires the approved managed DynamoDB table and endpoint.
"""

import re
from uuid import uuid4

from app.answer_cache import AnswerCacheBlocked, AnswerReceipt
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import GenerationError
from app.ledger import Change, ConcurrentWrite, DynamoLedgerStore
from app.model_gateway import safe_failure_code
from app.routed_source_reader import TextLookup


class TransactionalAnswerCache:
    def __init__(self, store, *, tenant_id, owner_id, namespace="LIVE"):
        if (namespace not in {"LIVE", "FIXTURE"} or not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", tenant_id)
                or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", owner_id)):
            raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_OWNER_OR_NAMESPACE")
        self.store, self.tenant_id, self.owner_id, self.namespace = store, tenant_id, owner_id, namespace
        self.partition = f"TENANT#{tenant_id}#USER#{owner_id}"

    def _key(self, context):
        cache_key = stable_id("routed-stage-cache-v1", context=context.model_dump(mode="json"))
        return (self.partition, "NATIVEANSWER#" + self.namespace + "#" + cache_key), cache_key

    def _commit(self, row, expected):
        if len(canonical_json(row)) > 192 * 1024:
            raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_ROW_BOUND")
        self.store.commit([Change((row["PK"], row["SK"]), expected, row)])

    def resolve_stage(self, context, *, system, prompt, max_output_tokens, binding_sha256, recipe_sha256,
        parser, gateway=None, dispatch=None):
        if context.stage != "GENERATE" or context.attempt != 1 or context.tenant_id != self.tenant_id:
            raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_CONTEXT")
        key, cache_key = self._key(context)
        identity = sha256_bytes(canonical_json({"binding": binding_sha256, "recipe": recipe_sha256,
            "system": system, "prompt": prompt, "max_output_tokens": max_output_tokens}))

        def validate_row(row):
            if (row.get("PK") != key[0] or row.get("SK") != key[1]
                    or row.get("record_type") != "NATIVE_ANSWER_RECEIPT_V1"
                    or row.get("tenant_id") != self.tenant_id or row.get("owner_id") != self.owner_id
                    or row.get("namespace") != self.namespace or row.get("identity_sha256") != identity):
                raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_TASK_OR_OWNER")

        def validate_receipt(receipt):
            if (receipt.namespace != self.namespace or receipt.cache_key != cache_key
                    or receipt.binding_sha256 != identity or receipt.recipe_sha256 != recipe_sha256
                    or receipt.operation_id != stable_id("op", **context.model_dump(mode="json"))):
                raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_RECEIPT_BINDING")
            return parser(receipt.raw_output, receipt.stop_reason)

        row = self.store.get(key)
        if row is not None:
            validate_row(row)
            if row.get("state") != "READY":
                raise AnswerCacheBlocked("TRANSACTIONAL_ANSWER_CACHE_RECONCILIATION_REQUIRED")
            raw = row.get("payload_json")
            if (type(raw) is not str or not 1 <= len(raw.encode()) <= 128 * 1024
                    or sha256_bytes(raw.encode()) != row.get("payload_sha256")):
                raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_PAYLOAD_CHECKSUM")
            receipt = AnswerReceipt.model_validate_json(raw)
            return TextLookup(validate_receipt(receipt), receipt, True)
        if gateway is None or dispatch is None:
            raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_MISS_NO_GATEWAY")
        if self.namespace != ("LIVE" if gateway.provider.live else "FIXTURE"):
            raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_GATEWAY_NAMESPACE")
        if gateway.provider.live and (not isinstance(self.store, DynamoLedgerStore)
                or self.store.table_name != "kg-r72-query-state"
                or self.store.client.meta.endpoint_url != "https://dynamodb.us-east-1.amazonaws.com"):
            raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_APPROVED_MANAGED_STORE_REQUIRED")
        lease = uuid4().hex
        row = {"PK": key[0], "SK": key[1], "version": 1, "record_type": "NATIVE_ANSWER_RECEIPT_V1",
            "tenant_id": self.tenant_id, "owner_id": self.owner_id, "namespace": self.namespace,
            "state": "IN_PROGRESS", "lease_owner": lease, "identity_sha256": identity}
        try:
            self._commit(row, None)
        except ConcurrentWrite:
            raise AnswerCacheBlocked("TRANSACTIONAL_ANSWER_CACHE_CLAIM_EXISTS") from None
        # An uncertain claim write propagates before dispatch. Never steal it.
        try:
            result = dispatch(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
            receipt = AnswerReceipt(namespace=self.namespace, cache_key=cache_key, operation_id=result.operation_id,
                recipe_sha256=recipe_sha256, binding_sha256=identity, raw_output=result.text or "",
                stop_reason=result.stop_reason or "MISSING", usage=result.usage,
                original_cost_micro_usd=result.actual_micro_usd)
            raw = receipt.model_dump_json()
            if len(raw.encode()) > 128 * 1024:
                raise GenerationError("TRANSACTIONAL_ANSWER_CACHE_RECEIPT_BOUND")
            row = row | {"version": 2, "payload_json": raw, "payload_sha256": sha256_bytes(raw.encode())}
            self._commit(row, 1)  # Durable raw first, qualification separately.
            value = validate_receipt(receipt)
            row = row | {"version": 3, "state": "READY"}
            self._commit(row, 2)
            return TextLookup(value, receipt, False)
        except Exception as exc:
            current = self.store.get(key)
            if current is not None:
                validate_row(current)
                if current.get("state") == "IN_PROGRESS" and current.get("lease_owner") == lease:
                    failed = current | {"version": current["version"]+1, "state": "FAILED",
                        "error_category": safe_failure_code(exc)}
                    try:
                        self._commit(failed, current["version"])
                    except ConcurrentWrite:
                        pass  # Another transition won; never overwrite it or dispatch again.
            raise
