import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.contracts import canonical_json, stable_id
from app.cost import TokenUsage
from app.extraction import ExtractionError
from app.model_gateway import CallContext
from app.semantic_extraction_cache_v10 import SemanticExtractionCache
from app.semantic_extraction_v10 import semantic_recipe
from tests.test_extraction import aliases
from tests.test_semantic_extraction_v10 import ep as endpoint
from tests.test_semantic_extraction_v10 import rel as relation

TEXT = "CloudWatch supports TLS."


class Gateway:
    provider = SimpleNamespace(live=False)
    routes = SimpleNamespace(generation_model_id="deepseek-flash", generation_invocation_id="deepseek-flash",
        source_region="external-default", account_model_metadata_sha256="0" * 64)

    def __init__(self, raw=None, *, unknown=False):
        self.calls, self.unknown = 0, unknown
        self.raw = raw or canonical_json({"relationships": [relation(TEXT, endpoint("CloudWatch", "SERVICE_PRODUCT"),
            endpoint("TLS", "PROTOCOL"), predicate="SUPPORTS", trigger_quote="supports")],
            "actions": [], "permission_requirements": [], "deferred_statements": [], "coverage_note": "HAS_CANDIDATES"}).decode()

    def generate(self, context, **kwargs):
        assert kwargs["max_output_tokens"] == 2048
        self.calls += 1
        if self.unknown:
            raise RuntimeError("UNKNOWN_SPEND_FIXTURE")
        return SimpleNamespace(operation_id=stable_id("op", **context.model_dump()), text=self.raw, stop_reason="end_turn",
            usage=TokenUsage(input_tokens=1000, output_tokens=200), actual_micro_usd=540)


def inputs():
    return CallContext(tenant_id="demo", run_id="v10-fixture", request_id="source", stage="EXTRACT"), semantic_recipe(aliases(), metadata_sha256="0" * 64)


def test_native_v10_durable_receipt_replay_is_separate_and_free(tmp_path):
    cache, gateway = SemanticExtractionCache(tmp_path, namespace="FIXTURE"), Gateway()
    context, recipe = inputs()
    first = cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    second = cache.resolve(context.model_copy(update={"request_id": "other-occurrence"}), text=TEXT, recipe=recipe)
    assert cache.path.name == "semantic_extractions_v10.sqlite3"
    assert gateway.calls == 1 and second.cache_hit and second.new_cost_micro_usd == 0
    assert first.receipt == second.receipt and second.validation.accepted_relationships[0]["evidence"]["quote"] == TEXT


@pytest.mark.parametrize("failure", ["schema", "unknown"])
def test_failed_or_uncertain_native_v10_never_redispatches(tmp_path, failure):
    cache, gateway = SemanticExtractionCache(tmp_path, namespace="FIXTURE"), Gateway(raw='{"extra":true}' if failure == "schema" else None, unknown=failure == "unknown")
    context, recipe = inputs()
    with pytest.raises((RuntimeError, ExtractionError)):
        cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    with cache._db() as db:
        state, payload = db.execute("SELECT state,payload FROM extractions").fetchone()
    assert state == "FAILED"
    assert (payload is not None) is (failure == "schema")
    if payload:
        assert json.loads(payload)["cost_micro_usd"] == 540
    with pytest.raises(ExtractionError, match="RECONCILIATION"):
        cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    assert gateway.calls == 1


def test_live_namespace_and_corrupted_cache_do_not_invoke_provider(tmp_path):
    context, recipe = inputs()
    gateway = Gateway()
    with pytest.raises(ExtractionError, match="NAMESPACE"):
        SemanticExtractionCache(tmp_path, namespace="LIVE").resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    cache = SemanticExtractionCache(tmp_path, namespace="FIXTURE")
    cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    with cache._db() as db:
        db.execute("UPDATE extractions SET payload=?", (b'{}',))
    with pytest.raises(ExtractionError, match="CHECKSUM"):
        cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    assert gateway.calls == 1


def test_concurrent_source_claim_still_dispatches_once(tmp_path):
    context, recipe = inputs()
    cache, gateway = SemanticExtractionCache(tmp_path, namespace="FIXTURE"), Gateway()
    def call(index):
        try:
            return cache.resolve(context.model_copy(update={"request_id": f"caller-{index}"}), text=TEXT, recipe=recipe, gateway=gateway)
        except ExtractionError:
            return None
    with ThreadPoolExecutor(max_workers=3) as pool:
        outcomes = list(pool.map(call, range(3)))
    assert gateway.calls == 1 and any(outcomes)
