import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from app.contracts import canonical_json, stable_id
from app.cost import TokenUsage
from app.extraction import ExtractionError
from app.model_gateway import CallContext
from app.semantic_extraction import semantic_recipe
from app.semantic_extraction_cache import SemanticExtractionCache
from tests.test_extraction import aliases

TEXT = "Service A supports TLS."


class Gateway:
    provider = SimpleNamespace(live=False)
    routes = SimpleNamespace(generation_model_id="deepseek-flash", generation_invocation_id="deepseek-flash",
        source_region="external-default", account_model_metadata_sha256="0" * 64)

    def __init__(self, raw=None):
        self.calls = 0
        self.raw = raw or json.dumps({"relationships": [{"evidence_quote": TEXT,
            "subject": {"name": "Service A", "entity_type": "RESOURCE", "occurrence": 1},
            "object": {"name": "TLS", "entity_type": "PROTOCOL", "occurrence": 1},
            "predicate": "SUPPORTS", "trigger_quote": "supports", "voice": "ACTIVE", "polarity": "POSITIVE",
            "modality": "ASSERTED", "qualifiers": []}], "permission_requirements": []})

    def generate(self, context, **kwargs):
        assert kwargs["max_output_tokens"] == 2048
        self.calls += 1
        return SimpleNamespace(operation_id=stable_id("op", **context.model_dump()), text=self.raw,
            stop_reason="end_turn", usage=TokenUsage(input_tokens=900, output_tokens=300), actual_micro_usd=700)


def inputs():
    return CallContext(tenant_id="demo", run_id="semantic-fixture", request_id="source", stage="EXTRACT"), semantic_recipe(aliases(), metadata_sha256="0" * 64)


def test_raw_reuse_keeps_original_paid_receipt_and_accepts_different_source_occurrence_context(tmp_path):
    cache, gateway = SemanticExtractionCache(tmp_path, namespace="FIXTURE"), Gateway()
    context, recipe = inputs()
    first = cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    hit = cache.resolve(context.model_copy(update={"request_id": "duplicate-occurrence"}), text=TEXT, recipe=recipe)
    assert gateway.calls == 1 and hit.cache_hit and hit.new_cost_micro_usd == 0
    assert first.receipt == hit.receipt and len(hit.validation.accepted_relationships) == 1
    assert hit.receipt["context"]["request_id"] == "source"


def test_settled_invalid_json_keeps_original_receipt_and_never_redispatches(tmp_path):
    cache, gateway = SemanticExtractionCache(tmp_path, namespace="FIXTURE"), Gateway(raw='{"bad":true}')
    context, recipe = inputs()
    with pytest.raises(ExtractionError, match="SCHEMA"):
        cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    with cache._db() as db:
        state, payload = db.execute("SELECT state,payload FROM extractions").fetchone()
    assert state == "FAILED" and json.loads(payload)["cost_micro_usd"] == 700
    with pytest.raises(ExtractionError, match="RECONCILIATION"):
        cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    assert gateway.calls == 1


def test_unknown_dispatch_is_not_an_automatic_paid_retry(tmp_path):
    cache, gateway = SemanticExtractionCache(tmp_path, namespace="FIXTURE"), Gateway()
    context, recipe = inputs()
    def uncertain(*args, **kwargs):
        gateway.calls += 1
        raise RuntimeError("UNKNOWN_SPEND_FIXTURE")
    gateway.generate = uncertain
    with pytest.raises(RuntimeError):
        cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    with pytest.raises(ExtractionError, match="RECONCILIATION"):
        cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    assert gateway.calls == 1


def test_namespace_and_checksum_corruption_never_reach_provider(tmp_path):
    cache, gateway = SemanticExtractionCache(tmp_path, namespace="LIVE"), Gateway()
    context, recipe = inputs()
    with pytest.raises(ExtractionError, match="NAMESPACE"):
        cache.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    fixture = SemanticExtractionCache(tmp_path, namespace="FIXTURE")
    fixture.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    with fixture._db() as db:
        db.execute("UPDATE extractions SET payload=?", (canonical_json({"tampered": True}),))
    with pytest.raises(ExtractionError, match="CHECKSUM"):
        fixture.resolve(context, text=TEXT, recipe=recipe, gateway=gateway)
    assert gateway.calls == 1


def test_concurrent_same_source_dispatches_at_most_once(tmp_path):
    cache, gateway = SemanticExtractionCache(tmp_path, namespace="FIXTURE"), Gateway()
    context, recipe = inputs()
    def run(i):
        try:
            return cache.resolve(context.model_copy(update={"request_id": f"caller-{i}"}), text=TEXT, recipe=recipe, gateway=gateway)
        except ExtractionError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, (1, 2)))
    assert gateway.calls == 1 and any(r is not None for r in results)
