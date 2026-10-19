import json
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.contracts import (
    EmbeddingRecipe,
    ExternalEmbeddingRecipe,
    SnapshotManifest,
    embedding_cache_key,
)
from app.embeddings import EmbeddingCache, EmbeddingCacheError, titan_recipe
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.extraction import (
    AliasDictionary,
    deepseek_extraction_recipe,
    extraction_recipe,
    raw_cache_key,
    raw_recipe,
)
from app.extraction_cache import ExtractionCache
from app.fixtures import fixture_content, fixture_recipes, fixture_vector
from app.ledger import Ledger, MemoryLedgerStore
from app.model_gateway import CallContext
from app.snapshot import build_snapshot, load_snapshot
from scripts.check_external_ledger import verify

ROOT = Path(__file__).resolve().parents[1]


class Stub:
    live = False

    def __init__(self):
        self.calls = []

    def request(self, provider, method, path, body):
        body = json.loads(body)
        self.calls.append(body)
        if provider == "VOYAGE":
            return {"model": "voyage-4-lite", "usage": {"total_tokens": 10},
                    "data": [{"index": 0, "embedding": [1.0] + [0.0] * 1023}]}
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": '{"entities":[],"relationships":[]}',
                }}]}


def configured():
    sheet = ExternalPriceSheet.model_validate_json((ROOT / 'configs/prices/external-direct-2026-10-17.json').read_bytes())
    ledger = Ledger(MemoryLedgerStore(), clock=lambda: sheet.checked_at + timedelta(seconds=1))
    ledger.initialize_month("kg-r72", sheet.checked_at.strftime("%Y-%m"))
    ledger.initialize_run("kg-r72", "tenant_fixture", "external-cache", cap_micro_usd=500_000)
    ledger.initialize_external_funding("kg-r72", cap_micro_usd=2_000_000)
    return ExternalModelGateway(ledger, Stub(), ExternalRoutes(account_model_metadata_sha256="0" * 64),
                                sheet.model_copy(update={"purpose": "FIXTURE"}))


def context(stage="EMBED", request="req1"):
    return CallContext(tenant_id="tenant_fixture", run_id="external-cache", request_id=request, stage=stage)


def recipe(role="document"):
    return ExternalEmbeddingRecipe(input_role=role,
                                   input_recipe_version="title-section-text-v1" if role == "document" else "question-text-v1")


def test_roles_and_providers_have_distinct_keys_and_persisted_recipes(tmp_path):
    gateway = configured()
    cache = EmbeddingCache(tmp_path, namespace="FIXTURE")
    document = cache.resolve(context(), text="Same text", recipe=recipe(), gateway=gateway)
    query = cache.resolve(context(request="req2"), text="Same text", recipe=recipe("query"), gateway=gateway)
    assert document.record.cache_key != query.record.cache_key
    assert document.record.cache_key != embedding_cache_key("tenant_fixture", "Same text", titan_recipe())
    reopened = EmbeddingCache(tmp_path, namespace="FIXTURE").resolve(context(), text="Same text", recipe=recipe())
    assert reopened.cache_hit and reopened.new_cost_micro_usd == 0
    assert type(reopened.record.recipe) is ExternalEmbeddingRecipe
    assert [call["input_type"] for call in gateway.provider.calls] == ["document", "query"]


def test_legacy_shape_remains_exact_and_external_role_is_required_before_dispatch(tmp_path):
    legacy = titan_recipe()
    assert "input_role" not in legacy.model_dump() and "schema_version" not in legacy.model_dump()
    disguised = EmbeddingRecipe(version="invalid-voyage-legacy", model_id="voyage-4-lite",
                                invocation_id="voyage-4-lite", source_region="external-default")
    gateway = configured()
    with pytest.raises(EmbeddingCacheError, match="ROLE_RECIPE_REQUIRED"):
        EmbeddingCache(tmp_path, namespace="FIXTURE").resolve(context(), text="Text", recipe=disguised, gateway=gateway)
    assert not gateway.provider.calls
    with pytest.raises(ValidationError, match="role/input recipe"):
        ExternalEmbeddingRecipe(input_role="query", input_recipe_version="title-section-text-v1")


def test_deepseek_extraction_raw_cache_is_new_but_alias_changes_reuse_raw_output(tmp_path):
    gateway = configured()
    aliases = AliasDictionary(version="fixture-external-alias-v1", entries=())
    second_aliases = AliasDictionary(version="fixture-external-alias-v2", entries=())
    current = deepseek_extraction_recipe(aliases, metadata_sha256="0" * 64)
    changed = deepseek_extraction_recipe(second_aliases, metadata_sha256="0" * 64)
    old = extraction_recipe(aliases)
    assert raw_cache_key("tenant_fixture", "Same text", current) != raw_cache_key("tenant_fixture", "Same text", old)
    source = fixture_content()
    chunk, document = source.chunks[0], source.documents[0]
    cache = ExtractionCache(tmp_path, namespace="FIXTURE")
    first = cache.resolve(context("EXTRACT"), chunk=chunk, document=document, recipe=current,
                          aliases=aliases, gateway=gateway)
    second = cache.resolve(context("EXTRACT"), chunk=chunk, document=document, recipe=changed,
                           aliases=second_aliases)
    assert not first.cache_hit and second.cache_hit
    assert len(gateway.provider.calls) == 1
    assert first.record.attempts[0].operation_id == second.record.attempts[0].operation_id
    assert raw_recipe(current)["parameters"]["thinking"] == {"type": "disabled"}


def test_four_item_managed_probe_script_can_be_checked_offline():
    ledger = Ledger(MemoryLedgerStore())
    report = verify(ledger)
    assert report["status"] == "PASS" and report["authorized_dispatches"] == 1
    assert report["model_calls"] == 0


def test_external_snapshot_roundtrip_keeps_provider_role_identity(tmp_path):
    content = fixture_content()
    extraction, _ = fixture_recipes()
    # These are explicitly synthetic FIXTURE vectors, never a Voyage inference claim.
    vectors = {chunk.chunk_id: fixture_vector(chunk.text) for chunk in content.chunks}
    result = build_snapshot(tmp_path, content, vectors, tenant_id="tenant_fixture", purpose="FIXTURE",
                            extraction_recipe=extraction, embedding_recipe=recipe(), corpus_version="fixture",
                            normalizer_version="normalize-v1", chunking_version="chunk-v1", tokenizer_version="fixture-words-v1")
    loaded = load_snapshot(result, tenant_id="tenant_fixture", embedding_model_id="voyage-4-lite", allow_fixture=True)
    assert type(loaded.manifest.embedding_recipe) is ExternalEmbeddingRecipe
    payload = loaded.manifest.model_dump(mode="json")
    payload["embedding_recipe"] = recipe("query").model_dump(mode="json")
    payload["snapshot_version"] = SnapshotManifest.version_for(payload)
    with pytest.raises(ValidationError, match="document embedding recipe"):
        SnapshotManifest.model_validate_json(json.dumps(payload))
