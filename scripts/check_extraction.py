"""Offline extraction integration receipt. Never constructs an AWS client."""

import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.contracts import sha256_bytes
from app.cost import PriceSheet
from app.extraction import AliasDictionary, ExtractionOutput, extraction_recipe
from app.extraction_cache import ExtractionCache
from app.fixtures import FIXTURE_TENANT, fixture_content
from app.ledger import Ledger, MemoryLedgerStore
from app.model_gateway import CallContext, FixtureProvider, ModelGateway, ModelRoutes


def main():
    content = fixture_content()
    chunk = content.chunks[0]
    document = next(d for d in content.documents if d.document_version_id == chunk.document_version_id)
    text = chunk.text

    def span(quote):
        start = text.index(quote)
        return {"start_char": start, "end_char": start + len(quote), "quote": quote}

    raw = json.dumps({
        "entities": [{"local_id": f"e{i}", "name": name, "entity_type": "CONCEPT",
                      "mentions": [span(name)]} for i, name in enumerate(("Service A", "Service B"))],
        "relationships": [{"subject": "e0", "object": "e1", "predicate": "USES",
                           "polarity": "POSITIVE", "qualifiers": [], "evidence": span(text)}],
    })

    class RecordedProvider(FixtureProvider):
        calls = 0

        def count_input(self, route, body):
            return 100

        def invoke(self, route, body):
            self.calls += 1
            return {"usage": {"input_tokens": 100, "output_tokens": 80}, "stop_reason": "end_turn",
                    "content": [{"type": "text", "text": "{}" if self.calls == 1 else raw}]}

    now = datetime.now(UTC)
    ledger = Ledger(MemoryLedgerStore(), clock=lambda: now)
    ledger.initialize_month("kg-r72", now.strftime("%Y-%m"))
    ledger.initialize_run("kg-r72", FIXTURE_TENANT, "extraction-offline", cap_micro_usd=100_000)
    prices = PriceSheet.model_validate_json(Path("configs/prices/aws-us-east-1-2026-10-16.json").read_bytes())
    routes = ModelRoutes.model_validate_json(Path("configs/models/bedrock-us-east-1-v2.json").read_bytes())
    provider = RecordedProvider()
    gateway = ModelGateway(ledger, provider, routes, prices.model_copy(update={"purpose": "FIXTURE"}))
    aliases = AliasDictionary(version="synthetic-empty-v1", entries=())
    recipe = extraction_recipe(aliases)
    context = CallContext(tenant_id=FIXTURE_TENANT, run_id="extraction-offline",
                          request_id="synthetic-one", stage="EXTRACT")
    with tempfile.TemporaryDirectory(prefix="kg-extraction-") as directory:
        cache = ExtractionCache(Path(directory), namespace="FIXTURE")
        kwargs = {"chunk": chunk, "document": document, "recipe": recipe, "aliases": aliases}
        first = cache.resolve(context, **kwargs, gateway=gateway, allow_repair=True)
        replay = ExtractionCache(Path(directory), namespace="FIXTURE").resolve(context, **kwargs)
        revised = AliasDictionary(version="synthetic-empty-v2", entries=())
        rebuilt = cache.resolve(context, **{**kwargs, "aliases": revised,
                                           "recipe": extraction_recipe(revised)})
    checks = {
        "two_distinct_settled_operations": len({a.operation_id for a in first.record.attempts}) == 2
            and all(ledger.operation(a.operation_id).status == "SETTLED" for a in first.record.attempts),
        "cache_reopen_has_no_new_charge": replay.cache_hit and replay.new_cost_micro_usd == 0,
        "alias_rebuild_reuses_raw_output": rebuilt.cache_hit and rebuilt.new_cost_micro_usd == 0
            and rebuilt.canonical.cache_key != first.canonical.cache_key,
        "document_relative_source_spans": all(
            document.normalized_text[m.start_char:m.end_char] == m.quote for m in first.canonical.mentions),
        "unknown_entities_require_review": len(first.canonical.review_entity_ids) == 2,
        "semantic_review_still_pending": first.canonical.semantic_review == "PENDING",
    }
    report = {
        "status": "PASS" if all(checks.values()) else "FAIL", "checked_at": now.isoformat(),
        "purpose": "SYNTHETIC_OFFLINE_ONLY_NOT_A_MODEL_QUALITY_BENCHMARK", "checks": checks,
        "fixture_provider_calls": provider.calls, "real_model_calls": 0,
        "new_fixture_cost_micro_usd": first.new_cost_micro_usd,
        "notice": "Fixture costs use test receipts and a memory ledger; not AWS usage or an invoice.",
        "entities": len(first.canonical.entities), "mentions": len(first.canonical.mentions),
        "assertions": len(first.canonical.assertions),
        "source_hashes": {name: sha256_bytes(Path(name).read_bytes()) for name in (
            "app/extraction.py", "app/extraction_cache.py", "app/model_gateway.py",
            "configs/extraction/aws-approved-aliases-v1.json")},
    }
    schema = Path("schemas/extraction/ExtractionOutput.v1.schema.json")
    schema.parent.mkdir(parents=True, exist_ok=True)
    schema.write_text(json.dumps(ExtractionOutput.model_json_schema(), indent=2, sort_keys=True) + "\n",
                      encoding="utf-8")
    target = Path("results/extraction_offline_integration.json")
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
