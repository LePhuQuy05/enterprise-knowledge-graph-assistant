"""Inventory every source occurrence and price full V4 extraction, without calls.

Pending source units are not promoted from older recipes or from mere mentions.
All duplicate occurrences remain bound to their original document and offsets.
"""

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

from app.contracts import ExtractionRecipe, canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.external_gateway import ExternalPriceSheet
from app.extraction import AliasDictionary, deepseek_anchored_recipe, extraction_request
from app.ingestion.text import ProxyTokenizer
from scripts.run_collection_benchmark import ROOT, load_inputs


def plan_inventory(collection, recipe, rate):
    docs, units = {}, {}
    for ref in collection.manifest.shards:
        shard = collection.get_shard(ref.shard_id)
        docs.update(shard.documents)
        for chunk in shard.chunks.values():
            digest = chunk.text_sha256
            if digest not in units:
                units[digest] = {"text": chunk.text, "occurrences": []}
            if units[digest]["text"] != chunk.text:
                raise ValueError("GRAPH_INVENTORY_TEXT_HASH_COLLISION")
            document = shard.documents[chunk.document_version_id]
            if document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text:
                raise ValueError("GRAPH_INVENTORY_SOURCE_SLICE")
            units[digest]["occurrences"].append({"chunk_id": chunk.chunk_id,
                "document_version_id": chunk.document_version_id, "source_uri": document.source_uri,
                "start_char": chunk.start_char, "end_char": chunk.end_char,
                "service_group": urlsplit(document.source_uri).path.strip("/").split("/")[0]})
    tokenizer, prepared = ProxyTokenizer(), []
    for digest, value in sorted(units.items()):
        # Source-bound V4 prompt, not the old V3 estimate. These are proxy counts,
        # not provider tokens or a reduced admission reservation.
        system, prompt = extraction_request(value["text"], recipe)
        input_proxy = tokenizer.count(system + "\n" + prompt)
        prepared.append({"unit_id": stable_id("graph-full-v2", source=collection.manifest_sha256,
            text=digest, recipe=recipe.fingerprint), "text_sha256": digest, "input_proxy_tokens": input_proxy,
            "status": "PENDING", "occurrences": sorted(value["occurrences"], key=lambda x: x["chunk_id"])})
    return docs, prepared


def make_plan(output, aliases_path=None):
    _, _, collection, _, _ = load_inputs()
    recipe_raw = (ROOT / "results/collection-graph/anchored-v4-plan-v2.json").read_bytes()
    recipe = ExtractionRecipe.model_validate(json.loads(recipe_raw)["extraction_recipe"])
    aliases = AliasDictionary.load(aliases_path) if aliases_path else None
    if aliases:
        recipe = deepseek_anchored_recipe(aliases, metadata_sha256=recipe.parameters["model_metadata_sha256"])
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    rate = prices.rate_for(recipe.model_id, recipe.invocation_id, recipe.source_region, now=datetime.now(UTC), live=False)
    docs, units = plan_inventory(collection, recipe, rate)
    occurrence_count = sum(len(u["occurrences"]) for u in units)
    if len(docs) != 316 or occurrence_count != 2978 or len(units) != 2897:
        raise ValueError("GRAPH_INVENTORY_EXPECTED_PINNED_COLLECTION_CHANGED")
    groups = defaultdict(set)
    for unit in units:
        for occurrence in unit["occurrences"]:
            groups[occurrence["service_group"]].add(occurrence["document_version_id"])
    forecasts = {str(tokens): sum(rate.quote(TokenUsage(input_tokens=u["input_proxy_tokens"], output_tokens=tokens))
        for u in units) for tokens in (512, 1024, 2048)}
    output.mkdir(parents=True, exist_ok=False)
    raw = b"".join(canonical_json(u) + b"\n" for u in units)
    (output / "source-units.jsonl").write_bytes(raw)
    graph_receipt = json.loads((ROOT / "results/collection-graph/curated-subset-v1.json").read_bytes())
    overlay = json.loads(Path(graph_receipt["graph_path"]).read_bytes())
    actual_assertion_docs = {a["document_version_id"] for a in overlay["assertions"]}
    result = {"schema_version": 2, "status": "FULL_SOURCE_INVENTORY_READY_EXTRACTION_NOT_EXECUTED",
        "collection_manifest_sha256": collection.manifest_sha256, "documents": len(docs), "chunks": occurrence_count,
        "unique_texts": len(units), "service_groups": len(groups), "documents_by_group": {g: len(v) for g, v in sorted(groups.items())},
        "unit_statuses": dict(Counter(u["status"] for u in units)), "source_units_sha256": sha256_bytes(raw),
        "extraction_recipe": recipe.model_dump(mode="json"), "recipe_sha256": recipe.fingerprint,
        "aliases_sha256": sha256_bytes(aliases_path.read_bytes()) if aliases_path else None,
        "aliases_review": "PROJECT_DICTIONARY_AI_SOURCE_AUDIT_REQUIRED_NOT_HUMAN_GOLD" if aliases else "LEGACY_V1",
        "input_proxy_tokens": sum(u["input_proxy_tokens"] for u in units),
        "provider_exact_tokens": False, "forecast_micro_usd_by_assumed_output_tokens": forecasts,
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "price_expires_at": prices.expires_at.isoformat(), "cost_basis": prices.cost_basis,
        "forecast_excludes": ["repairs", "semantic_review", "answer_generation"],
        "current_overlay_assertions": len(overlay["assertions"]), "current_overlay_source_documents": len(actual_assertion_docs),
        "proposed_tranche_units": 100, "full_batch_authorized": False, "ledger_caps_changed": False,
        "provider_calls": 0, "aws_calls": 0, "planned_at": datetime.now(UTC).isoformat(),
        "notes": ["PENDING refers to this full V4 recipe inventory, not proof that older V3 attempts never existed.",
            "Do not use proxy token counts to reduce the full-context admission reservation.",
            "Capacity/ontology/semantic and funding gates must pass before paid batches; source inventory alone is not a full graph."]}
    (output / "plan.json").write_bytes(canonical_json(result) + b"\n")
    print(json.dumps({k: result[k] for k in ("status", "documents", "chunks", "unique_texts", "input_proxy_tokens",
        "forecast_micro_usd_by_assumed_output_tokens", "provider_calls")}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--aliases", type=Path)
    args = parser.parse_args()
    make_plan(args.output_dir, args.aliases)


if __name__ == "__main__":
    main()
