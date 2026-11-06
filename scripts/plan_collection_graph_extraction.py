"""Freeze source-stratified graph inputs and token scenarios; no model or AWS calls."""

import json
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit

from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalRoutes
from app.extraction import AliasDictionary, deepseek_evidence_recipe, extraction_request
from app.ingestion.text import ProxyTokenizer

ROOT = Path(__file__).resolve().parents[1]


def make_plan(documents, chunks, recipe):
    docs = {d["document_version_id"]: d for d in documents}
    groups = defaultdict(list)
    for c in chunks:
        if 700 <= len(c["text"]) <= 3000:
            group = urlsplit(docs[c["document_version_id"]]["source_uri"]).path.split("/")[1]
            groups[group].append(c)
    selection = []
    for group, candidates in sorted(groups.items()):
        chosen_documents = set()
        for c in sorted(candidates, key=lambda c: (sha256_bytes(c["text"].encode()), c["chunk_id"])):
            if c["document_version_id"] in chosen_documents:
                continue
            selection.append({"group": group, "chunk_id": c["chunk_id"], "document_version_id": c["document_version_id"],
                "text_sha256": c["text_sha256"], "source_uri": docs[c["document_version_id"]]["source_uri"]})
            chosen_documents.add(c["document_version_id"])
            if len(chosen_documents) == 2:
                break
    if len(groups) != 24 or len(selection) != 48:
        raise ValueError("GRAPH_PILOT_REQUIRES_48_SOURCES_24_GROUPS")
    tokenizer = ProxyTokenizer()
    unique = {c["text_sha256"]: c for c in chunks}

    def scenario(values):
        inputs = sum(tokenizer.count("\n".join(extraction_request(c["text"], recipe))) for c in values)
        return {"dispatches_assuming_no_cache": len(values), "input_proxy_tokens": inputs,
            "provider_exact_tokens": False, "usd_scenarios_assumed_output_per_chunk": {
                str(output): round((inputs * .30 + len(values) * output * 1.20) / 1_000_000, 6)
                for output in (512, 1024, 2048)}}
    by_id = {c["chunk_id"]: c for c in chunks}
    return {"schema_version": 1, "status": "FROZEN_LOCAL_PLAN_NOT_EXECUTED", "corpus_documents": len(docs),
        "corpus_chunks": len(chunks), "unique_texts": len(unique), "selection_policy": "24_URI_NAMESPACES_2_DISTINCT_DOCS_SOURCE_TEXT_HASH_V1",
        "selection": selection, "extraction_recipe": recipe.model_dump(mode="json"),
        "pilot_scenario": scenario([by_id[s["chunk_id"]] for s in selection]), "full_scenario": scenario(unique.values()),
        "run_proposed": "collection316-graph-pilot-v1", "run_cap_proposed_micro_usd": 500000,
        "observed_cost_stop_between_calls_micro_usd": 150000, "max_dispatches": 48, "format_repair": False,
        "existing_shared_funding_cap_micro_usd": 2000000, "funding_cap_raise": False,
        "full_batch_authorized_by_this_file": False, "run_initialized": False,
        "conservative_single_dispatch_reservation_micro_usd": 317031,
        "explanation": "Forecast uses proxy input tokens and assumed output length. Gateway still reserves the accepted 1,048,576-token ceiling, not this proxy estimate. Full extraction must be re-estimated after pilot outcomes and remains unexecuted.",
        "semantic_review": "PENDING", "aws_calls": 0, "model_calls": 0}


def main():
    bundle = Path(json.loads((ROOT / "results/workbench-cloud/assets-build-v1.json").read_bytes())["bundle_directory"])
    documents, chunks = [], []
    for directory in (bundle / "collection").iterdir():
        if directory.is_dir():
            documents.extend(json.loads(line) for line in (directory / "documents.jsonl").read_text(encoding="utf-8").splitlines())
            chunks.extend(json.loads(line) for line in (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines())
    aliases = AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    plan = make_plan(documents, chunks, deepseek_evidence_recipe(aliases, metadata_sha256=routes.account_model_metadata_sha256))
    collection = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    plan["collection_manifest_sha256"] = sha256_bytes(Path(collection["collection_path"]).read_bytes())
    raw = canonical_json(plan) + b"\n"
    output = ROOT / "results/collection-graph/extraction-plan-v1.json"
    if output.exists() and output.read_bytes() != raw:
        raise ValueError("FROZEN_GRAPH_EXTRACTION_PLAN_CHANGED")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(raw)
    print(json.dumps({k: plan[k] for k in ("status", "corpus_documents", "corpus_chunks", "unique_texts", "pilot_scenario", "full_scenario", "model_calls")}))


if __name__ == "__main__":
    main()
