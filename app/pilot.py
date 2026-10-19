"""Bounded pilot selection and cache-backed execution; no account setup or network discovery."""

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.contracts import ExternalEmbeddingRecipe, canonical_json, embedding_input, sha256_bytes
from app.corpus import LocalCorpus
from app.embeddings import EmbeddingCache
from app.extraction import AliasDictionary, deepseek_extraction_recipe, deepseek_quote_recipe
from app.extraction_cache import ExtractionCache
from app.model_gateway import CallContext


def select_chunks(corpus: LocalCorpus, manifest: dict, *, limit: int = 10) -> list[dict]:
    """Round-robin sources, one complete short chunk/doc; not a benchmark sample."""
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("PILOT_LIMIT_1_TO_10")
    docs = {d.source_uri: d for d in corpus.documents}
    groups = {service: [] for service in ("Lambda", "S3", "IAM", "KMS")}
    for source in manifest["sources"]:
        doc = docs.get(source["url"])
        if doc is None or source["service"] not in groups:
            continue
        candidates = sorted((c for c in corpus.chunks if c.document_version_id == doc.document_version_id
                             and 160 <= len(c.text) <= 1600), key=lambda c: (c.position, c.chunk_id))
        if candidates:
            chunk = candidates[0]
            groups[source["service"]].append({"service": source["service"], "source_uri": doc.source_uri,
                                               "document_version_id": doc.document_version_id,
                                               "chunk_id": chunk.chunk_id, "text_sha256": sha256_bytes(chunk.text.encode()),
                                               "characters": len(chunk.text), "position": chunk.position})
    chosen = []
    for index in range(max((len(g) for g in groups.values()), default=0)):
        for group in groups.values():
            if index < len(group):
                chosen.append(group[index])
                if len(chosen) == limit:
                    return chosen
    raise ValueError("NOT_ENOUGH_DISTINCT_SHORT_DOCUMENTS_FOR_PILOT")


def pilot_plan(corpus: LocalCorpus, selection: list[dict], aliases: AliasDictionary, routes) -> dict:
    embedding = ExternalEmbeddingRecipe(input_role="document", input_recipe_version="title-section-text-v1")
    extraction = deepseek_extraction_recipe(aliases, metadata_sha256=routes.account_model_metadata_sha256)
    payload = {"version": "external-pilot10-v1", "corpus_version": corpus.version,
               "tenant_id": corpus.tenant_id, "selection_rule": "service-round-robin-first-complete-chunk-160-to-1600-chars-v1",
               "selection": selection, "embedding_recipe": embedding.model_dump(mode="json"),
               "extraction_recipe": extraction.model_dump(mode="json"), "allow_one_settled_format_repair": True,
               "run_cap_micro_usd": 500_000, "semantic_review": "PENDING", "benchmark_sample": False}
    return payload | {"plan_sha256": sha256_bytes(canonical_json(payload))}


def revised_quote_plan(corpus: LocalCorpus, prior: dict, unknown_chunk_id: str,
                       aliases: AliasDictionary, routes) -> dict:
    """A different frozen experiment, retaining the original allowance and tombstones."""
    chunks = {c.chunk_id: c for c in corpus.chunks}
    docs = {d.document_version_id: d for d in corpus.documents}
    unknown = chunks[unknown_chunk_id]
    excluded_input = embedding_input(docs[unknown.document_version_id], unknown)
    selection = []
    for row in prior["selection"]:
        candidates = sorted((c for c in corpus.chunks if c.document_version_id == row["document_version_id"]
                             and 160 <= len(c.text) <= 1600
                             and embedding_input(docs[c.document_version_id], c) != excluded_input),
                            key=lambda c: (c.chunk_id == row["chunk_id"], c.position, c.chunk_id))
        if not candidates:
            raise ValueError("NO_ALTERNATIVE_TO_UNKNOWN_INPUT")
        chunk = candidates[0]
        selection.append(row | {"chunk_id": chunk.chunk_id, "text_sha256": sha256_bytes(chunk.text.encode()),
                                "characters": len(chunk.text), "position": chunk.position})
    payload = {k: v for k, v in prior.items() if k != "plan_sha256"}
    payload.update(version="external-pilot10-v2", selection=selection, request_id_prefix="v2-",
                   extraction_recipe=deepseek_quote_recipe(aliases, metadata_sha256=routes.account_model_metadata_sha256).model_dump(mode="json"),
                   prior_plan_sha256=prior["plan_sha256"], excluded_unknown_chunk_id=unknown_chunk_id,
                   selection_rule="same-doc-next-complete-short-chunk-excluding-unknown-input-v2",
                   voyage_pacing={"source": "USER_REPORTED_2026-10-17", "rpm": 3, "tpm": 10000,
                                  "seconds_after_previous_completion": 21, "scope": "LOCAL_SINGLE_WORKER_ONLY"})
    return payload | {"plan_sha256": sha256_bytes(canonical_json(payload))}


def run_selected(corpus: LocalCorpus, plan: dict, *, aliases: AliasDictionary, gateway,
                 run_id: str, cache_directory: Path, artifact_directory: Path, checkpoint,
                 stages: tuple[str, ...] = ("EMBED", "EXTRACT")) -> dict:
    """Stop on any failure; cache/ledger retain completed and ambiguous work durably."""
    payload = {k: v for k, v in plan.items() if k != "plan_sha256"}
    if sha256_bytes(canonical_json(payload)) != plan["plan_sha256"] or corpus.version != plan["corpus_version"]:
        raise ValueError("PILOT_PLAN_OR_CORPUS_CHANGED")
    if stages not in {("EMBED", "EXTRACT"), ("EXTRACT",)}:
        raise ValueError("PILOT_STAGE_SELECTION_INVALID")
    selection = plan["selection"]
    if not 1 <= len(selection) <= 10 or len({r["chunk_id"] for r in selection}) != len(selection):
        raise ValueError("PILOT_SELECTION_LIMIT_OR_DUPLICATES")
    docs = {d.document_version_id: d for d in corpus.documents}
    chunks = {c.chunk_id: c for c in corpus.chunks}
    for row in selection:
        chunk = chunks[row["chunk_id"]]
        if chunk.document_version_id != row["document_version_id"] or sha256_bytes(chunk.text.encode()) != row["text_sha256"]:
            raise ValueError("PILOT_SOURCE_BINDING_CHANGED")
    namespace = "LIVE" if gateway.provider.live else "FIXTURE"
    embeddings = EmbeddingCache(cache_directory, namespace=namespace)
    extractions = ExtractionCache(cache_directory, namespace=namespace)
    embedding_recipe = ExternalEmbeddingRecipe.model_validate_json(canonical_json(plan["embedding_recipe"]))
    factory = deepseek_quote_recipe if plan["extraction_recipe"]["version"] == "deepseek-quotes-v2" else deepseek_extraction_recipe
    extraction_recipe = factory(aliases, metadata_sha256=gateway.routes.account_model_metadata_sha256)
    if extraction_recipe.model_dump(mode="json") != plan["extraction_recipe"]:
        raise ValueError("PILOT_RECIPE_CHANGED")
    artifact_directory.mkdir(parents=True, exist_ok=True)
    report = {"started_at": datetime.now(UTC).isoformat(), "status": "RUNNING", "namespace": namespace,
              "run_id": run_id, "plan_sha256": plan["plan_sha256"], "embeddings": [], "extractions": [],
              "semantic_review": "PENDING", "provider_invoice_micro_usd": None, "stages": list(stages)}
    checkpoint(report)
    vectors, canonical_rows = [], []
    prefix = plan.get("request_id_prefix", "")
    for row in selection if "EMBED" in stages else ():
        chunk = chunks[row["chunk_id"]]
        result = embeddings.resolve(CallContext(tenant_id=corpus.tenant_id, run_id=run_id,
                                                request_id="embed-" + prefix + chunk.chunk_id, stage="EMBED"),
                                    text=embedding_input(docs[chunk.document_version_id], chunk),
                                    recipe=embedding_recipe, gateway=gateway)
        vectors.append(result.record.vector)
        report["embeddings"].append({"chunk_id": chunk.chunk_id, "cache_hit": result.cache_hit,
                                     "operation_id": result.record.operation_id,
                                     "new_budget_cost_micro_usd": result.new_cost_micro_usd})
        checkpoint(report)
    if "EMBED" in stages:
        np.save(artifact_directory / "vectors.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False)
    for row in selection:
        chunk = chunks[row["chunk_id"]]
        result = extractions.resolve(CallContext(tenant_id=corpus.tenant_id, run_id=run_id,
                                                 request_id="extract-" + prefix + chunk.chunk_id, stage="EXTRACT"),
                                     chunk=chunk, document=docs[chunk.document_version_id], recipe=extraction_recipe,
                                     aliases=aliases, gateway=gateway,
                                     allow_repair=plan["allow_one_settled_format_repair"])
        canonical = {name: [record.model_dump(mode="json") for record in getattr(result.canonical, name)]
                     for name in ("entities", "mentions", "assertions")}
        canonical.update(chunk_id=chunk.chunk_id, review_entity_ids=result.canonical.review_entity_ids,
                         semantic_review="PENDING", offset_repairs=result.offset_repairs)
        canonical_rows.append(canonical)
        (artifact_directory / "canonical.jsonl").write_bytes(b"".join(canonical_json(r) + b"\n" for r in canonical_rows))
        report["extractions"].append({"chunk_id": chunk.chunk_id, "cache_hit": result.cache_hit,
                                      "operation_ids": [a.operation_id for a in result.record.attempts],
                                      "attempts": len(result.record.attempts), "offset_repairs": len(result.offset_repairs),
                                      "entities": len(result.canonical.entities), "assertions": len(result.canonical.assertions),
                                      "new_budget_cost_micro_usd": result.new_cost_micro_usd})
        checkpoint(report)
    report.update(status="STRUCTURAL_PASS_SEMANTIC_PENDING" if "EMBED" in stages else "EXTRACTION_STRUCTURAL_PASS_EMBEDDINGS_PENDING", chunks=len(selection),
                  completed_at=datetime.now(UTC).isoformat(),
                  new_budget_cost_micro_usd=sum(r["new_budget_cost_micro_usd"] for r in (
                      *report["embeddings"], *report["extractions"],
                  )),
                  artifacts={p.name: sha256_bytes(p.read_bytes()) for p in artifact_directory.iterdir() if p.is_file()})
    checkpoint(report)
    return report
