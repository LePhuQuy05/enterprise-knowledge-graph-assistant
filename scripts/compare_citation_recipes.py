"""Fixed-context development comparison; no retrieval/rewrite/holdout quality claims."""

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
from pydantic import Field, StringConstraints

from app.answer_cache import AnswerCache
from app.collection_answering import prepare_collection_answer
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import Record, canonical_json, sha256_bytes, stable_id
from app.embeddings import EmbeddingCache, query_embedding_input
from app.external_gateway import ExternalRoutes
from app.generation import GenerationRecipeV9, GenerationRecipeV10, generation_request
from app.local_bge import BgeEncoder
from app.model_gateway import safe_failure_code
from scripts.answer_local import ROOT, RUN, managed_gateway

VERSION = "citation-completeness-v10-development-v1"
OWNER = "local-ldh-owner"
Text = Annotated[str, StringConstraints(min_length=1, max_length=2000)]


class ComparisonCase(Record):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9-]{1,60}$")]
    language: Literal["en", "vi"]
    question: Text
    search_query: Text
    mode: Literal["hybrid", "bm25"] = "hybrid"
    baseline_receipt: str | None = None
    review_focus: tuple[str, ...] = ()


class ComparisonCases(Record):
    schema_version: Literal[1] = 1
    purpose: Literal["DEVELOPMENT_FIXED_CONTEXT_NOT_GOLD_OR_HOLDOUT"]
    cases: Annotated[tuple[ComparisonCase, ...], Field(min_length=1, max_length=6)]


def prepare_pair(collection, lexical, case, query_vector, *, routes, baseline=None):
    """One retrieval/packing result for both recipes; metadata never enters the prompt."""
    recipes = (GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256),
               GenerationRecipeV10(model_metadata_sha256=routes.account_model_metadata_sha256))
    identity = {"version": VERSION, "case": case.id, "question": case.question,
                "search_query": case.search_query, "language": case.language,
                "collection_sha": collection.manifest_sha256, "mode": case.mode}
    baseline_id = stable_id("cmp9", **identity, recipe=recipes[0].fingerprint)
    if baseline is not None:
        if (baseline["status"] != "COMPLETED" or baseline["input"]["original_question"] != case.question
                or baseline["answer_language"] != case.language or baseline["search_query"] != case.search_query
                or baseline["collection_manifest_sha256"] != collection.manifest_sha256):
            raise ValueError("COMPARISON_BASELINE_BINDING_MISMATCH")
        selected = next(row for row in baseline["methods"] if row["effective_method"] == case.mode)
        if selected["config"]["generation_recipe_sha256"] != recipes[0].fingerprint:
            raise ValueError("COMPARISON_BASELINE_RECIPE_MISMATCH")
        baseline_id = selected["response"]["request_id"]
    view, prepared = prepare_collection_answer(collection, lexical, case.question, query_vector,
        tenant_id=collection.manifest.tenant_id, owner_id=OWNER, run_id=RUN, recipe=recipes[0],
        mode=case.mode, request_id=baseline_id, search_query=case.search_query)
    if baseline is not None and list(prepared.packed.chunk_ids) != selected["context_chunk_ids"]:
        raise ValueError("COMPARISON_BASELINE_CONTEXT_MISMATCH")
    candidate_id = stable_id("cmp10", **identity, recipe=recipes[1].fingerprint)
    candidate = replace(prepared, context=prepared.context.model_copy(update={"request_id": candidate_id}),
                        binding=prepared.binding.model_copy(update={"request_id": candidate_id}), recipe=recipes[1])
    prompts = [generation_request(case.question, view, item.packed, collection.manifest.tenant_id,
                                 item.recipe, answer_language=case.language) for item in (prepared, candidate)]
    if prompts[0][1] != prompts[1][1] or prompts[0][0] == prompts[1][0]:
        raise ValueError("COMPARISON_NOT_SAME_PAYLOAD_DIFFERENT_POLICY")
    return view, (prepared, candidate), {
        "case": case.model_dump(mode="json"), "original_baseline_reused": baseline is not None,
        "collection_manifest_sha256": collection.manifest_sha256,
        "context_chunk_ids": list(prepared.packed.chunk_ids), "context_proxy_tokens": prepared.packed.proxy_tokens,
        "untrusted_payload_sha256": sha256_bytes(prompts[0][1].encode()),
        "variants": [{"name": name, "recipe": item.recipe.model_dump(mode="json"),
            "recipe_sha256": item.recipe.fingerprint, "request_id": item.context.request_id,
            "system_sha256": sha256_bytes(prompt[0].encode())}
            for name, item, prompt in zip(("v9", "v10"), (prepared, candidate), prompts, strict=True)],
    }


def execute_pair(view, pair, *, cache, language, gateway=None, expected_baseline=None, checkpoint=None):
    rows = []
    for name, prepared in zip(("v9", "v10"), pair, strict=True):
        try:
            lookup = cache.resolve(prepared.context, question=prepared.question, snapshot=view,
                packed=prepared.packed, binding=prepared.binding, recipe=prepared.recipe,
                answer_language=language, gateway=None)  # Prefer existing receipts even in execute mode.
        except Exception as exc:
            if str(exc) != "ANSWER_CACHE_MISS_NO_GATEWAY" or gateway is None:
                raise
            lookup = cache.resolve(prepared.context, question=prepared.question, snapshot=view,
                packed=prepared.packed, binding=prepared.binding, recipe=prepared.recipe,
                answer_language=language, gateway=gateway)
        response = lookup.response.model_copy(update={"retrieval_backend": "collection_numpy_global_bm25"})
        if name == "v9" and expected_baseline is not None and response.model_dump(mode="json") != expected_baseline:
            raise ValueError("COMPARISON_REUSED_BASELINE_RESPONSE_CHANGED")
        rows.append({"variant": name, "status": "STRUCTURAL_PASS_SEMANTIC_PENDING",
            "recipe_sha256": prepared.recipe.fingerprint, "cache_hit": lookup.cache_hit,
            "response": response.model_dump(mode="json"), "new_cost_micro_usd": lookup.new_cost_micro_usd,
            "receipt": lookup.receipt.model_dump(mode="json") if lookup.receipt else None,
            "deterministic_no_evidence": lookup.deterministic_no_evidence})
        if checkpoint:
            checkpoint(rows[-1])
    return rows


def pin_comparison_plan(target, plan):
    """Preserve the first plan and verify its stored body, not just its claimed digest."""
    identity = {key: value for key, value in plan.items() if key not in {"created_at", "plan_sha256"}}
    digest = sha256_bytes(canonical_json(identity))
    if target.exists():
        old = json.loads(target.read_bytes())
        old_identity = {key: value for key, value in old.items() if key not in {"created_at", "plan_sha256"}}
        if (sha256_bytes(canonical_json(old_identity)) != old["plan_sha256"]
                or old["plan_sha256"] != digest):
            raise ValueError("COMPARISON_PLAN_IMMUTABLE")
        return old
    pinned = {**plan, "plan_sha256": digest}
    target.write_bytes(canonical_json(pinned) + b"\n")
    return pinned


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results/citation-v10")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.execute and (args.output_dir / "live.json").exists():
        raise ValueError("COMPARISON_LIVE_RECEIPT_IMMUTABLE_USE_OFFLINE_REPLAY")
    cases = ComparisonCases.model_validate_json(args.cases.read_bytes())
    if len({case.id for case in cases.cases}) != len(cases.cases):
        raise ValueError("COMPARISON_DUPLICATE_CASE_IDS")
    paths = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id=encoder.recipe("document").model_id, cache_shards=4)
    if collection.manifest.embedding_recipe != encoder.recipe("document"):
        raise ValueError("COMPARISON_ENCODER_MISMATCH")
    embedding_cache = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE")
    cache = AnswerCache(ROOT / "data/cache/collection-answers", namespace="LIVE")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    ready, entries = [], []
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        for case in cases.cases:
            baseline = json.loads((ROOT / case.baseline_receipt).read_bytes()) if case.baseline_receipt else None
            vector = None
            if case.mode != "bm25":
                vector = embedding_cache.resolve_local("demo", text=query_embedding_input(case.search_query),
                    recipe=encoder.recipe("query"), encoder=encoder).record.vector
                vector = np.asarray(vector, dtype=np.float32)
            view, pair, entry = prepare_pair(collection, lexical, case, vector, routes=routes, baseline=baseline)
            ready.append((case, view, pair, baseline))
            entries.append(entry)
    plan = {"version": VERSION, "purpose": cases.purpose, "created_at": datetime.now(UTC).isoformat(),
            "collection_version": collection.manifest.collection_version, "cases": entries,
            "query_policy": "FIXED_ENGLISH_DEVELOPMENT_SEARCH_NOT_END_TO_END_VI_REWRITE",
            "max_new_generation_calls": sum(0 if not pair[0].packed.chunk_ids else 1 if old else 2
                                            for _, _, pair, old in ready),
            "quality_metrics": None, "human_reviewed": False}
    # Keep a stable plan independent of its first planning timestamp.
    target = args.output_dir / "plan.json"
    plan = pin_comparison_plan(target, plan)
    gateway = managed_gateway() if args.execute else None
    before_dispatch = gateway.provider.dispatch_attempts if gateway else 0
    results = []
    for case, view, pair, baseline in ready:
        expected = next(row["response"] for row in baseline["methods"] if row["effective_method"] == case.mode) if baseline else None
        completed_variants = []
        try:
            rows = execute_pair(view, pair, cache=cache, language=case.language, gateway=gateway,
                                expected_baseline=expected, checkpoint=completed_variants.append)
            results.append({"case_id": case.id, "status": "PAIRED_STRUCTURAL_PASS", "variants": rows})
        except Exception as exc:
            results.append({"case_id": case.id, "status": "BLOCKED_OR_FAILED", "variants": completed_variants,
                            "failed_variant_cost_micro_usd": None, "error_category": safe_failure_code(exc)})
            break  # Never continue paid dispatches after any uncertain/failed stage.
        report = {"version": VERSION, "plan_sha256": plan["plan_sha256"], "results": results,
                  "dispatches_this_invocation": gateway.provider.dispatch_attempts - before_dispatch if gateway else 0,
                  "new_known_cost_micro_usd": sum(row["new_cost_micro_usd"] for result in results for row in result.get("variants", [])),
                  "semantic_review": "NOT_HUMAN_REVIEWED", "quality_metrics": None, "cloud_deployed": False}
        (args.output_dir / ("live.json" if args.execute else "replay.json")).write_bytes(canonical_json(report) + b"\n")
    report = {"version": VERSION, "plan_sha256": plan["plan_sha256"], "results": results,
              "dispatches_this_invocation": gateway.provider.dispatch_attempts - before_dispatch if gateway else 0,
              "new_known_cost_micro_usd": sum(row["new_cost_micro_usd"] for result in results for row in result.get("variants", [])),
              "cost_complete": all(result["status"] == "PAIRED_STRUCTURAL_PASS" for result in results),
              "semantic_review": "NOT_HUMAN_REVIEWED", "quality_metrics": None, "cloud_deployed": False}
    (args.output_dir / ("live.json" if args.execute else "replay.json")).write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({"paired_cases": len(results), "all_cases": len(ready),
        "dispatches": report["dispatches_this_invocation"], "known_cost_micro_usd": report["new_known_cost_micro_usd"],
        "cost_complete": report["cost_complete"], "plan": str(target)}))
    return 0 if report["cost_complete"] and len(results) == len(ready) else 2


if __name__ == "__main__":
    raise SystemExit(main())
