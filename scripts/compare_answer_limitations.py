"""Frozen-context V9/V11 development comparison; six new generation calls at most."""

import argparse
import json
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from app.answer_cache import AnswerCache
from app.answer_pipeline import PreparedAnswer
from app.collection_answering import _evidence_view
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context import pack_context
from app.contracts import Record, Sha256, canonical_json, sha256_bytes, stable_id
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
from app.generation import (
    AnswerBinding,
    GenerationRecipeV9,
    GenerationRecipeV11,
    generation_request,
)
from app.graph_retrieval import GraphRetrievalResult
from app.model_gateway import CallContext, safe_failure_code
from app.rag_methods import MethodResultV1
from scripts.answer_local import ROOT, RUN, managed_gateway
from scripts.reconcile_balanced_query_pilot import validate_counter_baseline
from scripts.run_reranker_answer_pilot import write_once

VERSION = "frozen-context-answer-limitations-v11-development-v1"
OWNER = "local-ldh-owner"
IMPLEMENTATIONS = ("app/generation.py", "app/answer_assessment.py", "app/answer_cache.py",
                   "app/contracts.py", "scripts/compare_answer_limitations.py")


class FrozenCase(Record):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9-]{1,80}$")]
    language: Literal["en", "vi"]
    question: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
    baseline_receipt: Annotated[str, StringConstraints(
        pattern=r"^results/context-allocation/answer-pilot-v1/case-[123]\.json$")]
    baseline_receipt_sha256: Sha256
    method_index: Annotated[int, Field(ge=0, le=1)]
    effective_method: Literal["hybrid_multi_query", "hybrid_multi_query_balanced"]
    context_sources_sha256: Sha256
    context_chunk_ids_sha256: Sha256


class FrozenCases(Record):
    schema_version: Literal[1]
    purpose: Literal["DEVELOPMENT_FROZEN_CONTEXT_NOT_GOLD_OR_HOLDOUT"]
    cases: Annotated[tuple[FrozenCase, ...], Field(min_length=6, max_length=6)]


def load_cases(path):
    data = FrozenCases.model_validate_json(path.read_bytes())
    slots = {(row.baseline_receipt, row.method_index) for row in data.cases}
    required = {(f"results/context-allocation/answer-pilot-v1/case-{i}.json", j)
                for i in range(1, 4) for j in range(2)}
    if slots != required or len({row.id for row in data.cases}) != 6:
        raise ValueError("LIMITATIONS_CASE_SLOTS_OR_IDS")
    for row in data.cases:
        if row.effective_method != ("hybrid_multi_query", "hybrid_multi_query_balanced")[row.method_index]:
            raise ValueError("LIMITATIONS_CASE_METHOD_BINDING")
    return data


def source_records(view, packed):
    rows = []
    for source_id, key in zip(packed.source_ids, packed.chunk_ids, strict=True):
        chunk = view.chunks[key]
        doc, ref = view.documents[chunk.document_version_id], view.source_refs[key]
        if doc.normalized_text[chunk.start_char:chunk.end_char] != chunk.text:
            raise ValueError("LIMITATIONS_SOURCE_SLICE")
        rows.append({"source_id": source_id, "chunk_id": key, "document_version_id": chunk.document_version_id,
            "shard_id": ref.shard_id, "shard_manifest_sha256": ref.manifest_sha256,
            "source_uri": doc.source_uri, "title": doc.title, "section_path": list(chunk.section_path),
            "start_char": chunk.start_char, "end_char": chunk.end_char,
            "text_sha256": sha256_bytes(chunk.text.encode()), "text": chunk.text})
    return rows


def prepare_pair(collection, lexical, case, routes, cache):
    raw = (ROOT / case.baseline_receipt).read_bytes()
    if sha256_bytes(raw) != case.baseline_receipt_sha256:
        raise ValueError("LIMITATIONS_BASELINE_HASH")
    baseline = json.loads(raw)
    if (baseline["status"] != "COMPLETED" or not baseline["cost_complete"]
            or baseline["input"]["original_question"] != case.question
            or baseline["answer_language"] != case.language
            or baseline["collection_manifest_sha256"] != collection.manifest_sha256):
        raise ValueError("LIMITATIONS_BASELINE_INPUT_BINDING")
    row = baseline["methods"][case.method_index]
    MethodResultV1.model_validate_json(canonical_json(row))
    if (row["effective_method"] != case.effective_method or row["status"] != "COMPLETED"
            or sha256_bytes(canonical_json(row["context_sources"])) != case.context_sources_sha256
            or sha256_bytes(canonical_json(row["context_chunk_ids"])) != case.context_chunk_ids_sha256):
        raise ValueError("LIMITATIONS_BASELINE_CONTEXT_BINDING")
    if (row["config"]["collection_manifest_sha256"] != collection.manifest_sha256
            or row["config"]["lexical_index_version"] != lexical.manifest.index_version
            or any(row["config"][key] != value for key, value in {
                "max_context_chunks": 8, "context_proxy_token_budget": 4096, "max_output_tokens": 800}.items())
            or row["response"]["mode_requested"] != case.effective_method
            or row["response"]["mode_used"] != case.effective_method or row["response"]["used_graph"]):
        raise ValueError("LIMITATIONS_BASELINE_RUNTIME_CONFIGURATION")
    old_recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    new_recipe = GenerationRecipeV11(model_metadata_sha256=routes.account_model_metadata_sha256)
    if row["config"]["generation_recipe_sha256"] != old_recipe.fingerprint:
        raise ValueError("LIMITATIONS_BASELINE_GENERATION_RECIPE")
    ids = tuple(row["ranked_chunk_ids"])
    view = _evidence_view(collection, lexical, ids)
    packed = pack_context(view, GraphRetrievalResult(ids, ids, (), (), case.effective_method, False, (), 0, 0))
    if (list(packed.chunk_ids) != row["context_chunk_ids"] or packed.proxy_tokens != row["context_proxy_tokens"]
            or source_records(view, packed) != row["context_sources"]):
        raise ValueError("LIMITATIONS_PACKED_RECONSTRUCTION")
    binding = AnswerBinding(request_id=row["response"]["request_id"], tenant_id="demo", owner_id=OWNER,
        mode_requested=case.effective_method, mode_used=case.effective_method, used_graph=False)
    old = PreparedAnswer(CallContext(tenant_id="demo", run_id=RUN, request_id=binding.request_id,
        stage="GENERATE"), binding, packed, case.question, old_recipe, ids)
    old_lookup = cache.resolve(old.context, question=case.question, snapshot=view, packed=packed,
        binding=binding, recipe=old_recipe, answer_language=case.language, gateway=None)
    response = old_lookup.response.model_copy(update={"retrieval_backend": "collection_numpy_global_bm25"})
    if not old_lookup.cache_hit or response.model_dump(mode="json") != row["response"]:
        raise ValueError("LIMITATIONS_V9_REPLAY_MISMATCH")
    request_id = stable_id("cmp11", version=VERSION, case=case.id, baseline_request=binding.request_id,
        context_sha=sha256_bytes(packed.serialized.encode()), recipe=new_recipe.fingerprint,
        tenant="demo", owner=OWNER)
    new = replace(old, context=old.context.model_copy(update={"request_id": request_id}),
        binding=binding.model_copy(update={"request_id": request_id}), recipe=new_recipe)
    prompts = [generation_request(case.question, view, packed, "demo", recipe, answer_language=case.language)
               for recipe in (old_recipe, new_recipe)]
    if prompts[0][1] != prompts[1][1] or prompts[0][0] == prompts[1][0]:
        raise ValueError("LIMITATIONS_NOT_SAME_UNTRUSTED_EVIDENCE")
    entry = {"case": case.model_dump(mode="json"), "context_chunk_ids": list(packed.chunk_ids),
        "context_proxy_tokens": packed.proxy_tokens, "serialized_sha256": sha256_bytes(packed.serialized.encode()),
        "untrusted_payload_sha256": sha256_bytes(prompts[0][1].encode()),
        "variants": [{"name": name, "recipe": prepared.recipe.model_dump(mode="json"),
            "recipe_sha256": prepared.recipe.fingerprint, "request_id": prepared.context.request_id,
            "system_sha256": sha256_bytes(prompt[0].encode())}
            for name, prepared, prompt in zip(("v9", "v11"), (old, new), prompts, strict=True)]}
    return view, new, old_lookup, entry


class CandidateGateway:
    """Only six pinned GENERATE operations, each at most once; underlying ledger unchanged."""

    def __init__(self, gateway, allowed):
        self._gateway, self.allowed = gateway, allowed
        self.provider, self.routes = gateway.provider, gateway.routes
        self.seen, self.halted = set(), False

    def generate(self, context, *, system, prompt, max_output_tokens=800):
        digest = sha256_bytes(canonical_json([system, prompt]))
        key = context.request_id
        if (self.halted or context.stage != "GENERATE" or context.attempt != 1
                or context.run_id != RUN or context.tenant_id != "demo"
                or type(max_output_tokens) is not int or max_output_tokens != 800
                or key in self.seen or len(self.seen) >= 6 or self.allowed.get(key) != digest):
            self.halted = True
            raise ValueError("LIMITATIONS_DISPATCH_NOT_PREREGISTERED_OR_REPEATED")
        self.seen.add(key)
        try:
            return self._gateway.generate(context, system=system, prompt=prompt, max_output_tokens=800)
        except Exception:
            self.halted = True
            raise


def checkpoint_lookup(lookup, *, name, backend=True):
    response = lookup.response.model_copy(update={"retrieval_backend": "collection_numpy_global_bm25"}) if backend else lookup.response
    return {"variant": name, "response": response.model_dump(mode="json"), "cache_hit": lookup.cache_hit,
        "new_cost_micro_usd": lookup.new_cost_micro_usd,
        "receipt": lookup.receipt.model_dump(mode="json") if lookup.receipt else None}


def preflight(routes, price_path):
    now = datetime.now(UTC).isoformat()
    provider = deepseek_preflight(OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}))
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("LIMITATIONS_PROVIDER_NOT_READY")
    gateway = managed_gateway(prices_path=price_path)
    ledger = gateway.ledger
    counters = {"run": ledger.store.get(ledger.run_key(RUN)),
        "funding": ledger.store.get(ledger.external_funding_key("kg-r72")),
        "month": ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))}
    validate_counter_baseline(counters, now)
    if {key: row["reserved_micro_usd"] for key, row in counters.items()} != {"run": 640, "funding": 640, "month": 1250}:
        raise ValueError("LIMITATIONS_UNEXPECTED_LEGACY_RESERVATIONS")
    return gateway, {"created_at": now, "provider": provider, "counters": counters,
                     "status": "READY_READ_ONLY_NO_INFERENCE", "model_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true")
    mode.add_argument("--preflight", action="store_true")
    mode.add_argument("--replay", action="store_true")
    parser.add_argument("--cases", type=Path, default=ROOT / "configs/evaluation/answer-limitations-development-v1.json")
    parser.add_argument("--directory", type=Path, default=ROOT / "results/answer-limitations/comparison-v1")
    args = parser.parse_args(argv)
    directory = args.directory
    if not args.replay and ((directory / "execution-started.json").exists()
                           or (directory / "summary.json").exists() or any(directory.glob("case-*.json"))):
        raise ValueError("LIMITATIONS_EXECUTION_ALREADY_STARTED_USE_REPLAY")
    data = load_cases(args.cases)
    pipeline_path = ROOT / "results/corpus-expansion-v2/pipeline.json"
    paths = json.loads(pipeline_path.read_bytes())
    route_path, price_path = ROOT / "configs/models/external-direct-v1.json", ROOT / "configs/prices/external-direct-2026-10-24.json"
    routes = ExternalRoutes.model_validate_json(route_path.read_bytes())
    prices = ExternalPriceSheet.model_validate_json(price_path.read_bytes())
    if not args.replay:
        prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region,
                        now=datetime.now(UTC), live=True)
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    cache = AnswerCache(ROOT / "data/cache/collection-answers", namespace="LIVE")
    entries, ready, allowed = [], [], {}
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        for case in data.cases:
            view, new, old_lookup, entry = prepare_pair(collection, lexical, case, routes, cache)
            entries.append(entry)
            ready.append((case, view, new, old_lookup))
            allowed[new.context.request_id] = sha256_bytes(canonical_json(generation_request(
                case.question, view, new.packed, "demo", new.recipe, answer_language=case.language)))
    pins = {path: sha256_bytes((ROOT / path).read_bytes()) for path in IMPLEMENTATIONS}
    baselines = {case.baseline_receipt: case.baseline_receipt_sha256 for case in data.cases}
    plan = {"version": VERSION, "purpose": data.purpose, "cases": entries,
        "cases_sha256": sha256_bytes(args.cases.read_bytes()), "pipeline_sha256": sha256_bytes(pipeline_path.read_bytes()),
        "collection_manifest_sha256": collection.manifest_sha256, "routes_sha256": sha256_bytes(route_path.read_bytes()),
        "price_sheet_sha256": sha256_bytes(price_path.read_bytes()), "implementation_sha256": pins,
        "baseline_artifact_sha256": baselines, "new_generation_call_cap": 6, "max_output_tokens": 800,
        "run_id": RUN, "sequential_only": True, "same_spans_and_untrusted_payload": True,
        "new_retrieval_calls": 0, "new_encoder_calls": 0, "new_planner_calls": 0,
        "illustrative_forecast_micro_usd": 20160,
        "forecast_basis": "6x8000 input +800 output at peak cache miss; estimate, not invoice or aggregate reservation authorization",
        "quality_metrics": None, "human_reviewed": False, "gold": False, "holdout": False, "cloud_deployed": False}
    plan_path = directory / "plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_bytes()) != plan:
            raise ValueError("LIMITATIONS_IMMUTABLE_PLAN_MISMATCH")
    elif args.replay:
        raise ValueError("LIMITATIONS_REPLAY_REQUIRES_EXISTING_PLAN")
    else:
        write_once(plan_path, plan)
    if not any((args.execute, args.preflight, args.replay)):
        print(json.dumps({"status": "PLANNED_NO_MODEL_OR_AWS_CALLS", "cases": 6}))
        return 0
    gateway, start_calls = None, 0
    if not args.replay:
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        try:
            raw_gateway, report = preflight(routes, price_path)
        except Exception as exc:
            write_once(directory / f"preflight-{stamp}.json", {"status": "BLOCKED_BEFORE_EXECUTION",
                "error_category": safe_failure_code(exc), "model_calls": 0})
            print(json.dumps({"status": "BLOCKED_BEFORE_EXECUTION", "error_category": safe_failure_code(exc), "model_calls": 0}))
            return 2
        preflight_path = directory / f"preflight-{stamp}.json"
        write_once(preflight_path, report)
        if args.preflight:
            print(json.dumps({"status": report["status"], "model_calls": 0}))
            return 0
        write_once(directory / "execution-started.json", {"created_at": datetime.now(UTC).isoformat(),
            "plan_sha256": sha256_bytes(plan_path.read_bytes()), "preflight_path": str(preflight_path),
            "preflight_sha256": sha256_bytes(preflight_path.read_bytes()), "resume_allowed": False})
        gateway = CandidateGateway(raw_gateway, allowed)
        start_calls = gateway.provider.dispatch_attempts
    results, error = [], None
    try:
        for index, (case, view, new, old_lookup) in enumerate(ready, 1):
            if any(sha256_bytes((ROOT / path).read_bytes()) != digest for path, digest in baselines.items()):
                raise ValueError("LIMITATIONS_BASELINE_CHANGED_DURING_EXECUTION")
            lookup = cache.resolve(new.context, question=case.question, snapshot=view, packed=new.packed,
                binding=new.binding, recipe=new.recipe, answer_language=case.language, gateway=gateway)
            record = {"case_id": case.id, "baseline": checkpoint_lookup(old_lookup, name="v9"),
                "candidate": checkpoint_lookup(lookup, name="v11"), "context_sources": source_records(view, new.packed)}
            if args.replay:
                original = json.loads((directory / f"case-{index}.json").read_bytes())
                if (record["case_id"] != original["case_id"] or record["context_sources"] != original["context_sources"]
                        or any(record[key][field] != original[key][field] for key in ("baseline", "candidate")
                               for field in ("variant", "response", "receipt"))
                        or not lookup.cache_hit or lookup.new_cost_micro_usd != 0):
                    raise ValueError("LIMITATIONS_REPLAY_MISMATCH")
            else:
                write_once(directory / f"case-{index}.json", record)
            results.append(record)
            print(json.dumps({"case": index, "status": "REPLAYED" if args.replay else "COMPLETED",
                              "new_cost_micro_usd": lookup.new_cost_micro_usd}), flush=True)
    except Exception as exc:
        error = safe_failure_code(exc)
    report = {"status": "COMPLETED" if len(results) == 6 and error is None else "PARTIAL_STOPPED_NO_RETRY",
        "created_at": datetime.now(UTC).isoformat(), "plan_sha256": sha256_bytes(plan_path.read_bytes()),
        "error_category": error, "cases": len(results), "new_retrieval_calls": 0, "new_encoder_calls": 0,
        "new_planner_calls": 0, "dispatches_this_invocation": gateway.provider.dispatch_attempts - start_calls if gateway else 0,
        "new_known_cost_micro_usd": sum(row["candidate"]["new_cost_micro_usd"] for row in results),
        "cost_complete": error is None, "quality_metrics": None, "human_reviewed": False,
        "gold": False, "holdout": False, "cloud_deployed": False,
        "cost_basis": "PUBLISHED_PEAK_CACHE_MISS_UPPER_BOUND_NOT_INVOICE"}
    write_once(directory / ("replay.json" if args.replay else "summary.json"), report)
    print(json.dumps(report))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
