"""Synthetic workflow guards only; never reads actual experiment outputs or calls AWS."""

import copy
import json
from contextlib import nullcontext
from dataclasses import replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.context import pack_context
from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes, text_hash
from app.generation import (
    AnswerBinding,
    GenerationRecipeV9,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.graph_retrieval import GraphRetrievalResult
from app.model_gateway import CallContext
from app.multi_query import MultiQueryTraceV1, PlanningTraceV1, QueryRankingV1
from app.query_planning import PlannerRecipe
from app.rag_methods import MethodResultV1, MultiQueryMethodConfigV1
from app.snapshot import SnapshotContent
from scripts import compare_answer_limitations as workflow


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def cases_payload():
    return {"schema_version": 1, "purpose": "DEVELOPMENT_FROZEN_CONTEXT_NOT_GOLD_OR_HOLDOUT",
        "cases": [{"id": f"synthetic-{index}-{method}", "language": "vi" if index == 2 else "en",
            "question": f"Synthetic question {index}",
            "baseline_receipt": f"results/context-allocation/answer-pilot-v1/case-{index}.json",
            "baseline_receipt_sha256": "a" * 64, "method_index": method,
            "effective_method": ("hybrid_multi_query", "hybrid_multi_query_balanced")[method],
            "context_sources_sha256": "b" * 64, "context_chunk_ids_sha256": "c" * 64}
            for index in range(1, 4) for method in range(2)]}


def test_cases_accept_only_six_frozen_runtime_slots(tmp_path):
    data = cases_payload()
    path = tmp_path / "cases.json"
    write(path, data)
    parsed = workflow.load_cases(path)
    assert parsed.model_dump(mode="json") == data
    assert len({(case.baseline_receipt, case.method_index) for case in parsed.cases}) == 6


@pytest.mark.parametrize("location,key", [("root", "answers"), ("root", "quality_metrics"),
    ("case", "expected_facts"), ("case", "gold_answer"), ("case", "target_chunk_ids"),
    ("case", "review_focus"), ("case", "manual_prompt")])
def test_cases_reject_oracles_and_review_metadata(tmp_path, location, key):
    data = cases_payload()
    (data if location == "root" else data["cases"][0])[key] = "Do not leak review content"
    path = tmp_path / "cases.json"
    write(path, data)
    with pytest.raises(ValidationError):
        workflow.load_cases(path)


@pytest.mark.parametrize("mutation", ["few", "many", "duplicate_id", "duplicate_slot", "bad_method",
    "bad_language", "long_question", "empty_question", "blank_question", "bad_path", "path_traversal", "bad_hash",
    "bool_index", "index_out_of_range"])
def test_cases_reject_invalid_bounds_and_duplicate_slots(tmp_path, mutation):
    data = cases_payload()
    first = data["cases"][0]
    if mutation == "few":
        data["cases"].pop()
    elif mutation == "many":
        data["cases"].append(copy.deepcopy(first))
    elif mutation == "duplicate_id":
        data["cases"][1]["id"] = first["id"]
    elif mutation == "duplicate_slot":
        data["cases"][1].update(baseline_receipt=first["baseline_receipt"], method_index=0,
                                effective_method=first["effective_method"])
    elif mutation == "bad_method":
        first["effective_method"] = "hybrid_multi_query_balanced"
    elif mutation == "bad_language":
        first["language"] = "auto"
    elif mutation == "long_question":
        first["question"] = "x" * 2001
    elif mutation == "empty_question":
        first["question"] = ""
    elif mutation == "blank_question":
        first["question"] = " \t\n "
    elif mutation == "bad_path":
        first["baseline_receipt"] = "results/another-experiment/case-1.json"
    elif mutation == "path_traversal":
        first["baseline_receipt"] = "../results/context-allocation/answer-pilot-v1/case-1.json"
    elif mutation == "bad_hash":
        first["baseline_receipt_sha256"] = "not-a-sha256"
    elif mutation == "bool_index":
        first["method_index"] = False
    else:
        first["method_index"] = 2
    path = tmp_path / "cases.json"
    write(path, data)
    with pytest.raises((ValidationError, ValueError)):
        workflow.load_cases(path)


@pytest.mark.parametrize("marker", ["execution-started.json", "summary.json", "case-1.json"])
@pytest.mark.parametrize("flag", [[], ["--execute"], ["--preflight"]])
def test_existing_execution_blocks_before_model_cache_or_inputs(tmp_path, monkeypatch, marker, flag):
    directory = tmp_path / "immutable"
    directory.mkdir()
    original = b'{"synthetic":"must remain unchanged"}\n'
    (directory / marker).write_bytes(original)

    def forbidden(*args, **kwargs):
        pytest.fail("A started execution must fail before preparing inputs or a gateway")

    for name in ("load_cases", "AnswerCache", "load_collection", "managed_gateway", "preflight"):
        monkeypatch.setattr(workflow, name, forbidden)
    with pytest.raises(ValueError, match="EXECUTION_ALREADY_STARTED_USE_REPLAY"):
        workflow.main(["--directory", str(directory), *flag])
    assert (directory / marker).read_bytes() == original
    assert len(list(directory.iterdir())) == 1


def gateway_fixture(*, count=6, fail=False):
    calls = []

    def generate(context, **kwargs):
        calls.append((context, kwargs))
        if fail:
            raise TimeoutError("synthetic ambiguity")
        return "synthetic result"

    raw = SimpleNamespace(generate=generate, provider=SimpleNamespace(live=False), routes=object())
    digest = sha256_bytes(canonical_json(["Pinned system", "Pinned untrusted prompt"]))
    allowed = {f"candidate-{index}": digest for index in range(count)}
    return workflow.CandidateGateway(raw, allowed), calls


def context(request="candidate-0", **kwargs):
    return CallContext(tenant_id="demo", run_id=workflow.RUN, request_id=request, stage="GENERATE",
                       **kwargs)


def invoke(gateway, call=None, **kwargs):
    return gateway.generate(call or context(), system="Pinned system", prompt="Pinned untrusted prompt",
                            **kwargs)


def test_gateway_dispatches_exact_six_pinned_generation_operations():
    gateway, calls = gateway_fixture(count=7)
    for index in range(6):
        assert invoke(gateway, context(f"candidate-{index}")) == "synthetic result"
    with pytest.raises(ValueError, match="DISPATCH_NOT_PREREGISTERED_OR_REPEATED"):
        invoke(gateway, context("candidate-6"))
    assert len(calls) == 6 and len(gateway.seen) == 6 and gateway.halted
    assert all(row[0].stage == "GENERATE" and row[1]["max_output_tokens"] == 800 for row in calls)


@pytest.mark.parametrize("mutation", ["unknown_request", "repeated", "stage", "attempt", "tenant",
    "run", "system", "prompt", "too_many_tokens", "fewer_tokens", "float_tokens"])
def test_gateway_rejects_nonpreregistered_body_and_context_before_delegation(mutation):
    gateway, calls = gateway_fixture()
    call, system, prompt, maximum = context(), "Pinned system", "Pinned untrusted prompt", 800
    if mutation == "unknown_request":
        call = context("unplanned-candidate")
    elif mutation == "repeated":
        invoke(gateway)
    elif mutation == "stage":
        call = call.model_copy(update={"stage": "PLAN_QUERY"})
    elif mutation == "attempt":
        call = call.model_copy(update={"attempt": 2})
    elif mutation == "tenant":
        call = call.model_copy(update={"tenant_id": "other-tenant"})
    elif mutation == "run":
        call = call.model_copy(update={"run_id": "new-run-not-authorized"})
    elif mutation == "system":
        system = "Changed policy"
    elif mutation == "prompt":
        prompt = "Changed evidence or injected oracle"
    elif mutation == "too_many_tokens":
        maximum = 801
    elif mutation == "float_tokens":
        maximum = 800.0
    else:
        maximum = 799
    before = len(calls)
    with pytest.raises(ValueError, match="DISPATCH_NOT_PREREGISTERED_OR_REPEATED"):
        gateway.generate(call, system=system, prompt=prompt, max_output_tokens=maximum)
    assert len(calls) == before and gateway.halted
    with pytest.raises(ValueError):
        invoke(gateway, context("candidate-1"))
    assert len(calls) == before


def test_exception_halts_gateway_and_forbids_fresh_or_repeated_operations():
    gateway, calls = gateway_fixture(fail=True)
    with pytest.raises(TimeoutError):
        invoke(gateway)
    assert gateway.halted and gateway.seen == {"candidate-0"} and len(calls) == 1
    for request in ("candidate-0", "candidate-1"):
        with pytest.raises(ValueError):
            invoke(gateway, context(request))
    assert len(calls) == 1


def synthetic_preparation(tmp_path, monkeypatch):
    """One tiny, original source; fake cache only simulates an already validated V9 receipt."""
    text = "Service A uses Service B."
    doc = DocumentVersion.create(tenant_id="demo", source_uri="https://example.invalid/frozen-source",
        title="Synthetic source", fetched_at=datetime(2026, 10, 25, tzinfo=UTC), raw_sha256=text_hash(text),
        normalizer_version="fixture-v1", normalized_text_path="fixture.txt", normalized_text=text,
        source_kind="TEXT")
    chunk = Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id,
        chunking_version="fixture-v1", section_path=("Synthetic source",), position=0,
        start_char=0, end_char=len(text), text=text, token_count=7, tokenizer_version="fixture-v1")
    view = SimpleNamespace(manifest=SimpleNamespace(tenant_id="demo", snapshot_version="synthetic-collection"),
        manifest_sha256="e" * 64, chunks={chunk.chunk_id: chunk},
        documents={doc.document_version_id: doc}, entities={}, content=SnapshotContent((), ()),
        source_refs={chunk.chunk_id: SimpleNamespace(shard_id="synthetic-shard", manifest_sha256="d" * 64)})
    ids = (chunk.chunk_id,)
    packed = pack_context(view, GraphRetrievalResult(ids, ids, (), (), "hybrid_multi_query", False, (), 0, 0))
    recipe = GenerationRecipeV9(model_metadata_sha256="0" * 64)
    binding = AnswerBinding(request_id="old-validated-request", tenant_id="demo", owner_id=workflow.OWNER,
        mode_requested="hybrid_multi_query", mode_used="hybrid_multi_query", used_graph=False)
    parsed = parse_generated(json.dumps({"answer_status": "ANSWERED", "reason": "NONE", "claims": [
        {"text": text, "evidence": [{"source_id": "s1", "span_id": "s1p1"}]}]}), "end_turn", recipe)
    response = resolve_answer(parsed, snapshot=view, packed=packed, binding=binding, recipe=recipe).model_copy(
        update={"retrieval_backend": "collection_numpy_global_bm25"})
    planner = PlannerRecipe()
    config = MultiQueryMethodConfigV1(fusion="RRF60", document_recipe_sha256="a" * 64,
        query_recipe_sha256="b" * 64, generation_recipe_sha256=recipe.fingerprint,
        collection_manifest_sha256=view.manifest_sha256, lexical_index_version="synthetic-lexical",
        planner_recipe=planner, planner_recipe_sha256=planner.fingerprint)
    query = "How does Service A use Service B?"
    trace = MultiQueryTraceV1(base_search_query=query, queries=(QueryRankingV1(query=query, kind="BASE",
        ranked_chunk_ids=ids),), ranked_chunk_ids=ids, unique_union_count=1, truncated_count=0)
    planning = PlanningTraceV1(recipe_sha256=planner.fingerprint, subqueries=(), cache_hit=True,
        new_cost_micro_usd=0, receipt={"recipe_sha256": planner.fingerprint, "cost_micro_usd": 1})
    row = MethodResultV1(requested_method="hybrid_multi_query", effective_method="hybrid_multi_query",
        status="COMPLETED", ranked_chunk_ids=ids, context_chunk_ids=ids,
        context_proxy_tokens=packed.proxy_tokens, context_sources=tuple(workflow.source_records(view, packed)),
        response=response, config=config, config_sha256=config.fingerprint,
        collection_version="synthetic-collection", multi_query_trace=trace, planning_trace=planning).model_dump(mode="json")
    baseline = {"status": "COMPLETED", "cost_complete": True,
        "input": {"original_question": "How does Service A use Service B?"}, "answer_language": "en",
        "collection_manifest_sha256": view.manifest_sha256, "methods": [row]}
    case = cases_payload()["cases"][0] | {"question": baseline["input"]["original_question"],
        "context_sources_sha256": sha256_bytes(canonical_json(row["context_sources"])),
        "context_chunk_ids_sha256": sha256_bytes(canonical_json(row["context_chunk_ids"]))}
    path = tmp_path / case["baseline_receipt"]
    write(path, baseline)
    case["baseline_receipt_sha256"] = sha256_bytes(path.read_bytes())
    observed = []
    lookup = SimpleNamespace(response=response, cache_hit=True, new_cost_micro_usd=0, receipt=None)

    def resolve(*args, **kwargs):
        observed.append((args, kwargs))
        assert kwargs["gateway"] is None, "Preparing V9 must never create a paid model call"
        return lookup

    def evidence(collection, lexical, ranked):
        assert ranked == ids, "Only original ranked IDs may rebuild the source view"
        return view

    monkeypatch.setattr(workflow, "ROOT", tmp_path)
    monkeypatch.setattr(workflow, "_evidence_view", evidence)
    collection = SimpleNamespace(manifest_sha256=view.manifest_sha256)
    lexical = SimpleNamespace(manifest=SimpleNamespace(index_version="synthetic-lexical"))
    return SimpleNamespace(view=view, packed=packed, baseline=baseline, case=case, path=path, lookup=lookup,
        cache=SimpleNamespace(resolve=resolve), observed=observed, collection=collection, lexical=lexical,
        routes=SimpleNamespace(account_model_metadata_sha256="0" * 64))


def prepare(fixture):
    return workflow.prepare_pair(fixture.collection, fixture.lexical,
        workflow.FrozenCase.model_validate(fixture.case), fixture.routes, fixture.cache)


def test_preparation_is_cache_only_and_preserves_exact_untrusted_data(tmp_path, monkeypatch):
    fixture = synthetic_preparation(tmp_path, monkeypatch)
    before = fixture.path.read_bytes()
    view, new, lookup, entry = prepare(fixture)
    assert view is fixture.view and lookup is fixture.lookup and lookup.new_cost_micro_usd == 0
    assert new.context.request_id != fixture.lookup.response.request_id
    assert new.binding.request_id == new.context.request_id and new.context.stage == "GENERATE"
    assert new.binding.owner_id == workflow.OWNER and new.context.run_id == workflow.RUN
    assert new.packed.chunk_ids == fixture.packed.chunk_ids
    assert new.packed.serialized == fixture.packed.serialized and new.recipe.max_output_tokens == 800
    assert len(fixture.observed) == 1 and fixture.observed[0][1]["gateway"] is None
    old_recipe = fixture.observed[0][1]["recipe"]
    prompts = [generation_request(new.question, view, new.packed, "demo", recipe)
               for recipe in (old_recipe, new.recipe)]
    assert prompts[0][1] == prompts[1][1] and prompts[0][0] != prompts[1][0]
    assert entry["untrusted_payload_sha256"] == sha256_bytes(prompts[0][1].encode())
    assert entry["variants"][0]["request_id"] == fixture.lookup.response.request_id
    assert entry["variants"][1]["request_id"] == new.context.request_id
    assert fixture.path.read_bytes() == before


@pytest.mark.parametrize("mutation,category", [("hash", "BASELINE_HASH"),
    ("question", "BASELINE_INPUT_BINDING"), ("language", "BASELINE_INPUT_BINDING"),
    ("collection", "BASELINE_INPUT_BINDING"), ("status", "BASELINE_INPUT_BINDING"),
    ("cost", "BASELINE_INPUT_BINDING"), ("method", "METHOD_ID_MISMATCH"),
    ("sources", "BASELINE_CONTEXT_BINDING"), ("ids", "BASELINE_CONTEXT_BINDING"),
    ("recipe", "BASELINE_GENERATION_RECIPE"), ("proxy", "PACKED_RECONSTRUCTION"),
    ("source_text", "PACKED_RECONSTRUCTION")])
def test_preparation_rejects_changed_pins_and_source_reconstruction(tmp_path, monkeypatch, mutation, category):
    fixture = synthetic_preparation(tmp_path, monkeypatch)
    old = fixture.baseline
    row = old["methods"][0]
    if mutation == "hash":
        fixture.case["baseline_receipt_sha256"] = "f" * 64
    elif mutation == "question":
        old["input"]["original_question"] = "Changed question"
    elif mutation == "language":
        old["answer_language"] = "vi"
    elif mutation == "collection":
        old["collection_manifest_sha256"] = "f" * 64
    elif mutation == "status":
        old["status"] = "FAILED"
    elif mutation == "cost":
        old["cost_complete"] = False
    elif mutation == "method":
        row["effective_method"] = "hybrid_multi_query_balanced"
    elif mutation == "sources":
        fixture.case["context_sources_sha256"] = "f" * 64
    elif mutation == "ids":
        fixture.case["context_chunk_ids_sha256"] = "f" * 64
    elif mutation == "recipe":
        row["config"]["generation_recipe_sha256"] = "f" * 64
        row["config_sha256"] = sha256_bytes(canonical_json(row["config"]))
    elif mutation == "proxy":
        row["context_proxy_tokens"] += 1
    else:
        row["context_sources"][0]["text"] = "Changed synthetic text"
        fixture.case["context_sources_sha256"] = sha256_bytes(canonical_json(row["context_sources"]))
    if mutation != "hash":
        write(fixture.path, old)
        fixture.case["baseline_receipt_sha256"] = sha256_bytes(fixture.path.read_bytes())
    with pytest.raises(ValueError, match=category):
        prepare(fixture)
    assert not fixture.observed


@pytest.mark.parametrize("mutation", ["response", "not_cached"])
def test_baseline_must_be_an_exact_already_cached_response(tmp_path, monkeypatch, mutation):
    fixture = synthetic_preparation(tmp_path, monkeypatch)
    if mutation == "response":
        fixture.lookup.response = fixture.lookup.response.model_copy(update={"answer": "Changed answer"})
    else:
        fixture.lookup.cache_hit = False
    with pytest.raises(ValueError, match="V9_REPLAY_MISMATCH"):
        prepare(fixture)
    assert len(fixture.observed) == 1 and fixture.observed[0][1]["gateway"] is None


def test_old_cache_failure_is_never_retried_with_a_gateway(tmp_path, monkeypatch):
    fixture = synthetic_preparation(tmp_path, monkeypatch)
    calls = []

    def miss(*args, **kwargs):
        calls.append(kwargs["gateway"])
        raise ValueError("ANSWER_CACHE_MISS_NO_GATEWAY")

    fixture.cache.resolve = miss
    with pytest.raises(ValueError, match="ANSWER_CACHE_MISS_NO_GATEWAY"):
        prepare(fixture)
    assert calls == [None]


@pytest.mark.parametrize("key,value", [("max_context_chunks", 7), ("context_proxy_token_budget", 4095),
    ("max_output_tokens", 799), ("collection_manifest_sha256", "f" * 64),
    ("lexical_index_version", "different-lexical-version")])
def test_old_runtime_fields_cannot_silently_use_new_fixed_packer_defaults(tmp_path, monkeypatch, key, value):
    fixture = synthetic_preparation(tmp_path, monkeypatch)
    row = fixture.baseline["methods"][0]
    row["config"][key] = value
    row["config_sha256"] = sha256_bytes(canonical_json(row["config"]))
    write(fixture.path, fixture.baseline)
    fixture.case["baseline_receipt_sha256"] = sha256_bytes(fixture.path.read_bytes())
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not fixture.observed


@pytest.mark.parametrize("key,value", [("mode_requested", "hybrid"), ("mode_used", "hybrid"),
    ("used_graph", True)])
def test_original_response_mode_and_no_graph_binding_are_enforced(tmp_path, monkeypatch, key, value):
    fixture = synthetic_preparation(tmp_path, monkeypatch)
    fixture.baseline["methods"][0]["response"][key] = value
    write(fixture.path, fixture.baseline)
    fixture.case["baseline_receipt_sha256"] = sha256_bytes(fixture.path.read_bytes())
    with pytest.raises(ValueError):
        prepare(fixture)
    assert not fixture.observed


def main_fixture(tmp_path, monkeypatch, *, failure_index=None):
    """In-process orchestration: fake gateways, fake cache and six synthetic prepared cases."""
    fixture = synthetic_preparation(tmp_path, monkeypatch)
    view, template, old_lookup, _ = prepare(fixture)
    payload = cases_payload()
    for case in payload["cases"]:
        path = tmp_path / case["baseline_receipt"]
        if not path.exists():
            write(path, {"synthetic": case["baseline_receipt"]})
        case["baseline_receipt_sha256"] = sha256_bytes(path.read_bytes())
    cases_path = tmp_path / "cases.json"
    write(cases_path, payload)
    write(tmp_path / "results/corpus-expansion-v2/pipeline.json",
          {"collection_path": "synthetic-collection.json", "lexical_manifest_path": "synthetic-lexical.json"})
    for relative in ("configs/models/external-direct-v1.json", "configs/prices/external-direct-2026-10-24.json"):
        write(tmp_path / relative, {"synthetic": relative})
    for relative in workflow.IMPLEMENTATIONS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"Synthetic pin for {relative}\n", encoding="utf-8")
    routes = SimpleNamespace(account_model_metadata_sha256="0" * 64, generation_model_id="deepseek-flash",
        generation_invocation_id="deepseek-flash", source_region="external-default")
    rates, dispatches, cache_calls, preflight_calls = [], [], [], []
    provider = SimpleNamespace(live=False, dispatch_attempts=0)

    def generate(call, **kwargs):
        provider.dispatch_attempts += 1
        dispatches.append((call, kwargs))
        return object()

    gateway = SimpleNamespace(provider=provider, routes=routes, generate=generate)
    receipt = SimpleNamespace(model_dump=lambda **kwargs: {"synthetic_receipt": True, "cost_micro_usd": 7})

    def resolve(call, **kwargs):
        cache_calls.append((call.request_id, kwargs["gateway"]))
        if kwargs["gateway"] is not None:
            system, prompt = generation_request(kwargs["question"], kwargs["snapshot"], kwargs["packed"],
                "demo", kwargs["recipe"], answer_language=kwargs["answer_language"])
            kwargs["gateway"].generate(call, system=system, prompt=prompt, max_output_tokens=800)
            if failure_index == len(cache_calls):
                raise TimeoutError("Synthetic output ambiguity; do not retry")
        return SimpleNamespace(response=old_lookup.response.model_copy(update={"request_id": call.request_id}),
            receipt=receipt, cache_hit=kwargs["gateway"] is None,
            new_cost_micro_usd=0 if kwargs["gateway"] is None else 7)

    def prepared(collection, lexical, case, selected_routes, cache):
        request = f"synthetic-new-{case.id}"
        new = replace(template, question=case.question,
            context=template.context.model_copy(update={"request_id": request}),
            binding=template.binding.model_copy(update={"request_id": request}))
        entry = {"synthetic_case": case.id, "question": case.question,
                 "context_sha256": sha256_bytes(new.packed.serialized.encode())}
        return view, new, old_lookup, entry

    def preflight(selected_routes, path):
        preflight_calls.append(path)
        return gateway, {"status": "READY_READ_ONLY_NO_INFERENCE", "model_calls": 0}

    monkeypatch.setattr(workflow, "ExternalRoutes", SimpleNamespace(model_validate_json=lambda _: routes))
    monkeypatch.setattr(workflow, "ExternalPriceSheet", SimpleNamespace(model_validate_json=lambda _:
        SimpleNamespace(rate_for=lambda *args, **kwargs: rates.append((args, kwargs)))))
    monkeypatch.setattr(workflow, "load_collection", lambda *args, **kwargs: fixture.collection)
    monkeypatch.setattr(workflow, "load_global_lexical_index", lambda *args, **kwargs: nullcontext(fixture.lexical))
    monkeypatch.setattr(workflow, "AnswerCache", lambda *args, **kwargs: SimpleNamespace(resolve=resolve))
    monkeypatch.setattr(workflow, "prepare_pair", prepared)
    monkeypatch.setattr(workflow, "preflight", preflight)
    return SimpleNamespace(directory=tmp_path / "comparison", cases_path=cases_path, rates=rates,
        dispatches=dispatches, cache_calls=cache_calls, preflight_calls=preflight_calls,
        argv=["--cases", str(cases_path), "--directory", str(tmp_path / "comparison")])


def test_default_main_only_locks_plan_without_candidate_cache_or_model_access(tmp_path, monkeypatch):
    fixture = main_fixture(tmp_path, monkeypatch)
    assert workflow.main(fixture.argv) == 0
    plan_bytes = (fixture.directory / "plan.json").read_bytes()
    plan = json.loads(plan_bytes)
    assert plan["new_generation_call_cap"] == 6 and plan["max_output_tokens"] == 800
    assert plan["new_encoder_calls"] == plan["new_retrieval_calls"] == plan["new_planner_calls"] == 0
    assert not fixture.dispatches and not fixture.preflight_calls and not fixture.cache_calls
    assert not (fixture.directory / "execution-started.json").exists()
    assert workflow.main(fixture.argv) == 0
    assert (fixture.directory / "plan.json").read_bytes() == plan_bytes


def test_changed_implementation_pin_blocks_before_any_preflight(tmp_path, monkeypatch):
    fixture = main_fixture(tmp_path, monkeypatch)
    assert workflow.main(fixture.argv) == 0
    original = (fixture.directory / "plan.json").read_bytes()
    (tmp_path / workflow.IMPLEMENTATIONS[0]).write_text("Changed after registration\n", encoding="utf-8")
    with pytest.raises(ValueError, match="IMMUTABLE_PLAN_MISMATCH"):
        workflow.main([*fixture.argv, "--execute"])
    assert not fixture.preflight_calls and not fixture.dispatches
    assert (fixture.directory / "plan.json").read_bytes() == original


def test_preflight_only_has_no_candidate_dispatch_or_execution_marker(tmp_path, monkeypatch):
    fixture = main_fixture(tmp_path, monkeypatch)
    assert workflow.main([*fixture.argv, "--preflight"]) == 0
    assert len(fixture.preflight_calls) == 1 and not fixture.dispatches and not fixture.cache_calls
    assert len(list(fixture.directory.glob("preflight-*.json"))) == 1
    assert not (fixture.directory / "execution-started.json").exists()
    assert not (fixture.directory / "summary.json").exists()


def test_execute_and_replay_are_immutable_and_replay_has_no_models_or_aws(tmp_path, monkeypatch):
    fixture = main_fixture(tmp_path, monkeypatch)
    assert workflow.main([*fixture.argv, "--execute"]) == 0
    summary = json.loads((fixture.directory / "summary.json").read_bytes())
    assert summary["status"] == "COMPLETED" and summary["dispatches_this_invocation"] == 6
    assert summary["cases"] == 6 and summary["new_known_cost_micro_usd"] == 42  # Synthetic units only.
    assert len(fixture.dispatches) == 6 and len(fixture.preflight_calls) == 1
    immutable = {path: path.read_bytes() for path in fixture.directory.glob("*.json")}
    fixture.rates.clear()
    with pytest.raises(ValueError, match="EXECUTION_ALREADY_STARTED"):
        workflow.main([*fixture.argv, "--execute"])
    assert workflow.main([*fixture.argv, "--replay"]) == 0
    replay = json.loads((fixture.directory / "replay.json").read_bytes())
    assert replay["status"] == "COMPLETED" and replay["dispatches_this_invocation"] == 0
    assert replay["new_known_cost_micro_usd"] == 0 and replay["cases"] == 6
    assert not fixture.rates and len(fixture.preflight_calls) == 1 and len(fixture.dispatches) == 6
    assert all(path.read_bytes() == raw for path, raw in immutable.items())


def test_main_stops_after_first_uncertain_candidate_preserving_completed_cases(tmp_path, monkeypatch):
    fixture = main_fixture(tmp_path, monkeypatch, failure_index=2)
    assert workflow.main([*fixture.argv, "--execute"]) == 2
    summary = json.loads((fixture.directory / "summary.json").read_bytes())
    assert summary["status"] == "PARTIAL_STOPPED_NO_RETRY" and not summary["cost_complete"]
    assert summary["cases"] == 1 and summary["dispatches_this_invocation"] == 2
    assert len(fixture.cache_calls) == len(fixture.dispatches) == 2
    assert (fixture.directory / "case-1.json").exists() and not (fixture.directory / "case-2.json").exists()
    assert (fixture.directory / "execution-started.json").exists()
    with pytest.raises(ValueError, match="EXECUTION_ALREADY_STARTED"):
        workflow.main([*fixture.argv, "--execute"])
    assert len(fixture.dispatches) == 2
