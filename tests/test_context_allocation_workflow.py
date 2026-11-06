import copy
import json
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import context_selection
from app.context_selection import ContextSelectionRecipe
from app.contracts import Chunk, DocumentVersion, canonical_json, sha256_bytes, text_hash
from app.multi_query import MultiQueryTraceV1, QueryRankingV1
from app.retrieval import reciprocal_rank_fusion
from app.snapshot import SnapshotContent
from scripts import compare_context_allocations as workflow


def controls_payload():
    return {"schema_version": 1, "version": "context-allocation-controls-v1",
        "purpose": "DEVELOPMENT_CONTEXT_ALLOCATION_INPUTS_NOT_GOLD_OR_HOLDOUT",
        "query_policy": "MANUAL_DEV_FIXED_NOT_LLM", "cases": [
            {"id": f"synthetic-{index}", "language": "vi" if index == 3 else "en",
             "question": f"Synthetic question {index}", "search_query": "Synthetic base query",
             "additional_search_queries": ["Synthetic condition query"]} for index in range(4)]}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json(value) + b"\n")


def test_read_controls_returns_only_frozen_runtime_inputs(tmp_path):
    data = controls_payload()
    path = tmp_path / "controls.json"
    write(path, data)
    assert workflow.read_controls(path) == data
    assert all(set(case) == {"id", "language", "question", "search_query", "additional_search_queries"}
               for case in data["cases"])


@pytest.mark.parametrize("location,key", [
    ("root", "answers"), ("root", "target_chunk_ids"), ("case", "answer"),
    ("case", "expected_facts"), ("case", "target_chunk_ids"), ("case", "reviewer_notes"),
])
def test_controls_reject_answer_target_and_review_extras(tmp_path, location, key):
    data = controls_payload()
    (data if location == "root" else data["cases"][0])[key] = ["Oracle content must not enter retrieval"]
    path = tmp_path / "controls.json"
    write(path, data)
    with pytest.raises(ValueError, match="ALLOCATION_CONTROL"):
        workflow.read_controls(path)


@pytest.mark.parametrize("mutation", [
    "too_few", "too_many", "duplicate_id", "blank_id", "wrong_language", "question_too_long",
    "base_too_long", "empty_question", "empty_base", "too_many_subqueries", "long_subquery",
    "empty_subquery", "duplicate_base", "duplicate_subqueries", "subqueries_not_list",
])
def test_controls_reject_duplicate_queries_and_bounds(tmp_path, mutation):
    data = controls_payload()
    case = data["cases"][0]
    if mutation == "too_few":
        data["cases"].pop()
    elif mutation == "too_many":
        data["cases"].append(copy.deepcopy(case) | {"id": "extra"})
    elif mutation == "duplicate_id":
        data["cases"][1]["id"] = case["id"]
    elif mutation == "blank_id":
        case["id"] = " \t"
    elif mutation == "wrong_language":
        case["language"] = "auto"
    elif mutation == "question_too_long":
        case["question"] = "x" * 2001
    elif mutation == "base_too_long":
        case["search_query"] = "x" * 2001
    elif mutation == "empty_question":
        case["question"] = " \t"
    elif mutation == "empty_base":
        case["search_query"] = " "
    elif mutation == "too_many_subqueries":
        case["additional_search_queries"] = ["First", "Second", "Third"]
    elif mutation == "long_subquery":
        case["additional_search_queries"] = ["x" * 241]
    elif mutation == "empty_subquery":
        case["additional_search_queries"] = [" \t"]
    elif mutation == "duplicate_base":
        case["additional_search_queries"] = ["  SYNTHETIC  base\tquery "]
    elif mutation == "duplicate_subqueries":
        case["additional_search_queries"] = ["Extra search", "EXTRA\t search "]
    else:
        case["additional_search_queries"] = "Not a list"
    path = tmp_path / "controls.json"
    write(path, data)
    with pytest.raises(ValueError, match="ALLOCATION_CONTROL"):
        workflow.read_controls(path)


def test_read_controls_accepts_exact_boundaries_and_zero_subqueries(tmp_path):
    data = controls_payload()
    data["cases"][0].update(question="x" * 2000, search_query="y" * 2000,
                            additional_search_queries=["a" * 240, "b" * 240])
    data["cases"][1]["additional_search_queries"] = []
    path = tmp_path / "controls.json"
    write(path, data)
    assert workflow.read_controls(path) == data


def test_existing_plan_blocks_before_inputs_models_collection_or_cache(tmp_path, monkeypatch):
    output = tmp_path / "existing"
    output.mkdir()
    original = b'{"already":"preregistered"}\n'
    (output / "plan.json").write_bytes(original)

    def forbidden(*args, **kwargs):
        pytest.fail("An existing immutable plan must stop before shared/model work.")

    for name in ("read_controls", "BgeEncoder", "EmbeddingCache", "load_collection", "load_global_lexical_index"):
        monkeypatch.setattr(workflow, name, forbidden)
    with pytest.raises(ValueError, match="ALREADY_PLANNED_NO_AUTOMATIC_RESUME"):
        workflow.main(["--directory", str(output)])
    assert (output / "plan.json").read_bytes() == original
    assert list(output.iterdir()) == [output / "plan.json"]


def synthetic_view_and_trace():
    from datetime import UTC, datetime

    texts = ["A shared synthetic condition.", "A shared synthetic condition.",
             *[f"Synthetic restriction number {index}." for index in range(10)]]
    normalized = "\n".join(texts)
    doc = DocumentVersion.create(tenant_id="demo", source_uri="https://example.invalid/context-fixture",
        title="Synthetic selection source", fetched_at=datetime(2026, 10, 25, tzinfo=UTC),
        raw_sha256=text_hash(normalized), normalizer_version="fixture-v1", normalized_text_path="fixture.txt",
        normalized_text=normalized, source_kind="TEXT")
    chunks, offset = [], 0
    for index, text in enumerate(texts):
        chunks.append(Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id,
            chunking_version="fixture-v1", section_path=("Synthetic conditions",), position=index,
            start_char=offset, end_char=offset + len(text), text=text,
            token_count=len(text.split()), tokenizer_version="fixture-words-v1"))
        offset += len(text) + 1
    ids = tuple(row.chunk_id for row in chunks)
    rankings = (QueryRankingV1(query="Synthetic base query", kind="BASE", ranked_chunk_ids=ids[:8]),
                QueryRankingV1(query="Synthetic condition query", kind="SUBQUERY", ranked_chunk_ids=ids[1:]))
    fused = tuple(reciprocal_rank_fusion([row.ranked_chunk_ids for row in rankings]))
    trace = MultiQueryTraceV1(base_search_query=rankings[0].query, queries=rankings,
        ranked_chunk_ids=fused, unique_union_count=len(fused), truncated_count=0)
    view = SimpleNamespace(chunks={row.chunk_id: row for row in chunks}, documents={doc.document_version_id: doc},
        content=SnapshotContent((), ()), entities={}, source_refs={key: SimpleNamespace(
            shard_id="synthetic-shard", manifest_sha256="a" * 64) for key in ids})
    return view, trace


def test_fixture_comparison_preserves_sources_and_reports_exposure_not_fact_coverage():
    view, trace = synthetic_view_and_trace()
    original_ids = trace.ranked_chunk_ids
    result = workflow.compare(view, trace)
    assert set(result) == {"Q_RRF", "RR_ID_ONLY", "RR_SCOPED_EXACT_DEDUP"}
    assert trace.ranked_chunk_ids == original_ids
    baseline = result["Q_RRF"]["context"]
    for policy in workflow.POLICIES:
        row = result[policy]
        selected = row["selection_trace"]
        assert selected["original_pool_ids"] == list(original_ids)
        assert set(selected["round_robin_chunk_ids"]) == set(original_ids)
        assert selected["recipe_sha256"] == ContextSelectionRecipe(policy=policy).fingerprint
        assert len(row["selector_observations_ms"]) == 3
        assert row["context"]["query_exposure_is_not_fact_coverage"] is True
        assert len(row["context"]["chunk_ids"]) <= 8 and row["context"]["proxy_tokens"] <= 4096
        assert row["added_ids"] == [key for key in row["context"]["chunk_ids"] if key not in baseline["chunk_ids"]]
        assert row["removed_ids"] == [key for key in baseline["chunk_ids"] if key not in row["context"]["chunk_ids"]]
        for source in row["context"]["sources"]:
            doc = view.documents[source["document_version_id"]]
            assert doc.normalized_text[source["start_char"]:source["end_char"]] == source["text"]
            assert source["text_sha256"] == text_hash(source["text"])
    assert result["RR_ID_ONLY"]["selection_trace"]["duplicates"] == []
    assert len(result["RR_SCOPED_EXACT_DEDUP"]["selection_trace"]["duplicates"]) == 1


def test_fixture_comparison_checks_repeat_determinism(monkeypatch):
    view, trace = synthetic_view_and_trace()
    original, count = workflow.select_context_order, 0

    def unstable(*args, **kwargs):
        nonlocal count
        count += 1
        result = original(*args, **kwargs)
        if count == 2:
            return result.model_copy(update={"ordered_chunk_ids": result.ordered_chunk_ids[::-1]})
        return result

    monkeypatch.setattr(workflow, "select_context_order", unstable)
    with pytest.raises(ValueError, match="NONDETERMINISTIC_TRACE"):
        workflow.compare(view, trace)


@pytest.fixture
def saved_experiment(tmp_path, monkeypatch):
    root, directory = tmp_path / "root", tmp_path / "experiment"
    source_bytes = Path(context_selection.__file__).read_bytes()
    (root / "app").mkdir(parents=True)
    (root / "app/context_selection.py").write_bytes(source_bytes)
    controls_path = root / "configs/evaluation/context-allocation-controls-v1.json"
    write(controls_path, controls_payload())
    view, trace = synthetic_view_and_trace()
    baseline = workflow.describe(view, trace, trace.ranked_chunk_ids)
    for index in range(1, 4):
        write(root / f"results/multi-query/pilot-v1/case-{index}.json", {
            "status": "COMPLETED", "pilot_case_id": f"synthetic-frozen-{index}",
            "input": {"original_question": f"Synthetic frozen question {index}"}, "answer_language": "en",
            "methods": [{"effective_method": "hybrid"}, {"effective_method": "hybrid_multi_query",
                "multi_query_trace": trace.model_dump(mode="json"), "context_chunk_ids": baseline["chunk_ids"]}]})
    write(root / "results/corpus-expansion-v2/pipeline.json", {
        "collection_path": str(root / "fixture-collection.json"),
        "lexical_manifest_path": str(root / "fixture-lexical.json")})
    collection = SimpleNamespace(manifest_sha256="c" * 64)
    encoder = SimpleNamespace(inference_calls=0, recipe=lambda role: SimpleNamespace(fingerprint="b" * 64))
    model_loads = []

    def fake_encoder(*args, **kwargs):
        model_loads.append(True)
        return encoder

    class FakeCache:
        def __init__(self, *args, **kwargs):
            pass

        def resolve_local(self, *args, **kwargs):
            return SimpleNamespace(record=SimpleNamespace(vector=[1.0, 0.0]))

    monkeypatch.setattr(workflow, "ROOT", root)
    monkeypatch.setattr(workflow, "CONTROLS", controls_path)
    monkeypatch.setattr(workflow, "BgeEncoder", fake_encoder)
    monkeypatch.setattr(workflow, "EmbeddingCache", FakeCache)
    monkeypatch.setattr(workflow, "load_collection", lambda *args, **kwargs: collection)
    monkeypatch.setattr(workflow, "load_global_lexical_index", lambda *args, **kwargs: nullcontext(object()))
    monkeypatch.setattr(workflow, "_evidence_view", lambda *args, **kwargs: view)
    monkeypatch.setattr(workflow, "retrieve_multi_query", lambda *args, **kwargs: trace)
    assert workflow.main(["--directory", str(directory)]) == 0
    result = json.loads((directory / "experiment.json").read_bytes())
    return root, directory, result, model_loads


def test_fixture_saved_verification_uses_no_encoder_and_preserves_receipts(saved_experiment):
    _, directory, result, model_loads = saved_experiment
    before = (directory / "experiment.json").read_bytes()
    assert workflow.verify(directory) == 0
    assert (directory / "experiment.json").read_bytes() == before
    assert len(model_loads) == 1
    report = json.loads((directory / "verification.json").read_bytes())
    assert report["status"] == "PASS_SAVED_TRACE_SOURCE_CONTEXT_RECONSTRUCTION"
    assert report["cases"] == 7 and report["variants_per_case"] == 3
    assert report["encoder_calls"] == report["model_calls"] == report["aws_calls"] == report["paid_calls"] == 0
    assert result["quality_metrics"] is None and not result["human_reviewed"] and not result["cloud_deployed"]


@pytest.mark.parametrize("mutation", ["trace", "trace_pool", "recipe", "order", "source", "plan_checksum"])
def test_saved_verification_rejects_changed_trace_recipe_source_or_binding(saved_experiment, mutation):
    _, directory, result, _ = saved_experiment
    case = result["cases"][0]
    if mutation == "trace":
        case["multi_query_trace"]["queries"][0]["query"] = "Changed original anchor"
    elif mutation == "trace_pool":
        case["multi_query_trace"]["ranked_chunk_ids"].reverse()
    elif mutation == "recipe":
        case["variants"]["RR_ID_ONLY"]["selection_trace"]["recipe"]["allocation"] = "FILL_BY_RELEVANCE_LABEL"
    elif mutation == "order":
        case["variants"]["RR_ID_ONLY"]["ordered_chunk_ids"].reverse()
    elif mutation == "source":
        case["variants"]["RR_ID_ONLY"]["context"]["sources"][0]["text"] = "Not an original source slice"
    else:
        result["plan_sha256"] = "0" * 64
    write(directory / "experiment.json", result)
    original = (directory / "experiment.json").read_bytes()
    with pytest.raises(ValueError):
        workflow.verify(directory)
    assert (directory / "experiment.json").read_bytes() == original
    assert not (directory / "verification.json").exists()


def test_saved_verification_rejects_changed_selector_implementation(saved_experiment):
    root, directory, _, _ = saved_experiment
    with (root / "app/context_selection.py").open("ab") as stream:
        stream.write(b"\n# Changed after preregistration\n")
    with pytest.raises(ValueError, match="IMPLEMENTATION_CHANGED"):
        workflow.verify(directory)


def test_saved_verification_rejects_changed_plan_recipe_even_with_updated_plan_checksum(saved_experiment):
    _, directory, result, _ = saved_experiment
    plan_path = directory / "plan.json"
    plan = json.loads(plan_path.read_bytes())
    plan["policies"][0]["allocation"] = "UNREGISTERED_POLICY"
    write(plan_path, plan)
    result["plan_sha256"] = sha256_bytes(plan_path.read_bytes())
    write(directory / "experiment.json", result)
    with pytest.raises(ValueError, match="ALLOCATION_"):
        workflow.verify(directory)


def test_saved_verification_rejects_changed_frozen_input_file(saved_experiment):
    root, directory, _, _ = saved_experiment
    path = root / "results/multi-query/pilot-v1/case-1.json"
    original = json.loads(path.read_bytes())
    original["input"]["original_question"] = "Changed after preregistration"
    write(path, original)
    with pytest.raises(ValueError, match="ALLOCATION_"):
        workflow.verify(directory)


def test_saved_verification_rejects_changed_controls_file(saved_experiment):
    root, directory, _, _ = saved_experiment
    path = root / "configs/evaluation/context-allocation-controls-v1.json"
    controls = json.loads(path.read_bytes())
    controls["cases"][0]["question"] = "A different task after preregistration"
    write(path, controls)
    with pytest.raises(ValueError, match="ALLOCATION_"):
        workflow.verify(directory)


def test_saved_verification_rejects_internally_consistent_but_unregistered_frozen_trace(saved_experiment):
    _, directory, result, _ = saved_experiment
    view, original = synthetic_view_and_trace()
    changed = original.model_dump(mode="json")
    changed["base_search_query"] = changed["queries"][0]["query"] = "Changed valid original search task"
    trace = MultiQueryTraceV1.model_validate_json(canonical_json(changed))
    result["cases"][0]["multi_query_trace"] = trace.model_dump(mode="json")
    result["cases"][0]["variants"] = workflow.compare(view, trace)
    write(directory / "experiment.json", result)
    with pytest.raises(ValueError, match="ALLOCATION_"):
        workflow.verify(directory)


@pytest.mark.parametrize("mutation", ["missing_case", "duplicate_case", "case_id", "phase"])
def test_saved_verification_requires_exact_registered_case_set(saved_experiment, mutation):
    _, directory, result, _ = saved_experiment
    if mutation == "missing_case":
        result["cases"].pop()
    elif mutation == "duplicate_case":
        result["cases"].append(copy.deepcopy(result["cases"][0]))
    elif mutation == "case_id":
        result["cases"][0]["id"] = "unregistered-case"
    else:
        result["cases"][0]["phase"] = "HOLDOUT_HUMAN_REVIEWED"
    write(directory / "experiment.json", result)
    with pytest.raises(ValueError, match="ALLOCATION_"):
        workflow.verify(directory)


@pytest.mark.parametrize("mutation", ["base_query", "subquery", "question", "language"])
def test_saved_verification_binds_manual_cases_to_every_registered_runtime_input(saved_experiment, mutation):
    _, directory, result, _ = saved_experiment
    case = result["cases"][3]
    if mutation in {"base_query", "subquery"}:
        changed = copy.deepcopy(case["multi_query_trace"])
        index = 0 if mutation == "base_query" else 1
        changed["queries"][index]["query"] = "An unregistered but valid search angle"
        if index == 0:
            changed["base_search_query"] = changed["queries"][0]["query"]
        trace = MultiQueryTraceV1.model_validate_json(canonical_json(changed))
        view, _ = synthetic_view_and_trace()
        case["multi_query_trace"] = trace.model_dump(mode="json")
        case["variants"] = workflow.compare(view, trace)
    elif mutation == "question":
        case["question"] = "An unregistered task"
    else:
        case["language"] = "vi"
    write(directory / "experiment.json", result)
    with pytest.raises(ValueError, match="ALLOCATION_"):
        workflow.verify(directory)
