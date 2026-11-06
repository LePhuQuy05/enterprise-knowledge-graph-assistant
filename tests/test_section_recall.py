"""Source/namespace/scope/budget controls for the local section candidate."""

import json
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app.contracts import canonical_json, sha256_bytes
from app.faceted_query_v5 import FacetPlanV5, FacetRecipeV5, validate_facets
from app.fixtures import fixture_vector
from app.generation import GenerationError, GenerationRecipeV9
from app.graph_collection_migration import collection_records
from app.section_recall import (
    SectionPool,
    SectionRecipe,
    _diverse_order,
    build_section_pool,
    literal_probes,
    make_index,
    metadata_rows,
    pack_section_pool,
)
from tests.test_authorization_evidence import auth_case as auth_case
from tests.test_document_recall_v2 import multi_chunk_case as multi_chunk_case

QUESTION = "Compare S3 invocation and Lambda execution role across accounts."


def plan(collection, question=QUESTION):
    raw = {"facet_1": {"focus_quote": "S3 invocation", "search_query": "S3 invocation permissions for reading objects"},
        "facet_2": {"focus_quote": "Lambda execution role", "search_query": "Lambda execution role key policy"}, "facet_3": None}
    return FacetPlanV5(origin="FIXTURE", collection_manifest_sha256=collection.manifest_sha256,
        question_sha256=sha256_bytes(question.encode()), search_query_sha256=sha256_bytes(question.encode()), language="en",
        facets=validate_facets(json.dumps(raw), question, question), recipe_sha256=FacetRecipeV5().fingerprint,
        operation_id="fixture", receipt_sha256="0"*64)


def case_inputs(case):
    docs, chunks, _ = collection_records(case.collection)
    rows = metadata_rows(docs, chunks, "demo")
    vectors = np.asarray([fixture_vector(r["metadata_text"]) for r in rows], dtype=np.float32)
    index = make_index(case.collection, rows, vectors)
    p = plan(case.collection)
    queries = literal_probes(case.collection, QUESTION, QUESTION, "en", p)
    qvectors = tuple(fixture_vector(q) for q in queries)
    base = tuple(chunks)[:40]
    return index, p, qvectors, base


def test_task_probes_keep_original_and_copy_only_literal_foci_not_freeform_suggestions(auth_case):
    p = plan(auth_case.collection)
    queries = literal_probes(auth_case.collection, QUESTION, QUESTION, "en", p)
    assert queries == (QUESTION, "S3 invocation", "Lambda execution role")
    assert "reading objects" not in " ".join(queries) and "key policy" not in " ".join(queries)
    assert QUESTION.endswith("across accounts.")


@pytest.mark.parametrize("question", [
    "Do NOT allow a role to read S3. Compare S3 invocation and Lambda execution role across accounts.",
    "Which policies apply? Compare S3 invocation and Lambda execution role across accounts.",
])
def test_negation_conditions_and_policy_scope_remain_in_original_query(auth_case, question):
    p = plan(auth_case.collection, question)
    assert literal_probes(auth_case.collection, question, question, "en", p)[0] == question
    # Probes are explicitly partial recall surfaces, never asserted equivalent rewrites or policy grants.


def test_fixture_cannot_be_relabelled_to_real_collection(auth_case):
    p = plan(auth_case.collection)
    real = NS(manifest_sha256=auth_case.collection.manifest_sha256, manifest=NS(purpose="DEMO"))
    with pytest.raises(ValueError, match="ORIGIN"):
        literal_probes(real, QUESTION, QUESTION, "en", p)


def test_metadata_deduplicates_title_and_groups_only_within_document(auth_case):
    docs, chunks, _ = collection_records(auth_case.collection)
    rows = metadata_rows(docs, chunks, "demo")
    assert {k for row in rows for k in row["chunk_ids"]} == set(chunks)
    for row in rows:
        assert len(row["headings"]) == len(set(row["headings"]))
        assert all(chunks[k].document_version_id == row["document_version_id"] for k in row["chunk_ids"])


@pytest.mark.parametrize("tamper", ["text", "tenant", "missing_document"])
def test_metadata_rejects_source_or_tenant_mismatch(auth_case, tamper):
    docs, chunks, _ = collection_records(auth_case.collection)
    docs, chunks = dict(docs), dict(chunks)
    key = next(iter(chunks))
    if tamper == "missing_document":
        del docs[chunks[key].document_version_id]
    else:
        chunks[key] = chunks[key].model_copy(update={"text": "forged"} if tamper == "text" else {"tenant_id": "other"})
    with pytest.raises(GenerationError):
        metadata_rows(docs, chunks, "demo")


@pytest.mark.parametrize("tamper", ["rows", "dimension", "nan", "norm"])
def test_metadata_index_rejects_unbound_rows_and_vectors(auth_case, tamper):
    docs, chunks, _ = collection_records(auth_case.collection)
    rows = metadata_rows(docs, chunks, "demo")
    vectors = np.asarray([fixture_vector(r["metadata_text"]) for r in rows], dtype=np.float32)
    if tamper == "rows":
        rows = ({**rows[0], "metadata_text": "new fact"}, *rows[1:])
    elif tamper == "dimension":
        vectors = vectors[:, :-1]
    elif tamper == "nan":
        vectors[0, 0] = np.nan
    else:
        vectors[0] *= 2
    with pytest.raises(GenerationError):
        make_index(auth_case.collection, rows, vectors)


def test_diversity_keeps_all_candidates_but_defers_repeated_document():
    rows = [{"document_version_id": "a", "id": 1}, {"document_version_id": "a", "id": 2},
        {"document_version_id": "b", "id": 3}]
    assert [r["id"] for r in _diverse_order(rows)] == [1, 3, 2]


@pytest.mark.parametrize("field_mode", ["BM25", "DENSE", "RRF"])
@pytest.mark.parametrize("admission", ["RANKED", "DOCUMENT_DIVERSE"])
def test_all_modes_pack_source_exact_same_pool_q_qb_without_graph(auth_case, field_mode, admission):
    index, p, vectors, base = case_inputs(auth_case)
    pool = build_section_pool(auth_case.collection, auth_case.lexical, index, question=QUESTION, search=QUESTION,
        language="en", plan=p, vectors=vectors, base_ids=base, recipe=SectionRecipe(field_mode=field_mode, admission=admission))
    assert len(pool.pool_ids) <= 40 and set(base[:16]) <= set(pool.pool_ids)
    trace = json.loads(pool.trace_json)
    assert not trace["source_annotations_used_for_ranking"] and trace["new_graph_records"] == 0
    assert trace["query_track"] == "DERIVED_LITERAL_RECALL_PROBES_NOT_NATIVE_REWRITES"
    fingerprints = []
    for balanced in (False, True):
        answer, audit = pack_section_pool(auth_case.collection, auth_case.lexical, pool, balanced=balanced,
            generation_recipe=GenerationRecipeV9(model_metadata_sha256="0"*64), run_id="fixture", owner_id="fixture")
        assert len(answer.packed.chunk_ids) <= 8 and answer.packed.proxy_tokens <= 4096
        assert not answer.binding.used_graph and not audit["hidden_fallbacks"] and not audit["context_assertion_ids"]
        assert json.loads(answer.packed.serialized)["untrusted_question"] == QUESTION
        assert all("metadata_text" not in source for source in json.loads(answer.packed.serialized)["untrusted_sources"])
        fingerprints.append(audit["pool_fingerprint"])
    assert fingerprints == [pool.fingerprint]*2


def test_query_vector_count_and_base_pool_are_checked(auth_case):
    index, p, vectors, base = case_inputs(auth_case)
    for change in ({"vectors": ()}, {"base_ids": ("not-a-source",)}, {"vectors": tuple(v*2 for v in vectors)}):
        args = dict(question=QUESTION, search=QUESTION, language="en", plan=p, vectors=vectors, base_ids=base, recipe=SectionRecipe())
        with pytest.raises(GenerationError):
            build_section_pool(auth_case.collection, auth_case.lexical, index, **(args | change))


def test_tampered_partial_window_and_pool_are_rejected(auth_case):
    index, p, vectors, base = case_inputs(auth_case)
    pool = build_section_pool(auth_case.collection, auth_case.lexical, index, question=QUESTION, search=QUESTION,
        language="en", plan=p, vectors=vectors, base_ids=base, recipe=SectionRecipe())
    bundles = json.loads(pool.bundles_json)
    window = next(b for b in bundles if b["kind"] == "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT")
    bundles.append({"bundle_id": "bad-singleton", "kind": "RANKED_ORIGINAL_CHUNK", "query_index": 0,
        "chunk_ids": window["chunk_ids"][:1]})
    for altered in (replace(pool, bundles_json=canonical_json(bundles).decode()), replace(pool, pool_ids=())):
        with pytest.raises(GenerationError):
            pack_section_pool(auth_case.collection, auth_case.lexical, altered, balanced=True,
                generation_recipe=GenerationRecipeV9(model_metadata_sha256="0"*64), run_id="fixture", owner_id="fixture")


@pytest.mark.parametrize("balanced", [False, True])
def test_multi_chunk_bundles_skip_nine_chunk_union_atomically(multi_chunk_case, balanced):
    from app.document_recall_v2 import document_window

    collection, lexical, chunks = multi_chunk_case
    ids = tuple(c.chunk_id for c in chunks)
    own = {c.chunk_id: c for c in chunks}
    windows = [{"bundle_id": f"window-{i}", "kind": "ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT",
        "query_index": i, "anchor_chunk_id": chunks[anchor].chunk_id,
        "chunk_ids": document_window(own, chunks[anchor].chunk_id, 2)} for i, anchor in enumerate((2, 6))]
    members = {k for b in windows for k in b["chunk_ids"]}
    singletons = [{"bundle_id": f"single-{i}", "kind": "RANKED_ORIGINAL_CHUNK", "query_index": 0,
        "chunk_ids": [key]} for i, key in enumerate(ids) if key not in members]
    recipe = SectionRecipe(radius=2)
    trace = {"recipe": recipe.model_dump(mode="json"), "native_v5_plan": plan(collection).model_dump(mode="json"),
        "queries": (QUESTION, "S3 invocation", "Lambda execution role"), "query_order": [1, 2, 0], "pool_ids": ids}
    pool = SectionPool(collection.manifest_sha256, QUESTION, QUESTION, "en", recipe, ids, ids,
        canonical_json(windows+singletons).decode(), canonical_json(trace).decode())
    answer, audit = pack_section_pool(collection, lexical, pool, balanced=balanced,
        generation_recipe=GenerationRecipeV9(model_metadata_sha256="0"*64), run_id="fixture", owner_id="fixture")
    decisions = {d["bundle_id"]: d for d in audit["decisions"]}
    assert sum(decisions[b["bundle_id"]]["reason"] == "SELECTED" for b in windows) == 1
    skipped = next(b for b in windows if decisions[b["bundle_id"]]["reason"] == "MAX_CHUNKS")
    selected = next(b for b in windows if decisions[b["bundle_id"]]["reason"] == "SELECTED")
    assert len(set(windows[0]["chunk_ids"]) | set(windows[1]["chunk_ids"])) == 9
    assert not (set(skipped["chunk_ids"]) - set(selected["chunk_ids"])) & set(answer.packed.chunk_ids)
    assert len(answer.packed.chunk_ids) == 8


@pytest.mark.parametrize("change", ["recipe", "task", "index"])
def test_changed_recipe_task_and_index_rejected(auth_case, change):
    index, p, vectors, base = case_inputs(auth_case)
    if change == "index":
        index.rows[0]["metadata_text"] = "modified"
        with pytest.raises(GenerationError, match="INDEX_MUTATED"):
            build_section_pool(auth_case.collection, auth_case.lexical, index, question=QUESTION, search=QUESTION,
                language="en", plan=p, vectors=vectors, base_ids=base, recipe=SectionRecipe())
        return
    pool = build_section_pool(auth_case.collection, auth_case.lexical, index, question=QUESTION, search=QUESTION,
        language="en", plan=p, vectors=vectors, base_ids=base, recipe=SectionRecipe())
    altered = replace(pool, recipe=SectionRecipe(radius=2)) if change == "recipe" else replace(pool, question="Different question?")
    with pytest.raises(ValueError):
        pack_section_pool(auth_case.collection, auth_case.lexical, altered, balanced=False,
            generation_recipe=GenerationRecipeV9(model_metadata_sha256="0"*64), run_id="fixture", owner_id="fixture")
