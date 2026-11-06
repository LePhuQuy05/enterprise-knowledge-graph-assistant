"""Source-exact bounded windows and same-pool Q/QB, using labelled fixtures."""

import json
from dataclasses import replace
from types import SimpleNamespace as NS

import numpy as np
import pytest

from app.contracts import canonical_json, sha256_bytes
from app.document_recall_v2 import (
    DocumentRecallRecipe,
    RecallPool,
    build_recall_pool,
    document_window,
    prepare_recall_answer,
    round_robin_bundles,
)
from app.faceted_query_v5 import FacetPlanV5, FacetRecipeV5, validate_facets
from app.fixtures import fixture_vector
from app.generation import GenerationError, GenerationRecipeV9
from tests.test_authorization_evidence import auth_case as auth_case


def fixture_plan(collection, question):
    rows = [{"focus_quote": "S3 invocation", "search_query": "S3 invoke Lambda permissions"},
        {"focus_quote": "Lambda execution role", "search_query": "Lambda execution role permissions"}]
    return FacetPlanV5(origin="FIXTURE", collection_manifest_sha256=collection.manifest_sha256,
        question_sha256=sha256_bytes(question.encode()), search_query_sha256=sha256_bytes(question.encode()), language="en",
        facets=validate_facets(json.dumps({"facet_1": rows[0], "facet_2": rows[1], "facet_3": None}), question, question), recipe_sha256=FacetRecipeV5().fingerprint,
        operation_id="fixture", receipt_sha256="0" * 64)


def pool(case, **overrides):
    question = "Compare S3 invocation and Lambda execution role permissions in this synthetic example."
    plan = fixture_plan(case.collection, question)
    vectors = (fixture_vector(question), *(fixture_vector(f.search_query) for f in plan.facets))
    args = dict(question=question, search=question, language="en", plan=plan, vectors=vectors)
    return build_recall_pool(case.collection, case.lexical, **(args | overrides))


@pytest.mark.parametrize("radius,expected", [(1, ("a", "b", "c")), (2, ("a", "b", "c", "d"))])
def test_neighbor_window_uses_document_offsets_not_input_order_and_never_crosses_documents(radius, expected):
    chunks = {k: NS(chunk_id=k, document_version_id="doc", start_char=i*10, end_char=i*10+10)
        for i,k in enumerate(("a","b","c","d"))}
    chunks = dict(reversed(list(chunks.items())))
    chunks["foreign"] = NS(chunk_id="foreign", document_version_id="other", start_char=11, end_char=25)
    assert document_window(chunks, "b", radius) == expected


@pytest.mark.parametrize("radius", [0,3,True,1.0])
def test_window_radius_is_bounded(radius):
    with pytest.raises(GenerationError):
        document_window({}, "missing", radius)


def test_round_robin_changes_order_only_and_preserves_every_bundle():
    bundles=[{"query_index":0,"id":"a"},{"query_index":0,"id":"b"},{"query_index":1,"id":"c"},{"query_index":2,"id":"d"}]
    ordered=round_robin_bundles(bundles,(1,2,0))
    assert [b["id"] for b in ordered]==["c","d","a","b"]
    assert sorted(b["id"] for b in ordered)==sorted(b["id"] for b in bundles)


def test_real_fixture_pool_and_q_qb_preserve_sources_budget_and_shared_fingerprint(auth_case):
    recalled=pool(auth_case)
    assert len(recalled.pool_ids)<=40 and len(set(recalled.pool_ids))==len(recalled.pool_ids)
    recipe=GenerationRecipeV9(model_metadata_sha256="0"*64)
    outputs=[]
    for balanced in (False,True):
        view,p,trace=prepare_recall_answer(auth_case.collection,auth_case.lexical,recalled,
            balanced=balanced,recipe=recipe,run_id="fixture",owner_id="fixture")
        assert len(p.packed.chunk_ids)<=8 and p.packed.proxy_tokens<=4096 and not p.binding.used_graph
        assert trace["pool_fingerprint"]==recalled.fingerprint
        assert p.retrieved_chunk_ids==recalled.pool_ids
        for s in trace["sources"]:
            ch=view.chunks[s["chunk_id"]]
            assert view.documents[ch.document_version_id].normalized_text[ch.start_char:ch.end_char]==ch.text
        outputs.append(trace)
    assert not outputs[0]["context_selection_used"] and outputs[1]["context_selection_used"]


def test_mismatched_or_nonfinite_vectors_rejected(auth_case):
    with pytest.raises(GenerationError,match="VECTOR_COUNT"):
        pool(auth_case,vectors=())
    bad=fixture_vector("bad")
    bad[0]=np.nan
    with pytest.raises(GenerationError,match="QUERY_VECTOR"):
        pool(auth_case,vectors=(bad,bad,bad))


def test_pool_trace_tamper_is_rejected_before_packing(auth_case):
    recalled=pool(auth_case)
    trace=json.loads(recalled.trace_json)
    trace["pool_chunk_ids"]=[]
    altered=replace(recalled,trace_json=canonical_json(trace).decode())
    with pytest.raises(GenerationError,match="POOL_OR_BUNDLE"):
        prepare_recall_answer(auth_case.collection,auth_case.lexical,altered,balanced=True,
            recipe=GenerationRecipeV9(model_metadata_sha256="0"*64),run_id="fixture",owner_id="fixture")


def test_partial_window_cannot_sneak_back_as_singleton(auth_case):
    recalled=pool(auth_case)
    bundles=json.loads(recalled.bundles_json)
    window=next(b for b in bundles if b["kind"]=="ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT")
    bundles.append({"bundle_id":"forbidden","kind":"RANKED_ORIGINAL_CHUNK","query_index":0,
        "chunk_ids":[window["chunk_ids"][0]],"anchor_chunk_id":window["chunk_ids"][0]})
    altered=replace(recalled,bundles_json=canonical_json(bundles).decode())
    with pytest.raises(GenerationError,match="PARTIAL_WINDOW"):
        prepare_recall_answer(auth_case.collection,auth_case.lexical,altered,balanced=True,
            recipe=GenerationRecipeV9(model_metadata_sha256="0"*64),run_id="fixture",owner_id="fixture")


def test_recipe_records_unchanged_serving_limits():
    r=DocumentRecallRecipe()
    assert (r.max_pool,r.max_chunks,r.user_message_proxy_tokens)==(40,8,4096)
    assert r.fingerprint!=DocumentRecallRecipe(radius=2).fingerprint


@pytest.mark.parametrize("spans,expected", [
    ([{"text":"actor condition"},{"text":"\nrequired qualifier"}], True),
    ([{"text":"actor condition"}], False),
    ([{"text":"required qualifier"}], False),
    ([{"text":"actor condition"},{"text":"invented qualifier"}], False),
])
def test_model_rendering_requires_all_nonwhitespace_source_text(spans,expected):
    from app.document_recall_v2 import _full_rendering

    view=NS(chunks={"source":NS(text="actor condition\nrequired qualifier")})
    prompt=json.dumps({"untrusted_sources":[{"source_id":"s1","spans":spans}]})
    assert _full_rendering(view,("source",),prompt) is expected


@pytest.fixture
def multi_chunk_case(tmp_path):
    from app.collection_lexical import build_global_lexical_index, load_global_lexical_index
    from app.collections import build_collection, load_collection
    from app.contracts import Chunk
    from app.fixtures import FIXTURE_EMBEDDING_MODEL, fixture_recipes
    from app.snapshot import SnapshotContent, build_snapshot
    from tests.test_extraction import source

    paragraphs=[f"Synthetic passage {i}. Actor reads a resource under an explicit condition. "*10+"\n\n" for i in range(12)]
    doc,_=source("".join(paragraphs),prefix="",suffix="atomic-document-windows")
    chunks=[]
    offset=0
    for i,text in enumerate(paragraphs):
        chunks.append(Chunk.create(tenant_id="demo",document_version_id=doc.document_version_id,
            chunking_version="fixture-v1",section_path=("Synthetic",),position=i,start_char=offset,end_char=offset+len(text),
            text=text,token_count=20,tokenizer_version="fixture-v1"))
        offset+=len(text)
    extraction,embedding=fixture_recipes()
    snapshot=build_snapshot(tmp_path/"snapshots",SnapshotContent((doc,),tuple(chunks)),
        {c.chunk_id:fixture_vector(c.text) for c in chunks},tenant_id="demo",purpose="FIXTURE",
        extraction_recipe=extraction,embedding_recipe=embedding,corpus_version="atomic-fixture-v1",
        normalizer_version="fixture-v1",chunking_version="fixture-v1",tokenizer_version="fixture-v1")
    path=build_collection(snapshot.parent,[snapshot],tenant_id="demo",embedding_model_id=FIXTURE_EMBEDDING_MODEL,allow_fixture=True)
    collection=load_collection(path,tenant_id="demo",embedding_model_id=FIXTURE_EMBEDDING_MODEL,allow_fixture=True)
    lexical_path=build_global_lexical_index(collection,tmp_path/"lexical")
    with load_global_lexical_index(lexical_path,collection) as lexical:
        yield collection,lexical,chunks


def atomic_pool(case):
    collection,_,chunks=case
    windows=[]
    for number,index in enumerate((2,6)):
        window=document_window({c.chunk_id:c for c in chunks},chunks[index].chunk_id,2)
        windows.append({"bundle_id":f"fixture-window-{number}","kind":"ORIGINAL_DOCUMENT_WINDOW_NOT_GRAPH_FACT",
            "query_index":number,"anchor_chunk_id":chunks[index].chunk_id,"chunk_ids":window})
    covered={key for w in windows for key in w["chunk_ids"]}
    singletons=[{"bundle_id":f"fixture-single-{i}","kind":"RANKED_ORIGINAL_CHUNK","query_index":0,
        "chunk_ids":[c.chunk_id],"anchor_chunk_id":c.chunk_id} for i,c in enumerate(chunks) if c.chunk_id not in covered]
    ids=tuple(c.chunk_id for c in chunks)
    return RecallPool(collection.manifest_sha256,"Explain this synthetic actor condition.","synthetic actor condition","en",
        DocumentRecallRecipe(radius=2),ids,canonical_json(windows+singletons).decode(),
        canonical_json({"pool_chunk_ids":ids,"query_order":[0,1]}).decode())


@pytest.mark.parametrize("balanced",[False,True])
def test_eight_chunk_limit_skips_entire_second_window_not_just_last_member(multi_chunk_case,balanced):
    collection,lexical,chunks=multi_chunk_case
    recalled=atomic_pool(multi_chunk_case)
    _,prepared,audit=prepare_recall_answer(collection,lexical,recalled,balanced=balanced,
        recipe=GenerationRecipeV9(model_metadata_sha256="0"*64),run_id="fixture",owner_id="fixture")
    first=json.loads(recalled.bundles_json)[0]
    assert set(first["chunk_ids"])<=set(prepared.packed.chunk_ids)
    second=next(d for d in audit["decisions"] if d["bundle_id"]=="fixture-window-1")
    assert second["reason"]=="MAX_CHUNKS" and second["proposed_chunk_count"]==9
    assert not {chunks[i].chunk_id for i in (5,6,7,8)}&set(prepared.packed.chunk_ids)
    assert len(prepared.packed.chunk_ids)==8


def test_window_member_tampering_is_not_a_valid_neighbor_bundle(multi_chunk_case):
    collection,lexical,chunks=multi_chunk_case
    recalled=atomic_pool(multi_chunk_case)
    bundles=json.loads(recalled.bundles_json)
    bundles[0]["chunk_ids"][0]=chunks[-1].chunk_id
    altered=replace(recalled,bundles_json=canonical_json(bundles).decode())
    with pytest.raises(GenerationError,match="NEIGHBOR_WINDOW_CHANGED"):
        prepare_recall_answer(collection,lexical,altered,balanced=False,
            recipe=GenerationRecipeV9(model_metadata_sha256="0"*64),run_id="fixture",owner_id="fixture")
