import json
from types import SimpleNamespace

import pytest

from app.answer_cache import AnswerCache
from app.collection_graph_runner import run_graph_method
from app.fixtures import fixture_vector
from app.generation import GenerationError, GenerationRecipeV9
from app.rewrite_cache import RewriteCache
from tests.test_collection_graph_runtime import graph_case as graph_case
from tests.test_external_pilot import configured


@pytest.fixture
def runtime(graph_case, tmp_path):
    collection = graph_case[0]
    encoder = SimpleNamespace(recipe=lambda role: collection.manifest.embedding_recipe)
    embeddings = SimpleNamespace(namespace="FIXTURE", resolve_local=lambda tenant, **kw:
        SimpleNamespace(record=SimpleNamespace(vector=fixture_vector(kw["text"]))))
    gateway = configured()
    gateway.ledger.initialize_run("kg-r72", collection.manifest.tenant_id, "graph-runner-fixture", cap_micro_usd=500000)

    def respond(provider, method, path, body):
        gateway.provider.calls += 1
        payload = json.loads(body)
        prompt = json.loads(payload["messages"][1]["content"])
        if "untrusted_sources" in prompt:
            source = prompt["untrusted_sources"][0]
            raw = {"answer_status": "ANSWERED", "reason": "NONE", "claims": [{
                "text": "Ví dụ từ nguồn." if "Vietnamese" in payload["messages"][0]["content"] else "Synthetic source example.",
                "evidence": [{"source_id": source["source_id"], "span_id": source["spans"][0]["span_id"]}]}]}
        else:
            raw = {"search_query": "Resource C Permission D Service A"}
        return {"model": "deepseek-flash", "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
            "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": json.dumps(raw)}}]}

    gateway.provider.request = respond
    return dict(tenant_id=collection.manifest.tenant_id, owner_id="fixture-owner", run_id="graph-runner-fixture",
        idempotency_key="graph-key", recipe=GenerationRecipeV9(model_metadata_sha256="0" * 64), routes=gateway.routes,
        encoder=encoder, embedding_cache=embeddings, answer_cache=AnswerCache(tmp_path / "answers", namespace="FIXTURE"),
        rewrite_cache=RewriteCache(tmp_path / "rewrites", namespace="FIXTURE"), allow_experimental=True), gateway


def run(case, params, question="Which service needs Resource C?", **changes):
    collection, lexical, graph, _ = case
    return run_graph_method(collection, lexical, graph, question, **(params | changes))


def test_graph_retrieval_and_cache_only_answer_never_create_provider_calls(graph_case, runtime):
    params, gateway = runtime
    retrieved = run(graph_case, params)
    assert retrieved["status"] == "RETRIEVED" and retrieved["response"] is None and retrieved["new_cost_micro_usd"] == 0
    assert retrieved["documents"] == graph_case[0].manifest.counts.documents
    planned = run(graph_case, params, idempotency_key="planned-answer", run_kind="answer")
    assert planned["status"] == "PLANNED_NO_MODEL_CALL" and gateway.provider.calls == 0


@pytest.mark.parametrize("language,question,calls", [("en", "Which service needs Resource C?", 1),
    ("vi", "Dịch vụ nào cần Resource C?", 2)])
def test_cited_graph_answers_and_vietnamese_rewrite_replay_without_a_gateway(graph_case, runtime, language, question, calls):
    params, gateway = runtime
    first = run(graph_case, params, question, run_kind="answer", language=language, gateway=gateway, hops=2)
    replay = run(graph_case, params, question, run_kind="answer", language=language, hops=2)
    assert first["status"] == replay["status"] == "COMPLETED"
    assert first["used_graph"] and first["response"]["mode_used"] == "graph_2hop"
    assert first["response"] == replay["response"] and replay["answer_cache_hit"]
    assert gateway.provider.calls == calls and replay["new_cost_micro_usd"] == 0
    assert first["response"]["answer_language"] == language
    assert tuple(s["chunk_id"] for s in first["context_sources"]) == first["trace"]["context_chunk_ids"]
    sources = {s["chunk_id"]: s for s in first["context_sources"]}
    for citation in first["response"]["citations"]:
        source = sources[citation["chunk_id"]]
        assert citation["quote"] in source["text"] and citation["document_version_id"] == source["document_version_id"]


def test_graph_idempotency_conflict_is_rejected_before_another_dispatch(graph_case, runtime):
    params, gateway = runtime
    run(graph_case, params, run_kind="answer", gateway=gateway)
    with pytest.raises(GenerationError, match="IDEMPOTENCY_CONFLICT"):
        run(graph_case, params, "An unrelated question", run_kind="answer", gateway=gateway)
    assert gateway.provider.calls == 1


def test_graph_failed_provider_intent_cannot_be_retried(graph_case, runtime):
    params, gateway = runtime

    def timeout(*args):
        gateway.provider.calls += 1
        raise TimeoutError("SYNTHETIC_ONLY")

    gateway.provider.request = timeout
    with pytest.raises(Exception):
        run(graph_case, params, run_kind="answer", gateway=gateway)
    with pytest.raises(GenerationError, match="RECONCILIATION_REQUIRED"):
        run(graph_case, params, run_kind="answer", gateway=gateway)
    assert gateway.provider.calls == 1


def test_graph_gateway_and_foreign_namespace_are_rejected_early(graph_case, runtime):
    params, gateway = runtime
    with pytest.raises(GenerationError, match="DISALLOWS_GATEWAY"):
        run(graph_case, params, gateway=gateway)
    with pytest.raises(GenerationError, match="NAMESPACE_MISMATCH"):
        run(graph_case, params, answer_cache=SimpleNamespace(namespace="LIVE"))
    assert gateway.provider.calls == 0
