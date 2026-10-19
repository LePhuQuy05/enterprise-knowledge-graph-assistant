import copy
import json
from types import SimpleNamespace

import pytest

from app.cost import TokenUsage
from app.generation import GenerationRecipeV9
from tests.test_aws_knowledge import item
from tests.test_query_api import USER_B, call, event
from tests.test_query_api import setup_api as setup_api  # noqa: F401


def configure(s, *, empty=False, fail=False):
    s.service.recipe = GenerationRecipeV9(model_metadata_sha256="0" * 64)
    s.searches, s.languages = [], []
    def search(phrase):
        s.searches.append(phrase)
        if fail:
            raise TimeoutError("synthetic private error")
        source = item()
        return [] if empty else [{"url": source["url"], "title": source["title"], "text": source["context"],
            "excerpt_sha256": "1" * 64}]
    s.service.knowledge_factory = lambda timeout: SimpleNamespace(search=search)
    previous = s.gateway.generate
    def generate(context, **kwargs):
        if context.stage == "REWRITE":
            s.gateway.calls += 1
            return SimpleNamespace(text='{"search_query":"Amazon RDS Multi-AZ versus Read Replica asynchronous replication"}',
                stop_reason="end_turn", operation_id="fixture-rewrite", usage=SimpleNamespace(model_dump=lambda **_: {"input_tokens": 50, "output_tokens": 30}),
                actual_micro_usd=25)
        s.languages.append(kwargs["system"])
        source = json.loads(kwargs["prompt"])["untrusted_sources"][0]
        if "spans" in source:
            s.gateway.calls += 1
            result = SimpleNamespace(text=json.dumps({"answer_status": "ANSWERED", "reason": "NONE", "claims": [{
                "text": "Amazon RDS uses asynchronous replication to update a read replica.",
                "evidence": [{"source_id": source["source_id"], "span_id": source["spans"][0]["span_id"]}]}]}),
                stop_reason="end_turn", operation_id="fixture-generate", usage=TokenUsage(input_tokens=500, output_tokens=100),
                actual_micro_usd=100)
        else:
            result = previous(context, **kwargs)
        if "tiếng Việt" in kwargs["system"]:
            data = json.loads(result.text)
            data["claims"][0]["text"] = "Amazon RDS sử dụng sao chép bất đồng bộ để cập nhật Read Replica."
            result.text = json.dumps(data, ensure_ascii=False)
        return result
    s.gateway.generate = generate
    return s


def request(question="How does RDS update a read replica?", **kwargs):
    return event(body={"question": question, "corpus_scope": "aws_official", **kwargs})


def test_wide_source_is_pinned_owner_bound_and_idempotent_without_model_or_search_replay(setup_api):
    s = configure(setup_api)
    req = request()
    status, data = call(s, req)
    assert status == 200 and data["status"] == "COMPLETED"
    response = data["response"]
    assert response["retrieval_backend"] == "aws_knowledge_mcp" and response["evidence_bundle_sha256"]
    assert not response["used_graph"]
    rid, chunk = data["request_id"], response["citations"][0]["chunk_id"]
    assert s.gateway.calls == 1 and len(s.searches) == 1
    assert call(s, req)[1]["idempotent_replay"]
    assert s.gateway.calls == 1 and len(s.searches) == 1
    path = "/v1/sources/" + chunk
    source = call(s, event("GET", path, params={"request_id": rid}))[1]
    assert source["text"] == response["citations"][0]["quote"]
    assert call(s, event("GET", path, user=USER_B, params={"request_id": rid}))[0] == 404
    assert "source_bundle" not in json.dumps(data) and "rewrite_receipt" not in json.dumps(data)
    # Live results changing later must not affect an old source lookup.
    s.service.knowledge_factory = lambda _: (_ for _ in ()).throw(AssertionError("must not search again"))
    assert call(s, event("GET", path, params={"request_id": rid}))[1] == source


def test_vietnamese_rewrite_separate_receipt_and_original_language_quotes(setup_api):
    s = configure(setup_api)
    req = request("RDS Multi-AZ khác gì với Read Replica?")
    status, data = call(s, req)
    assert status == 200 and s.gateway.calls == 2
    assert data["response"]["answer_language"] == "vi"
    assert "sử dụng" in data["response"]["answer"]
    assert data["response"]["citations"][0]["quote"].startswith("Amazon RDS uses")
    assert s.searches == ["Amazon RDS Multi-AZ versus Read Replica asynchronous replication"]
    row = s.repo.get(s.auth, data["request_id"])
    assert row["rewrite_receipt"]["operation_id"] == "fixture-rewrite"
    assert row["receipt"]["operation_id"] != row["rewrite_receipt"]["operation_id"]
    assert call(s, req)[1]["idempotent_replay"] and s.gateway.calls == 2 and len(s.searches) == 1
    changed = copy.deepcopy(req)
    body = json.loads(changed["body"])
    body["answer_language"] = "en"
    changed["body"] = json.dumps(body)
    assert call(s, changed)[0] == 409


def test_empty_live_search_abstains_in_vietnamese_without_generation(setup_api):
    s = configure(setup_api, empty=True)
    status, data = call(s, request("EC2 là gì?"))
    assert status == 200 and s.gateway.calls == 1
    assert data["response"]["answer_status"] == "ABSTAINED"
    assert "chưa cung cấp đủ" in data["response"]["answer"] and not data["response"]["citations"]


def test_search_failure_never_falls_back_to_ungrounded_model_and_does_not_replay(setup_api):
    s = configure(setup_api, fail=True)
    req = request()
    status, data = call(s, req)
    assert status == 503 and s.gateway.calls == 0
    assert "synthetic" not in json.dumps(data)
    assert call(s, req)[1]["status"] == "FAILED" and len(s.searches) == 1


def test_wide_scope_vector_mode_is_rejected_before_quota_or_model(setup_api):
    s = configure(setup_api)
    assert call(s, request(mode="vector"))[0] == 400
    assert s.repo.quota_remaining() == 300 and s.gateway.calls == 0 and not s.searches


def test_tampered_pinned_bundle_cannot_be_displayed(setup_api):
    s = configure(setup_api)
    _, data = call(s, request())
    rid = data["request_id"]
    row = s.state._items[(s.auth.partition, "REQUEST#" + rid)]
    row["source_bundle"]["documents"][0]["normalized_text"] = "forged"
    assert call(s, event("GET", "/v1/queries/" + rid))[0] == 503


@pytest.mark.parametrize("question,language,expected,calls", [
    ("RDS Multi AZ khac gi Read Replica?", "vi", "vi", 2),
    ("RDS Multi-AZ khác gì Read Replica?", "en", "en", 2),
    ("How does RDS update replicas?", "en", "en", 1),
])
def test_explicit_language_controls_and_rewrite_before_english_retrieval(setup_api, question, language, expected, calls):
    s = configure(setup_api)
    status, data = call(s, request(question, answer_language=language))
    assert status == 200 and s.gateway.calls == calls
    assert data["response"]["answer_language"] == expected
    assert bool("tiếng Việt" in s.languages[0]) == (expected == "vi")
