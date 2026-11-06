import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from types import SimpleNamespace

import pytest

from app.contracts import canonical_json, sha256_bytes, stable_id, text_hash
from app.cost import TokenUsage
from app.external_gateway import ExternalRoutes
from app.model_gateway import CallContext, ModelResult, UncertainModelCall
from app.query_language import RewriteError
from app.rewrite_cache import RewriteCache, RewriteCacheBlocked, RewriteCacheError

QUESTION = "Tại sao s3:GetObject gặp AccessDenied với kms:Decrypt?"
SEARCH = "Why does s3:GetObject encounter AccessDenied with kms:Decrypt?"


@pytest.fixture
def routes():
    return ExternalRoutes(account_model_metadata_sha256=text_hash("fixture-model-metadata"))


@pytest.fixture
def context():
    return CallContext(tenant_id="demo", run_id="rewrite-fixture", request_id="shared-rewrite",
                       stage="REWRITE")


class Gateway:
    def __init__(self, routes, *, live=False, text=None, stop_reason="end_turn", fail=None,
                 entered=None, release=None):
        self.routes = routes
        self.provider = SimpleNamespace(live=live)
        self.text = json.dumps({"search_query": SEARCH}) if text is None else text
        self.stop_reason = stop_reason
        self.fail = fail
        self.entered, self.release = entered, release
        self.calls = 0

    def generate(self, context, **kwargs):
        self.calls += 1
        assert context.stage == "REWRITE" and kwargs["max_output_tokens"] == 256
        if self.entered:
            self.entered.set()
            assert self.release.wait(5)
        if self.fail:
            raise self.fail
        return ModelResult(
            operation_id=stable_id("op", **context.model_dump(mode="json")),
            usage=TokenUsage(input_tokens=70, output_tokens=20), actual_micro_usd=45,
            text=self.text, stop_reason=self.stop_reason,
        )


def stored(cache):
    with sqlite3.connect(cache.path) as db:
        return db.execute("SELECT state,payload,payload_sha256,error_category FROM rewrites").fetchall()


def test_shared_rewrite_reopens_and_replays_without_gateway(tmp_path, routes, context):
    gateway = Gateway(routes)
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    first = cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    replay_routes = routes.model_copy(update={"live_inference_enabled": True,
                                              "live_inference_gate": "DIFFERENT_OPERATIONAL_FLAG"})
    for _method in ("lexical", "vector", "hybrid"):
        replay = RewriteCache(tmp_path, namespace="FIXTURE").resolve(
            context, question=QUESTION, routes=replay_routes)
        assert replay.search_query == first.search_query == SEARCH
        assert replay.cache_hit and replay.new_cost_micro_usd == 0
        assert replay.receipt == first.receipt
    assert first.new_cost_micro_usd == 45 and not first.cache_hit
    assert first.receipt["cost_micro_usd"] == 45 and gateway.calls == 1
    assert stored(cache)[0][0] == "READY"


@pytest.mark.parametrize("text,stop", [
    ("not JSON", "end_turn"),
    (json.dumps({"search_query": "Why is object access denied?"}), "end_turn"),
    (json.dumps({"search_query": SEARCH, "extra": "not allowed"}), "end_turn"),
    (json.dumps({"search_query": SEARCH}), "max_tokens"),
])
def test_settled_raw_is_durable_before_invalid_rewrite_blocks_retries(
    tmp_path, routes, context, monkeypatch, text, stop,
):
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    gateway = Gateway(routes, text=text, stop_reason=stop)
    import app.query_language as language

    original = language.strict_json
    observed = []

    def inspect_before_parse(raw):
        state, payload, digest, _ = stored(cache)[0]
        assert state == "IN_PROGRESS" and sha256_bytes(payload) == digest
        value = json.loads(payload)
        assert value["provider_receipt"]["raw_output"] == text
        assert value["provider_receipt"]["cost_micro_usd"] == 45
        observed.append(True)
        return original(raw)

    monkeypatch.setattr(language, "strict_json", inspect_before_parse)
    with pytest.raises(RewriteError):
        cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    state, payload, digest, error = stored(cache)[0]
    assert state == "FAILED" and error and sha256_bytes(payload) == digest
    assert json.loads(payload)["provider_receipt"]["raw_output"] == text
    if stop == "end_turn":
        assert observed
    with pytest.raises(RewriteCacheBlocked, match="RECONCILIATION"):
        RewriteCache(tmp_path, namespace="FIXTURE").resolve(
            context, question=QUESTION, routes=routes, gateway=gateway)
    assert gateway.calls == 1


def test_uncertain_call_stays_failed_and_never_dispatches_again(tmp_path, routes, context):
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    gateway = Gateway(routes, fail=UncertainModelCall("op-uncertain-fixture"))
    with pytest.raises(UncertainModelCall):
        cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    assert stored(cache)[0][:2] == ("FAILED", None)
    with pytest.raises(RewriteCacheBlocked):
        cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    assert gateway.calls == 1


def test_namespace_mismatch_fails_before_claim_or_dispatch(tmp_path, routes, context):
    for namespace, live in [("LIVE", False), ("FIXTURE", True)]:
        cache = RewriteCache(tmp_path / namespace, namespace=namespace)
        gateway = Gateway(routes, live=live)
        with pytest.raises(RewriteCacheError, match="NAMESPACE_MISMATCH"):
            cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
        assert stored(cache) == [] and gateway.calls == 0


def test_live_namespace_cannot_replay_fixture_receipt(tmp_path, routes, context):
    fixture = RewriteCache(tmp_path, namespace="FIXTURE")
    fixture.resolve(context, question=QUESTION, routes=routes, gateway=Gateway(routes))
    live = RewriteCache(tmp_path)
    with pytest.raises(RewriteCacheError, match="MISS_NO_GATEWAY"):
        live.resolve(context, question=QUESTION, routes=routes)


@pytest.mark.parametrize("change", ["question", "route"])
def test_same_operation_cannot_silently_change_payload_or_model(tmp_path, routes, context, change):
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    gateway = Gateway(routes)
    cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    question = QUESTION + " thêm điều kiện" if change == "question" else QUESTION
    changed_routes = routes.model_copy(update={"account_model_metadata_sha256": text_hash("new-model")})
    with pytest.raises(RewriteCacheBlocked, match="IDEMPOTENCY_CONFLICT"):
        cache.resolve(context, question=question,
                      routes=changed_routes if change == "route" else routes, gateway=gateway)
    assert gateway.calls == 1


def test_wrong_gateway_route_fails_before_claim(tmp_path, routes, context):
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    gateway = Gateway(routes.model_copy(update={"account_model_metadata_sha256": text_hash("wrong")}))
    with pytest.raises(RewriteCacheError, match="ROUTE_MISMATCH"):
        cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    assert gateway.calls == 0 and stored(cache) == []


def test_concurrent_cache_insert_dispatches_once(tmp_path, routes, context):
    entered, release = Event(), Event()
    gateway = Gateway(routes, entered=entered, release=release)
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    with ThreadPoolExecutor(max_workers=2) as executor:
        running = executor.submit(cache.resolve, context, question=QUESTION, routes=routes, gateway=gateway)
        assert entered.wait(5)
        try:
            with pytest.raises(RewriteCacheBlocked):
                RewriteCache(tmp_path, namespace="FIXTURE").resolve(
                    context, question=QUESTION, routes=routes, gateway=gateway)
        finally:
            release.set()
        assert running.result(timeout=5).search_query == SEARCH
    assert gateway.calls == 1


@pytest.mark.parametrize("tamper", ["checksum", "binding", "namespace", "operation", "oversize"])
def test_corrupt_ready_receipt_never_triggers_repair_call(tmp_path, routes, context, tamper):
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    gateway = Gateway(routes)
    cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    _, raw, _, _ = stored(cache)[0]
    value = json.loads(raw)
    if tamper == "checksum":
        raw += b" "
    elif tamper == "binding":
        value["binding_sha256"] = text_hash("other-request")
    elif tamper == "namespace":
        value["namespace"] = "LIVE"
    elif tamper == "operation":
        value["provider_receipt"]["operation_id"] = "op-wrong-request"
    else:
        value["provider_receipt"]["raw_output"] = "x" * 70_000
    if tamper != "checksum":
        raw = canonical_json(value)
    with sqlite3.connect(cache.path) as db:
        if tamper == "checksum":
            db.execute("UPDATE rewrites SET payload=?", (raw,))
        else:
            db.execute("UPDATE rewrites SET payload=?,payload_sha256=?", (raw, sha256_bytes(raw)))
    with pytest.raises(RewriteCacheError):
        cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    assert gateway.calls == 1


def test_durable_in_progress_claim_survives_process_restart(tmp_path, routes, context):
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    gateway = Gateway(routes)
    cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    with sqlite3.connect(cache.path) as db:
        db.execute("UPDATE rewrites SET state='IN_PROGRESS',payload=NULL,payload_sha256=NULL")
    with pytest.raises(RewriteCacheBlocked, match="RECONCILIATION"):
        RewriteCache(tmp_path, namespace="FIXTURE").resolve(
            context, question=QUESTION, routes=routes, gateway=gateway)
    assert gateway.calls == 1


@pytest.mark.parametrize("question,stage,attempt", [
    ("", "REWRITE", 1), (" ", "REWRITE", 1), ("x" * 2001, "REWRITE", 1),
    (QUESTION, "GENERATE", 1), (QUESTION, "REWRITE", 2),
])
def test_invalid_inputs_do_not_claim_or_dispatch(tmp_path, routes, context, question, stage, attempt):
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    gateway = Gateway(routes)
    with pytest.raises(RewriteCacheError, match="INPUT_INVALID"):
        cache.resolve(context.model_copy(update={"stage": stage, "attempt": attempt}),
                      question=question, routes=routes, gateway=gateway)
    assert gateway.calls == 0 and stored(cache) == []


def test_non_rewrite_provider_exception_is_persisted_without_its_message(tmp_path, routes, context):
    cache = RewriteCache(tmp_path, namespace="FIXTURE")
    gateway = Gateway(routes, fail=RuntimeError("PRIVATE_PROVIDER_ERROR_CONTENT"))
    with pytest.raises(RewriteCacheError, match="DISPATCH_FAILED"):
        cache.resolve(context, question=QUESTION, routes=routes, gateway=gateway)
    assert stored(cache)[0][3] == "RuntimeError"
    assert "PRIVATE_PROVIDER_ERROR_CONTENT" not in str(stored(cache))
