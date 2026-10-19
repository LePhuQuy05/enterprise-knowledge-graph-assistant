import base64
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from uuid import uuid4

import numpy as np
import pytest

from app.contracts import canonical_json
from app.cost import TokenUsage
from app.fixtures import fixture_vector
from app.generation import GenerationRecipe
from app.ledger import Change, MemoryLedgerStore
from app.query_api import QueryApi
from app.query_auth import ApiError, AuthContext, require_current_membership
from app.query_service import QueryService
from app.query_state import MemoryQueryStore, QueryRepository
from app.releases import ReleaseRegistry

USER_A = "11111111-1111-1111-1111-111111111111"
USER_B = "22222222-2222-2222-2222-222222222222"
PUBLIC = {"issuer": "https://issuer.invalid/pool", "client_id": "fixture-client"}
NOW = datetime(2026, 10, 18, tzinfo=UTC)


def put(store, row):
    store.commit([Change((row["PK"], row["SK"]), None, row | {"version": 1})])


class Encoder:
    def encode(self, texts, role):
        assert role == "query"
        return np.asarray([fixture_vector(texts[0])], dtype=np.float32)


class Gateway:
    def __init__(self, recipe):
        self.calls, self.invalid, self.uncertain = 0, False, False
        self.routes = SimpleNamespace(account_model_metadata_sha256=recipe.model_metadata_sha256,
                                      generation_model_id=recipe.model_id)

    def generate(self, context, **kwargs):
        self.calls += 1
        if self.uncertain:
            raise TimeoutError("synthetic provider failure: must not reach API response")
        source = json.loads(kwargs["prompt"])["untrusted_sources"][0]
        raw = {"answer_status": "ANSWERED", "reason": "NONE", "claims": [{
            "text": source["untrusted_text"], "evidence": [{"source_id": source["source_id"],
            "quote": source["untrusted_text"]}]}]}
        return SimpleNamespace(operation_id="fixture-operation", text="invalid JSON" if self.invalid else json.dumps(raw),
            stop_reason="end_turn", usage=TokenUsage(input_tokens=100, output_tokens=10), actual_micro_usd=42)


@pytest.fixture
def setup_api(loaded):
    metadata, state = MemoryLedgerStore(), MemoryQueryStore()
    clock = [NOW]
    for user in (USER_A, USER_B):
        put(metadata, {"PK": "USER#" + user, "SK": "MEMBERSHIP", "record_type": "MEMBERSHIP",
            "schema_version": 1, "tenant_id": loaded.manifest.tenant_id, "status": "ACTIVE", "roles": ["reviewer"]})
    release = {"PK": "TENANT#" + loaded.manifest.tenant_id, "SK": "SNAPSHOT#" + loaded.manifest.snapshot_version,
        "record_type": "SNAPSHOT_RELEASE", "schema_version": 1, "tenant_id": loaded.manifest.tenant_id,
        "snapshot_version": loaded.manifest.snapshot_version, "manifest_sha256": loaded.manifest_sha256,
        "publication_status": "PUBLISHED", "revoked": False, "valid_for_history": True,
        "published_at": NOW.isoformat(), "asset_config": {"tenant_id": loaded.manifest.tenant_id,
            "snapshot_version": loaded.manifest.snapshot_version, "snapshot_manifest_sha256": loaded.manifest_sha256}}
    put(metadata, release)
    put(metadata, {"PK": release["PK"], "SK": "RELEASE", "record_type": "RELEASE_POINTER", "schema_version": 1,
        "tenant_id": loaded.manifest.tenant_id, "snapshot_version": loaded.manifest.snapshot_version,
        "manifest_sha256": loaded.manifest_sha256, "generation": 1})
    recipe = GenerationRecipe(model_metadata_sha256="0" * 64)
    gateway = Gateway(recipe)
    repo = QueryRepository(state, clock=lambda: clock[0])
    service = QueryService(repo, ReleaseRegistry(metadata), lambda _: (Encoder(), loaded), lambda timeout: gateway, recipe,
                           authorization_check=lambda auth: require_current_membership(auth, metadata))
    return SimpleNamespace(api=QueryApi(service, metadata, PUBLIC), service=service, repo=repo, state=state,
        metadata=metadata, release=release, gateway=gateway, clock=clock,
        auth=AuthContext(loaded.manifest.tenant_id, USER_A))


def event(method="POST", path="/v1/queries", *, user=USER_A, body=None, key=None, params=None):
    return {"requestContext": {"http": {"method": method}, "authorizer": {"jwt": {"claims": {
        "sub": user, "exp": "9999999999", "iss": PUBLIC["issuer"], "client_id": PUBLIC["client_id"],
        "scope": "kg/read kg/query", "token_use": "access"}}}}, "rawPath": path,
        "headers": {"content-type": "application/json", "idempotency-key": key or str(uuid4())},
        "body": json.dumps(body or {"question": "How does Service A use Service B?"}),
        "queryStringParameters": params}


def call(setup, request, remaining=23):
    raw = setup.api.handler(request, remaining_seconds=lambda: remaining)
    return raw["statusCode"], json.loads(raw["body"])


def test_only_valid_status_enrolls_and_does_not_consume_query_or_model_budget(setup_api):
    s = setup_api
    new_user = "33333333-3333-4333-8333-333333333333"
    called = []
    def provision(sub, *, cognito_username):
        called.append(sub)
        put(s.metadata, {"PK": "USER#" + sub, "SK": "MEMBERSHIP", "record_type": "MEMBERSHIP",
            "schema_version": 1, "tenant_id": s.auth.tenant_id, "status": "ACTIVE", "roles": ["reviewer"]})
    s.api.enroll_member = provision
    assert call(s, event("GET", "/v1/status", user=new_user, params={"tenant": "attacker"}))[0] == 400
    assert call(s, event("GET", "/v1/queries", user=new_user))[0] == 403
    assert call(s, event(user=new_user))[0] == 403
    assert called == []
    assert call(s, event("GET", "/v1/status", user=new_user))[0] == 200
    assert called == [new_user]
    assert call(s, event("GET", "/v1/queries", user=new_user))[1]["items"] == []
    assert s.repo.quota_remaining() == 300 and s.gateway.calls == 0


def test_query_receipt_replay_history_and_source_are_owner_bound(setup_api):
    s = setup_api
    request = event()
    status, data = call(s, request)
    assert status == 200 and data["status"] == "COMPLETED" and not data["idempotent_replay"]
    rid = data["request_id"]
    assert data["response"]["request_id"] == rid and s.gateway.calls == 1
    stored = s.repo.get(s.auth, rid)
    assert stored["receipt"]["raw_output"] and stored["receipt"]["original_cost_micro_usd"] == 42
    assert call(s, request)[1]["idempotent_replay"] and s.gateway.calls == 1 and s.repo.quota_remaining() == 299
    assert call(s, event("GET", "/v1/queries"))[1]["items"][0]["request_id"] == rid
    assert call(s, event("GET", "/v1/queries", user=USER_B))[1]["items"] == []
    assert call(s, event("GET", "/v1/queries/" + rid, user=USER_B))[0] == 404
    chunk_id = data["response"]["citations"][0]["chunk_id"]
    path = "/v1/sources/" + chunk_id
    assert call(s, event("GET", path, params={"request_id": rid}))[1]["text"]
    assert call(s, event("GET", path, user=USER_B, params={"request_id": rid}))[0] == 404
    other = next(c for c in s.service.runtime_loader(s.release)[1].chunks if c not in stored["allowed_chunk_ids"])
    assert call(s, event("GET", "/v1/sources/" + other, params={"request_id": rid}))[0] == 404
    assert "raw_output" not in json.dumps(data) and "receipt" not in json.dumps(data)


@pytest.mark.parametrize("bad", ["missing", "id_token", "issuer", "client", "expired", "scope", "membership", "foreign_tenant"])
def test_auth_rejections_happen_before_admission_and_provider(setup_api, bad):
    s, request = setup_api, event()
    claims = request["requestContext"]["authorizer"]["jwt"]["claims"]
    if bad == "missing":
        request["requestContext"].pop("authorizer")
    elif bad == "membership":
        s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]["status"] = "DISABLED"
    elif bad == "foreign_tenant":
        s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]["tenant_id"] = "other"
    else:
        field, value = {"id_token": ("token_use", "id"), "issuer": ("iss", "other"),
            "client": ("client_id", "other"), "expired": ("exp", "1"), "scope": ("scope", "kg/read")}[bad]
        claims[field] = value
    assert call(s, request)[0] in {401, 403, 503} and s.gateway.calls == 0
    assert s.repo.quota_remaining() == 300


@pytest.mark.parametrize("bad", ["extra", "graph", "long", "empty", "duplicate_json", "nan", "oversize", "key", "content_type", "base64"])
def test_bad_input_rejected_without_model_call(setup_api, bad):
    s, request = setup_api, event()
    if bad in {"extra", "graph", "long", "empty"}:
        payload = {"question": "Question"}
        payload.update({"extra": {"tenant_id": "forged"}, "graph": {"mode": "graph_2hop"},
                        "long": {"question": "x" * 2001}, "empty": {"question": " "}}[bad])
        request["body"] = json.dumps(payload)
    elif bad in {"duplicate_json", "nan", "oversize"}:
        request["body"] = {"duplicate_json": '{"question":"a","question":"b"}', "nan": '{"question":NaN}',
                           "oversize": " " * 17000}[bad]
    elif bad == "key":
        request["headers"]["idempotency-key"] = "not-a-uuid"
    elif bad == "content_type":
        request["headers"]["content-type"] = "text/plain"
    else:
        request["isBase64Encoded"] = True
    assert call(s, request)[0] == 400 and s.gateway.calls == 0 and s.repo.quota_remaining() == 300


def test_atomic_duplicate_and_monthly_299_boundary(setup_api):
    s = setup_api
    key = str(uuid4())
    payload = {"question": "same", "mode": "hybrid"}
    with ThreadPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(lambda _: s.repo.admit(s.auth, key, payload, s.release, "0" * 64), range(8)))
    assert sum(fresh for _, fresh in rows) == 1 and s.repo.quota_remaining() == 299
    with pytest.raises(ApiError, match="IDEMPOTENCY_CONFLICT"):
        s.repo.admit(s.auth, key, payload | {"question": "different"}, s.release, "0" * 64)
    quota_key = s.repo.quota_key("2026-10")
    s.state._items[quota_key]["count"] = 299

    def admit_new(_):
        try:
            return s.repo.admit(s.auth, str(uuid4()), payload, s.release, "0" * 64)[1]
        except ApiError as exc:
            assert exc.code in {"QUERY_QUOTA_EXCEEDED", "STATE_BUSY"}
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(admit_new, range(8))) == 1
    assert s.repo.quota_remaining() == 0
    s.clock[0] = datetime(2026, 11, 1, tzinfo=UTC)
    assert s.repo.quota_remaining() == 300  # Project-wide UTC month, not per user.


@pytest.mark.parametrize("kind", ["invalid", "uncertain", "deadline"])
def test_failure_is_durable_and_same_key_never_dispatches_again(setup_api, kind):
    s, request = setup_api, event()
    s.gateway.invalid, s.gateway.uncertain = kind == "invalid", kind == "uncertain"
    code, data = call(s, request, remaining=1 if kind == "deadline" else 23)
    assert code in {502, 503, 504}
    row = s.repo.get(s.auth, data["request_id"])
    assert row["status"] == "FAILED"
    if kind == "invalid":
        assert row["receipt"]["raw_output"] == "invalid JSON"
    calls = s.gateway.calls
    assert call(s, request)[1]["status"] == "FAILED" and s.gateway.calls == calls
    assert s.repo.quota_remaining() == 299
    assert "synthetic provider failure" not in json.dumps(data)


def test_revocation_blocks_response_source_and_history(setup_api):
    s = setup_api
    _, data = call(s, event())
    release_key = s.release["PK"], s.release["SK"]
    s.metadata._items[release_key]["revoked"] = True
    rid, chunk_id = data["request_id"], data["response"]["citations"][0]["chunk_id"]
    assert call(s, event("GET", "/v1/queries/" + rid))[0] == 503
    assert call(s, event("GET", "/v1/sources/" + chunk_id, params={"request_id": rid}))[0] == 503
    assert call(s, event("GET", "/v1/queries"))[1]["items"] == []
    assert s.gateway.calls == 1


def test_cursor_cross_owner_expiry_tampering_and_ttl_validity(setup_api):
    s = setup_api
    for _ in range(3):
        call(s, event())
        s.clock[0] += timedelta(seconds=1)
    _, first = call(s, event("GET", "/v1/queries", params={"limit": "1"}))
    cursor = first["next_cursor"]
    assert cursor
    assert call(s, event("GET", "/v1/queries", user=USER_B, params={"cursor": cursor}))[0] == 400
    value = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    value["SK"] = "REQUEST#arbitrary"
    bad = base64.urlsafe_b64encode(canonical_json(value)).decode().rstrip("=")
    assert call(s, event("GET", "/v1/queries", params={"cursor": bad}))[0] == 400
    s.clock[0] += timedelta(hours=2)
    assert call(s, event("GET", "/v1/queries", params={"cursor": cursor}))[0] == 400
    s.clock[0] += timedelta(days=31)
    assert call(s, event("GET", "/v1/queries"))[1]["items"] == []


def test_processing_timeout_and_expired_idempotency_are_not_resumed(setup_api):
    s, key = setup_api, str(uuid4())
    row, _ = s.repo.admit(s.auth, key, {"question": "x", "mode": "hybrid"}, s.release, "0" * 64)
    s.clock[0] += timedelta(seconds=30)
    assert s.repo.get(s.auth, row["request_id"])["error_code"] == "RECONCILIATION_REQUIRED"
    s.clock[0] += timedelta(days=8)
    with pytest.raises(ApiError, match="IDEMPOTENCY_KEY_EXPIRED"):
        s.repo.admit(s.auth, key, {"question": "x", "mode": "hybrid"}, s.release, "0" * 64)
    assert s.gateway.calls == 0


def test_corrupt_response_is_not_replayed(setup_api):
    s, request = setup_api, event()
    _, data = call(s, request)
    key = s.auth.partition, "REQUEST#" + data["request_id"]
    s.state._items[key]["response"]["answer"] = "forged answer"
    assert call(s, request)[0] == 503 and s.gateway.calls == 1


def test_owner_and_quota_are_not_selected_by_body(setup_api):
    s = setup_api
    bad = event(body={"question": "x", "owner_id": USER_B, "tenant_id": "other"})
    assert call(s, bad)[0] == 400
    assert call(s, event("GET", "/v1/status"))[1]["quota_remaining"] == 300
    assert s.gateway.calls == 0


def test_instrumentation_correlates_without_payload_or_extra_dispatch(setup_api):
    from app.query_telemetry import QueryTelemetry

    s, rows = setup_api, []
    trace = QueryTelemetry(invocation_id="11111111-2222-3333-4444-555555555555", sink=rows.append)
    s.api.telemetry = s.service.telemetry = trace
    request = event(body={"question": "PRIVATE_QUESTION_CANARY_71a9"})
    status, data = call(s, request)
    assert status == 200 and data["status"] == "COMPLETED" and s.gateway.calls == 1
    stages = {r.get("stage") for r in rows}
    assert {"authorize", "admission", "runtime_load", "query_embedding", "retrieval_context", "model_call",
            "receipt_write", "answer_validation", "final_write"} <= stages
    receipt = next(r for r in rows if r["event"] == "query_model_receipt")
    assert receipt["input_tokens"] == 100 and receipt["output_tokens"] == 10 and receipt["cost_micro_usd"] == 42
    assert rows[-1]["answer_status"] == "ANSWERED" and rows[-1]["request_id"] == data["request_id"]
    wire = json.dumps(rows)
    assert "PRIVATE_QUESTION_CANARY_71a9" not in wire and USER_A not in wire
    assert all(key not in wire for key in ("raw_output", "untrusted_sources", "Authorization", "tenant_id", "user_id"))
    before = s.gateway.calls
    s.api.telemetry = s.service.telemetry = QueryTelemetry(sink=rows.append)
    assert call(s, request)[1]["idempotent_replay"] is True and s.gateway.calls == before
    assert not any(r["event"] == "query_stage" and r.get("stage") == "model_call" for r in rows[-4:])


@pytest.mark.parametrize("when", ["during_runtime", "during_gateway_setup", "after_provider", "during_source_read"])
def test_membership_revoked_in_flight_blocks_dispatch_or_response(setup_api, when):
    s = setup_api
    member = s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]
    request = event()
    if when == "during_source_read":
        _, completed = call(s, request)
        request = event("GET", "/v1/sources/" + completed["response"]["citations"][0]["chunk_id"],
                        params={"request_id": completed["request_id"]})
    if when in {"during_runtime", "during_source_read"}:
        original = s.service.runtime_loader

        def revoke(release):
            result = original(release)
            member["status"] = "DISABLED"
            return result

        s.service.runtime_loader = revoke
    elif when == "during_gateway_setup":
        def revoke(timeout):
            member["status"] = "DISABLED"
            return s.gateway

        s.service.gateway_factory = revoke
    else:
        original = s.gateway.generate

        def revoke(*args, **kwargs):
            result = original(*args, **kwargs)
            member["status"] = "DISABLED"
            return result

        s.gateway.generate = revoke
    code, data = call(s, request)
    assert code == 403 and data["error"]["code"] == "ACTIVE_MEMBERSHIP_REQUIRED"
    assert not any(k in data for k in ("response", "text", "question"))
    assert s.gateway.calls == (0 if when in {"during_runtime", "during_gateway_setup"} else 1)
    if when != "during_source_read":
        row = s.repo.get(s.auth, data["request_id"])
        assert row["status"] == "FAILED" and ("receipt" in row) == (when == "after_provider")
        member["status"] = "ACTIVE"
        before = s.gateway.calls
        assert call(s, request)[1]["status"] == "FAILED" and s.gateway.calls == before


@pytest.mark.parametrize("roles", ["not-a-reviewer", {"reviewer": False}, None, ["not-reviewer"]])
def test_malformed_membership_roles_fail_closed(setup_api, roles):
    s = setup_api
    s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]["roles"] = roles
    assert call(s, event())[0] == 403 and s.gateway.calls == 0


@pytest.mark.parametrize("change", ["tenant", "release", "membership_store"])
def test_policy_change_during_cold_load_prevents_model_dispatch(setup_api, change):
    s = setup_api
    loader = s.service.runtime_loader

    def changed(release):
        result = loader(release)
        if change == "tenant":
            s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]["tenant_id"] = "other"
        elif change == "release":
            s.metadata._items[(release["PK"], release["SK"])]["revoked"] = True
        else:
            original = s.metadata.get

            def get(key):
                if key[0].startswith("USER#"):
                    raise OSError("SYNTHETIC_PRIVATE_STORE_ERROR")
                return original(key)

            s.metadata.get = get
        return result

    s.service.runtime_loader = changed
    code, data = call(s, event())
    assert code == (403 if change == "tenant" else 503) and s.gateway.calls == 0
    assert s.repo.get(s.auth, data["request_id"])["status"] == "FAILED"
    assert "SYNTHETIC_PRIVATE_STORE_ERROR" not in json.dumps(data)


def test_revocation_after_provider_keeps_receipt_but_never_exposes_answer(setup_api):
    s, request = setup_api, event()
    generate = s.gateway.generate

    def revoke(*args, **kwargs):
        result = generate(*args, **kwargs)
        s.metadata._items[(s.release["PK"], s.release["SK"])]["revoked"] = True
        return result

    s.gateway.generate = revoke
    code, data = call(s, request)
    assert code == 503 and data["error"]["code"] == "SNAPSHOT_UNAVAILABLE" and "response" not in data
    row = s.repo.get(s.auth, data["request_id"])
    assert row["status"] == "FAILED" and row["receipt"]["original_cost_micro_usd"] == 42
    assert s.gateway.calls == 1


def test_deadline_consumed_by_runtime_stops_before_gateway_setup(setup_api):
    s = setup_api
    remaining = iter([23, 6])
    setups = []
    s.service.gateway_factory = lambda timeout: setups.append(timeout)
    raw = s.api.handler(event(), remaining_seconds=lambda: next(remaining))
    assert raw["statusCode"] == 504 and s.gateway.calls == 0 and setups == []
    assert s.repo.quota_remaining() == 299  # An admitted attempt still counts.


def test_concurrent_identical_http_intents_dispatch_one_provider_call(setup_api):
    s, request = setup_api, event()
    entered, finish = threading.Event(), threading.Event()
    generate = s.gateway.generate

    def blocking(*args, **kwargs):
        entered.set()
        assert finish.wait(5)
        return generate(*args, **kwargs)

    s.gateway.generate = blocking
    with ThreadPoolExecutor(max_workers=6) as pool:
        first = pool.submit(call, s, request)
        assert entered.wait(5)
        try:
            duplicates = list(pool.map(lambda _: call(s, request), range(5)))
            assert all(code == 409 and data["error"]["code"] == "REQUEST_IN_PROGRESS"
                       for code, data in duplicates)
        finally:
            finish.set()
        assert first.result()[1]["status"] == "COMPLETED"
    assert s.gateway.calls == 1 and s.repo.quota_remaining() == 299


def test_membership_recheck_accepts_real_dynamodb_string_set_roles(setup_api):
    s = setup_api
    s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]["roles"] = {"reviewer"}
    assert call(s, event())[0] == 200 and s.gateway.calls == 1


def test_gateway_setup_consumes_deadline_and_cannot_start_a_late_call(setup_api):
    s, remaining = setup_api, [23]

    def slow_gateway(timeout):
        remaining[0] = 1
        return s.gateway

    s.service.gateway_factory = slow_gateway
    raw = s.api.handler(event(), remaining_seconds=lambda: remaining[0])
    assert raw["statusCode"] == 504 and s.gateway.calls == 0
