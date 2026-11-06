import copy
import json

import pytest
from pydantic import ValidationError

from app.method_catalog import PublicMethodCatalog, deployment_method_catalog
from tests.test_query_api import USER_A, call, event
from tests.test_query_api import setup_api as setup_api


@pytest.mark.parametrize("official", [False, True])
def test_capabilities_match_deployed_service_and_do_not_advertise_local_results(official):
    data = deployment_method_catalog("snap_" + "a" * 64, official_search_available=official)
    methods = {r["method_id"]: r for r in data["methods"]}
    assert len(methods) == 8
    assert {k for k, v in methods.items() if v["state"] == "AVAILABLE"} == {"vector", "hybrid"}
    assert methods["vector"]["available_scopes"] == ["snapshot"]
    assert methods["hybrid"]["available_scopes"] == ["snapshot"] + (["aws_official"] if official else [])
    assert not data["collection_serving_enabled"] and not data["async_runs_enabled"]
    for method, row in methods.items():
        assert row["quality_metrics"] is None
        if row["state"] == "NOT_SERVED":
            assert row["available_scopes"] == [] and row["run_kinds"] == [] and row["execution_endpoint"] is None
            assert row["unavailable_reason"] == (
                "GRAPH_ANSWER_GATE_CLOSED" if method.startswith("graph_") else "COLLECTION_SERVING_NOT_ENABLED")
    data["methods"][0]["available_scopes"].append("snapshot")
    assert deployment_method_catalog("snap_x", official_search_available=official)["methods"][0]["available_scopes"] == []


@pytest.mark.parametrize("bad", [None, 0, 1, "true", [], {}])
def test_server_capability_requires_a_real_bool(bad):
    with pytest.raises(ValueError, match="CAPABILITY_TYPE"):
        deployment_method_catalog("snap_x", official_search_available=bad)


@pytest.mark.parametrize("bad", ["duplicate_id", "missing_scope", "missing_endpoint", "duplicate_scope",
                                "unserved_endpoint", "unserved_scope", "invented_score", "extra_secret"])
def test_catalog_contract_rejects_contradictions_and_private_fields(bad):
    data = deployment_method_catalog("snap_x", official_search_available=False)
    lexical, dense = data["methods"][:2]
    if bad == "duplicate_id":
        lexical["method_id"] = dense["method_id"]
    elif bad == "missing_scope":
        dense["available_scopes"] = []
    elif bad == "missing_endpoint":
        dense["execution_endpoint"] = None
    elif bad == "duplicate_scope":
        dense["available_scopes"] *= 2
    elif bad == "unserved_endpoint":
        lexical["execution_endpoint"] = "/v1/queries"
    elif bad == "unserved_scope":
        lexical["available_scopes"] = ["snapshot"]
    elif bad == "invented_score":
        dense["quality_metrics"] = {"accuracy": 1}
    else:
        data["secret_arn"] = "fixture"
    with pytest.raises(ValidationError):
        PublicMethodCatalog.model_validate(data)


def test_authenticated_catalog_reads_published_release_without_compute_writes_or_admission(setup_api, monkeypatch):
    s = setup_api
    def forbidden(*args, **kwargs):
        pytest.fail("Catalog must not load runtime, make model calls, enroll or admit requests")
    monkeypatch.setattr(s.service, "runtime_loader", forbidden)
    monkeypatch.setattr(s.service, "gateway_factory", forbidden)
    monkeypatch.setattr(s.service.repo, "admit", forbidden)
    s.api.enroll_member = forbidden
    state, metadata = copy.deepcopy(s.state._items), copy.deepcopy(s.metadata._items)
    for official in (False, True):
        s.service.knowledge_factory = forbidden if official else None
        raw = s.api.handler(event("GET", "/v1/rag-methods"), remaining_seconds=lambda: 1)
        assert raw["statusCode"] == 200 and raw["headers"]["cache-control"] == "no-store"
        data = json.loads(raw["body"])
        assert data["active_snapshot_version"] == s.release["snapshot_version"]
        assert "aws_official" in data["methods"][2]["available_scopes"] if official else (
            data["methods"][2]["available_scopes"] == ["snapshot"])
        assert not {"tenant_id", "user_id", "secret_arn", "raw_output", "model_id"} & set(data)
    assert s.state._items == state and s.metadata._items == metadata
    assert s.repo.quota_remaining() == 300 and s.gateway.calls == 0


@pytest.mark.parametrize("bad", ["missing", "id_token", "issuer", "client", "expired", "scope", "membership"])
def test_catalog_requires_read_scope_and_current_membership(setup_api, bad):
    s, request = setup_api, event("GET", "/v1/rag-methods")
    claims = request["requestContext"]["authorizer"]["jwt"]["claims"]
    if bad == "missing":
        request["requestContext"].pop("authorizer")
    elif bad == "membership":
        s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]["status"] = "DISABLED"
    else:
        key, value = {"id_token": ("token_use", "id"), "issuer": ("iss", "other"),
            "client": ("client_id", "other"), "expired": ("exp", "1"), "scope": ("scope", "kg/query")}[bad]
        claims[key] = value
    assert call(s, request)[0] in {401, 403}
    assert s.repo.quota_remaining() == 300 and s.gateway.calls == 0 and not s.state._items


def test_catalog_reauthorizes_after_read_and_does_not_enroll_unknown_users(setup_api, monkeypatch):
    s, observed = setup_api, []
    s.api.enroll_member = lambda *a, **k: observed.append(a)
    assert call(s, event("GET", "/v1/rag-methods", user="33333333-3333-4333-8333-333333333333"))[0] == 403
    assert observed == []
    current = s.service.releases.current
    def revoke(tenant):
        release = current(tenant)
        s.metadata._items[("USER#" + USER_A, "MEMBERSHIP")]["status"] = "DISABLED"
        return release
    monkeypatch.setattr(s.service.releases, "current", revoke)
    assert call(s, event("GET", "/v1/rag-methods"))[0] == 403


@pytest.mark.parametrize("params", [{"tenant_id": "other"}, {"enable_graph": "true"}, {"mode": "hybrid_multi_query"}])
def test_catalog_cannot_be_enabled_or_reconfigured_by_query_parameters(setup_api, params):
    assert call(setup_api, event("GET", "/v1/rag-methods", params=params))[0] == 400
    assert setup_api.gateway.calls == 0 and setup_api.repo.quota_remaining() == 300


def test_catalog_fails_closed_on_revoked_release_and_has_no_post_action(setup_api):
    s = setup_api
    assert call(s, event("POST", "/v1/rag-methods"))[0] == 404
    s.metadata._items[(s.release["PK"], s.release["SK"])]["revoked"] = True
    assert call(s, event("GET", "/v1/rag-methods"))[0] == 503
    assert s.gateway.calls == 0 and s.repo.quota_remaining() == 300


@pytest.mark.parametrize("mode", ["bm25", "hybrid_reranked", "hybrid_multi_query", "hybrid_multi_query_balanced",
                                 "graph_1hop", "graph_2hop"])
def test_catalog_never_opens_unsupported_serving_methods(setup_api, mode):
    assert call(setup_api, event("GET", "/v1/rag-methods"))[0] == 200
    status, data = call(setup_api, event(body={"question": "Service B", "mode": mode}))
    assert status == 400 and data["error"]["code"] == "MODE_UNAVAILABLE"
    assert setup_api.gateway.calls == 0 and setup_api.repo.quota_remaining() == 300
