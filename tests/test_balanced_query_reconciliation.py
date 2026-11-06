"""Offline guards for the QB audit; no provider, encoder or AWS requests."""

import copy
import json
import sqlite3

import pytest

from app.contracts import Record, canonical_json, sha256_bytes, stable_id
from app.model_gateway import CallContext
from app.query_planning import PlannerRecipe, QueryPlanError


@pytest.fixture
def audit():
    # Import only once the real driver exists; never replace its check_pair with a stub.
    from scripts import reconcile_balanced_query_pilot

    return reconcile_balanced_query_pilot


class SyntheticReceipt(Record):
    namespace: str
    cache_key: str


def _save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = canonical_json(value)
    path.write_bytes(raw)
    return sha256_bytes(raw)


def _cache(path, *, state="READY", intent=None, saved=None, digest=None, context="context"):
    intent = {"context": "synthetic"} if intent is None else intent
    raw = canonical_json({"namespace": "LIVE", "cache_key": "key"} if saved is None else saved)
    with sqlite3.connect(path) as db:
        for table in ("answers", "query_plans", "rewrites"):
            db.execute(f"CREATE TABLE {table} (namespace TEXT,cache_key TEXT,state TEXT,intent BLOB,"
                       "payload BLOB,payload_sha256 TEXT,context_key TEXT)")
            db.execute(f"INSERT INTO {table} VALUES (?,?,?,?,?,?,?)", ("LIVE", "key", state,
                canonical_json(intent), raw, sha256_bytes(raw) if digest is None else digest, context))
    return path


def test_json_guard_reads_exact_bytes_and_digest(audit, tmp_path):
    path = tmp_path / "receipt.json"
    raw = b'{"question":"AWS", "status":"READY"}\n'
    path.write_bytes(raw)
    value, digest = audit.read_json(path, maximum=len(raw))
    assert value == {"question": "AWS", "status": "READY"}
    assert digest == sha256_bytes(raw)


@pytest.mark.parametrize("kind", ["missing", "directory", "empty", "oversized", "invalid"])
def test_json_guard_rejects_missing_empty_large_and_invalid(audit, tmp_path, kind):
    path = tmp_path / "input.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "empty":
        path.write_bytes(b"")
    elif kind == "oversized":
        path.write_bytes(b'{"extra":true}')
    elif kind == "invalid":
        path.write_bytes(b"xx")
    with pytest.raises((ValueError, json.JSONDecodeError)):
        audit.read_json(path, maximum=2)


def test_json_and_connection_refuse_symlinks(audit, tmp_path):
    target = tmp_path / "actual.json"
    target.write_bytes(b"{}")
    alias = tmp_path / "alias.json"
    try:
        alias.symlink_to(target)
    except OSError:
        pytest.skip("Windows account cannot create symlinks")
    with pytest.raises(ValueError, match="UNSAFE"):
        audit.read_json(alias)
    with pytest.raises(ValueError, match="LOCAL_CACHE_PATH"):
        audit.connection(alias)


def test_sqlite_connection_is_read_only_and_does_not_create_missing_database(audit, tmp_path):
    path = _cache(tmp_path / "cache.sqlite3")
    original = path.read_bytes()
    with audit.connection(path) as db:
        assert db.execute("SELECT count(*) FROM answers").fetchone()[0] == 1
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM answers")
    assert path.read_bytes() == original
    missing = tmp_path / "missing.sqlite3"
    with pytest.raises(ValueError, match="LOCAL_CACHE_PATH"):
        audit.connection(missing)
    assert not missing.exists()


@pytest.mark.parametrize("table", ["answers", "query_plans", "rewrites"])
def test_cache_receipt_requires_ready_exact_intent_context_checksum_and_identity(audit, tmp_path, table):
    path = _cache(tmp_path / "cache.sqlite3")
    original = path.read_bytes()
    with audit.connection(path) as db:
        saved = audit.cached_row(db, table, "key", {"context": "synthetic"}, SyntheticReceipt,
                                 context_key="context")
    assert saved.model_dump() == {"namespace": "LIVE", "cache_key": "key"}
    assert path.read_bytes() == original


@pytest.mark.parametrize("changes,expected", [
    ({"state": "FAILED"}, "STATE_OR_INTENT"),
    ({"state": "IN_PROGRESS"}, "STATE_OR_INTENT"),
    ({"intent": {"context": "different"}}, "STATE_OR_INTENT"),
    ({"digest": "f" * 64}, "PAYLOAD_CHECKSUM"),
    ({"context": "different"}, "CONTEXT_KEY"),
    ({"saved": {"namespace": "FIXTURE", "cache_key": "key"}}, "NAMESPACE_OR_KEY"),
    ({"saved": {"namespace": "LIVE", "cache_key": "other"}}, "NAMESPACE_OR_KEY"),
])
def test_cache_guard_rejects_forgery_or_unsettled_state(audit, tmp_path, changes, expected):
    path = _cache(tmp_path / "cache.sqlite3", **changes)
    with audit.connection(path) as db, pytest.raises(ValueError, match=expected):
        audit.cached_row(db, "query_plans", "key", {"context": "synthetic"}, SyntheticReceipt,
                         context_key="context")


def test_cache_table_allowlist_rejects_interpolation_before_query(audit):
    class NeverQuery:
        def execute(self, *args):
            raise AssertionError("invalid table must never execute SQL")

    with pytest.raises(ValueError, match="CACHE_TABLE"):
        audit.cached_row(NeverQuery(), "answers; DROP TABLE answers", "key", {}, SyntheticReceipt)


def test_cache_guard_rejects_missing_or_oversized_payload(audit, tmp_path):
    path = _cache(tmp_path / "cache.sqlite3")
    with audit.connection(path) as db, pytest.raises(ValueError, match="STATE_OR_INTENT"):
        audit.cached_row(db, "answers", "missing", {"context": "synthetic"}, SyntheticReceipt)
    raw = b"x" * (128 * 1024 + 1)
    with sqlite3.connect(path) as db:
        db.execute("UPDATE answers SET payload=?,payload_sha256=?", (raw, sha256_bytes(raw)))
    with audit.connection(path) as db, pytest.raises(ValueError, match="PAYLOAD_CHECKSUM"):
        audit.cached_row(db, "answers", "key", {"context": "synthetic"}, SyntheticReceipt)


def test_expectation_binds_shared_planner_to_operation_context_and_body_hash(audit):
    context = CallContext(tenant_id="demo", run_id="synthetic-run", request_id="synthetic-q-family", stage="PLAN_QUERY")
    raw = audit.body("system", "prompt", 256)
    receipt = {"operation_id": stable_id("op", **context.model_dump(mode="json")),
               "cost_micro_usd": 42, "usage": {"input_tokens": 100, "output_tokens": 10}}
    result = audit.expectation(context, raw, receipt, case=1, method="shared_q_family", output=256)
    assert result["input_sha256"] == sha256_bytes(raw)
    assert result["context"] == context.model_dump(mode="json")
    assert result["method"] == "shared_q_family" and result["output_bound"] == 256
    with pytest.raises(ValueError, match="RECEIPT_OPERATION_ID"):
        audit.expectation(context, raw, receipt | {"operation_id": "different-op"},
                          case=1, method="shared_q_family", output=256)


@pytest.mark.parametrize("tamper", [None, "raw_sha", "operation", "recipe"])
def test_saved_ready_planner_still_requires_raw_output_context_and_recipe_binding(audit, tmp_path, tamper):
    recipe = PlannerRecipe()
    context = CallContext(tenant_id="demo", run_id="synthetic-run", request_id="synthetic-plan", stage="PLAN_QUERY")
    raw = canonical_json({"subqueries": ["Service A configuration"]})
    provider = {"operation_id": stable_id("op", **context.model_dump(mode="json")),
        "raw_output": raw.decode(), "raw_output_sha256": sha256_bytes(raw), "raw_output_bytes": len(raw),
        "raw_output_complete": True, "usage": {"input_tokens": 100, "output_tokens": 10},
        "cost_micro_usd": 42, "stop_reason": "end_turn", "prompt_version": recipe.prompt_version,
        "recipe_sha256": recipe.fingerprint}
    if tamper == "raw_sha":
        provider["raw_output_sha256"] = "f" * 64
    elif tamper == "operation":
        provider["operation_id"] = "other-operation"
    elif tamper == "recipe":
        provider["recipe_sha256"] = "f" * 64
    saved = {"schema_version": 1, "namespace": "LIVE", "cache_key": "key",
             "binding_sha256": "0" * 64, "provider_receipt": provider}
    intent = {"context": context.model_dump(mode="json"), "synthetic": True}
    path = _cache(tmp_path / "cache.sqlite3", intent=intent, saved=saved)
    with audit.connection(path) as db:
        loaded = audit.cached_row(db, "query_plans", "key", intent, audit.PlannerReceipt,
                                  context_key="context")
    if tamper is None:
        assert audit.parse_saved_plan(loaded.provider_receipt, context, "Service B", recipe) == ("Service A configuration",)
    else:
        # A self-consistent SQLite payload checksum does not establish semantic receipt binding.
        with pytest.raises(QueryPlanError, match="QUERY_PLAN_"):
            audit.parse_saved_plan(loaded.provider_receipt, context, "Service B", recipe)


def _old_artifacts(root):
    for folder in ("reranker-answering", "multi-query"):
        directory = root / "results" / folder / "pilot-v1"
        cases = []
        for number in range(1, 4):
            digest = _save(directory / f"case-{number}.json", {"case": number, "family": folder})
            replay = _save(directory / f"case-{number}-replay.json", {"case": number, "cache_hit": True})
            cases.append({"case": number, "live_sha256": digest, "replay_sha256": replay})
        _save(directory / "reconciliation.json", {"status": "PASS_SOURCE_CACHE_LEDGER_REPLAY", "cases": cases})
        _save(directory / "plan.json", {"plan": folder})
        _save(directory / "summary.json", {"status": "COMPLETED"})


def test_previous_artifacts_capture_both_method_families_and_detect_changed_snapshot(audit, tmp_path, monkeypatch):
    _old_artifacts(tmp_path)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    first = audit.old_r_hashes()
    assert len(first) == 18
    assert {name.split("/")[0] for name in first} == {"R", "Q"}
    path = tmp_path / "results/multi-query/pilot-v1/summary.json"
    _save(path, {"status": "CHANGED_AFTER_CAPTURE"})
    assert audit.old_r_hashes() != first


@pytest.mark.parametrize("family", ["reranker-answering", "multi-query"])
def test_previous_live_hash_must_match_earlier_verified_audit(audit, tmp_path, monkeypatch, family):
    _old_artifacts(tmp_path)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    _save(tmp_path / "results" / family / "pilot-v1/case-2.json", {"tampered": True})
    with pytest.raises(ValueError, match="PREVIOUS_LIVE_CHANGED"):
        audit.old_r_hashes()


def test_previous_family_must_have_verified_status(audit, tmp_path, monkeypatch):
    _old_artifacts(tmp_path)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "results/multi-query/pilot-v1/reconciliation.json"
    row = json.loads(path.read_bytes())
    _save(path, row | {"status": "BLOCKED_AUDIT_NO_RETRY"})
    with pytest.raises(ValueError, match="PREVIOUS_NOT_VERIFIED"):
        audit.old_r_hashes()


@pytest.mark.parametrize("family", ["reranker-answering", "multi-query"])
def test_previous_replay_hash_when_recorded_is_verified(audit, tmp_path, monkeypatch, family):
    _old_artifacts(tmp_path)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    _save(tmp_path / "results" / family / "pilot-v1/case-1-replay.json", {"tampered": True})
    with pytest.raises(ValueError, match="PREVIOUS_REPLAY_CHANGED"):
        audit.old_r_hashes()


def test_legacy_r_audit_without_replay_hash_still_captured_for_preexecution_pin(audit, tmp_path, monkeypatch):
    _old_artifacts(tmp_path)
    monkeypatch.setattr(audit, "ROOT", tmp_path)
    path = tmp_path / "results/reranker-answering/pilot-v1/reconciliation.json"
    original = json.loads(path.read_bytes())
    for row in original["cases"]:
        row.pop("replay_sha256")
    _save(path, original)
    pins = audit.old_r_hashes()
    assert len(pins) == 18
    assert pins["R/case-1-replay.json"] == sha256_bytes(
        (tmp_path / "results/reranker-answering/pilot-v1/case-1-replay.json").read_bytes())


def _counters(audit):
    common = {"record_type": "BUDGET_COUNTER", "project_id": "kg-r72", "halted": False,
              "settled_micro_usd": 1234, "reserved_micro_usd": 640, "version": 5,
              "gross_stop_micro_usd": None, "other_cost_estimate_micro_usd": 0,
              "tenant_id": None, "period": None}
    return {"run": common | {"PK": f"RUN#{audit.RUN}", "SK": "BUDGET", "cap_micro_usd": 500_000,
                             "tenant_id": "demo"},
        "funding": common | {"PK": "FUNDING#kg-r72#external-r72-v1", "SK": "BUDGET", "cap_micro_usd": 2_000_000},
        "month": common | {"PK": "BUDGET#kg-r72#2026-10", "SK": "MODEL", "cap_micro_usd": 12_000_000,
            "period": "2026-10", "reserved_micro_usd": 1250, "gross_stop_micro_usd": 15_000_000,
            "other_cost_estimate_micro_usd": 3_000_000}}


def test_baseline_requires_real_scopes_and_accepts_preserved_nonzero_reservations(audit):
    counters = _counters(audit)
    original = copy.deepcopy(counters)
    audit.validate_counter_baseline(counters, "2026-10-25T00:00:00+00:00")
    assert counters == original  # Validation cannot reset or refund earlier reservations.


@pytest.mark.parametrize("mutation", ["empty", "missing", "extra", "none", "run_scope", "funding_scope",
    "month_scope", "tenant", "cap", "halted", "negative", "bool", "fraction", "version", "over_cap", "gross"])
def test_counter_baseline_rejects_missing_changed_or_unsafe_scope(audit, mutation):
    counters = _counters(audit)
    if mutation == "empty":
        counters = {}
    elif mutation == "missing":
        counters.pop("month")
    elif mutation == "extra":
        counters["other"] = counters["month"]
    elif mutation == "none":
        counters["funding"] = None
    elif mutation == "run_scope":
        counters["run"]["PK"] = "RUN#another-run"
    elif mutation == "funding_scope":
        counters["funding"]["PK"] = "FUNDING#kg-r72#another-funding"
    elif mutation == "month_scope":
        counters["month"]["period"] = "2026-11"
    elif mutation == "tenant":
        counters["run"]["tenant_id"] = "other"
    elif mutation == "cap":
        counters["run"]["cap_micro_usd"] += 1
    elif mutation == "halted":
        counters["month"]["halted"] = True
    elif mutation == "negative":
        counters["run"]["reserved_micro_usd"] = -1
    elif mutation == "bool":
        counters["run"]["settled_micro_usd"] = True
    elif mutation == "fraction":
        counters["funding"]["reserved_micro_usd"] = 0.5
    elif mutation == "version":
        counters["month"]["version"] = None
    elif mutation == "over_cap":
        counters["run"]["settled_micro_usd"] = counters["run"]["cap_micro_usd"]
    else:
        counters["month"]["gross_stop_micro_usd"] = 100_000_000
    with pytest.raises(ValueError, match="QB_AUDIT_COUNTER"):
        audit.validate_counter_baseline(counters, "2026-10-25T00:00:00+00:00")
