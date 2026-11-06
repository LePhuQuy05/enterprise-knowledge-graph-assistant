"""Synthetic source/READY-receipt/ledger checks; no real answers or services."""

import copy
import json
import sqlite3
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from app.answer_cache import AnswerReceipt
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.generation import (
    GenerationRecipeV9,
    GenerationRecipeV11,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.ledger import ExternalOperationSpec, Operation
from app.model_gateway import CallContext
from scripts import reconcile_answer_limitations as audit
from tests.test_balanced_query_reconciliation import _counters
from tests.test_generation import setup


def _run_records():
    plan = {"version": audit.VERSION, "new_generation_call_cap": 6, "max_output_tokens": 800,
        "run_id": audit.RUN, "sequential_only": True, "same_spans_and_untrusted_payload": True,
        "new_retrieval_calls": 0, "new_encoder_calls": 0, "new_planner_calls": 0, "cases": [{}] * 6}
    marker = {"plan_sha256": "a" * 64, "resume_allowed": False, "created_at": "2026-10-25T00:00:00+00:00"}
    summary = {"status": "COMPLETED", "cost_complete": True, "error_category": None,
        "plan_sha256": "a" * 64, "cases": 6, "quality_metrics": None, "human_reviewed": False,
        "gold": False, "holdout": False, "cloud_deployed": False, "new_retrieval_calls": 0,
        "new_encoder_calls": 0, "new_planner_calls": 0, "dispatches_this_invocation": 6,
        "new_known_cost_micro_usd": 100, "created_at": "2026-10-25T00:10:00+00:00"}
    replay = summary | {"dispatches_this_invocation": 0, "new_known_cost_micro_usd": 0,
                        "created_at": "2026-10-25T00:11:00+00:00"}
    return plan, marker, summary, replay


def test_run_requires_six_new_generations_and_zero_cost_replay():
    records = _run_records()
    audit.validate_run_records(*records, "a" * 64)


@pytest.mark.parametrize("where,key,value", [
    (0, "new_generation_call_cap", 7), (0, "max_output_tokens", 1600), (0, "new_retrieval_calls", 1),
    (0, "same_spans_and_untrusted_payload", False), (1, "resume_allowed", True),
    (2, "dispatches_this_invocation", 5), (2, "cases", 5), (2, "cost_complete", False),
    (2, "gold", True), (2, "quality_metrics", {"accuracy": 1}), (2, "error_category", "TIMEOUT"),
    (3, "dispatches_this_invocation", 1), (3, "new_known_cost_micro_usd", 1),
    (3, "created_at", "2026-10-25T00:09:00+00:00"),
])
def test_run_rejects_hidden_work_incomplete_receipts_and_invalid_labels(where, key, value):
    records = _run_records()
    records[where][key] = value
    with pytest.raises(ValueError, match="LIMITATIONS_AUDIT_"):
        audit.validate_run_records(*records, "a" * 64)


def _ready_fixture(loaded, path, *, candidate=True, language="en", state="READY", abstain_reason=None):
    packed, binding, _ = setup(loaded)
    recipe = (GenerationRecipeV11 if candidate else GenerationRecipeV9)(model_metadata_sha256="0" * 64)
    context = CallContext(tenant_id=binding.tenant_id, run_id=audit.RUN,
                          request_id=binding.request_id, stage="GENERATE")
    data = {"answer_status": "PARTIAL" if candidate else "ANSWERED",
        "reason": "INSUFFICIENT_EVIDENCE" if candidate else "NONE", "claims": [{
            "text": "Service A uses Service B.", "evidence": [{"source_id": "s1", "span_id": "s1p1"}]}]}
    if candidate:
        data["limitations"] = [{"code": "ACCOUNT_STATE_UNAVAILABLE", "question_part": "My account configuration"}]
    if abstain_reason is not None:
        code = {"INSUFFICIENT_EVIDENCE": "MISSING_SUPPLIED_EVIDENCE",
                "CONFLICTING_EVIDENCE": "CONFLICTING_SUPPLIED_EVIDENCE", "OUT_OF_SCOPE": "OUT_OF_SCOPE"}[abstain_reason]
        data.update(answer_status="ABSTAINED", reason=abstain_reason, claims=[],
                    limitations=[{"code": code, "question_part": "Missing requested part"}])
    raw = canonical_json(data).decode()
    system, prompt = generation_request("How?", loaded, packed, binding.tenant_id, recipe, answer_language=language)
    binding_sha = sha256_bytes(canonical_json(binding.model_dump(mode="json")))
    key = stable_id("answer", context=context.model_dump(mode="json"), binding_sha=binding_sha,
        request_sha=sha256_bytes(canonical_json([system, prompt])), recipe=recipe.fingerprint,
        snapshot_sha=loaded.manifest_sha256)
    receipt = AnswerReceipt(namespace="LIVE", cache_key=key,
        operation_id=stable_id("op", **context.model_dump(mode="json")), recipe_sha256=recipe.fingerprint,
        binding_sha256=binding_sha, raw_output=raw, stop_reason="end_turn",
        usage=TokenUsage(input_tokens=100, output_tokens=50), original_cost_micro_usd=90)
    response = resolve_answer(parse_generated(raw, "end_turn", recipe), snapshot=loaded, packed=packed,
        binding=binding, recipe=recipe, answer_language=language).model_copy(update={
            "answer_language": language, "retrieval_backend": "collection_numpy_global_bm25"})
    intent = {"context": context.model_dump(mode="json"), "binding_sha256": binding_sha,
              "recipe_sha256": recipe.fingerprint}
    payload = receipt.model_dump_json().encode()
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE answers (namespace TEXT,cache_key TEXT,state TEXT,intent BLOB,"
                   "payload BLOB,payload_sha256 TEXT)")
        db.execute("INSERT INTO answers VALUES (?,?,?,?,?,?)", ("LIVE", key, state, canonical_json(intent),
                   payload, sha256_bytes(payload)))
    result = {"receipt": receipt.model_dump(mode="json"), "response": response.model_dump(mode="json")}
    return dict(view=loaded, packed=packed, binding=binding, context=context, question="How?",
                language=language, recipe=recipe, result=result)


@pytest.mark.parametrize("candidate", [False, True])
@pytest.mark.parametrize("language", ["en", "vi"])
def test_ready_raw_reconstruction_matches_both_versions_and_localized_templates(loaded, tmp_path, candidate, language):
    path = tmp_path / "answers.sqlite3"
    params = _ready_fixture(loaded, path, candidate=candidate, language=language)
    before = path.read_bytes()
    with audit.connection(path) as db:
        receipt, _, _, response = audit.rebuild_cached_answer(db, **params)
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            db.execute("DELETE FROM answers")
    assert receipt.namespace == "LIVE" and path.read_bytes() == before
    assert response.citations[0].quote == "Service A uses Service B."
    assert (response.assessment is not None) == candidate
    if candidate:
        assert response.assessment.limitations[0].message == audit.MESSAGES[language]["ACCOUNT_STATE_UNAVAILABLE"][0]
        assert response.assessment.limitations[0].message not in response.answer
    else:
        assert "assessment" not in response.model_dump(mode="json")


@pytest.mark.parametrize("reason", ["INSUFFICIENT_EVIDENCE", "CONFLICTING_EVIDENCE", "OUT_OF_SCOPE"])
def test_v11_vietnamese_abstention_keeps_reason_specific_text_instead_of_legacy_override(loaded, tmp_path, reason):
    from app.generation import ABSTENTION_TEXT_VI_V11

    path = tmp_path / "answers.sqlite3"
    params = _ready_fixture(loaded, path, language="vi", abstain_reason=reason)
    with audit.connection(path) as db:
        _, _, _, response = audit.rebuild_cached_answer(db, **params)
    assert response.answer == ABSTENTION_TEXT_VI_V11[reason]
    assert response.answer_status == "ABSTAINED" and not response.claims and not response.citations


@pytest.mark.parametrize("state", ["FAILED", "IN_PROGRESS"])
def test_even_valid_saved_raw_cannot_make_an_unready_cache_entry_pass(loaded, tmp_path, state):
    path = tmp_path / "answers.sqlite3"
    params = _ready_fixture(loaded, path, state=state)
    with audit.connection(path) as db, pytest.raises(ValueError, match="STATE_OR_INTENT"):
        audit.rebuild_cached_answer(db, **params)


@pytest.mark.parametrize("tamper", ["response", "template", "origin", "raw_receipt", "source", "owner", "request", "checksum"])
def test_reconstruction_rejects_changed_claims_trusted_templates_receipts_and_scope(loaded, tmp_path, tamper):
    path = tmp_path / "answers.sqlite3"
    params = _ready_fixture(loaded, path)
    if tamper == "response":
        params["result"]["response"]["answer"] = "Account is correct."
    elif tamper == "template":
        params["result"]["response"]["assessment"]["limitations"][0]["next_step"] = "Send your AWS secret."
    elif tamper == "origin":
        params["result"]["response"]["assessment"]["limitations"][0]["reported_by"] = "SYSTEM_POLICY"
    elif tamper == "raw_receipt":
        params["result"]["receipt"]["raw_output"] += " "
    elif tamper == "source":
        params["result"]["response"]["citations"][0]["quote"] = "Unrelated invented text."
    elif tamper == "owner":
        params["binding"] = params["binding"].model_copy(update={"owner_id": "another-owner"})
    elif tamper == "request":
        params["context"] = params["context"].model_copy(update={"request_id": "another-request"})
    else:
        with sqlite3.connect(path) as db:
            db.execute("UPDATE answers SET payload_sha256=?", ("f" * 64,))
    with audit.connection(path) as db, pytest.raises(ValueError):
        audit.rebuild_cached_answer(db, **params)


def _operation_fixture():
    routes = ExternalRoutes.model_validate_json((audit.ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((audit.ROOT / audit.PRICE_PATH).read_bytes())
    stamp = datetime(2026, 10, 25, 0, 5, tzinfo=UTC)
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id,
                           routes.source_region, now=stamp, live=True)
    context = CallContext(tenant_id="demo", run_id=audit.RUN, request_id="synthetic-v11-audit", stage="GENERATE")
    usage = TokenUsage(input_tokens=100, output_tokens=10)
    receipt = {"operation_id": stable_id("op", **context.model_dump(mode="json")),
               "cost_micro_usd": rate.quote(usage), "usage": usage.model_dump(mode="json")}
    expected = audit.expectation(context, audit.body("system", "prompt", 800), receipt,
                                 case="synthetic", method="v11", output=800)
    spec = ExternalOperationSpec(**context.model_dump(mode="json"),
        input_sha256=expected["input_sha256"], model_id=routes.generation_model_id,
        invocation_id=routes.generation_invocation_id, source_region=routes.source_region,
        price_sheet_version=prices.version, provider="DEEPSEEK", max_cost_micro_usd=rate.quote(TokenUsage(
            input_tokens=routes.generation_context_ceiling, output_tokens=800)))
    op = Operation(PK="OP#" + spec.operation_id, operation_id=spec.operation_id, spec=spec,
        spec_sha256=spec.fingerprint, period="2026-10", version=3, status="SETTLED", call_started=True,
        dispatch_id="a" * 32, created_at=stamp, updated_at=stamp, usage=usage,
        actual_micro_usd=receipt["cost_micro_usd"])
    _, marker, summary, _ = _run_records()
    return op, expected, prices, routes, marker, summary


def test_ledger_operation_matches_exact_context_body_usage_route_price_and_window():
    audit.validate_operation(*_operation_fixture())


@pytest.mark.parametrize("tamper", ["unknown", "not_dispatched", "bound", "body", "cost", "usage", "output",
    "request", "run", "stage", "route", "sheet", "max_cost", "funding", "before", "after"])
def test_ledger_rejects_wrong_operation_or_unsettled_unknown_usage(tamper):
    op, expected, prices, routes, marker, summary = _operation_fixture()
    if tamper == "unknown":
        op = op.model_copy(update={"status": "UNKNOWN_SPEND"})
    elif tamper == "not_dispatched":
        op = op.model_copy(update={"call_started": False, "dispatch_id": None})
    elif tamper == "bound":
        op = op.model_copy(update={"bound_exceeded": True})
    elif tamper == "body":
        expected["input_sha256"] = "f" * 64
    elif tamper == "cost":
        expected["cost_micro_usd"] += 1
    elif tamper == "usage":
        expected["usage"]["input_tokens"] += 1
    elif tamper == "output":
        op = op.model_copy(update={"usage": TokenUsage(input_tokens=100, output_tokens=801)})
    elif tamper in {"request", "run", "stage"}:
        field = {"request": "request_id", "run": "run_id", "stage": "stage"}[tamper]
        op = op.model_copy(update={"spec": op.spec.model_copy(update={field: "other"})})
    elif tamper in {"route", "sheet", "max_cost", "funding"}:
        field = {"route": "source_region", "sheet": "price_sheet_version", "max_cost": "max_cost_micro_usd",
                 "funding": "funding_id"}[tamper]
        op = op.model_copy(update={"spec": op.spec.model_copy(update={field: 1 if tamper == "max_cost" else "other"})})
    elif tamper == "before":
        marker["created_at"] = "2026-10-25T00:06:00+00:00"
    else:
        summary["created_at"] = "2026-10-25T00:04:00+00:00"
    with pytest.raises(ValueError, match="LEDGER_OPERATION"):
        audit.validate_operation(op, expected, prices, routes, marker, summary)


def test_counter_delta_is_only_six_new_receipts_and_preserves_old_caps_and_reservations():
    baseline = _counters(audit)
    after = copy.deepcopy(baseline)
    for row in after.values():
        row["version"] += 12
        row["settled_micro_usd"] += 123
    audit.validate_counter_baseline(baseline, "2026-10-25T00:00:00+00:00")
    audit.validate_after_counters(baseline, after, 123, 6)
    assert baseline["run"]["reserved_micro_usd"] == 640 and baseline["month"]["reserved_micro_usd"] == 1250


@pytest.mark.parametrize("tamper", ["empty", "missing", "delta", "version", "cap", "reserved", "halted", "scope"])
def test_counters_reject_concurrent_extra_spend_resets_or_changed_caps(tamper):
    baseline = _counters(audit)
    after = copy.deepcopy(baseline)
    for row in after.values():
        row["version"] += 12
        row["settled_micro_usd"] += 123
    if tamper == "empty":
        after = {}
    elif tamper == "missing":
        after.pop("funding")
    else:
        key = {"delta": "settled_micro_usd", "version": "version", "cap": "cap_micro_usd",
               "reserved": "reserved_micro_usd", "halted": "halted", "scope": "tenant_id"}[tamper]
        after["run"][key] = True if tamper == "halted" else "another-tenant" if tamper == "scope" else after["run"][key] + 1
    with pytest.raises(ValueError, match="COUNTER"):
        audit.validate_after_counters(baseline, after, 123, 6)


@pytest.mark.parametrize("service,operation,accepted", [
    ("sts", "GetCallerIdentity", True), ("dynamodb", "GetItem", True),
    ("signin", "CreateOAuth2Token", True), ("dynamodb", "UpdateItem", False),
    ("dynamodb", "TransactWriteItems", False), ("bedrock-runtime", "InvokeModel", False),
    ("sts", "GetItem", False), ("dynamodb", "GetCallerIdentity", False),
])
def test_aws_guard_allows_only_specific_service_operation_pairs(service, operation, accepted):
    report = {"aws_operations": []}
    model = SimpleNamespace(service_model=SimpleNamespace(service_name=service), name=operation)
    if accepted:
        audit.reads_only(report, model)
        assert report["aws_operations"] == [{"service": service, "operation": operation}]
    else:
        with pytest.raises(ValueError, match="AWS_MUTATION_FORBIDDEN"):
            audit.reads_only(report, model)
        assert not report["aws_operations"]


def test_local_only_does_not_create_aws_session_write_receipt_or_leave_global_guards(tmp_path, monkeypatch, capsys):
    from app.external_http import OfficialHttpsTransport

    original = OfficialHttpsTransport.request

    def local(report, **kwargs):
        report["new_upper_bound_micro_usd"] = 123
        with pytest.raises(ValueError, match="FORBIDDEN"):
            OfficialHttpsTransport.request(None)
        return [object()] * 6, None, None, None, None, None

    def forbidden_aws(*args, **kwargs):
        raise AssertionError("Local-only audit must not create an AWS session")

    monkeypatch.setattr(audit, "audit_local", local)
    monkeypatch.setattr(audit.boto3, "Session", forbidden_aws)
    assert audit.main(["--local-only", "--directory", str(tmp_path)]) == 0
    assert json.loads(capsys.readouterr().out)["status"] == "LOCAL_CHECKS_PASSED_AWS_PENDING"
    assert not (tmp_path / "reconciliation.json").exists()
    assert OfficialHttpsTransport.request is original


def test_existing_audit_receipt_is_never_overwritten(tmp_path):
    path = tmp_path / "reconciliation.json"
    path.write_bytes(b'{"status":"original"}')
    with pytest.raises(ValueError, match="RECEIPT_IMMUTABLE"):
        audit.main(["--local-only", "--directory", str(tmp_path)])
    assert path.read_bytes() == b'{"status":"original"}'


def test_review_criteria_hash_is_pinned_separately_and_cannot_change_after_outputs(tmp_path, monkeypatch):
    path = tmp_path / "criteria.json"
    raw = canonical_json({"label_status": "AI_DRAFT", "review_instructions": ["Keep negation"]})
    path.write_bytes(raw)
    monkeypatch.setattr(audit, "EXPECTED_CRITERIA_SHA256", sha256_bytes(raw))
    saved, digest = audit.load_review_criteria(path)
    assert saved["label_status"] == "AI_DRAFT" and digest == sha256_bytes(raw)
    path.write_bytes(canonical_json(saved | {"review_instructions": ["Ignore missed facts"]}))
    with pytest.raises(ValueError, match="PREREGISTERED_CRITERIA_HASH"):
        audit.load_review_criteria(path)
