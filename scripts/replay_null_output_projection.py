"""Evaluate a separately versioned NULL-extra projection over recorded outputs.

No model, AWS, retry, cache reset or historical result mutation. Derived results
stay outside native answer caches and must not be counted as native passes.
"""

import argparse
import json
import sqlite3
from contextlib import ExitStack
from unittest.mock import patch

from app.contracts import canonical_json, sha256_bytes, stable_id
from app.generation import GenerationRecipeV9, resolve_answer
from app.structured_output_projection import project_null_extras, projection_recipe
from scripts.pilot_coverage_protocol import CACHE, cases
from scripts.pilot_coverage_protocol import OUTPUT as NATIVE
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT

OUTPUT = ROOT / "results/coverage-null-projection-v1"
CODE = ("app/structured_output_projection.py", "scripts/replay_null_output_projection.py")


def states():
    with sqlite3.connect(f"file:{(CACHE / 'answers.sqlite3').as_posix()}?mode=ro", uri=True) as db:
        return db.execute("SELECT cache_key,state,payload_sha256 FROM answers WHERE namespace='LIVE' ORDER BY cache_key").fetchall()


def evaluate(*, replay=False):
    before = states()
    recipe = projection_recipe()
    scope = {"version": "recorded-null-extra-projection-evaluation-v1",
        "native_scope_sha256": sha256_bytes((NATIVE / "scope.json").read_bytes()),
        "native_ledger_audit_sha256": sha256_bytes((NATIVE / "ledger-audit.json").read_bytes()),
        "projection_recipe": recipe, "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "native_cache_states_before": before, "original_failed_entries_must_remain_failed": True,
        "new_model_calls": 0, "aws_calls": 0, "serving_published": False}
    immutable(OUTPUT / "plan.json", scope)
    results, negatives = [], []
    for case in cases():
        p = case["prepared"]
        success_path = NATIVE / "answers" / (case["id"] + ".json")
        failed = not success_path.exists()
        source_path = NATIVE / "failed-answers" / (case["id"] + ".json") if failed else success_path
        native = json.loads(source_path.read_bytes())
        receipt = native["receipt"]
        projection = project_null_extras(receipt["raw_output"], receipt["stop_reason"], p.recipe)
        request_id = stable_id("derived-output-projection-v1", native_request=p.binding.request_id,
            raw=projection.raw_sha256, policy=projection.policy_sha256, projected=projection.projected_sha256)
        warnings = p.binding.warnings + (("OUTPUT_NULL_EXTRAS_PROJECTED_LOCAL_DERIVATION",) if projection.dropped_null_fields else ())
        binding = p.binding.model_copy(update={"request_id": request_id, "warnings": tuple(dict.fromkeys(warnings))})
        response = resolve_answer(projection.parsed, snapshot=case["view"], packed=p.packed, binding=binding,
            recipe=p.recipe, answer_language=case["language"])
        row = {"id": case["id"], "original_native_status": "FAILED_VALIDATION" if failed else "READY",
            "native_receipt_file_sha256": sha256_bytes(source_path.read_bytes()), "native_operation_id": receipt["operation_id"],
            "raw_sha256": projection.raw_sha256, "projected_sha256": projection.projected_sha256,
            "projection_policy_sha256": projection.policy_sha256, "preserved_business_fields_sha256": projection.preserved_fields_sha256,
            "dropped_null_fields": projection.dropped_null_fields, "canonical_projected_output": projection.canonical_output,
            "derived_response": response.model_dump(mode="json"), "derived_not_native_schema_pass": True,
            "native_cache_updated": False, "model_called": False}
        immutable(OUTPUT / "derived-answers" / (case["id"] + ".json"), row)
        results.append({"id": case["id"], "native_status": row["original_native_status"],
            "dropped_null_fields": projection.dropped_null_fields, "derived_status_self_report": response.answer_status,
            "source_validation_passed": True, "limitations": len(response.assessment.limitations)})
    # Actual historical missing-reason failures are negative controls. A NULL
    # metadata compatibility rule cannot invent the absent mandatory reason.
    legacy = ROOT / "results/source-message-answers-v2/failed-answers"
    for path in sorted(legacy.glob("*.json")):
        row = json.loads(path.read_bytes())
        try:
            project_null_extras(row["receipt"]["raw_output"], row["receipt"]["stop_reason"], GenerationRecipeV9(model_metadata_sha256="0" * 64))
        except ValueError as exc:
            negatives.append({"id": path.stem, "status": "STILL_REJECTED", "failure": str(exc), "original_file_sha256": sha256_bytes(path.read_bytes())})
        else:
            raise ValueError("NULL_PROJECTION_INVENTED_MISSING_REQUIRED_FIELD")
    if states() != before or len(results) != 8 or len(negatives) != 5:
        raise ValueError("NULL_PROJECTION_CACHE_MUTATED_OR_CONTROL_DENOMINATOR")
    raw = canonical_json({"results": results, "historical_negative_controls": negatives}) + b"\n"
    report = {"status": "SEPARATE_RECORDED_OUTPUT_PROJECTION_CHECKED", "native_cells": 8,
        "native_strict_schema_valid": sum(r["native_status"] == "READY" for r in results),
        "native_strict_schema_failed": sum(r["native_status"] != "READY" for r in results),
        "derived_source_validated": len(results), "outputs_with_null_projection": sum(bool(r["dropped_null_fields"]) for r in results),
        "historical_missing_reason_still_rejected": len(negatives), "native_cache_unchanged": True,
        "new_model_calls": 0, "aws_calls": 0, "new_cost_micro_usd": 0,
        "human_reviewed": 0, "answer_accuracy_measured": False, "serving_published": False, "receipts_sha256": sha256_bytes(raw)}
    if replay:
        if (OUTPUT / "receipts.json").read_bytes() != raw:
            raise ValueError("NULL_PROJECTION_EXACT_REPLAY_CHANGED")
        immutable(OUTPUT / "replay.json", report | {"status": "RECORDED_PROJECTION_AND_NEGATIVE_CONTROLS_EXACT_REPLAY"})
    else:
        (OUTPUT / "receipts.json").write_bytes(raw)
        immutable(OUTPUT / "report.json", report)
    print(json.dumps(report))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay", action="store_true")
    with ExitStack() as stack:
        for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
            stack.enter_context(patch(target, side_effect=AssertionError("NULL_PROJECTION_REPLAY_NETWORK_BLOCKED")))
        evaluate(replay=parser.parse_args().replay)
