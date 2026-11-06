"""Replay valid responses and reproduce FAILED raw schema outputs without mutation."""

import json
import sqlite3
from unittest.mock import patch

from app.answer_cache import AnswerCache, AnswerReceipt
from app.contracts import canonical_json, sha256_bytes
from app.generation import parse_generated
from scripts.compare_required_reason_protocol import CACHE, CODE, OUTPUT, protocol_cases
from scripts.prepare_authorization_evidence import ROOT


def replay():
    scope = json.loads((OUTPUT / "scope.json").read_bytes())
    if scope["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
        raise ValueError("REASON_PROTOCOL_REPLAY_CODE_CHANGED")
    cache, results = AnswerCache(CACHE, namespace="LIVE"), []
    with patch("socket.socket", side_effect=AssertionError("REASON_PROTOCOL_REPLAY_NETWORK_FORBIDDEN")):
        _, cases = protocol_cases()
        for case in cases:
            p = case["prepared"]
            failed_path = OUTPUT / "failed-answers" / (case["id"] + ".json")
            if failed_path.exists():
                original = json.loads(failed_path.read_bytes())
                intent = canonical_json({"context": p.context.model_dump(mode="json"),
                    "binding_sha256": sha256_bytes(canonical_json(p.binding.model_dump(mode="json"))),
                    "recipe_sha256": p.recipe.fingerprint})
                with sqlite3.connect(f"file:{cache.path.as_posix()}?mode=ro", uri=True) as db:
                    row = db.execute("SELECT state,payload,payload_sha256 FROM answers WHERE namespace='LIVE' AND intent=?", (intent,)).fetchone()
                if not row or row[0] != "FAILED" or sha256_bytes(row[1]) != row[2]:
                    raise ValueError("REASON_PROTOCOL_REPLAY_FAILED_RECEIPT_BINDING")
                receipt = AnswerReceipt.model_validate_json(row[1])
                if canonical_json(receipt.model_dump(mode="json")) != canonical_json(original["receipt"]):
                    raise ValueError("REASON_PROTOCOL_REPLAY_RAW_FAILURE_CHANGED")
                try:
                    parse_generated(receipt.raw_output, receipt.stop_reason, p.recipe)
                except ValueError as exc:
                    diagnostic = str(exc)
                else:
                    raise ValueError("REASON_PROTOCOL_REPLAY_FAILURE_NOT_REPRODUCED")
                results.append({"id": case["id"], "status": "FAILED_VALIDATION", "raw_error": diagnostic,
                    "receipt_unchanged": True, "cache_state_unchanged": True, "repaired": False})
                continue
            lookup = cache.resolve(p.context, question=p.question, snapshot=case["view"], packed=p.packed,
                binding=p.binding, recipe=p.recipe, gateway=None, answer_language=case["language"])
            original = json.loads((OUTPUT / "answers" / (case["id"] + ".json")).read_bytes())
            if (not lookup.cache_hit or canonical_json(lookup.response.model_dump(mode="json")) != canonical_json(original["response"])
                    or canonical_json(lookup.receipt.model_dump(mode="json")) != canonical_json(original["receipt"])):
                raise ValueError("REASON_PROTOCOL_REPLAY_RESPONSE_CHANGED")
            results.append({"id": case["id"], "status": lookup.response.answer_status, "cache_hit": True})
    report = {"status": "ALL_32_CONTEXT_RESPONSE_AND_RAW_FAILURE_REPLAY_MATCH", "cases": results,
        "matrix_cells": len(results), "ready_cache_hits": sum(r.get("cache_hit", False) for r in results),
        "raw_schema_failures_reproduced": sum(r["status"] == "FAILED_VALIDATION" for r in results),
        "network_blocked": True, "new_model_calls": 0, "new_cost_micro_usd": 0, "aws_calls": 0,
        "repairs": 0, "failed_state_resets": 0}
    (OUTPUT / "replay-v2.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))


if __name__ == "__main__":
    replay()
