"""RC20 paired-language fixed-evidence evaluation; semantic labels remain AI_DRAFT."""

import argparse
import json
import time
from datetime import UTC, datetime

from app.aws_knowledge import AwsKnowledgeClient, create_bundle, load_bundle, pack_bundle
from app.contracts import canonical_json, sha256_bytes
from app.generation import (
    AnswerBinding,
    GenerationRecipeV8,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.model_gateway import CallContext
from app.release_experiment import ReleaseExperiment
from scripts.answer_local import ROOT, RUN, managed_gateway
from scripts.run_chatbot_release_smoke import EXPERIMENT, save

DIRECTORY = ROOT / "data/experiments/recruiter-chatbot-release-v1/rc20-v8"
OUTPUT = ROOT / "results/chatbot-release/rc20-v8"
CATALOG = ROOT / "configs/evaluation/chatbot-release-v1.json"
CODE = ("app/generation.py", "app/aws_knowledge.py", "app/query_language.py", "app/release_experiment.py", "app/source_scope.py")


def catalog():
    value = json.loads(CATALOG.read_bytes())
    cases = value["cases"]
    if len(cases) != 20 or len({c["case_id"] for c in cases}) != 20 or value["label_status"] != "AI_DRAFT":
        raise ValueError("RC20_CATALOG_INVALID")
    counts = {(split, family): sum(c["split"] == split and c["family"] == family for c in cases)
              for split in ("development", "release") for family in ("factual", "workflow", "troubleshooting", "missing")}
    if list(counts.values()) != [3, 2, 2, 1, 3, 3, 3, 3]:
        raise ValueError("RC20_SPLITS_CHANGED")
    return value


def current_plan(value, recipe):
    return {"version": value["version"], "catalog_sha256": sha256_bytes(CATALOG.read_bytes()),
        "label_status": value["label_status"], "recipe": recipe.model_dump(mode="json"),
        "recipe_sha256": recipe.fingerprint, "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "price_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes()),
        "paired_languages_same_evidence": True, "maximum_generation_operations": 40,
        "limits": "AI_DRAFT labels; exactness is not entailment; 20 intents not 40 independent samples"}


def bundle_for(case, entry):
    record = entry["bundle"]
    return load_bundle(record, entry["bundle_sha256"], tenant_id="demo", owner_id="release-evaluator",
        request_id=case["case_id"], snapshot_version="rc20-fixed-evidence-policy")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--replay", action="store_true")
    parser.add_argument("--split", choices=("development", "release", "all"), default="all")
    parser.add_argument("--case", choices=["rc" + str(i).zfill(2) for i in range(1,21)])
    args = parser.parse_args()
    value = catalog()
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV8(model_metadata_sha256=routes["account_model_metadata_sha256"])
    plan = current_plan(value, recipe)
    if args.prepare:
        if (DIRECTORY / "plan.json").exists():
            raise ValueError("RC20_ALREADY_FROZEN")
        entries = []
        for case in value["cases"]:
            target = DIRECTORY / (case["case_id"] + "-evidence.json")
            if target.exists():
                entry = json.loads(target.read_bytes())
                bundle_for(case, entry)
            else:
                # Missing-data controls get a real documentation distractor, never
                # private account data or a synthetic zero-context easy pass.
                phrase = case["en"] if case["family"] != "missing" else "AWS IAM roles identity permissions documentation"
                started = time.monotonic()
                items = AwsKnowledgeClient(timeout=8).search(phrase)
                bundle = create_bundle(items, tenant_id="demo", owner_id="release-evaluator", request_id=case["case_id"],
                    snapshot_version="rc20-fixed-evidence-policy", search_phrase=phrase)
                if not bundle.chunks:
                    raise ValueError("RC20_EMPTY_EVIDENCE_NO_PAID_CALL")
                entry = {"case_id": case["case_id"], "bundle": bundle.record, "bundle_sha256": bundle.sha256,
                    "retrieval_seconds": round(time.monotonic() - started, 3), "label_status": "AI_DRAFT",
                    "gold_source_anchors": case["source_anchors"], "source_review": "PENDING"}
                save(target, entry)
            entries.append({"case_id": case["case_id"], "sha256": sha256_bytes(target.read_bytes())})
            print(json.dumps({"prepared": case["case_id"]}), flush=True)
        plan["evidence_files"] = entries
        save(DIRECTORY / "plan.json", plan)
        save(OUTPUT / "plan.json", plan)
        print(json.dumps({"status": "RC20_FROZEN_AI_DRAFT", "model_operations": 0}))
        return 0
    frozen = json.loads((DIRECTORY / "plan.json").read_bytes())
    if {k: v for k, v in frozen.items() if k != "evidence_files"} != plan:
        raise ValueError("RC20_PLAN_CHANGED_NO_DISPATCH")
    for entry in frozen["evidence_files"]:
        if sha256_bytes((DIRECTORY / (entry["case_id"] + "-evidence.json")).read_bytes()) != entry["sha256"]:
            raise ValueError("RC20_EVIDENCE_CHANGED_NO_DISPATCH")
    gateway = managed_gateway() if args.execute else None
    report = {"status": "INCOMPLETE", "split": args.split, "label_status": "AI_DRAFT", "human_review": "PENDING",
        "scope": "FIXED_EVIDENCE_GENERATION_NOT_LIVE_RETRIEVAL_OR_BROWSER", "cases": [], "new_model_operations": 0,
        "plan_sha256": sha256_bytes(canonical_json(frozen))}
    try:
        with ReleaseExperiment(EXPERIMENT, gateway) as experiment:
            for case in value["cases"]:
                if args.split != "all" and case["split"] != args.split:
                    continue
                if args.case and case["case_id"] != args.case:
                    continue
                evidence = json.loads((DIRECTORY / (case["case_id"] + "-evidence.json")).read_bytes())
                bundle = bundle_for(case, evidence)
                packed = pack_bundle(bundle)
                for language in ("en", "vi"):
                    request_id = "rc20-v8-" + case["case_id"] + "-" + language
                    path = DIRECTORY / (request_id + ".json")
                    if args.replay or path.exists():
                        receipt = json.loads(path.read_bytes())
                    else:
                        system, prompt = generation_request(case[language], bundle, packed, "demo", recipe, answer_language=language)
                        started = time.monotonic()
                        result = experiment.generate(CallContext(tenant_id="demo", run_id=RUN, request_id=request_id, stage="EVALUATE"),
                            system=system, prompt=prompt, max_output_tokens=800)
                        receipt = {"case_id": case["case_id"], "language": language, "raw_output": result.text,
                            "stop_reason": result.stop_reason, "operation_id": result.operation_id,
                            "cost_micro_usd": result.actual_micro_usd, "usage": result.usage.model_dump(mode="json"),
                            "seconds": round(time.monotonic() - started, 3), "bundle_sha256": bundle.sha256,
                            "recipe_sha256": recipe.fingerprint, "created_at": datetime.now(UTC).isoformat()}
                        save(path, receipt)  # Retain invalid, paid output before parsing.
                    if receipt["bundle_sha256"] != bundle.sha256 or receipt["recipe_sha256"] != recipe.fingerprint:
                        raise ValueError("RC20_RECEIPT_BINDING_INVALID")
                    binding = AnswerBinding(request_id=request_id, tenant_id="demo", owner_id="release-evaluator",
                        mode_requested="hybrid", mode_used="hybrid", used_graph=False)
                    answer = resolve_answer(parse_generated(receipt["raw_output"], receipt["stop_reason"], recipe),
                        snapshot=bundle, packed=packed, binding=binding, recipe=recipe)
                    answer = answer.model_copy(update={"answer_language": language})
                    if language == "vi" and answer.answer_status == "ABSTAINED":
                        answer = answer.model_copy(update={"answer": "Tài liệu được tìm thấy chưa cung cấp đủ bằng chứng để trả lời câu hỏi này."})
                    save(OUTPUT / (request_id + ".json"), {"case_id": case["case_id"], "language": language,
                        "expected": case["expected"], "required_facts": case["required_facts"], "source_anchors": case["source_anchors"],
                        "label_status": "AI_DRAFT", "semantic_review": "PENDING", "response": answer.model_dump(mode="json")})
                    row = {"case_id": case["case_id"], "split": case["split"], "language": language,
                        "structural_status": "PASS", "answer_status": answer.answer_status, "citations": len(answer.citations),
                        "cost_micro_usd": receipt["cost_micro_usd"], "seconds": receipt["seconds"], "semantic_review": "PENDING"}
                    report["cases"].append(row)
                    save(OUTPUT / ("replay-" + args.split + ".json" if args.replay else "report-" + args.split + ".json"), report)
                    print(json.dumps(row), flush=True)
            report["status"] = "STRUCTURAL_PASS_SEMANTIC_REVIEW_PENDING"
    except Exception as exc:
        report.update(status="STOPPED_NO_AUTOMATIC_RETRY", error_category=type(exc).__name__)
        if isinstance(exc, ValueError) and str(exc).isupper():
            report["safe_error"] = str(exc)
    finally:
        if gateway:
            report["new_model_operations"] = gateway.provider.dispatch_attempts
        report["settled_case_cost_micro_usd"] = sum(r["cost_micro_usd"] for r in report["cases"])
        save(OUTPUT / ("replay-" + args.split + ".json" if args.replay else "report-" + args.split + ".json"), report)
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}))
    return int(report["status"] != "STRUCTURAL_PASS_SEMANTIC_REVIEW_PENDING")


if __name__ == "__main__":
    raise SystemExit(main())
