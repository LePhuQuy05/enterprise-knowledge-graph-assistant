"""New V4 experiment, never re-dispatch the historical failed request."""

import argparse
import json
from datetime import UTC, datetime

from app.aws_knowledge import AwsKnowledgeClient, create_bundle, load_bundle, pack_bundle
from app.contracts import sha256_bytes
from app.generation import (
    AnswerBinding,
    GenerationRecipeV4,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.model_gateway import CallContext
from app.query_language import rewrite_for_search
from app.release_experiment import ReleaseExperiment
from scripts.answer_local import ROOT, RUN, managed_gateway

OUT = ROOT / "results/chatbot-release/smoke-v4"
EXPERIMENT = ROOT / "data/experiments/recruiter-chatbot-release-v1"
CASES = [("rds-vi", "Amazon RDS Multi-AZ khác gì với Read Replica?", "vi"),
         ("vpc-en", "How do security groups differ from network ACLs in an Amazon VPC?", "en")]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--execute", action="store_true")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV4(model_metadata_sha256=routes["account_model_metadata_sha256"])
    if not args.execute and not args.replay:
        print(json.dumps({"status": "PLANNED", "max_operations": 3, "recipe": recipe.fingerprint}))
        return
    if args.execute:
        if (OUT / "intent.json").exists():
            raise ValueError("EXISTING_SMOKE_INTENT_NO_RETRY")
        gateway = managed_gateway()
        save(OUT / "intent.json", {"created_at": datetime.now(UTC).isoformat(), "maximum_new_operations": 3,
            "recipe": recipe.model_dump(mode="json"), "price_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes()),
            "scope": "LOCAL_LIVE_RETRIEVAL_AND_GENERATION_NOT_HTTP_OR_BROWSER"})
    else:
        gateway = None
    report = {"status": "INCOMPLETE", "cases": [], "new_operations": 0,
        "scope": "LOCAL_LIVE_RETRIEVAL_AND_GENERATION_NOT_HTTP_OR_BROWSER", "semantic_review": "PENDING"}
    try:
        with ReleaseExperiment(EXPERIMENT, gateway) as experiment:
            for name, question, language in CASES:
                path = OUT / (name + ".json")
                if args.replay:
                    receipt = json.loads(path.read_bytes())
                    source = receipt["bundle"]
                    bundle = load_bundle(source, receipt["bundle_sha256"], tenant_id="demo", owner_id="release-operator",
                        request_id=name + "-v4", snapshot_version="official-release-policy-fixture")
                else:
                    phrase = question
                    if language == "vi":
                        phrase, rewrite = rewrite_for_search(question, gateway=experiment, tenant_id="demo",
                            request_id=name + "-v4", run_id=RUN)
                        save(OUT / (name + "-rewrite.json"), rewrite)
                    items = AwsKnowledgeClient(timeout=6).search(phrase)
                    bundle = create_bundle(items, tenant_id="demo", owner_id="release-operator", request_id=name + "-v4",
                        snapshot_version="official-release-policy-fixture", search_phrase=phrase)
                    if not bundle.chunks:
                        raise ValueError("EMPTY_SMOKE_EVIDENCE_NO_GENERATION")
                    packed = pack_bundle(bundle)
                    system, prompt = generation_request(question, bundle, packed, "demo", recipe, answer_language=language)
                    result = experiment.generate(CallContext(tenant_id="demo", run_id=RUN, request_id=name + "-v4", stage="GENERATE"),
                        system=system, prompt=prompt, max_output_tokens=800)
                    receipt = {"question": question, "language": language, "bundle": bundle.record, "bundle_sha256": bundle.sha256,
                        "raw_output": result.text, "stop_reason": result.stop_reason, "operation_id": result.operation_id,
                        "usage": result.usage.model_dump(mode="json"), "cost_micro_usd": result.actual_micro_usd,
                        "recipe_sha256": recipe.fingerprint}
                    save(path, receipt)
                binding = AnswerBinding(request_id=name + "-v4", tenant_id="demo", owner_id="release-operator",
                    mode_requested="hybrid", mode_used="hybrid", used_graph=False)
                answer = resolve_answer(parse_generated(receipt["raw_output"], receipt["stop_reason"]), snapshot=bundle,
                    packed=pack_bundle(bundle), binding=binding)
                save(OUT / (name + "-answer.json"), answer.model_dump(mode="json"))
                report["cases"].append({"case": name, "structural_status": "PASS", "answer_status": answer.answer_status,
                    "citations": len(answer.citations), "cost_micro_usd": receipt["cost_micro_usd"], "language_requested": language})
                print(json.dumps(report["cases"][-1]), flush=True)
            report["status"] = "PASS_STRUCTURAL_SMOKE_REVIEW_PENDING"
    except Exception as exc:
        report.update(status="STOPPED_NO_AUTOMATIC_RETRY", error_category=type(exc).__name__)
        if isinstance(exc, ValueError) and str(exc).isupper():
            report["safe_error"] = str(exc)
    finally:
        if gateway:
            report["new_operations"] = gateway.provider.dispatch_attempts
        save(OUT / ("replay.json" if args.replay else "report.json"), report)
    print(json.dumps(report, ensure_ascii=False))
    return int(report["status"] != "PASS_STRUCTURAL_SMOKE_REVIEW_PENDING")


if __name__ == "__main__":
    raise SystemExit(main())
