"""One new recipe experiment against the recorded citation-count development failure."""

import argparse
import json

from app.contracts import canonical_json, sha256_bytes
from app.generation import (
    AnswerBinding,
    GenerationRecipeV9,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.model_gateway import CallContext
from app.release_experiment import ReleaseExperiment
from scripts.answer_local import ROOT, RUN, managed_gateway
from scripts.evaluate_chatbot_release import bundle_for, catalog
from scripts.run_chatbot_release_smoke import EXPERIMENT, save

OUT = ROOT / "results/chatbot-release/span-v3-smoke"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--execute", action="store_true")
    group.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV9(model_metadata_sha256=routes["account_model_metadata_sha256"])
    case = next(c for c in catalog()["cases"] if c["case_id"] == "rc16")
    entry = json.loads((EXPERIMENT / "rc20-v8/rc16-evidence.json").read_bytes())
    bundle = bundle_for(case, entry)
    from app.aws_knowledge import pack_bundle

    packed = pack_bundle(bundle)
    request_id = "span-v3-rc16-vi"
    if args.execute:
        if (OUT / "intent.json").exists():
            raise ValueError("EXISTING_SPAN_V3_INTENT_NO_RETRY")
        gateway = managed_gateway()
        system, prompt = generation_request(case["vi"], bundle, packed, "demo", recipe, answer_language="vi")
        save(OUT / "intent.json", {"recipe": recipe.model_dump(mode="json"), "bundle_sha256": bundle.sha256,
            "input_sha256": sha256_bytes(canonical_json([system, prompt])), "maximum_new_model_operations": 1,
            "scope": "FIXED_EVIDENCE_GENERATION_REGRESSION_NOT_HTTP"})
        with ReleaseExperiment(EXPERIMENT, gateway) as experiment:
            result = experiment.generate(CallContext(tenant_id="demo", run_id=RUN, request_id=request_id, stage="EVALUATE"),
                system=system, prompt=prompt, max_output_tokens=800)
        receipt = {"raw_output": result.text, "stop_reason": result.stop_reason, "operation_id": result.operation_id,
            "cost_micro_usd": result.actual_micro_usd, "usage": result.usage.model_dump(mode="json"),
            "recipe_sha256": recipe.fingerprint, "bundle_sha256": bundle.sha256}
        save(OUT / "receipt.json", receipt)
    else:
        receipt = json.loads((OUT / "receipt.json").read_bytes())
    if receipt["recipe_sha256"] != recipe.fingerprint or receipt["bundle_sha256"] != bundle.sha256:
        raise ValueError("SPAN_V3_RECEIPT_BINDING_MISMATCH")
    response = resolve_answer(parse_generated(receipt["raw_output"], receipt["stop_reason"], recipe), snapshot=bundle,
        packed=packed, binding=AnswerBinding(request_id=request_id, tenant_id="demo", owner_id="release-evaluator",
            mode_requested="hybrid", mode_used="hybrid", used_graph=False), recipe=recipe)
    save(OUT / "answer.json", response.model_copy(update={"answer_language": "vi"}).model_dump(mode="json"))
    report = {"status": "PASS_SPAN_V3_STRUCTURAL_SMOKE", "new_model_operations": int(args.execute),
        "citations": len(response.citations), "answer_status": response.answer_status,
        "cost_micro_usd": receipt["cost_micro_usd"], "semantic_review": "PENDING", "full_rc20_gate": "NOT_MET"}
    save(OUT / ("report.json" if args.execute else "replay.json"), report)
    print(json.dumps(report))


if __name__ == "__main__":
    raise SystemExit(main())
