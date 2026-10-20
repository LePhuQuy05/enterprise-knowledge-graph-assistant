"""Paired EN/VI fixed-evidence V9 answer experiment; never a semantic accuracy claim."""

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path

from app.context import pack_context
from app.contracts import canonical_json, sha256_bytes
from app.generation import (
    AnswerBinding,
    GenerationError,
    GenerationRecipeV9,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.graph_retrieval import GraphRetrievalResult
from app.model_gateway import CallContext
from app.release_experiment import ReleaseExperiment
from scripts.answer_local import managed_gateway
from scripts.evaluate_development import load_inputs, save
from scripts.evaluate_expanded import ROOT, read_pair
from scripts.evaluate_expanded import replay as replay_retrieval


class ExpandedAnswerExperiment(ReleaseExperiment):
    MAX_OPERATIONS = 100
    BATCH_STOP = 250_000
    OBLIGATION_CAP = 750_000


def settings():
    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipeV9(model_metadata_sha256=routes["account_model_metadata_sha256"])
    return recipe, {
        "recipe": recipe.model_dump(mode="json"), "recipe_sha256": recipe.fingerprint,
        "price_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes()),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in (
            "app/generation.py", "app/context.py", "app/external_gateway.py", "app/release_experiment.py",
            "scripts/evaluate_expanded_answers.py")},
        "max_operations": 100, "output_tokens_per_call": 800, "batch_stop_micro_usd": 250_000,
        "obligation_cap_micro_usd": 750_000, "managed_run_cap_micro_usd": 500_000,
        "shared_external_funding_cap_micro_usd": 2_000_000,
        "scope": "FIXED_ENGLISH_HYBRID_CONTEXT_PAIRED_LANGUAGE_GENERATION_NOT_END_TO_END_LIVE_SEARCH",
        "human_semantic_review": "PENDING", "labels": "AI_DRAFT_NOT_HOLDOUT",
    }


def packed_for(row, graph):
    ids = tuple(row["ranked_chunk_ids"])
    packed = pack_context(graph, GraphRetrievalResult(ids, ids, (), (), "hybrid", False, (), 0, 0))
    if list(packed.chunk_ids) != row["context_chunk_ids"]:
        raise ValueError("ANSWER_EXPERIMENT_CONTEXT_CHANGED")
    return packed


def prepare(directory):
    datasets, hashes = read_pair(directory)
    _, graph = load_inputs(datasets["en"])
    replay_retrieval(directory, datasets, hashes, graph)
    output = directory / "answers-v1"
    if (output / "plan.json").exists():
        raise ValueError("ANSWER_PLAN_ALREADY_FROZEN")
    output.mkdir(parents=True, exist_ok=True)
    recipe, policy = settings()
    ranking = {r["question_id"]: r for r in map(json.loads, (directory / "retrieval.jsonl").read_bytes().splitlines())
               if r["language"] == "en" and r["system"] == "B2_HYBRID"}
    inputs = []
    for en, vi in zip(datasets["en"].items, datasets["vi"].items, strict=True):
        packed = packed_for(ranking[en.question_id], graph)
        for lang, item in (("en", en), ("vi", vi)):
            system, prompt = generation_request(item.question, graph, packed, "demo", recipe, answer_language=lang)
            inputs.append({"scenario_id": en.question_id, "question_id": item.question_id, "language": lang,
                "category": item.category, "answerable_in_corpus": item.answerable,
                "context_complete_for_english_retrieval": ranking[en.question_id]["metrics"]["context_complete"],
                "system": system, "prompt": prompt, "context_chunk_ids": packed.chunk_ids})
    save(output / "inputs.json", inputs)
    manifest = json.loads((directory / "manifest.json").read_bytes())
    plan = {**policy, "created_at": datetime.now(UTC).isoformat(), "dataset_sha256": hashes,
        "retrieval_sha256": manifest["retrieval_sha256"], "graph_manifest_sha256": graph.manifest_sha256,
        "inputs_sha256": sha256_bytes((output / "inputs.json").read_bytes()),
        "managed_run_id": "expanded-eval-" + manifest["configuration_sha256"][:20],
        "planned_inputs": len(inputs), "unique_scenarios": len(datasets["en"].items)}
    save(output / "plan.json", plan)
    return {"status": "PREPARED_NO_MODEL_CALLS", "inputs": len(inputs), "directory": str(output)}


def checked_inputs(directory):
    output = directory / "answers-v1"
    plan = json.loads((output / "plan.json").read_bytes())
    recipe, policy = settings()
    if any(plan[key] != value for key, value in policy.items()):
        raise ValueError("ANSWER_PLAN_POLICY_CHANGED_NO_DISPATCH")
    if sha256_bytes((output / "inputs.json").read_bytes()) != plan["inputs_sha256"]:
        raise ValueError("ANSWER_INPUTS_CHANGED_NO_DISPATCH")
    datasets, hashes = read_pair(directory)
    _, graph = load_inputs(datasets["en"])
    if hashes != plan["dataset_sha256"] or graph.manifest_sha256 != plan["graph_manifest_sha256"]:
        raise ValueError("ANSWER_SNAPSHOT_BINDING_CHANGED")
    inputs = json.loads((output / "inputs.json").read_bytes())
    expected = {(lang, item.question_id) for lang, ds in datasets.items() for item in ds.items}
    keys = [(row["language"], row["question_id"]) for row in inputs]
    if len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError("ANSWER_INPUT_PAIR_SET_CHANGED")
    return output, plan, recipe, inputs, datasets, graph


def score_receipt(receipt, entry, packed, graph, recipe, request_id):
    if (receipt["input_sha256"] != sha256_bytes(canonical_json(entry)) or
            receipt["recipe_sha256"] != recipe.fingerprint):
        raise ValueError("ANSWER_RECEIPT_BINDING_INVALID")
    row = {key: entry[key] for key in ("scenario_id", "question_id", "language", "category", "answerable_in_corpus",
                                      "context_complete_for_english_retrieval")}
    row.update(cost_upper_bound_micro_usd=receipt["cost_upper_bound_micro_usd"], seconds=receipt["seconds"],
               semantic_correctness="NOT_SCORED_REQUIRES_REVIEW", citation_entailment="NOT_SCORED_REQUIRES_REVIEW")
    try:
        binding = AnswerBinding(request_id=request_id, tenant_id="demo", owner_id="expanded-evaluator",
                                mode_requested="hybrid", mode_used="hybrid", used_graph=False)
        answer = resolve_answer(parse_generated(receipt["raw_output"], receipt["stop_reason"], recipe),
            snapshot=graph, packed=packed, binding=binding, recipe=recipe)
        row.update(structural_status="PASS", answer_status=answer.answer_status, claims=len(answer.claims),
                   citations=len(answer.citations), response=answer.model_dump(mode="json"))
    except GenerationError as exc:
        row.update(structural_status="FAIL", safe_error=str(exc), answer_status=None, claims=None, citations=None)
    return row


def summary(rows, inputs):
    if len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("DUPLICATE_ANSWER_EVALUATION_ROWS")
    languages = {}
    for lang in ("en", "vi"):
        own = [r for r in rows if r["language"] == lang]
        missing = [r for r in own if not r["answerable_in_corpus"]]
        languages[lang] = {"attempted": len(own), "planned": sum(i["language"] == lang for i in inputs),
            "structural_pass": sum(r["structural_status"] == "PASS" for r in own),
            "structural_fail": sum(r["structural_status"] == "FAIL" for r in own),
            "unanswerable_attempted": len(missing),
            "unanswerable_returned_abstained": sum(r["answer_status"] == "ABSTAINED" for r in missing),
            "semantic_correctness": None, "citation_entailment": None}
    return {"attempted": len(rows), "planned": len(inputs), "unique_scenarios_attempted": len({r["scenario_id"] for r in rows}),
        "languages": languages, "settled_published_peak_cost_upper_bound_micro_usd": sum(r["cost_upper_bound_micro_usd"] for r in rows),
        "scope": "FIXED_ENGLISH_HYBRID_CONTEXT_PAIRED_LANGUAGE_GENERATION_NOT_END_TO_END_LIVE_SEARCH",
        "quality_verdict": "HUMAN_SEMANTIC_REVIEW_PENDING_NOT_AN_ACCURACY_SCORE"}


def evaluate(directory, *, execute=False, limit=None, canary=False):
    output, plan, recipe, inputs, datasets, graph = checked_inputs(directory)
    rows, calls = [], 0
    if canary:
        en = datasets["en"].items
        selected = {next(i.question_id for i in en if i.category == category)
                    for category in ("FACTUAL", "MULTI_HOP", "UNANSWERABLE")}
        inputs = [i for i in inputs if i["scenario_id"] in selected]
    journal = ROOT / "data/experiments" / plan["managed_run_id"]
    gateway = managed_gateway() if execute else None
    if gateway:
        # A separate run isolates the experiment; shared one-time funding and monthly
        # caps remain unchanged. Idempotent initialization cannot reset past spending.
        gateway.ledger.initialize_run("kg-r72", "demo", plan["managed_run_id"], cap_micro_usd=500_000)
    experiment = ExpandedAnswerExperiment(journal, gateway) if execute else None
    ranking = {r["question_id"]: r for r in map(json.loads, (directory / "retrieval.jsonl").read_bytes().splitlines())
               if r["language"] == "en" and r["system"] == "B2_HYBRID"}
    report = {"status": "INCOMPLETE", "new_model_operations": 0, "plan_sha256": sha256_bytes(canonical_json(plan))}
    entered = False
    try:
        if experiment:
            experiment.__enter__()
            entered = True
        for entry in inputs:
            request_id = "expanded-v1-" + entry["question_id"]
            path = output / (request_id + ".json")
            packed = packed_for(ranking[entry["scenario_id"]], graph)
            # A frozen input must actually be the prompt being executed or replayed.
            item = next(i for i in datasets[entry["language"]].items if i.question_id == entry["question_id"])
            system, prompt = generation_request(item.question, graph, packed, "demo", recipe, answer_language=entry["language"])
            if (entry["system"], entry["prompt"]) != (system, prompt):
                raise ValueError("ANSWER_FROZEN_PROMPT_CHANGED")
            if path.exists():
                receipt = json.loads(path.read_bytes())
            elif not execute:
                continue  # NOT_RUN remains visible in the denominator.
            else:
                if limit is not None and calls >= limit:
                    continue
                started = time.monotonic()
                result = experiment.generate(CallContext(tenant_id="demo", run_id=plan["managed_run_id"],
                    request_id=request_id, stage="EVALUATE"), system=system, prompt=prompt, max_output_tokens=800)
                calls += 1
                receipt = {"input_sha256": sha256_bytes(canonical_json(entry)), "recipe_sha256": recipe.fingerprint,
                    "raw_output": result.text, "stop_reason": result.stop_reason, "operation_id": result.operation_id,
                    "observed_model": result.observed_model, "usage": result.usage.model_dump(mode="json"),
                    "cost_upper_bound_micro_usd": result.actual_micro_usd, "seconds": (time.monotonic() - started),
                    "created_at": datetime.now(UTC).isoformat()}
                save(path, receipt)  # Durable paid output before strict parsing, including invalid output.
            row = score_receipt(receipt, entry, packed, graph, recipe, request_id)
            rows.append(row)
            save(output / (request_id + "-scored.json"), row)
            print(json.dumps({k: row[k] for k in ("question_id", "language", "structural_status", "answer_status")}), flush=True)
        report["status"] = "COMPLETE_STRUCTURAL_REVIEW_PENDING" if len(rows) == len(inputs) else "PARTIAL_NOT_RUN_RETAINED"
    except Exception as exc:
        report.update(status="STOPPED_NO_AUTOMATIC_RETRY", error_category=type(exc).__name__)
        # Provider exceptions may contain sensitive details; never print arbitrary text.
        if isinstance(exc, ValueError) and str(exc).isupper() and len(str(exc)) < 160:
            report["safe_error"] = str(exc)
    finally:
        if entered:
            experiment.__exit__(None, None, None)
        report.update(summary(rows, inputs))
        report["new_model_operations"] = gateway.provider.dispatch_attempts if gateway else 0
        report["cases"] = [{k: v for k, v in row.items() if k != "response"} for row in rows]
        save(output / ("canary-report.json" if canary else "report.json"), report)
    return {k: v for k, v in report.items() if k != "cases"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--execute", action="store_true")
    actions.add_argument("--replay", action="store_true")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    directory = args.directory.resolve()
    if not directory.is_relative_to((ROOT / "results/evaluation-expanded").resolve()):
        raise ValueError("ANSWER_EXPERIMENT_OUTSIDE_RESULTS")
    if args.limit is not None and not 1 <= args.limit <= 100:
        raise ValueError("ANSWER_EXPERIMENT_LIMIT_INVALID")
    result = prepare(directory) if args.prepare else evaluate(directory, execute=args.execute, limit=args.limit, canary=args.canary)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
