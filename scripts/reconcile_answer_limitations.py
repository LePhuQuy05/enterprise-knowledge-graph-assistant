"""Read-only frozen-context V9/V11 receipt/source/ledger audit; never inference."""

import argparse
import json
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import boto3
from botocore.config import Config

from app.answer_assessment import MESSAGES, build_assessment
from app.answer_cache import AnswerReceipt
from app.collection_answering import _evidence_view
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context import pack_context
from app.contracts import canonical_json, sha256_bytes, stable_id
from app.cost import TokenUsage
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.generation import (
    AnswerBinding,
    GenerationRecipeV9,
    GenerationRecipeV11,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.graph_retrieval import GraphRetrievalResult
from app.ledger import DynamoLedgerStore, Ledger
from app.model_gateway import CallContext, safe_failure_code
from scripts.answer_local import ROOT, RUN
from scripts.compare_answer_limitations import (
    IMPLEMENTATIONS,
    OWNER,
    VERSION,
    load_cases,
    source_records,
)
from scripts.reconcile_balanced_query_pilot import (
    body,
    cached_row,
    connection,
    expectation,
    read_json,
    validate_counter_baseline,
)
from scripts.run_reranker_answer_pilot import write_once

DIRECTORY = ROOT / "results/answer-limitations/comparison-v1"
PRICE_PATH = "configs/prices/external-direct-2026-10-24.json"
CRITERIA_PATH = "results/answer-limitations/source-review-criteria-v1.json"
EXPECTED_CRITERIA_SHA256 = "7d5b5b37c979850bbb323574d80692a6aaa53ec0ccad90ffeb1f1c5721161c73"


def require(condition, category):
    if not condition:
        raise ValueError("LIMITATIONS_AUDIT_" + category)


def load_review_criteria(path):
    """Pin the separately preregistered review file without feeding facts to generation."""
    criteria, digest = read_json(path)
    require(digest == EXPECTED_CRITERIA_SHA256, "PREREGISTERED_CRITERIA_HASH")
    return criteria, digest


def validate_run_records(plan, marker, summary, replay, plan_hash):
    require(plan["version"] == VERSION and plan["new_generation_call_cap"] == 6
            and plan["max_output_tokens"] == 800 and plan["run_id"] == RUN
            and plan["sequential_only"] and plan["same_spans_and_untrusted_payload"]
            and all(plan[key] == 0 for key in ("new_retrieval_calls", "new_encoder_calls", "new_planner_calls")),
            "PLAN_CONSTRAINTS")
    require(len(plan["cases"]) == 6 and marker["plan_sha256"] == plan_hash
            and marker["resume_allowed"] is False, "PLAN_OR_MARKER")
    for record in (summary, replay):
        require(record["status"] == "COMPLETED" and record["cost_complete"]
                and record["error_category"] is None and record["plan_sha256"] == plan_hash
                and record["cases"] == 6 and record["quality_metrics"] is None
                and record["human_reviewed"] is False and record["gold"] is False
                and record["holdout"] is False and record["cloud_deployed"] is False
                and all(record[key] == 0 for key in ("new_retrieval_calls", "new_encoder_calls", "new_planner_calls")),
                "INCOMPLETE_OR_MISLABELED_RUN")
    require(summary["dispatches_this_invocation"] == 6
            and replay["dispatches_this_invocation"] == replay["new_known_cost_micro_usd"] == 0,
            "REPLAY_OR_DISPATCH_COUNTS")
    start, end, repeated = (datetime.fromisoformat(row["created_at"]) for row in (marker, summary, replay))
    require(start.utcoffset().total_seconds() == end.utcoffset().total_seconds() == repeated.utcoffset().total_seconds() == 0
            and start <= end <= repeated, "RUN_TIMESTAMPS")


def validate_assessment(response, output, recipe, language):
    if not isinstance(recipe, GenerationRecipeV11):
        require("assessment" not in response.model_dump(mode="json"), "LEGACY_RESPONSE_SHAPE")
        return
    expected = build_assessment(output.limitations, language=language)
    require(response.assessment == expected and response.answer_language == language,
            "LOCALIZED_ASSESSMENT")
    assessment = response.assessment
    require(assessment.policy_origin == "SYSTEM_POLICY" and assessment.account_state_access == "NOT_CONNECTED"
            and assessment.source_scope == "SUPPLIED_DOCUMENTATION_ONLY", "SYSTEM_SCOPE_POLICY")
    for original, gap in zip(output.limitations, assessment.limitations, strict=True):
        message, next_step = MESSAGES[language][original.code]
        require(gap.reported_by == "MODEL_REPORTED" and gap.question_part == original.question_part
                and gap.message == message and gap.next_step == next_step, "LIMITATION_TEMPLATE_BODY")


def rebuild_cached_answer(db, *, view, packed, binding, context, question, language, recipe, result):
    """Revalidate raw READY bytes and rebuild output; this function cannot dispatch."""
    system, prompt = generation_request(question, view, packed, binding.tenant_id, recipe,
                                         answer_language=language)
    binding_sha = sha256_bytes(canonical_json(binding.model_dump(mode="json")))
    key = stable_id("answer", context=context.model_dump(mode="json"), binding_sha=binding_sha,
        request_sha=sha256_bytes(canonical_json([system, prompt])), recipe=recipe.fingerprint,
        snapshot_sha=view.manifest_sha256)
    intent = {"context": context.model_dump(mode="json"), "binding_sha256": binding_sha,
              "recipe_sha256": recipe.fingerprint}
    saved = cached_row(db, "answers", key, intent, AnswerReceipt)
    require(saved == AnswerReceipt.model_validate_json(canonical_json(result["receipt"]))
            and saved.binding_sha256 == binding_sha and saved.recipe_sha256 == recipe.fingerprint
            and saved.operation_id == stable_id("op", **context.model_dump(mode="json")), "RECEIPT_BINDING")
    output = parse_generated(saved.raw_output, saved.stop_reason, recipe)
    response = resolve_answer(output, snapshot=view, packed=packed, binding=binding,
                              recipe=recipe, answer_language=language)
    validate_assessment(response, output, recipe, language)
    changes = {"retrieval_backend": "collection_numpy_global_bm25", "answer_language": language}
    if language == "vi" and response.answer_status == "ABSTAINED" and not isinstance(recipe, GenerationRecipeV11):
        changes["answer"] = "Tài liệu được tìm thấy chưa cung cấp đủ bằng chứng để trả lời câu hỏi này."
    response = response.model_copy(update=changes)
    require(response.model_dump(mode="json") == result["response"], "RESPONSE_RAW_RECONSTRUCTION")
    for citation in response.citations:
        chunk = view.chunks[citation.chunk_id]
        doc = view.documents[chunk.document_version_id]
        require(citation.chunk_id in packed.chunk_ids and citation.document_version_id == doc.document_version_id
                and citation.source_uri == doc.source_uri and citation.title == doc.title
                and citation.section_path == chunk.section_path and citation.quote in chunk.text,
                "CITATION_SOURCE_PROVENANCE")
    return saved, system, prompt, response


def audit_local(report, *, directory=None, root=None):
    root, directory = root or ROOT, directory or DIRECTORY
    plan, plan_hash = read_json(directory / "plan.json")
    marker, marker_hash = read_json(directory / "execution-started.json")
    summary, summary_hash = read_json(directory / "summary.json")
    replay, replay_hash = read_json(directory / "replay.json")
    validate_run_records(plan, marker, summary, replay, plan_hash)
    preflight, preflight_hash = read_json(Path(marker["preflight_path"]))
    require(preflight_hash == marker["preflight_sha256"]
            and preflight["status"] == "READY_READ_ONLY_NO_INFERENCE" and preflight["model_calls"] == 0,
            "EXECUTION_PREFLIGHT")
    require(datetime.fromisoformat(preflight["created_at"]) <= datetime.fromisoformat(marker["created_at"]),
            "PREFLIGHT_TIMESTAMP")
    validate_counter_baseline(preflight["counters"], preflight["created_at"])
    require({key: row["reserved_micro_usd"] for key, row in preflight["counters"].items()}
            == {"run": 640, "funding": 640, "month": 1250}, "LEGACY_RESERVATIONS")
    config_path = root / "configs/evaluation/answer-limitations-development-v1.json"
    require(sha256_bytes(config_path.read_bytes()) == plan["cases_sha256"], "CONFIG_HASH")
    data = load_cases(config_path)
    require(len(plan["cases"]) == len(data.cases), "CONFIG_CASE_COUNT")
    pipeline_path = root / "results/corpus-expansion-v2/pipeline.json"
    paths, pipeline_hash = read_json(pipeline_path)
    route_path, price_path = root / "configs/models/external-direct-v1.json", root / PRICE_PATH
    routes = ExternalRoutes.model_validate_json(route_path.read_bytes())
    prices = ExternalPriceSheet.model_validate_json(price_path.read_bytes())
    require(pipeline_hash == plan["pipeline_sha256"] and sha256_bytes(route_path.read_bytes()) == plan["routes_sha256"]
            and sha256_bytes(price_path.read_bytes()) == plan["price_sheet_sha256"], "PIPELINE_ROUTE_PRICE_HASHES")
    require(routes.generation_model_id == routes.generation_invocation_id == "deepseek-flash"
            and preflight["provider"]["available"]
            and preflight["provider"]["metadata_sha256"] == routes.account_model_metadata_sha256, "PROVIDER_ROUTE")
    require(set(plan["implementation_sha256"]) == set(IMPLEMENTATIONS)
            and all(sha256_bytes((root / path).read_bytes()) == digest
                    for path, digest in plan["implementation_sha256"].items()), "IMPLEMENTATION_PINS")
    originals = {row.baseline_receipt: row.baseline_receipt_sha256 for row in data.cases}
    require(plan["baseline_artifact_sha256"] == originals
            and all(sha256_bytes((root / path).read_bytes()) == digest for path, digest in originals.items()),
            "ORIGINAL_RECEIPT_HASHES")
    criteria, criteria_hash = load_review_criteria(root / CRITERIA_PATH)
    require(criteria["runtime_config_binding"] == {"path": config_path.relative_to(root).as_posix(),
        "sha256": plan["cases_sha256"]} and criteria["label_status"] == "AI_DRAFT"
        and criteria["human_reviewed"] is criteria["gold"] is criteria["holdout"] is False,
        "CRITERIA_BINDING_OR_LABELS")
    require(all(sha256_bytes((root / row["path"]).read_bytes()) == row["sha256"]
                for row in criteria["input_provenance"]), "CRITERIA_PROVENANCE_HASHES")
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    collection.validate_all()
    require(collection.manifest_sha256 == plan["collection_manifest_sha256"], "COLLECTION_HASH")
    recipes = (GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256),
               GenerationRecipeV11(model_metadata_sha256=routes.account_model_metadata_sha256))
    pending, cases, source_count, citations = [], [], 0, 0
    with connection(root / "data/cache/collection-answers/answers.sqlite3") as db, \
            load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        for number, (case, entry) in enumerate(zip(data.cases, plan["cases"], strict=True), 1):
            require(entry["case"] == case.model_dump(mode="json"), "PLAN_CASE_BINDING")
            original = json.loads((root / case.baseline_receipt).read_bytes())
            method = original["methods"][case.method_index]
            require(original["status"] == method["status"] == "COMPLETED" and original["cost_complete"]
                    and original["input"]["original_question"] == case.question
                    and original["answer_language"] == case.language
                    and original["collection_manifest_sha256"] == collection.manifest_sha256
                    and method["effective_method"] == case.effective_method
                    and method["config"]["generation_recipe_sha256"] == recipes[0].fingerprint
                    and sha256_bytes(canonical_json(method["context_sources"])) == case.context_sources_sha256
                    and sha256_bytes(canonical_json(method["context_chunk_ids"])) == case.context_chunk_ids_sha256,
                    "ORIGINAL_INPUT_CONTEXT_BINDING")
            ids = tuple(method["ranked_chunk_ids"])
            view = _evidence_view(collection, lexical, ids)
            packed = pack_context(view, GraphRetrievalResult(ids, ids, (), (), case.effective_method, False, (), 0, 0))
            sources = source_records(view, packed)
            require(list(packed.chunk_ids) == method["context_chunk_ids"] == entry["context_chunk_ids"]
                    and packed.proxy_tokens == method["context_proxy_tokens"] == entry["context_proxy_tokens"]
                    and sources == method["context_sources"]
                    and sha256_bytes(packed.serialized.encode()) == entry["serialized_sha256"],
                    "FROZEN_CONTEXT_RECONSTRUCTION")
            for chunk_id in packed.chunk_ids:
                chunk = view.chunks[chunk_id]
                document = view.documents[chunk.document_version_id]
                require(sha256_bytes(chunk.text.encode()) == chunk.text_sha256
                        and sha256_bytes(document.normalized_text.encode()) == document.normalized_sha256,
                        "SOURCE_TEXT_HASHES")
            live, live_hash = read_json(directory / f"case-{number}.json")
            require(live["case_id"] == case.id and live["context_sources"] == sources, "LIVE_CONTEXT_CASE")
            require(live["baseline"]["response"] == method["response"]
                    and live["baseline"]["cache_hit"] and live["baseline"]["new_cost_micro_usd"] == 0
                    and not live["candidate"]["cache_hit"], "REUSED_BASELINE_OR_NEW_CANDIDATE")
            require(len(entry["variants"]) == 2, "VARIANT_COUNT")
            prompts = []
            for variant, recipe, result in zip(entry["variants"], recipes,
                    (live["baseline"], live["candidate"]), strict=True):
                name = "v11" if isinstance(recipe, GenerationRecipeV11) else "v9"
                require(variant["name"] == result["variant"] == name
                        and variant["recipe"] == recipe.model_dump(mode="json")
                        and variant["recipe_sha256"] == recipe.fingerprint, "VARIANT_RECIPE")
                expected_id = method["response"]["request_id"] if name == "v9" else stable_id("cmp11",
                    version=VERSION, case=case.id, baseline_request=method["response"]["request_id"],
                    context_sha=entry["serialized_sha256"], recipe=recipe.fingerprint, tenant="demo", owner=OWNER)
                require(variant["request_id"] == expected_id == result["response"]["request_id"],
                        "VARIANT_REQUEST_ID")
                binding = AnswerBinding(request_id=expected_id, tenant_id="demo", owner_id=OWNER,
                    mode_requested=case.effective_method, mode_used=case.effective_method, used_graph=False)
                context = CallContext(tenant_id="demo", run_id=RUN, request_id=expected_id, stage="GENERATE")
                receipt, system, prompt, response = rebuild_cached_answer(db, view=view, packed=packed,
                    binding=binding, context=context, question=case.question, language=case.language,
                    recipe=recipe, result=result)
                require(sha256_bytes(system.encode()) == variant["system_sha256"]
                        and sha256_bytes(prompt.encode()) == entry["untrusted_payload_sha256"], "PROMPT_HASHES")
                prompts.append((system, prompt))
                require(result["new_cost_micro_usd"] == (receipt.original_cost_micro_usd if name == "v11" else 0),
                        "INCREMENTAL_COST")
                if name == "v11":
                    pending.append(expectation(context, body(system, prompt, 800), {
                        "operation_id": receipt.operation_id, "cost_micro_usd": receipt.original_cost_micro_usd,
                        "usage": receipt.usage.model_dump(mode="json")}, case=case.id, method=name, output=800))
                citations += len(response.citations)
            require(prompts[0][1] == prompts[1][1] and prompts[0][0] != prompts[1][0], "SAME_EVIDENCE_POLICY_ONLY")
            source_count += len(sources) * 2
            cases.append({"case_id": case.id, "live_sha256": live_hash, "baseline_sha256": case.baseline_receipt_sha256,
                "method": case.effective_method, "context_sources_sha256": case.context_sources_sha256,
                "context_chunk_ids_sha256": case.context_chunk_ids_sha256,
                "same_untrusted_payload_and_spans": True, "raw_response_and_assessment_reconstructed": True})
    total = sum(row["cost_micro_usd"] for row in pending)
    require(len(pending) == len({row["operation_id"] for row in pending}) == 6
            and all(row["context"]["stage"] == "GENERATE" for row in pending)
            and total == summary["new_known_cost_micro_usd"], "NEW_OPERATION_TOTALS")
    require(all(sha256_bytes((root / path).read_bytes()) == digest for path, digest in originals.items()),
            "ORIGINALS_CHANGED_DURING_AUDIT")
    report.update(local_checks="PASS", plan_sha256=plan_hash, summary_sha256=summary_hash,
        replay_sha256=replay_hash, marker_sha256=marker_hash, execution_preflight_sha256=preflight_hash,
        source_criteria_sha256=criteria_hash, original_artifact_sha256=originals, cases=cases,
        source_appearances=source_count, citation_appearances=citations,
        new_upper_bound_micro_usd=total, provider_invoice_micro_usd=None,
        price_sheet_version=prices.version, expected_stage_counts={"GENERATE": 6},
        historical_v9_receipts_rebuilt=6, new_v11_receipts_rebuilt=6)
    return pending, prices, routes, preflight["counters"], marker, summary


def validate_operation(operation, expected, prices, routes, marker, summary):
    op = operation
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id,
                           routes.source_region, now=op.created_at, live=True)
    require(op.status == "SETTLED" and op.call_started and op.dispatch_id is not None and not op.bound_exceeded
            and op.spec.project_id == "kg-r72" and op.spec.provider == "DEEPSEEK"
            and op.spec.billing_source == "DIRECT_API" and op.spec.funding_id == "external-r72-v1"
            and op.spec.model_id == routes.generation_model_id and op.spec.invocation_id == routes.generation_invocation_id
            and op.spec.source_region == routes.source_region and op.spec.price_sheet_version == prices.version
            and all(getattr(op.spec, key) == value for key, value in expected["context"].items())
            and op.operation_id == expected["operation_id"] and op.spec.input_sha256 == expected["input_sha256"]
            and op.usage.model_dump(mode="json") == expected["usage"]
            and op.actual_micro_usd == expected["cost_micro_usd"] == rate.quote(op.usage)
            and 0 < op.usage.input_tokens <= routes.generation_context_ceiling
            and 0 <= op.usage.output_tokens <= expected["output_bound"] == 800
            and op.spec.max_cost_micro_usd == rate.quote(TokenUsage(
                input_tokens=routes.generation_context_ceiling, output_tokens=800))
            and datetime.fromisoformat(marker["created_at"]) <= op.created_at <= op.updated_at
                <= datetime.fromisoformat(summary["created_at"]), "LEDGER_OPERATION")


def validate_after_counters(baseline, counters, total, count):
    require(set(counters) == set(baseline) == {"run", "funding", "month"}, "AFTER_COUNTER_SCOPES")
    for label, after in counters.items():
        before = baseline[label]
        require(after is not None and not after["halted"]
                and after["settled_micro_usd"] - before["settled_micro_usd"] == total
                and after["version"] - before["version"] == 2 * count, "COUNTER_DELTA_OR_HALT")
        require(all(after[key] == before[key] for key in ("PK", "SK", "project_id", "tenant_id", "period",
            "cap_micro_usd", "reserved_micro_usd", "other_cost_estimate_micro_usd", "gross_stop_micro_usd")),
            "COUNTER_CAP_RESERVATION_OR_SCOPE")


def reads_only(report, model, **kwargs):
    service, operation = model.service_model.service_name, model.name
    require((service == "sts" and operation == "GetCallerIdentity")
            or (service == "dynamodb" and operation == "GetItem")
            or (service == "signin" and operation == "CreateOAuth2Token"), "AWS_MUTATION_FORBIDDEN")
    report["aws_operations"].append({"service": service, "operation": operation})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-only", action="store_true")
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    args = parser.parse_args(argv)
    target = args.directory / "reconciliation.json"
    require(not target.exists(), "RECEIPT_IMMUTABLE")
    report = {"status": "CHECKING", "checked_at": datetime.now(UTC).isoformat(),
        "model_calls_this_command": 0, "encoder_calls_this_command": 0, "retrieval_calls_this_command": 0,
        "planner_calls_this_command": 0, "aws_resource_mutations_this_command": 0,
        "local_caches_opened_read_only": True, "aws_operations": [], "human_reviewed": False,
        "quality_metrics": None, "gold": False, "holdout": False, "cloud_deployed": False}

    def forbidden(*args, **kwargs):
        raise ValueError("LIMITATIONS_AUDIT_INFERENCE_ENCODER_OR_RETRIEVAL_FORBIDDEN")

    try:
        with ExitStack() as guards:
            for name in ("app.external_http.OfficialHttpsTransport.request",
                    "app.external_gateway.ExternalModelGateway.generate", "app.local_bge.BgeEncoder.__init__",
                    "app.reranking.OnnxCrossEncoder.__init__", "app.multi_query.retrieve_multi_query",
                    "app.collection_lexical.retrieve_collection_hybrid", "app.collection_answering.retrieve_multi_query",
                    "app.collection_answering.retrieve_collection_hybrid"):
                guards.enter_context(patch(name, forbidden))
            pending, prices, routes, baseline, marker, summary = audit_local(report, directory=args.directory)
            if args.local_only:
                print(json.dumps({"status": "LOCAL_CHECKS_PASSED_AWS_PENDING", "operations": len(pending),
                                  "upper_bound_micro_usd": report["new_upper_bound_micro_usd"]}))
                return 0
            session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
            session.events.register("before-call.*.*", lambda model, **kwargs: reads_only(report, model, **kwargs))
            cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=30)
            require(session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"),
                    "AWS_IDENTITY")
            client = session.client("dynamodb", config=cfg)
            require(client.meta.endpoint_url == "https://dynamodb.us-east-1.amazonaws.com", "LEDGER_ENDPOINT")
            ledger = Ledger(DynamoLedgerStore(client, "kg-r72-query-state"))
            reconciled = []
            for expected in pending:
                op = ledger.operation(expected["operation_id"])
                validate_operation(op, expected, prices, routes, marker, summary)
                reconciled.append({key: expected[key] for key in ("case", "method", "operation_id",
                    "input_sha256", "cost_micro_usd", "usage")} | {"stage": "GENERATE", "status": op.status})
            counters = {label: ledger.store.get((row["PK"], row["SK"])) for label, row in baseline.items()}
            validate_after_counters(baseline, counters, report["new_upper_bound_micro_usd"], len(pending))
            require(all(sha256_bytes((ROOT / path).read_bytes()) == digest
                        for path, digest in report["original_artifact_sha256"].items()), "ORIGINALS_CHANGED")
            report.update(status="PASS_SOURCE_CACHE_LEDGER_REPLAY", reconciled_operations=reconciled,
                reconciled_operation_count=len(reconciled), counters=counters,
                legacy_caps_preserved=True, legacy_reservations_preserved=True, originals_unchanged=True)
    except Exception as exc:
        report.update(status="BLOCKED_AUDIT_NO_RETRY", error_category=safe_failure_code(exc))
        if isinstance(exc, ValueError) and str(exc).startswith("LIMITATIONS_AUDIT_"):
            report["failed_check"] = str(exc)
        if args.local_only:
            print(json.dumps({key: report[key] for key in ("status", "error_category", "failed_check") if key in report}))
            return 2
    write_once(target, report)
    print(json.dumps({key: report[key] for key in ("status", "error_category", "failed_check",
        "source_appearances", "citation_appearances", "new_upper_bound_micro_usd",
        "reconciled_operation_count") if key in report}))
    return 0 if report["status"] == "PASS_SOURCE_CACHE_LEDGER_REPLAY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
