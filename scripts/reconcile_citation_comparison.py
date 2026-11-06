"""Read-only provenance, durable-cache, replay and ledger audit of the V9/V10 dev comparison."""

import json
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config

from app.answer_cache import AnswerReceipt
from app.collection_answering import prepare_collection_answer
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import (
    LocalBgeRecipe,
    canonical_json,
    embedding_cache_key,
    sha256_bytes,
    stable_id,
)
from app.embeddings import CachedEmbedding, query_embedding_input
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.generation import (
    GenerationRecipeV9,
    GenerationRecipeV10,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.ledger import DynamoLedgerStore, Ledger
from app.model_gateway import safe_failure_code
from scripts.answer_local import ROOT, RUN
from scripts.compare_citation_recipes import OWNER, ComparisonCase

DIRECTORY = ROOT / "results/citation-v10"


def require(condition, category):
    if not condition:
        raise ValueError(category)


def read_query_vector(db, case, recipe):
    if case.mode == "bm25":
        return None
    text = query_embedding_input(case.search_query)
    key = embedding_cache_key("demo", text, recipe)
    row = db.execute("SELECT state,payload,payload_sha256 FROM embeddings WHERE namespace='LIVE' AND cache_key=?",
                     (key,)).fetchone()
    require(row is not None and row[0] == "READY" and sha256_bytes(row[1]) == row[2],
            "RECONCILIATION_QUERY_CACHE_STATE_OR_CHECKSUM")
    record = CachedEmbedding.model_validate_json(row[1])
    require(record.namespace == "LIVE" and record.tenant_id == "demo" and record.recipe == recipe
            and record.cache_key == key and record.input_text == text, "RECONCILIATION_QUERY_CACHE_IDENTITY")
    return np.asarray(record.vector, dtype=np.float32)


def validate_replay(live, replay):
    require(replay["plan_sha256"] == live["plan_sha256"] and replay["cost_complete"]
            and replay["dispatches_this_invocation"] == replay["new_known_cost_micro_usd"] == 0
            and len(replay["results"]) == len(live["results"]), "RECONCILIATION_REPLAY_RUN")
    for original, repeated in zip(live["results"], replay["results"], strict=True):
        require(original["case_id"] == repeated["case_id"] and original["status"] == repeated["status"]
                and len(original["variants"]) == len(repeated["variants"]), "RECONCILIATION_REPLAY_CASE")
        for before, after in zip(original["variants"], repeated["variants"], strict=True):
            same = ("variant", "status", "recipe_sha256", "response", "receipt", "deterministic_no_evidence")
            require(all(before[key] == after[key] for key in same) and after["new_cost_micro_usd"] == 0
                    and (after["receipt"] is None or after["cache_hit"]), "RECONCILIATION_REPLAY_VARIANT")


def reconcile(report):
    raw = (DIRECTORY / "live.json").read_bytes()
    report["original_live_sha256"] = sha256_bytes(raw)
    live = json.loads(raw)
    plan = json.loads((DIRECTORY / "plan.json").read_bytes())
    identity = {key: value for key, value in plan.items() if key not in {"created_at", "plan_sha256"}}
    require(sha256_bytes(canonical_json(identity)) == plan["plan_sha256"] == live["plan_sha256"],
            "RECONCILIATION_PLAN_HASH")
    require(live["cost_complete"] and len(live["results"]) == len(plan["cases"]) == 6,
            "RECONCILIATION_INCOMPLETE_CASES")
    validate_replay(live, json.loads((DIRECTORY / "replay.json").read_bytes()))
    report["replay_matches_live"] = True
    report["plan_sha256"] = plan["plan_sha256"]
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes())
    recipes = (GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256),
               GenerationRecipeV10(model_metadata_sha256=routes.account_model_metadata_sha256))
    paths = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    collection.validate_all()
    require(collection.manifest.collection_version == plan["collection_version"], "RECONCILIATION_COLLECTION")
    query_recipe = LocalBgeRecipe.model_validate(collection.manifest.embedding_recipe.model_dump() | {
        "input_role": "query", "input_recipe_version": "question-text-v1"})
    embedding_uri = (ROOT / "data/cache/local-bge/embeddings.sqlite3").resolve().as_uri() + "?mode=ro"
    answer_uri = (ROOT / "data/cache/collection-answers/answers.sqlite3").resolve().as_uri() + "?mode=ro"
    ready_operations = []
    with sqlite3.connect(embedding_uri, uri=True) as embeddings, sqlite3.connect(answer_uri, uri=True) as answers, \
            load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        for case_plan, case_result in zip(plan["cases"], live["results"], strict=True):
            case = ComparisonCase.model_validate_json(canonical_json(case_plan["case"]))
            require(case_result["case_id"] == case.id and case_result["status"] == "PAIRED_STRUCTURAL_PASS"
                    and len(case_result["variants"]) == len(case_plan["variants"]) == 2,
                    "RECONCILIATION_CASE_ID_OR_STATUS")
            require(case_plan["collection_manifest_sha256"] == collection.manifest_sha256,
                    "RECONCILIATION_COLLECTION_HASH")
            vector = read_query_vector(embeddings, case, query_recipe)
            view, prepared = prepare_collection_answer(collection, lexical, case.question, vector,
                tenant_id="demo", owner_id=OWNER, run_id=RUN, recipe=recipes[0], mode=case.mode,
                request_id=case_plan["variants"][0]["request_id"], search_query=case.search_query)
            require(list(prepared.packed.chunk_ids) == case_plan["context_chunk_ids"]
                    and prepared.packed.proxy_tokens == case_plan["context_proxy_tokens"],
                    "RECONCILIATION_FROZEN_CONTEXT")
            source_rows = []
            for chunk_id in prepared.packed.chunk_ids:
                chunk = view.chunks[chunk_id]
                doc = view.documents[chunk.document_version_id]
                source = view.source_refs[chunk_id]
                require(doc.normalized_text[chunk.start_char:chunk.end_char] == chunk.text
                        and sha256_bytes(chunk.text.encode()) == chunk.text_sha256
                        and sha256_bytes(doc.normalized_text.encode()) == doc.normalized_sha256,
                        "RECONCILIATION_SOURCE_SLICE_OR_HASH")
                source_rows.append({"chunk_id": chunk_id, "document_version_id": doc.document_version_id,
                    "text_sha256": chunk.text_sha256, "shard_id": source.shard_id,
                    "shard_manifest_sha256": source.manifest_sha256})
            report["source_appearances"] += len(source_rows) * 2
            report["contexts"].append({"case_id": case.id, "sources": source_rows,
                                        "context_proxy_tokens": prepared.packed.proxy_tokens})
            for variant_plan, result, recipe in zip(case_plan["variants"], case_result["variants"], recipes, strict=True):
                candidate = replace(prepared, context=prepared.context.model_copy(update={
                    "request_id": variant_plan["request_id"]}), binding=prepared.binding.model_copy(update={
                    "request_id": variant_plan["request_id"]}), recipe=recipe)
                system, prompt = generation_request(case.question, view, candidate.packed, "demo", recipe,
                                                     answer_language=case.language)
                require(recipe.model_dump(mode="json") == variant_plan["recipe"]
                        and recipe.fingerprint == variant_plan["recipe_sha256"] == result["recipe_sha256"]
                        and sha256_bytes(system.encode()) == variant_plan["system_sha256"]
                        and sha256_bytes(prompt.encode()) == case_plan["untrusted_payload_sha256"]
                        and result["response"]["request_id"] == variant_plan["request_id"],
                        "RECONCILIATION_RECIPE_OR_PROMPT")
                if result["receipt"] is None:
                    require(result["deterministic_no_evidence"] and not prepared.packed.chunk_ids
                            and result["new_cost_micro_usd"] == 0 and result["response"]["answer_status"] == "ABSTAINED"
                            and not result["response"]["citations"] and not result["response"]["claims"],
                            "RECONCILIATION_EMPTY_CONTROL")
                    report["deterministic_no_evidence_variants"] += 1
                    continue
                receipt = AnswerReceipt.model_validate_json(canonical_json(result["receipt"]))
                binding_sha = sha256_bytes(canonical_json(candidate.binding.model_dump(mode="json")))
                cache_key = stable_id("answer", context=candidate.context.model_dump(mode="json"),
                    binding_sha=binding_sha, request_sha=sha256_bytes(canonical_json([system, prompt])),
                    recipe=recipe.fingerprint, snapshot_sha=collection.manifest_sha256)
                row = answers.execute("SELECT state,intent,payload,payload_sha256 FROM answers "
                    "WHERE namespace='LIVE' AND cache_key=?", (cache_key,)).fetchone()
                require(row is not None and row[0] == "READY" and sha256_bytes(row[2]) == row[3],
                        "RECONCILIATION_ANSWER_CACHE_STATE_OR_CHECKSUM")
                cached = AnswerReceipt.model_validate_json(row[2])
                require(cached == receipt and receipt.namespace == "LIVE" and receipt.cache_key == cache_key
                        and receipt.recipe_sha256 == recipe.fingerprint and receipt.binding_sha256 == binding_sha
                        and receipt.operation_id == stable_id("op", **candidate.context.model_dump())
                        and json.loads(row[1]) == {"context": candidate.context.model_dump(mode="json"),
                            "binding_sha256": binding_sha, "recipe_sha256": recipe.fingerprint},
                        "RECONCILIATION_DURABLE_RECEIPT_BINDING")
                response = resolve_answer(parse_generated(receipt.raw_output, receipt.stop_reason, recipe),
                    snapshot=view, packed=candidate.packed, binding=candidate.binding, recipe=recipe)
                updates = {"retrieval_backend": "collection_numpy_global_bm25", "answer_language": case.language}
                if case.language == "vi" and response.answer_status == "ABSTAINED":
                    updates["answer"] = "Tài liệu được tìm thấy chưa cung cấp đủ bằng chứng để trả lời câu hỏi này."
                require(response.model_copy(update=updates).model_dump(mode="json") == result["response"],
                        "RECONCILIATION_RESPONSE_REBUILD")
                for citation in response.citations:
                    chunk = view.chunks[citation.chunk_id]
                    doc = view.documents[chunk.document_version_id]
                    require(citation.chunk_id in candidate.packed.chunk_ids
                            and citation.document_version_id == doc.document_version_id
                            and citation.source_uri == doc.source_uri and citation.title == doc.title
                            and citation.section_path == chunk.section_path and citation.quote in chunk.text,
                            "RECONCILIATION_CITATION_SOURCE")
                    report["citation_appearances"] += 1
                body = canonical_json({"model": routes.generation_model_id,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "thinking": {"type": "disabled"}, "temperature": 0, "max_tokens": recipe.max_output_tokens,
                    "stream": False, "response_format": {"type": "json_object"}})
                reused = case_plan["original_baseline_reused"] and result["variant"] == "v9"
                require(result["new_cost_micro_usd"] == (0 if reused else receipt.original_cost_micro_usd)
                        and (not reused or result["cache_hit"]), "RECONCILIATION_INCREMENTAL_COST")
                ready_operations.append((case.id, result["variant"], receipt, sha256_bytes(body), reused))
    report["local_source_cache_checks"] = "PASS"
    require(len(ready_operations) == len({row[2].operation_id for row in ready_operations}) == 10
            and report["deterministic_no_evidence_variants"] == 2, "RECONCILIATION_COUNTS")
    report["phase"] = "AWS_READ_ONLY"
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    config = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=30)
    report["aws_read_attempts"] += 1
    require(session.client("sts", config=config).get_caller_identity()["Arn"].endswith(":user/kg-dev"),
            "RECONCILIATION_AWS_IDENTITY")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=config), "kg-r72-query-state"))
    for case_id, variant, receipt, payload_sha, reused in ready_operations:
        report["aws_read_attempts"] += 1
        operation = ledger.operation(receipt.operation_id)
        rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id,
                               routes.source_region, now=operation.created_at, live=True)
        require(operation.status == "SETTLED" and operation.call_started and not operation.bound_exceeded
                and operation.spec.run_id == RUN and operation.spec.tenant_id == "demo"
                and operation.spec.stage == "GENERATE" and operation.spec.provider == "DEEPSEEK"
                and operation.spec.model_id == routes.generation_model_id
                and operation.spec.invocation_id == routes.generation_invocation_id
                and operation.spec.input_sha256 == payload_sha and operation.spec.price_sheet_version == prices.version
                and operation.spec.funding_id == "external-r72-v1" and operation.usage == receipt.usage
                and operation.actual_micro_usd == receipt.original_cost_micro_usd == rate.quote(operation.usage),
                "RECONCILIATION_OPERATION_STATE_USAGE_OR_COST")
        report["operations"].append({"case_id": case_id, "variant": variant,
            "operation_id": operation.operation_id, "status": operation.status,
            "historical_reused": reused, "usage": operation.usage.model_dump(mode="json"),
            "budget_cost_micro_usd": operation.actual_micro_usd,
            "request_id": operation.spec.request_id, "input_sha256": operation.spec.input_sha256,
            "recipe_sha256": receipt.recipe_sha256, "binding_sha256": receipt.binding_sha256})
    new = [row for row in report["operations"] if not row["historical_reused"]]
    old = [row for row in report["operations"] if row["historical_reused"]]
    new_cost = sum(row["budget_cost_micro_usd"] for row in new)
    historical_cost = sum(row["budget_cost_micro_usd"] for row in old)
    require(len(new) == live["dispatches_this_invocation"] == plan["max_new_generation_calls"] == 9
            and len(old) == 1 and new_cost == live["new_known_cost_micro_usd"] == 12546
            and historical_cost == 1545, "RECONCILIATION_CALL_OR_COST_TOTALS")
    report.update({"reconciled_generation_operations": 10, "new_generation_calls": 9,
        "historical_reused_calls": 1, "new_budget_upper_bound_micro_usd": new_cost,
        "historical_reused_budget_micro_usd": historical_cost,
        "all_receipt_budget_micro_usd": new_cost + historical_cost, "price_sheet_version": prices.version})
    before = json.loads((DIRECTORY / "preflight.json").read_bytes())["counters"]
    keys = {"run": ledger.run_key(RUN), "funding": ledger.external_funding_key("kg-r72"),
            "month": ledger.month_key("kg-r72", before["month"]["period"])}
    for label, key in keys.items():
        report["aws_read_attempts"] += 1
        after = ledger.store.get(key)
        require(after is not None and after["settled_micro_usd"] - before[label]["settled_micro_usd"] == new_cost
                and after["reserved_micro_usd"] == before[label]["reserved_micro_usd"]
                and after["cap_micro_usd"] == before[label]["cap_micro_usd"]
                and after["halted"] == before[label]["halted"], "RECONCILIATION_COUNTER_DELTA_OR_RESERVE")
        report["counters"][label] = after
    require(report["counters"]["run"]["settled_micro_usd"] == 139832
            and report["counters"]["funding"]["settled_micro_usd"] == 264744
            and report["counters"]["run"]["reserved_micro_usd"] == 640
            and report["counters"]["funding"]["reserved_micro_usd"] == 640
            and report["counters"]["month"]["reserved_micro_usd"] == 1250,
            "RECONCILIATION_EXPECTED_COUNTER_VALUES")
    report["original_reservations_preserved"] = True
    report["status"] = "PASS_SOURCE_CACHE_LEDGER_REPLAY"
    report["phase"] = "COMPLETE"


def main():
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "BLOCKED_OR_FAILED",
        "phase": "LOCAL_SOURCE_CACHE", "aws_read_only": True, "local_caches_opened_read_only": True,
        "model_calls_this_command": 0, "aws_mutations_this_command": 0, "query_encoder_calls_this_command": 0,
        "aws_read_attempts": 0, "operations": [], "contexts": [], "counters": {},
        "source_appearances": 0, "citation_appearances": 0, "deterministic_no_evidence_variants": 0,
        "provider_invoice_micro_usd": None, "cost_basis": "PUBLISHED_PEAK_CACHE_MISS_UPPER_BOUND",
        "semantic_review": "NOT_HUMAN_REVIEWED", "quality_metrics": None}
    try:
        reconcile(report)
    except Exception as exc:
        report["error_category"] = safe_failure_code(exc)
        if isinstance(exc, ValueError) and str(exc).startswith("RECONCILIATION_"):
            report["failed_check"] = str(exc)
    report["original_live_sha256_unchanged"] = (
        sha256_bytes((DIRECTORY / "live.json").read_bytes()) == report.get("original_live_sha256"))
    if not report["original_live_sha256_unchanged"]:
        report["status"] = "BLOCKED_OR_FAILED"
        report["failed_check"] = "RECONCILIATION_ORIGINAL_LIVE_CHANGED"
    (DIRECTORY / "reconciliation.json").write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({key: report.get(key) for key in ("status", "phase", "error_category", "failed_check",
        "reconciled_generation_operations", "new_budget_upper_bound_micro_usd", "historical_reused_budget_micro_usd",
        "source_appearances", "citation_appearances", "original_live_sha256_unchanged")}))
    return 0 if report["status"] == "PASS_SOURCE_CACHE_LEDGER_REPLAY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
