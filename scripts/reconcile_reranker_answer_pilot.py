"""Read-only cache/source/ledger audit; saved R logits, no ONNX or provider calls."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import boto3
from botocore.config import Config

from app.answer_cache import AnswerReceipt
from app.collection_answering import _evidence_view
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context import pack_context
from app.contracts import canonical_json, sha256_bytes
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.generation import (
    AnswerBinding,
    GenerationRecipeV9,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.graph_retrieval import GraphRetrievalResult
from app.ledger import DynamoLedgerStore, Ledger
from app.rag_methods import MethodResultV1
from scripts.answer_local import ROOT, RUN
from scripts.replay_collection_run import assert_replay
from scripts.run_reranker_answer_pilot import check_pair, write_once


def require(condition, category):
    if not condition:
        raise ValueError(category)


def main():
    directory = ROOT / "results/reranker-answering/pilot-v1"
    require(not (directory / "reconciliation.json").exists(), "R_AUDIT_RECEIPT_IMMUTABLE")
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")

    def reads_only(model, **kwargs):
        if model.service_model.service_name in {"sts", "dynamodb"}:
            require(model.name in {"GetCallerIdentity", "GetItem"}, "R_AUDIT_RESOURCE_MUTATION_FORBIDDEN")

    session.events.register("before-call.*.*", reads_only)
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=30)
    require(session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"),
            "R_AUDIT_IDENTITY_MISMATCH")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes())
    recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    plan = json.loads((directory / "plan.json").read_bytes())
    summary = json.loads((directory / "summary.json").read_bytes())
    paths = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    require(collection.manifest_sha256 == plan["collection_sha256"] and recipe.fingerprint == plan["generation_recipe_sha256"],
            "R_AUDIT_PINNED_ARTIFACT_MISMATCH")
    answer_uri = (ROOT / "data/cache/collection-answers/answers.sqlite3").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(answer_uri, uri=True) as db:
        rows = db.execute("SELECT state,intent,payload,payload_sha256 FROM answers WHERE namespace='LIVE'").fetchall()
    operations, cases, sources_count, citations_count = [], [], 0, 0
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        for number in range(1, 4):
            raw = (directory / f"case-{number}.json").read_bytes()
            live = json.loads(raw)
            replay = json.loads((directory / f"case-{number}-replay.json").read_bytes())
            check_pair(live)
            assert_replay(live, replay)
            require(replay["replay_verified_against_sha256"] == sha256_bytes(raw), "R_AUDIT_REPLAY_ORIGINAL_HASH")
            for method in live["methods"]:
                MethodResultV1.model_validate_json(canonical_json(method))
                ids = tuple(method["ranked_chunk_ids"])
                view = _evidence_view(collection, lexical, ids)
                retrieval = GraphRetrievalResult(ids, ids, (), (), method["effective_method"], False, (), 0, 0)
                packed = pack_context(view, retrieval)
                require(list(packed.chunk_ids) == method["context_chunk_ids"]
                        and packed.proxy_tokens == method["context_proxy_tokens"], "R_AUDIT_CONTEXT_MISMATCH")
                for source in method["context_sources"]:
                    chunk = view.chunks[source["chunk_id"]]
                    document = view.documents[chunk.document_version_id]
                    ref = view.source_refs[chunk.chunk_id]
                    require(source["text"] == document.normalized_text[source["start_char"]:source["end_char"]]
                            and source["document_version_id"] == chunk.document_version_id
                            and source["text_sha256"] == sha256_bytes(chunk.text.encode())
                            and source["source_uri"] == document.source_uri and source["title"] == document.title
                            and source["section_path"] == list(chunk.section_path)
                            and source["shard_id"] == ref.shard_id and source["shard_manifest_sha256"] == ref.manifest_sha256,
                            "R_AUDIT_SOURCE_PROVENANCE_MISMATCH")
                    sources_count += 1
                response = method["response"]
                matches = [row for row in rows if json.loads(row[1])["context"]["request_id"] == response["request_id"]]
                require(len(matches) == 1 and matches[0][0] == "READY", "R_AUDIT_CACHE_STATE")
                _, _, payload, checksum = matches[0]
                require(sha256_bytes(payload) == checksum, "R_AUDIT_CACHE_CHECKSUM")
                receipt = AnswerReceipt.model_validate_json(payload)
                binding = AnswerBinding(request_id=response["request_id"], tenant_id="demo", owner_id="local-ldh-owner",
                    mode_requested=method["effective_method"], mode_used=method["effective_method"], used_graph=False)
                require(receipt.binding_sha256 == sha256_bytes(canonical_json(binding.model_dump(mode="json")))
                        and receipt.recipe_sha256 == recipe.fingerprint, "R_AUDIT_CACHE_BINDING")
                resolved = resolve_answer(parse_generated(receipt.raw_output, receipt.stop_reason, recipe),
                    snapshot=view, packed=packed, binding=binding, recipe=recipe)
                resolved = resolved.model_copy(update={"retrieval_backend": "collection_numpy_global_bm25",
                    "answer_language": live["answer_language"]})
                require(resolved.model_dump(mode="json") == response, "R_AUDIT_RESPONSE_MISMATCH")
                citations_count += len(resolved.citations)
                system, prompt = generation_request(live["input"]["original_question"], view, packed, "demo", recipe,
                                                     answer_language=live["answer_language"])
                body = canonical_json({"model": routes.generation_model_id,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "thinking": {"type": "disabled"}, "temperature": 0, "max_tokens": 800,
                    "stream": False, "response_format": {"type": "json_object"}})
                op = ledger.operation(receipt.operation_id)
                rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id,
                                       routes.source_region, now=op.created_at, live=True)
                require(op.status == "SETTLED" and op.spec.run_id == RUN and op.spec.tenant_id == "demo"
                        and op.spec.request_id == response["request_id"] and op.spec.input_sha256 == sha256_bytes(body)
                        and op.spec.stage == "GENERATE" and op.spec.provider == "DEEPSEEK"
                        and op.spec.model_id == recipe.model_id and op.spec.invocation_id == recipe.invocation_id
                        and op.spec.price_sheet_version == prices.version and op.usage == receipt.usage
                        and op.actual_micro_usd == receipt.original_cost_micro_usd == method["new_cost_micro_usd"]
                        and op.actual_micro_usd == rate.quote(op.usage), "R_AUDIT_LEDGER_OPERATION_MISMATCH")
                operations.append({"case": number, "method": method["effective_method"], "operation_id": op.operation_id,
                    "stage": op.spec.stage, "status": op.status, "cost_micro_usd": op.actual_micro_usd,
                    "usage": op.usage.model_dump(mode="json")})
            rewrite = live.get("rewrite_receipt")
            if rewrite:
                op = ledger.operation(rewrite["operation_id"])
                rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id,
                                       routes.source_region, now=op.created_at, live=True)
                require(op.status == "SETTLED" and op.spec.stage == "REWRITE" and op.spec.run_id == RUN
                        and op.spec.request_id == live["run_identity"] and op.spec.provider == "DEEPSEEK"
                        and op.spec.price_sheet_version == prices.version and op.actual_micro_usd == rewrite["cost_micro_usd"]
                        and op.usage.model_dump(mode="json") == rewrite["usage"]
                        and op.actual_micro_usd == rate.quote(op.usage), "R_AUDIT_REWRITE_MISMATCH")
                operations.append({"case": number, "method": "shared", "operation_id": op.operation_id,
                    "stage": op.spec.stage, "status": op.status, "cost_micro_usd": op.actual_micro_usd,
                    "usage": op.usage.model_dump(mode="json")})
            cases.append({"case": number, "live_sha256": sha256_bytes(raw), "same_pool_verified": True,
                "replay_local_reranker_onnx_calls": replay["local_reranker_onnx_calls_this_invocation"],
                "replay_paid_calls": 0, "replay_aws_calls": 0})
    total = sum(op["cost_micro_usd"] for op in operations)
    require(len(operations) == len({op["operation_id"] for op in operations}) == summary["model_dispatch_attempts"] == 7
            and total == summary["new_known_cost_micro_usd"], "R_AUDIT_OPERATION_COUNT_OR_COST")
    counters = {"run": ledger.store.get(ledger.run_key(RUN)),
        "funding": ledger.store.get(ledger.external_funding_key("kg-r72")),
        "month": ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))}
    baseline = json.loads((ROOT / "results/reranker-answering/preflight-ledger-recheck.json").read_bytes())["counters"]
    for label, after in counters.items():
        require(after["settled_micro_usd"] - baseline[label]["settled_micro_usd"] == total
                and after["reserved_micro_usd"] == baseline[label]["reserved_micro_usd"]
                and after["cap_micro_usd"] == baseline[label]["cap_micro_usd"] and not after["halted"],
                "R_AUDIT_COUNTER_DELTA_OR_CAP_MISMATCH")
    report = {"status": "PASS_SOURCE_CACHE_LEDGER_REPLAY", "checked_at": datetime.now(UTC).isoformat(),
        "model_calls_this_command": 0, "onnx_calls_this_command": 0, "aws_resource_mutations_this_command": 0,
        "local_cache_read_only": True, "reconciled_operations": operations, "cases": cases,
        "source_appearances": sources_count, "citation_appearances": citations_count,
        "new_upper_bound_micro_usd": total, "provider_invoice_micro_usd": None,
        "price_sheet_version": prices.version, "counters": counters, "legacy_reservations_preserved": True,
        "human_reviewed": False, "quality_metrics": None}
    write_once(directory / "reconciliation.json", report)
    print(json.dumps({key: report[key] for key in ("status", "source_appearances", "citation_appearances",
        "new_upper_bound_micro_usd", "legacy_reservations_preserved")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
