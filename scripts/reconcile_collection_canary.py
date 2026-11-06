"""Read-only source/cache/managed-ledger reconciliation for the bounded L/D/H canary."""

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import boto3
import numpy as np
from botocore.config import Config

from app.answer_cache import AnswerReceipt
from app.collection_answering import prepare_collection_answer
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import canonical_json, embedding_cache_key, sha256_bytes
from app.embeddings import CachedEmbedding, query_embedding_input
from app.external_gateway import ExternalPriceSheet, ExternalRoutes
from app.generation import GenerationRecipeV9, generation_request, parse_generated, resolve_answer
from app.ledger import DynamoLedgerStore, Ledger
from app.local_bge import BgeEncoder
from scripts.answer_local import ROOT, RUN
from scripts.replay_collection_run import assert_replay


def main():
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=30)
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("RECONCILIATION_IDENTITY_MISMATCH")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes())
    recipe = GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256)
    encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    paths = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(paths["collection_path"]), tenant_id="demo",
                                 embedding_model_id=encoder.recipe("document").model_id, cache_shards=4)
    operations, source_count, citation_count = [], 0, 0
    answer_uri = (ROOT / "data/cache/collection-answers/answers.sqlite3").resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(answer_uri, uri=True) as db:
        cache_rows = db.execute("SELECT state,intent,payload,payload_sha256 FROM answers WHERE namespace='LIVE'").fetchall()
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        for language in ("en", "vi"):
            live = json.loads((ROOT / f"results/collection-answering/ecs-{language}-controlled-live.json").read_bytes())
            replay = json.loads((ROOT / f"results/collection-answering/ecs-{language}-offline-replay.json").read_bytes())
            assert_replay(live, replay)
            text = query_embedding_input(live["search_query"])
            query_recipe = encoder.recipe("query")
            key = embedding_cache_key("demo", text, query_recipe)
            embedding_uri = (ROOT / "data/cache/local-bge/embeddings.sqlite3").resolve().as_uri() + "?mode=ro"
            with sqlite3.connect(embedding_uri, uri=True) as db:
                row = db.execute("SELECT state,payload,payload_sha256 FROM embeddings WHERE namespace='LIVE' AND cache_key=?",
                                 (key,)).fetchone()
            if not row or row[0] != "READY" or sha256_bytes(row[1]) != row[2]:
                raise ValueError("RECONCILIATION_QUERY_CACHE_MISSING_OR_CORRUPT")
            vector = CachedEmbedding.model_validate_json(row[1])
            if vector.cache_key != key or vector.recipe != query_recipe or vector.input_text != text:
                raise ValueError("RECONCILIATION_QUERY_CACHE_IDENTITY")
            query_vector = np.asarray(vector.vector, dtype=np.float32)
            for method in live["methods"]:
                response = method["response"]
                view, prepared = prepare_collection_answer(collection, lexical, live["input"]["original_question"],
                    query_vector, tenant_id="demo", owner_id="local-ldh-owner", run_id=RUN, recipe=recipe,
                    mode=method["effective_method"], search_query=live["search_query"], request_id=response["request_id"])
                if list(prepared.packed.chunk_ids) != method["context_chunk_ids"]:
                    raise ValueError("RECONCILIATION_CONTEXT_MISMATCH")
                for source in method["context_sources"]:
                    chunk = view.chunks[source["chunk_id"]]
                    doc = view.documents[chunk.document_version_id]
                    ref = view.source_refs[chunk.chunk_id]
                    if (source["text"] != doc.normalized_text[source["start_char"]:source["end_char"]]
                            or source["text_sha256"] != sha256_bytes(chunk.text.encode())
                            or source["source_uri"] != doc.source_uri or source["title"] != doc.title
                            or source["shard_id"] != ref.shard_id or source["shard_manifest_sha256"] != ref.manifest_sha256):
                        raise ValueError("RECONCILIATION_SOURCE_MISMATCH")
                    source_count += 1
                matches = [row for row in cache_rows if json.loads(row[1])["context"]["request_id"] == response["request_id"]]
                if len(matches) != 1 or matches[0][0] != "READY":
                    raise ValueError("RECONCILIATION_CACHE_STATE")
                _, intent, payload, checksum = matches[0]
                if sha256_bytes(payload) != checksum:
                    raise ValueError("RECONCILIATION_CACHE_CHECKSUM")
                receipt = AnswerReceipt.model_validate_json(payload)
                if (receipt.namespace != "LIVE" or receipt.recipe_sha256 != recipe.fingerprint
                        or receipt.binding_sha256 != sha256_bytes(canonical_json(prepared.binding.model_dump(mode="json")))):
                    raise ValueError("RECONCILIATION_CACHE_BINDING")
                resolved = resolve_answer(parse_generated(receipt.raw_output, receipt.stop_reason, recipe),
                    snapshot=view, packed=prepared.packed, binding=prepared.binding, recipe=recipe)
                resolved = resolved.model_copy(update={"retrieval_backend": "collection_numpy_global_bm25", "answer_language": language})
                if resolved.model_dump(mode="json") != response:
                    raise ValueError("RECONCILIATION_RESPONSE_MISMATCH")
                citation_count += len(resolved.citations)
                system, prompt = generation_request(prepared.question, view, prepared.packed, "demo", recipe,
                                                    answer_language=language)
                body = canonical_json({"model": routes.generation_model_id,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                    "thinking": {"type": "disabled"}, "temperature": 0, "max_tokens": 800,
                    "stream": False, "response_format": {"type": "json_object"}})
                op = ledger.operation(receipt.operation_id)
                rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id,
                                       routes.source_region, now=op.created_at, live=True)
                if (op.status != "SETTLED" or op.spec.run_id != RUN or op.spec.tenant_id != "demo"
                        or op.spec.request_id != response["request_id"] or op.spec.input_sha256 != sha256_bytes(body)
                        or op.spec.stage != "GENERATE" or op.spec.provider != "DEEPSEEK"
                        or op.spec.model_id != recipe.model_id or op.spec.invocation_id != recipe.invocation_id
                        or op.spec.price_sheet_version != prices.version
                        or op.usage != receipt.usage or op.actual_micro_usd != receipt.original_cost_micro_usd
                        or op.actual_micro_usd != method["new_cost_micro_usd"]
                        or op.actual_micro_usd != rate.quote(op.usage)):
                    raise ValueError("RECONCILIATION_OPERATION_MISMATCH")
                operations.append({"language": language, "method": method["effective_method"],
                    "operation_id": op.operation_id, "status": op.status, "stage": op.spec.stage,
                    "usage": op.usage.model_dump(), "cost_micro_usd": op.actual_micro_usd})
            rewrite = live.get("rewrite_receipt")
            if rewrite:
                op = ledger.operation(rewrite["operation_id"])
                rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id,
                                       routes.source_region, now=op.created_at, live=True)
                if (op.status != "SETTLED" or op.spec.stage != "REWRITE" or op.spec.run_id != RUN
                        or op.spec.provider != "DEEPSEEK" or op.spec.model_id != recipe.model_id
                        or op.spec.price_sheet_version != prices.version
                        or op.actual_micro_usd != rewrite["cost_micro_usd"] or op.usage.model_dump(mode="json") != rewrite["usage"]
                        or op.actual_micro_usd != rate.quote(op.usage)):
                    raise ValueError("RECONCILIATION_REWRITE_MISMATCH")
                operations.append({"language": language, "method": "shared", "operation_id": op.operation_id,
                    "status": op.status, "stage": op.spec.stage, "usage": op.usage.model_dump(), "cost_micro_usd": op.actual_micro_usd})
    if len(operations) != 7 or len({row["operation_id"] for row in operations}) != 7:
        raise ValueError("RECONCILIATION_OPERATION_COUNT")
    total = sum(row["cost_micro_usd"] for row in operations)
    counters = {"run": ledger.store.get(ledger.run_key(RUN)),
        "funding": ledger.store.get(ledger.external_funding_key("kg-r72")),
        "month": ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))}
    baseline = json.loads((ROOT / "results/collection-answering/preflight-live.json").read_bytes())["counters"]
    for label, after in counters.items():
        if (after["settled_micro_usd"] - baseline[label]["settled_micro_usd"] != total
                or after["reserved_micro_usd"] != baseline[label]["reserved_micro_usd"]
                or after["cap_micro_usd"] != baseline[label]["cap_micro_usd"]):
            raise ValueError("RECONCILIATION_COUNTER_DELTA")
    report = {"checked_at": datetime.now(UTC).isoformat(), "status": "PASS_SOURCE_CACHE_LEDGER_REPLAY",
        "aws_read_only": True, "local_caches_opened_read_only": True, "local_report_written": True,
        "model_calls_this_command": 0, "aws_mutations_this_command": 0,
        "reconciled_inference_calls": len(operations), "generation_calls": 6, "shared_rewrite_calls": 1,
        "operations": operations, "source_appearances": source_count, "citation_appearances": citation_count,
        "new_budget_upper_bound_micro_usd": total, "provider_invoice_micro_usd": None,
        "price_sheet_version": prices.version, "counters": counters, "original_reservations_preserved": True,
        "semantic_review": "NOT_HUMAN_REVIEWED", "quality_metrics": None}
    path = ROOT / "results/collection-answering/canary-reconciliation.json"
    path.write_bytes(canonical_json(report) + b"\n")
    print(json.dumps({key: report[key] for key in ("status", "reconciled_inference_calls", "source_appearances",
        "citation_appearances", "new_budget_upper_bound_micro_usd", "original_reservations_preserved")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
