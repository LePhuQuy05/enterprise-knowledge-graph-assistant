"""Four native V5 facet calls, then bounded Q/QB recall arms; no answers.

Default freezes scope. Execute has no retry/resume of uncertain calls. Replay
uses the new V5 cache only, blocks sockets and compares exact source receipts.
"""

import argparse
import json
from collections import Counter
from contextlib import ExitStack
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.collection_answering import _evidence_view
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.cost import TokenUsage
from app.document_recall_v2 import DocumentRecallRecipe, build_recall_pool, prepare_recall_answer
from app.embeddings import query_embedding_input
from app.external_gateway import ExternalRoutes
from app.faceted_query_v5 import SYSTEM, FacetPlanCacheV5, preparation_context
from app.graph_collection_migration import collection_records
from app.graph_retrieval import GraphRetrievalResult
from app.model_gateway import safe_failure_code
from app.source_dependency_audit import SourceDependency, audit_source_dependencies
from app.source_message_packing import pack_source_message
from scripts.audit_collection319_methods import source_reachability
from scripts.check_collection319_methods import QUESTIONS, preparation_payload, provider_body
from scripts.migrate_collection319_graph import load_migrated
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT
from scripts.run_collection_benchmark import OfflineRuntime

RUN = "literal-facet-recall-v5-canary-v1"
OUTPUT = ROOT / "results/facet-recall-v5-canary-v1"
CACHE = ROOT / "data/cache/facet-recall-v5-canary-v1"
PARENT = ROOT / "results/collection319-methods-v1"
STOP = 20000
CODE = ("app/faceted_query_v5.py", "app/document_recall_v2.py", "scripts/run_facet_recall_v5.py",
    "app/query_planning_v2.py", "app/query_preparation_v3.py", "app/source_dependency_audit.py",
    "app/source_message_packing.py", "app/generation.py", "app/external_gateway.py",
    "app/collection_answering.py", "app/collection_lexical.py", "app/collections.py")


def inputs():
    payload = json.loads((PARENT / "receipts.json").read_bytes())
    parent_report = json.loads((PARENT / "report.json").read_bytes())
    parent_scope = json.loads((PARENT / "scope.json").read_bytes())
    if (parent_report["receipts_sha256"] != sha256_bytes((PARENT / "receipts.json").read_bytes())
            or any(sha256_bytes((ROOT / p).read_bytes()) != h for p, h in parent_scope["code_sha256"].items())):
        raise ValueError("FACET_PARENT_RECEIPTS_OR_CODE_CHANGED")
    rewrites = {(r["intent"], r["language"]): r for r in payload["preparations"] if r["stage"] == "REWRITE"}
    tasks = []
    for intent, languages in QUESTIONS.items():
        for language, question in languages.items():
            prior = rewrites.get((intent, language))
            search = prior["validated"]["search_query"] if prior else question
            if prior and (prior["validated"]["question_sha256"] != sha256_bytes(question.encode())
                    or json.loads(prior["receipt"]["raw_output"])["search_query"] != search):
                raise ValueError("FACET_PARENT_REWRITE_BINDING")
            tasks.append({"intent": intent, "language": language, "question": question, "search": search,
                "reused_v3_rewrite_receipt_sha256": sha256_bytes(canonical_json(prior["receipt"])) if prior else None})
    return tasks, payload


def scope(collection, tasks):
    result = {"version": RUN, "purpose": "EXPOSED_RECALL_QUALIFICATION_NOT_ANSWER_ACCURACY",
        "scope_revision": 1,
        "predecessor_v4_scope_sha256": sha256_bytes((ROOT / "results/facet-recall-v4-canary-v1/scope-v2.json").read_bytes()),
        "predecessor_v4_stop_audit_sha256": sha256_bytes((ROOT / "results/facet-recall-v4-canary-v1/stopped-ledger-audit.json").read_bytes()),
        "v4_failed_receipts_reused_or_repaired": False,
        "collection_manifest_sha256": collection.manifest_sha256, "tasks": tasks,
        "native_facet_calls_max": 4, "generation_calls": 0, "extraction_calls": 0,
        "reused_native_v3_rewrites": 2, "parent_receipts_sha256": sha256_bytes((PARENT / "receipts.json").read_bytes()),
        "window_radii": [1, 2], "arms": ["V5_CHUNK_Q", "V5_CHUNK_QB", "V5_WINDOW_R1_Q", "V5_WINDOW_R1_QB", "V5_WINDOW_R2_Q", "V5_WINDOW_R2_QB"],
        "pool_cap": 40, "context_chunk_cap": 8, "user_message_proxy_tokens": 4096,
        "transient_query_occurrences_max": 160, "prior_transient_query_occurrences_max": 120,
        "title_documents_scored": collection.manifest.counts.documents, "document_window_cap": 6,
        "base_candidates_reserved": 16, "title_depth_per_query": 4,
        "run_cap_micro_usd": 500000, "observed_stop_micro_usd": STOP,
        "source_annotations_used_for_ranking": False, "new_graph_records": 0,
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE},
        "routes_sha256": sha256_bytes((ROOT / "configs/models/external-direct-v1.json").read_bytes()),
        "price_sheet_sha256": sha256_bytes((ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()),
        "source_review_annotations_sha256": sha256_bytes((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes()),
        "human_reviewed": 0, "serving_published": False}
    immutable(OUTPUT / "scope.json", result)
    return result


class ScopedGateway:
    def __init__(self, base, manifest, tasks):
        self.base, self.routes, self.provider = base, base.routes, base.provider
        self.allowed = {preparation_context(manifest, t["question"], t["search"], t["language"], RUN).model_dump_json():
            (SYSTEM, preparation_payload(t["question"], t["search"], t["language"]), 256) for t in tasks}
        self.seen, self.cost, self.halted = set(), 0, False

    def generate(self, context, *, system, prompt, max_output_tokens):
        key = context.model_dump_json()
        if (self.halted or key in self.seen or len(self.seen) >= 4 or self.cost >= STOP
                or self.allowed.get(key) != (system, prompt, max_output_tokens)):
            raise ValueError("FACET_NATIVE_DISPATCH_BOUND")
        self.seen.add(key)
        try:
            reply = self.base.generate(context, system=system, prompt=prompt, max_output_tokens=max_output_tokens)
        except Exception:
            self.halted = True
            raise
        self.cost += reply.actual_micro_usd
        return reply


def managed_gateway(frozen, tasks):
    import boto3
    from botocore.config import Config

    from app.external_gateway import ExternalModelGateway, ExternalPriceSheet
    from app.external_http import OfficialHttpsTransport, deepseek_preflight, load_private_key
    from app.funding_policy import TRANSITION_PROFILE, validate_external_funding
    from app.ledger import DynamoLedgerStore, Ledger

    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=5, read_timeout=15)
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise ValueError("FACET_OPERATOR")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    funding = validate_external_funding(ledger, profile=TRANSITION_PROFILE)
    month = ledger.store.get(ledger.month_key("kg-r72", datetime.now(UTC).strftime("%Y-%m")))
    if (funding.cap_micro_usd != 8000000 or funding.cap_micro_usd - funding.settled_micro_usd - funding.reserved_micro_usd < 500000
            or not month or month["cap_micro_usd"] != 12000000 or month["halted"]):
        raise ValueError("FACET_FUNDING_OR_MONTH_CAP")
    raw = (ROOT / "configs/prices/external-direct-2026-10-24.json").read_bytes()
    if sha256_bytes(raw) != frozen["price_sheet_sha256"]:
        raise ValueError("FACET_PRICE_PIN")
    prices = ExternalPriceSheet.model_validate_json(raw)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    rate = prices.rate_for(routes.generation_model_id, routes.generation_invocation_id, routes.source_region, now=datetime.now(UTC), live=True)
    reserve = rate.quote(TokenUsage(input_tokens=routes.generation_context_ceiling, output_tokens=256))
    if reserve + STOP > 500000:
        raise ValueError("FACET_RUN_ADMISSION")
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")}, timeout_seconds=25)
    provider = deepseek_preflight(transport)
    if not provider["available"] or provider["metadata_sha256"] != routes.account_model_metadata_sha256:
        raise ValueError("FACET_PROVIDER_CHANGED")
    ledger.initialize_run("kg-r72", "demo", RUN, cap_micro_usd=500000)
    counter = ledger.store.get(ledger.run_key(RUN))
    if counter["halted"] or counter["reserved_micro_usd"] or counter["settled_micro_usd"]:
        raise ValueError("FACET_RUN_PREVIOUSLY_ATTEMPTED_RECONCILE_FIRST")
    immutable(OUTPUT / "preflight.json", {"checked_at": datetime.now(UTC).isoformat(), "funding": funding.model_dump(mode="json"),
        "provider": provider, "worst_single_reservation_micro_usd": reserve, "observed_stop_micro_usd": STOP})
    base = ExternalModelGateway(ledger, transport, routes.model_copy(update={"live_inference_enabled": True,
        "live_inference_gate": "FOUR_NATIVE_TASK_FACET_CALLS_NO_ANSWERS_OR_EXTRACTION"}), prices)
    return ScopedGateway(base, frozen["collection_manifest_sha256"], tasks), ledger


def _round_robin_pool(rankings, pool):
    order, seen = [], set()
    for rank in range(40):
        for row in rankings:
            if rank < len(row["ranking"]):
                key = row["ranking"][rank]
                if key in pool and key not in seen:
                    order.append(key)
                    seen.add(key)
    if set(order) != set(pool):
        raise ValueError("FACET_CHUNK_QB_POOL")
    return tuple(order)


def run(*, execute=False, replay=False):
    collection, paths, _ = load_migrated()
    tasks, parent = inputs()
    frozen = scope(collection, tasks)
    if not execute and not replay:
        print(json.dumps({"status": "FROZEN_SCOPE_NO_CALLS", "native_calls_max": 4, "generation_calls": 0}))
        return
    gateway, ledger = managed_gateway(frozen, tasks) if execute else (None, None)
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    cache = FacetPlanCacheV5(CACHE, collection.manifest_sha256)
    preparations, rows = [], []
    documents, chunks, _ = collection_records(collection)
    controls = json.loads((ROOT / "results/permission-context-qualification-v1/receipts.json").read_bytes())
    dependencies = tuple(SourceDependency.model_validate_json(canonical_json(r["dependency"])) for r in controls["source_controls"])
    dependencies += (SourceDependency.model_validate_json(canonical_json(controls["parent_kms_dependency"])),)
    try:
        for task in tasks:
            context = preparation_context(collection.manifest_sha256, task["question"], task["search"], task["language"], RUN)
            lookup = cache.resolve(context, question=task["question"], search_query=task["search"], language=task["language"], routes=routes, gateway=gateway)
            if bool(gateway) == lookup.cache_hit:
                raise ValueError("FACET_NATIVE_OR_REPLAY_IDENTITY")
            prepared = task | {"context": context.model_dump(mode="json"), "plan": lookup.plan.model_dump(mode="json"),
                "receipt": lookup.receipt, "payload_sha256": sha256_bytes(provider_body(SYSTEM, preparation_payload(task["question"], task["search"], task["language"])))}
            immutable(OUTPUT / "preparations" / f"{task['intent']}-{task['language']}.json", prepared)
            preparations.append(prepared)
            print(json.dumps({"intent": task["intent"], "language": task["language"], "facets": len(lookup.plan.facets), "cache_hit": lookup.cache_hit}), flush=True)
        with ExitStack() as stack:
            for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
                stack.enter_context(patch(target, side_effect=AssertionError("FACET_RETRIEVAL_NETWORK_BLOCKED")))
            with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
                runtime = OfflineRuntime(collection, lexical, ())
                for task, native in zip(tasks, preparations, strict=True):
                    from app.faceted_query_v5 import FacetPlanV5

                    plan = FacetPlanV5.model_validate_json(canonical_json(native["plan"]))
                    queries = (task["search"], *(f.search_query for f in plan.facets))
                    vectors = tuple(np.asarray(runtime.cache.resolve_local("demo", text=query_embedding_input(query),
                        recipe=runtime.query_recipe, encoder=runtime.encoder).record.vector, dtype=np.float32) for query in queries)
                    for radius in (1, 2):
                        pool = build_recall_pool(collection, lexical, question=task["question"], search=task["search"], language=task["language"],
                            plan=plan, vectors=vectors, recipe=DocumentRecallRecipe(radius=radius))
                        trace = json.loads(pool.trace_json)
                        if radius == 1:
                            raw_pool = tuple(trace["original_fused_pool"])
                            for balanced in (False, True):
                                order = _round_robin_pool(trace["query_rankings"], raw_pool) if balanced else raw_pool
                                view = _evidence_view(collection, lexical, order)
                                result = GraphRetrievalResult(order, raw_pool, (), (), "hybrid_multi_query", False, (), 0, 0)
                                packing = pack_source_message(view, result, question=task["question"], recipe=runtime.recipe,
                                    tenant_id="demo", answer_language=task["language"])
                                rows.append(task | {"arm": "V5_CHUNK_QB" if balanced else "V5_CHUNK_Q", "pool_ids": raw_pool,
                                    "context_chunk_ids": packing.packed.chunk_ids, "source_message": packing.packed.serialized,
                                    "user_message_proxy_tokens": packing.packed.proxy_tokens,
                                    "context_audit": audit_source_dependencies(documents, chunks, packing.packed.chunk_ids, dependencies, tenant_id="demo"),
                                    "source_reachability": [source_reachability(dep, raw_pool, packing.packed.chunk_ids, chunks) for dep in dependencies],
                                    "graph_records_used": 0, "generation_executed": False})
                        for balanced in (False, True):
                            view, answer, audit = prepare_recall_answer(collection, lexical, pool, balanced=balanced,
                                recipe=runtime.recipe, run_id=RUN, owner_id="facet-recall-local")
                            rows.append(task | {"arm": f"V5_WINDOW_R{radius}_" + ("QB" if balanced else "Q"),
                                "pool_ids": pool.pool_ids, "pool_fingerprint": pool.fingerprint, "pool_trace": trace,
                                "bundles": json.loads(pool.bundles_json), "binding": answer.binding.model_dump(mode="json"),
                                "context_chunk_ids": answer.packed.chunk_ids, "source_message": answer.packed.serialized,
                                "user_message_proxy_tokens": answer.packed.proxy_tokens, "packing_audit": audit,
                                "context_audit": audit_source_dependencies(documents, chunks, answer.packed.chunk_ids, dependencies, tenant_id="demo"),
                                "source_reachability": [source_reachability(dep, pool.pool_ids, answer.packed.chunk_ids, chunks) for dep in dependencies],
                                "graph_records_used": 0, "generation_executed": False})
                    print(json.dumps({"intent": task["intent"], "language": task["language"], "recall_cells": len(rows)}), flush=True)
        immutable(OUTPUT / "receipts.json", {"native_preparations": preparations, "recall_cells": rows,
            "recorded_v3_q_qb_baseline": [r for r in parent["cells"][:32] if r["method"].startswith("hybrid_multi_query")]})
        cost = sum(r["receipt"]["cost_micro_usd"] for r in preparations)
        if ledger:
            for row in preparations:
                op = ledger.operation(row["receipt"]["operation_id"])
                if (op.status != "SETTLED" or not op.call_started or op.bound_exceeded or op.spec.run_id != RUN
                        or op.spec.input_sha256 != row["payload_sha256"] or op.usage.model_dump(mode="json") != row["receipt"]["usage"]
                        or op.actual_micro_usd != row["receipt"]["cost_micro_usd"]):
                    raise ValueError("FACET_OPERATION_LEDGER_MISMATCH")
            counter = ledger.store.get(ledger.run_key(RUN))
            funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
            if counter["reserved_micro_usd"] or counter["settled_micro_usd"] != cost:
                raise ValueError("FACET_RUN_COST_RECONCILIATION")
            immutable(OUTPUT / "ledger-audit.json", {"status": "FOUR_SETTLED_PAYLOAD_USAGE_COST_MATCH", "checked_at": datetime.now(UTC).isoformat(),
                "run_counter": counter, "funding_counter": funding,
                "headroom_micro_usd": funding["cap_micro_usd"] - funding["settled_micro_usd"] - funding["reserved_micro_usd"]})
        if frozen["code_sha256"] != {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE}:
            raise ValueError("FACET_CODE_CHANGED_DURING_RUN")
        summary = {"status": "FACET_RECALL_ARMS_REPLAY_MATCH" if replay else "FACET_RECALL_ARMS_RECORDED",
            "native_calls_this_invocation": len(gateway.seen) if gateway else 0, "native_preparation_receipts": len(preparations),
            "receipt_cost_micro_usd": cost, "recall_cells": len(rows), "recorded_v3_baseline_cells": 8,
            "receipts_sha256": sha256_bytes((OUTPUT / "receipts.json").read_bytes()),
            "annotation_statuses": dict(Counter(c["status"] for row in rows for c in row["context_audit"]["checks"])),
            "generation_calls": 0, "extraction_calls": 0, "quality_accuracy_measured": False, "human_reviewed": 0, "serving_published": False}
        immutable(OUTPUT / ("replay.json" if replay else "report.json"), summary)
        print(json.dumps(summary))
    except Exception as exc:
        immutable(OUTPUT / "stopped.json", {"status": "STOPPED_RAW_CACHE_RETAINED_NO_RETRY_OR_FALLBACK",
            "error": safe_failure_code(exc), "dispatch_attempts": len(gateway.seen) if gateway else 0,
            "observed_receipted_cost_micro_usd": gateway.cost if gateway else 0})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--execute", action="store_true")
    choice.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    with ExitStack() as stack:
        if args.replay:
            for target in ("socket.socket.connect", "socket.socket.connect_ex", "socket.create_connection"):
                stack.enter_context(patch(target, side_effect=AssertionError("FACET_REPLAY_NETWORK_BLOCKED")))
        run(execute=args.execute, replay=args.replay)
