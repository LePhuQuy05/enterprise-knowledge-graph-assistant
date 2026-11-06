"""Read-only reconciliation of a completed owned answer: sources, raw stage, usage and budgets."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.collection_answering import _evidence_view
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.context import pack_context
from app.contracts import canonical_json, sha256_bytes
from app.cost import TokenUsage
from app.generation import (
    AnswerBinding,
    GenerationRecipeV9,
    generation_request,
    parse_generated,
    resolve_answer,
)
from app.graph_retrieval import GraphRetrievalResult
from app.owned_runs import OwnedRunRepository
from app.query_state import QueryRepository
from app.run_artifacts import LocalRunBlobs, RunArtifactRepository
from app.run_budget import StageReceipt
from app.run_execution import validate_result
from app.run_storage import DynamoRunStore
from scripts.answer_local import ROOT, managed_gateway
from scripts.run_owned_answer_canary import AUTH, DIRECTORY, PRICES


def require(condition, code):
    if not condition:
        raise ValueError(code)


def audit(directory):
    plan_raw, marker_raw, summary_raw = [(directory / name).read_bytes()
        for name in ("plan.json", "execution-started.json", "summary.json")]
    plan, marker, summary = [json.loads(raw) for raw in (plan_raw, marker_raw, summary_raw)]
    require(summary["status"] == "COMPLETED" and summary["cost_complete"] and summary["new_model_calls"] == 1,
        "OWNED_AUDIT_COMPLETED_SINGLE_CALL_REQUIRED")
    require(marker["plan_sha256"] == sha256_bytes(plan_raw), "OWNED_AUDIT_PLAN_HASH")
    require(all(sha256_bytes((directory / "execution-runner.py" if path == "scripts/run_owned_answer_canary.py"
        else ROOT / path).read_bytes()) == digest for path, digest in plan["implementation_sha256"].items()),
        "OWNED_AUDIT_IMPLEMENTATION_CHANGED")
    gateway = managed_gateway(prices_path=PRICES)
    ledger = gateway.ledger
    repo = OwnedRunRepository(DynamoRunStore(ledger.store.client, ledger.store.table_name))
    parent = repo.get(AUTH, summary["run"]["run_id"])
    require(parent["request"] == summary["run"]["request"] and parent["release"] == plan["release"], "OWNED_AUDIT_RUN_PIN")
    child = repo.child(AUTH, parent["run_id"], "hybrid")
    artifacts = RunArtifactRepository(LocalRunBlobs(ROOT / "data/cache/owned-answer-canary/artifacts", read_only=True))
    result = validate_result(parent, "hybrid", json.loads(artifacts.read(AUTH, parent["run_id"], "hybrid", child["result_ref"])))
    paths = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
    collection = load_collection(Path(paths["collection_path"]), tenant_id=AUTH.tenant_id,
        embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    require(collection.manifest_sha256 == plan["collection_manifest_sha256"], "OWNED_AUDIT_COLLECTION_PIN")
    for source in result.context_sources:
        shard = collection.get_shard(source["shard_id"])
        chunk, doc = shard.chunks[source["chunk_id"]], shard.documents[source["document_version_id"]]
        require(source["shard_manifest_sha256"] == shard.manifest_sha256
            and source["text"] == chunk.text == doc.normalized_text[source["start_char"]:source["end_char"]]
            and source["source_uri"] == doc.source_uri and source["title"] == doc.title,
            "OWNED_AUDIT_ORIGINAL_DOCUMENT_SOURCE")
    stage = repo.store.get((AUTH.partition, f"RUNSTAGE#{parent['run_id']}#GENERATE:hybrid"))
    require(stage["status"] == "RECEIVED" and stage["admission_token"] == parent["admission_token"]
        and sha256_bytes(canonical_json(stage["receipt"])) == stage["receipt_sha256"], "OWNED_AUDIT_RAW_STAGE_HASH")
    receipt = StageReceipt.model_validate(stage["receipt"])
    operation = ledger.operation(receipt.operation_id)
    require(operation.status == "SETTLED" and operation.call_started and operation.usage == receipt.usage
        and operation.actual_micro_usd == receipt.cost_micro_usd == summary["new_model_cost_micro_usd"]
        and operation.spec.input_sha256 == summary["dispatch_body_sha256"][0], "OWNED_AUDIT_OPERATION_RECEIPT")
    rate = gateway.prices.rate_for(operation.spec.model_id, operation.spec.invocation_id,
        operation.spec.source_region, now=operation.created_at, live=True)
    require(rate.quote(receipt.usage) == receipt.cost_micro_usd, "OWNED_AUDIT_COST_RECOMPUTATION")
    require(operation.spec.max_cost_micro_usd == rate.quote(TokenUsage(input_tokens=1048576, output_tokens=800)),
        "OWNED_AUDIT_BOUND")
    with load_global_lexical_index(Path(paths["lexical_manifest_path"]), collection) as lexical:
        view = _evidence_view(collection, lexical, result.ranked_chunk_ids)
        packed = pack_context(view, GraphRetrievalResult(result.ranked_chunk_ids, result.ranked_chunk_ids, (), (), "hybrid", False, (), 0, 0))
        require(packed.chunk_ids == result.context_chunk_ids, "OWNED_AUDIT_CONTEXT_PACK")
        recipe = GenerationRecipeV9(model_metadata_sha256=gateway.routes.account_model_metadata_sha256)
        system, prompt = generation_request(parent["request"]["original_question"], view, packed, AUTH.tenant_id, recipe, answer_language="en")
        body = canonical_json({"model": "deepseek-flash", "messages": [{"role": "system", "content": system},
            {"role": "user", "content": prompt}], "thinking": {"type": "disabled"}, "temperature": 0,
            "max_tokens": 800, "stream": False, "response_format": {"type": "json_object"}})
        require(sha256_bytes(body) == operation.spec.input_sha256, "OWNED_AUDIT_EXACT_MODEL_INPUT")
        binding = AnswerBinding(request_id=result.response.request_id, tenant_id=AUTH.tenant_id, owner_id=AUTH.user_id,
            mode_requested="hybrid", mode_used="hybrid", used_graph=False)
        restored = resolve_answer(parse_generated(receipt.raw_output, receipt.stop_reason, recipe), snapshot=view,
            packed=packed, binding=binding, recipe=recipe, answer_language="en")
        restored = restored.model_copy(update={"answer_language": "en", "retrieval_backend": "collection_numpy_global_bm25"})
        require(restored == result.response, "OWNED_AUDIT_RAW_TO_CITED_ANSWER")
    before = marker["preflight"]["counters"]
    after = {name: ledger.store.get((row["PK"], row["SK"])) for name, row in before.items()}
    for name, previous in before.items():
        current = after[name]
        require(current["cap_micro_usd"] == previous["cap_micro_usd"]
            and current["settled_micro_usd"] >= previous["settled_micro_usd"] + receipt.cost_micro_usd
            and current["reserved_micro_usd"] == previous["reserved_micro_usd"], "OWNED_AUDIT_CAP_OR_OBLIGATION_CHANGED")
    require(repo.store.get(repo.active_keys(AUTH)[0])["count"] == 0, "OWNED_AUDIT_ACTIVE_CHILDREN")
    return {"status": "PASS_MANAGED_SOURCE_RECEIPT_INPUT_USAGE_LEDGER", "checked_at": datetime.now(UTC).isoformat(),
        "run_id": parent["run_id"], "collection_manifest_sha256": collection.manifest_sha256,
        "source_appearances": len(result.context_sources), "citations": len(result.response.citations),
        "claims": len(result.response.claims), "raw_receipt_sha256": stage["receipt_sha256"],
        "settled_micro_usd": receipt.cost_micro_usd, "usage": receipt.usage.model_dump(mode="json"),
        "reservation_micro_usd": operation.spec.max_cost_micro_usd, "counters": after,
        "quota_remaining": QueryRepository(repo.store).quota_remaining(), "active_children": 0,
        "new_model_calls": 0, "aws_reads": True, "writes": False, "quality_metrics": None, "human_reviewed": False,
        "cloud_deployed": False, "plan_sha256": sha256_bytes(plan_raw), "summary_sha256": sha256_bytes(summary_raw)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=DIRECTORY)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("OWNED_AUDIT_OUTPUT_EXISTS")
    result = audit(args.directory)
    from scripts.run_owned_answer_canary import write_once

    write_once(args.output, result)
    print(json.dumps({k: result[k] for k in ("status", "source_appearances", "citations", "settled_micro_usd", "new_model_calls")}))


if __name__ == "__main__":
    main()
