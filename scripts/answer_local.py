"""Answer from the frozen baseline locally; explicit paid miss, durable cache and no auto retry."""

import argparse
import json
from pathlib import Path

import boto3
from botocore.config import Config

from app.answer_cache import AnswerCache
from app.answer_pipeline import answer_prepared, prepare_answer
from app.embeddings import EmbeddingCache, query_embedding_input
from app.external_gateway import ExternalModelGateway, ExternalPriceSheet, ExternalRoutes
from app.external_http import OfficialHttpsTransport, load_private_key
from app.generation import GenerationError, GenerationRecipe
from app.ledger import DynamoLedgerStore, Ledger
from app.local_bge import BgeEncoder
from app.model_gateway import GatewayError, safe_failure_code
from app.snapshot import load_snapshot

ROOT = Path(__file__).resolve().parents[1]
RUN = "external-pilot10-r72-v1"  # One existing 0.50 USD run, shared 2 USD funding.


def managed_gateway():
    session = boto3.Session(profile_name="kg-dev", region_name="us-east-1")
    cfg = Config(retries={"total_max_attempts": 1}, connect_timeout=3, read_timeout=10)
    if not session.client("sts", config=cfg).get_caller_identity()["Arn"].endswith(":user/kg-dev"):
        raise GatewayError("Known developer identity required")
    ledger = Ledger(DynamoLedgerStore(session.client("dynamodb", config=cfg), "kg-r72-query-state"))
    run = ledger.store.get(ledger.run_key(RUN))
    funding = ledger.store.get(ledger.external_funding_key("kg-r72"))
    if not run or run["cap_micro_usd"] != 500_000 or run["halted"]:
        raise GatewayError("Existing pilot run must be present and not halted; never reset")
    if not funding or funding["cap_micro_usd"] != 2_000_000 or funding["halted"]:
        raise GatewayError("Existing funding must be present and not halted; never reset")
    routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    prices = ExternalPriceSheet.model_validate_json((ROOT / "configs/prices/external-direct-2026-10-17.json").read_bytes())
    transport = OfficialHttpsTransport({"DEEPSEEK": load_private_key(ROOT, "DEEPSEEK")})
    gateway = ExternalModelGateway(ledger, transport, routes.model_copy(update={
        "live_inference_enabled": True, "live_inference_gate": "EXPLICIT_LOCAL_ANSWER_CACHE_MISS_ONLY",
    }), prices)
    return gateway


def local_preparation(question, mode="hybrid", *, encoder=None):
    receipt = json.loads((ROOT / "results/bge_baseline.json").read_bytes())
    encoder = encoder or BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    snapshot = load_snapshot(Path(receipt["snapshot_directory"]), tenant_id="demo",
        embedding_model_id=encoder.recipe("document").model_id, trusted_manifest_sha256=receipt["manifest_sha256"])
    if snapshot.manifest.embedding_recipe != encoder.recipe("document"):
        raise GenerationError("QUERY_ENCODER_DOES_NOT_MATCH_SNAPSHOT")
    vector = EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE").resolve_local(
        "demo", text=query_embedding_input(question), recipe=encoder.recipe("query"), encoder=encoder)
    from numpy import asarray, float32

    routes = json.loads((ROOT / "configs/models/external-direct-v1.json").read_bytes())
    recipe = GenerationRecipe(model_metadata_sha256=routes["account_model_metadata_sha256"])
    prepared = prepare_answer(snapshot, question, asarray(vector.record.vector, dtype=float32),
        tenant_id="demo", owner_id="local-owner", run_id=RUN, recipe=recipe, mode=mode)
    return snapshot, prepared


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--mode", choices=("vector", "hybrid"), default="hybrid")
    parser.add_argument("--execute", action="store_true", help="Allow one paid DeepSeek call only on a new cache miss")
    args = parser.parse_args()
    snapshot, prepared = local_preparation(args.question, args.mode)
    cache = AnswerCache(ROOT / "data/cache/local-answers", namespace="LIVE")
    gateway = None
    try:
        result = answer_prepared(prepared, snapshot, cache)
    except GenerationError as exc:
        if str(exc) != "ANSWER_CACHE_MISS_NO_GATEWAY":
            raise
        if not args.execute:
            print(json.dumps({"status": "PLANNED_NO_MODEL_CALL", "request_id": prepared.binding.request_id,
                "mode": args.mode, "context_chunks": len(prepared.packed.chunk_ids),
                "context_proxy_tokens": prepared.packed.proxy_tokens, "paid_calls": 0,
                "next": "Use --execute for a paid cache miss; existing failed/unknown requests are never retried."}, indent=2))
            return 0
        gateway = managed_gateway()
        result = answer_prepared(prepared, snapshot, cache, gateway=gateway)
    print(json.dumps({"status": "LOCAL_ANSWER_STRUCTURAL_PASS_SEMANTIC_PENDING",
        "response": result.response.model_dump(mode="json"), "cache_hit": result.cache_hit,
        "new_budget_cost_micro_usd": result.new_cost_micro_usd,
        "model_dispatches": gateway.provider.dispatch_attempts if gateway else 0,
        "context_proxy_tokens": prepared.packed.proxy_tokens, "cloud_deployed": False}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_category": safe_failure_code(exc),
                          "instruction": "Inspect the local receipt; failed or uncertain calls are never retried automatically."}))
        raise SystemExit(1) from None
