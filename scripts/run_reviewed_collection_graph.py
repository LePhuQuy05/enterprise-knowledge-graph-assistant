"""Run any question on the collection316 graph. Default retrieval-only, no AWS/model.

--execute is for an explicitly funded local answer run; no allowance is initialized here.
Replay with the same idempotency key and a new output path omits --execute.
"""

import argparse
import json
from pathlib import Path

from app.answer_cache import AnswerCache
from app.collection_graph_runner import run_graph_method
from app.collection_graph_runtime import load_collection_graph
from app.collection_lexical import load_global_lexical_index
from app.collections import load_collection
from app.contracts import canonical_json
from app.embeddings import EmbeddingCache
from app.external_gateway import ExternalRoutes
from app.generation import GenerationRecipeV9
from app.local_bge import BgeEncoder
from app.model_gateway import safe_failure_code
from app.rewrite_cache import RewriteCache
from scripts.answer_local import ROOT, RUN, managed_gateway


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--hops", type=int, choices=(1, 2), default=1)
    parser.add_argument("--language", choices=("auto", "en", "vi"), default="auto")
    parser.add_argument("--run-kind", choices=("retrieval", "answer"), default="retrieval")
    parser.add_argument("--idempotency-key", required=True)
    parser.add_argument("--allow-experimental", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not args.allow_experimental:
        parser.error("--allow-experimental is required; AI source review is not human gold")
    if args.execute and args.run_kind != "answer":
        parser.error("retrieval-only cannot enable a gateway")
    if args.output.exists():
        raise ValueError("GRAPH_OUTPUT_EXISTS_KEEP_HISTORY_USE_NEW_REPLAY_PATH")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write(canonical_json({"status": "STARTING", "model_calls": 0, "automatic_retry": False}) + b"\n")
    gateway = None
    try:
        pipeline = json.loads((ROOT / "results/corpus-expansion-v2/pipeline.json").read_bytes())
        graph_receipt = json.loads((ROOT / "results/collection-graph/curated-subset-v1.json").read_bytes())
        collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
            embedding_model_id="BAAI/bge-small-en-v1.5", trusted_manifest_sha256=graph_receipt["collection_manifest_sha256"])
        graph = load_collection_graph(Path(graph_receipt["graph_path"]), collection,
            trusted_sha256=graph_receipt["graph_file_sha256"], allow_experimental=True)
        encoder = BgeEncoder(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
        routes = ExternalRoutes.model_validate_json((ROOT / "configs/models/external-direct-v1.json").read_bytes())
        gateway = managed_gateway(prices_path=ROOT / "configs/prices/external-direct-2026-10-24.json") if args.execute else None
        with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
            report = run_graph_method(collection, lexical, graph, args.question, tenant_id="demo", owner_id="local-graph-owner",
                run_id=RUN, idempotency_key=args.idempotency_key, recipe=GenerationRecipeV9(model_metadata_sha256=routes.account_model_metadata_sha256),
                routes=routes, encoder=encoder, embedding_cache=EmbeddingCache(ROOT / "data/cache/local-bge", namespace="LIVE"),
                answer_cache=AnswerCache(ROOT / "data/cache/collection-graph-answers", namespace="LIVE"),
                rewrite_cache=RewriteCache(ROOT / "data/cache/collection-graph-rewrites", namespace="LIVE"), gateway=gateway,
                hops=args.hops, language=args.language, run_kind=args.run_kind, allow_experimental=True)
        report["model_dispatch_attempts_this_invocation"] = gateway.provider.dispatch_attempts if gateway else 0
        args.output.write_bytes(canonical_json(report) + b"\n")
        print(json.dumps({"status": report["status"], "requested": report["requested_method"], "effective": report["effective_method"],
            "context_assertions": len(report["trace"]["context_assertion_ids"]), "model_dispatches": report["model_dispatch_attempts_this_invocation"],
            "new_cost_micro_usd": report["new_cost_micro_usd"]}))
    except Exception as exc:
        args.output.write_bytes(canonical_json({"status": "STOPPED_NO_AUTOMATIC_RETRY", "error_category": safe_failure_code(exc),
            "model_dispatch_attempts_this_invocation": gateway.provider.dispatch_attempts if gateway else 0,
            "cost_complete": gateway is None, "new_cost_micro_usd": 0 if gateway is None else None,
            "instruction": "Inspect cache intents/receipts and managed ledger before any retry."}) + b"\n")
        raise


if __name__ == "__main__":
    main()
