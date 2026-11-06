"""Exact ZIP + portable319 assets probe, offline and without real credentials.

Run inside the pinned Lambda image with /candidate and /inputs read-only and
/output writable. This validates construction/ranking, not paid answering/IAM.
"""

import hashlib
import json
import os
import resource
import socket
import sys
import time
import zipfile
from pathlib import Path


def forbidden(*args, **kwargs):
    raise AssertionError("SCOPED_PACKAGE_NETWORK_FORBIDDEN")


def main():
    started = time.perf_counter()
    socket.socket.connect = socket.socket.connect_ex = socket.create_connection = forbidden
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    os.environ["TOKENIZER_CACHE_DIR"] = "/tmp/tokenizer-cache"
    root, package = Path("/candidate"), Path("/tmp/scoped-package")
    pinned = json.loads((root/"package.json").read_bytes())
    archive_path = root/"scoped-runtime.zip"
    with archive_path.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != pinned["sha256"]:
            raise ValueError("SCOPED_PACKAGE_ZIP_PIN")
    with zipfile.ZipFile(archive_path) as archive:
        for item in archive.infolist():
            path = Path(item.filename)
            if path.is_absolute() or ".." in path.parts or "\\" in item.filename or item.flag_bits & 1:
                raise ValueError("SCOPED_PACKAGE_UNSAFE_MEMBER")
        archive.extractall(package)
    sys.path.insert(0, str(package))
    os.chdir(package)
    import boto3
    from pydantic import SecretStr

    from app.contracts import canonical_json, sha256_bytes
    from app.embeddings import EmbeddingCache
    from app.external_gateway import ExternalModelGateway
    from app.external_http import OfficialHttpsTransport
    from app.generation import GenerationRecipeV9
    from app.local_bge import BgeEncoder
    from app.profiled_generation import ProfiledGenerationRecipe
    from app.profiled_source_context import SourceContextProfile
    from app.rag_runner import _validate_encoder
    from app.reranking import OnnxCrossEncoder
    from app.scoped_cloud_assets import ScopedGraphBundle, open_scoped_graph
    from app.scoped_cloud_state import ScopedCloudDynamoRunStore
    from app.scoped_method_flow import METHODS, prepare_scoped_flow
    from app.scoped_owned_runs import ScopedRunRepository
    from app.scoped_query_v6 import ScopedPlanV6, validate_source_facets
    from app.scoped_run_cloud import load_cloud_config, make_scoped_quote_controller
    from app.scoped_run_executor import ScopedOwnedAnswerExecutor
    from app.workbench_assets import WorkbenchAssetBundle, open_workbench_assets

    for name, digest in pinned["source_sha256"].items():
        if sha256_bytes((package/name).read_bytes()) != digest:
            raise ValueError("SCOPED_PACKAGE_SOURCE_PIN")
    os.environ["KG_SCOPED_CONFIG_PATH"] = f"configs/releases/scoped-cloud-{pinned['config_sha256']}.json"
    os.environ["KG_SCOPED_CONFIG_SHA256"] = pinned["config_sha256"]
    config = load_cloud_config()
    b_root = root/"assets"/config["assets"]["version"]
    g_root = root/"assets"/config["graph_assets"]["version"]
    for directory, key in ((b_root, "assets"), (g_root, "graph_assets")):
        raw = (directory/"bundle.json").read_bytes()
        if len(raw) != config[key]["manifest_bytes"] or sha256_bytes(raw) != config[key]["manifest_sha256"]:
            raise ValueError("SCOPED_PACKAGE_ASSET_MANIFEST_PIN")
    base_bundle = WorkbenchAssetBundle.model_validate_json((b_root/"bundle.json").read_bytes())
    graph_bundle = ScopedGraphBundle.model_validate_json((g_root/"bundle.json").read_bytes())
    cells = []
    with open_workbench_assets(b_root, base_bundle, tenant_id="demo") as assets:
        graph = open_scoped_graph(g_root, graph_bundle, assets.collection)
        encoder = BgeEncoder(assets.root/"models/bge", assets.root/"models/bge/manifest.json", threads=2)
        from types import SimpleNamespace

        runtime = SimpleNamespace(encoder=encoder, query_recipe=_validate_encoder(assets.collection, encoder),
            cache=EmbeddingCache(Path("/tmp/scoped-vectors"), namespace="LIVE"),
            reranker=OnnxCrossEncoder(assets.root/"models/reranker", assets.root/"models/reranker/manifest.json", threads=2))
        # Dummy credentials only allow local SDK construction. No send/get call
        # occurs; socket methods above reject all actual network dispatch.
        db = boto3.client("dynamodb", region_name="us-east-1", aws_access_key_id="offline-fixture", aws_secret_access_key="offline-fixture")
        store = ScopedCloudDynamoRunStore(db, "kg-r72-query-state")
        controller = make_scoped_quote_controller(store, config["prices"])
        transport = OfficialHttpsTransport({"DEEPSEEK": SecretStr("offline-placeholder-no-provider-access")})
        controller.bind_gateway(ExternalModelGateway(controller.ledger, transport, controller.routes, controller.prices))
        runtime.recipe = GenerationRecipeV9(model_metadata_sha256=controller.routes.account_model_metadata_sha256)
        repo = ScopedRunRepository(store, clock=controller.ledger.clock, answer_budget=controller)
        executor = ScopedOwnedAnswerExecutor(assets.collection, assets.lexical, graph, runtime,
            controller=controller, repository=repo, authorize=lambda _: None)
        executor.check_release(config["release"])
        setup_seconds = time.perf_counter()-started
        for case in json.loads(Path("/inputs/preparations.json").read_bytes()):
            task, receipt = case["task"], case["receipt"]
            plan = ScopedPlanV6.model_validate_json(canonical_json(case["plan"]))
            if plan.receipt_sha256 != sha256_bytes(canonical_json(receipt)) or plan.facets != validate_source_facets(
                    receipt["raw_output"], task["question"], task["search"]):
                raise ValueError("SCOPED_PACKAGE_REPLAYED_PLAN_BINDING")
            for method in METHODS:
                flow = prepare_scoped_flow(assets.collection, assets.lexical, graph, runtime,
                    question=task["question"], search=task["search"], language=task["language"], method=method,
                    run_id="owned319-offline-preflight", owner_id="offline-source-check", recipe=ProfiledGenerationRecipe(
                        model_metadata_sha256=controller.routes.account_model_metadata_sha256),
                    profile=SourceContextProfile(max_chunks=12), plan=plan)
                expected = case["references"][method]
                if tuple(expected["context_chunk_ids"]) != flow.context.chunk_ids or expected["pool_sha256"] != flow.trace["pool_fingerprint"]:
                    raise ValueError("SCOPED_PACKAGE_WINDOWS_LINUX_RETRIEVAL_PARITY")
                if flow.trace["method_requested"] != flow.trace["method_used"] or flow.trace["hidden_fallbacks"]:
                    raise ValueError("SCOPED_PACKAGE_HIDDEN_FALLBACK")
                cells.append({"intent": task["intent"], "language": task["language"], "method": method,
                    "status": flow.status, "chunks": len(flow.context.chunk_ids), "used_graph": flow.binding.used_graph})
        if transport.dispatch_attempts:
            raise ValueError("SCOPED_PACKAGE_UNEXPECTED_PROVIDER_DISPATCH")
        report = {"status": "PACKAGED_CLOUD_RELEASE_AND_POOLS_NO_DISPATCH", "package_sha256": pinned["sha256"],
            "config_sha256": pinned["config_sha256"], "documents": executor.release.document_count,
            "chunks": executor.release.chunk_count, "graph_runtime_facts": len(graph.facts),
            "graph_source_documents": executor.release.graph_source_document_count, "cells": cells,
            "windows_linux_rank_context_parity": True, "hidden_fallbacks": 0, "setup_seconds": setup_seconds,
            "total_seconds": time.perf_counter()-started, "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "provider_calls": transport.dispatch_attempts, "aws_calls": 0, "real_credentials_mounted": False,
            "planner_origin": "REPLAYED_NATIVE_RECEIPTS_NOT_NEW_OWNED_PLANS", "native_answers": 0,
            "quality_metrics": None, "cloud_deployed": False}
    with Path("/output/native.json").open("xb") as stream:
        stream.write(canonical_json(report))
    print(json.dumps({k: v for k, v in report.items() if k != "cells"}), flush=True)


if __name__ == "__main__":
    main()
