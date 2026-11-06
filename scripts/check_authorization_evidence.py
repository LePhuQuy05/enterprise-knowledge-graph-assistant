"""Run/replay exposed local G1/G2 source controls; never call a provider."""

import argparse
import json
from collections import Counter
from pathlib import Path
from time import perf_counter

import numpy as np

from app.authorization_evidence import load_authorization_facts
from app.authorization_retrieval import combine_evidence_graph, prepare_authorization_answer
from app.collection_lexical import load_global_lexical_index
from app.contracts import canonical_json, sha256_bytes
from app.embeddings import query_embedding_input
from app.generation import generation_request
from app.reviewed_source_graph import load_reviewed_source_graph
from scripts.prepare_authorization_evidence import ROOT, load_sources
from scripts.run_collection_benchmark import OfflineRuntime, load_inputs

CONTROLS = (
    ("lambda-role", "When does Lambda automatically assume your execution role?"),
    ("lambda-trust", "What must the Lambda execution role trust policy specify about lambda.amazonaws.com?"),
    ("s3-invoke", "Which resource-based policy permission does Amazon S3 need to invoke Lambda?"),
    ("s3-sdk", "Which execution role permissions are needed if Lambda uses the AWS SDK to manage Amazon S3 resources?"),
    ("kms-decrypt", "Which AWS KMS permission is required to download an encrypted object?"),
    ("kms-cross-account", "Which AWS KMS key policy and IAM policy are required for cross-account access?"),
    ("permission-layers", "Explain Lambda trust policy, execution role and Amazon S3 resource-based policy permissions."),
    ("permission-vs-read", "Why can S3 invoke Lambda but the function cannot read an SSE-KMS object?"),
    ("native-two-source", "How does Secrets Manager publish native events to EventBridge, and how does EventBridge encrypt events when the target is another event bus in a different Region?"),
    ("metadata-negative", "Does EventBridge encrypt event metadata?"),
    ("outside-source-coverage", "How does Neptune Analytics handle a graph snapshot?"),
)
CODE_PATHS = ("app/authorization_evidence.py", "app/authorization_retrieval.py", "scripts/check_authorization_evidence.py",
    "scripts/prepare_authorization_evidence.py", "app/context.py", "app/generation.py", "app/method_execution.py")


def check(output, *, replay=False):
    collection, chunks, documents = load_sources()
    _, _, other, _, pipeline = load_inputs()
    if other.manifest_sha256 != collection.manifest_sha256:
        raise ValueError("AUTH_CONTROL_PIPELINE_COLLECTION_MISMATCH")
    selection = json.loads((ROOT / "configs/graph/reviewed-source-runtime-v1.json").read_bytes())
    reviewed = load_reviewed_source_graph(ROOT / selection["release_path"], collection,
        trusted_sha256=selection["release_sha256"], allow_experimental=True)
    auth_selection_path = ROOT / "configs/graph/authorization-evidence-local-v1.json"
    auth_selection = json.loads(auth_selection_path.read_bytes())
    if (auth_selection["version"] != "authorization-evidence-local-selection-v1"
            or auth_selection["source_manifest_sha256"] != collection.manifest_sha256
            or auth_selection["release_path"] != "results/authorization-evidence-v1/release.json"
            or auth_selection["serving_published"] is not False):
        raise ValueError("AUTH_LOCAL_SELECTION_BINDING")
    authorization_path = ROOT / auth_selection["release_path"]
    authorization = load_authorization_facts(authorization_path, collection, chunks, documents,
        trusted_sha256=auth_selection["release_sha256"], allow_experimental=True)
    graph = combine_evidence_graph(collection, reviewed, authorization)
    plan = {"version": "authorization-source-controls-v2", "controls": CONTROLS,
        "collection_manifest_sha256": collection.manifest_sha256, "authorization_release_sha256": authorization.release_sha256,
        "base_release_sha256": reviewed.release_sha256, "combined_release_sha256": graph.release_sha256,
        "authorization_selection_sha256": sha256_bytes(auth_selection_path.read_bytes()),
        "code_sha256": {p: sha256_bytes((ROOT / p).read_bytes()) for p in CODE_PATHS},
        "purpose": "EXPOSED_DEVELOPMENT_SOURCE_MECHANISMS_NOT_INDEPENDENT_BENCHMARK",
        "new_provider_calls": 0, "aws_calls": 0}
    plan_raw = canonical_json(plan) + b"\n"
    if replay:
        if (output / "plan.json").read_bytes() != plan_raw:
            raise ValueError("AUTH_CONTROLS_REPLAY_PLAN_CHANGED")
    else:
        output.mkdir(parents=True, exist_ok=False)
        (output / "plan.json").write_bytes(plan_raw)
    rows, timings = [], []
    with load_global_lexical_index(Path(pipeline["lexical_manifest_path"]), collection) as lexical:
        runtime = OfflineRuntime(collection, lexical, ())
        for name, question in CONTROLS:
            lookup = runtime.cache.resolve_local(collection.manifest.tenant_id, text=query_embedding_input(question),
                recipe=runtime.query_recipe, encoder=runtime.encoder)
            vector = np.asarray(lookup.record.vector, dtype=np.float32)
            for hops in (1, 2):
                started = perf_counter()
                result = prepare_authorization_answer(collection, lexical, graph, question, vector,
                    tenant_id=collection.manifest.tenant_id, owner_id="authorization-source-controls", run_id="authorization-v1",
                    recipe=runtime.recipe, hops=hops, allow_experimental=True)
                timings.append({"control": name, "hops": hops, "milliseconds": round((perf_counter() - started) * 1000, 3)})
                if result.prepared is None:
                    raise ValueError("AUTH_CONTROL_NOT_COMPLETED")
                if name in {"metadata-negative", "outside-source-coverage"} and result.outcome.mechanism_status != "GRAPH_ZERO_EXPANSION":
                    raise ValueError("AUTH_CONTROL_UNRELATED_PATH_EXPANDED")
                system, prompt = generation_request(question, result.view, result.prepared.packed,
                    collection.manifest.tenant_id, result.prepared.recipe)
                rows.append({"control": name, "outcome": result.outcome.model_dump(mode="json"), "trace": result.trace,
                    "packed": json.loads(result.prepared.packed.serialized), "prompt_sha256": sha256_bytes(prompt.encode()),
                    "system_sha256": sha256_bytes(system.encode()), "generation_executed": False})
            print(json.dumps({"control": name, "receipts": len(rows)}), flush=True)
    raw = b"".join(canonical_json(row) + b"\n" for row in rows)
    report = {"status": "LOCAL_AUTHORIZATION_G1_G2_MECHANISMS_NOT_SERVING_OR_ANSWER_ACCURACY", "receipts": len(rows),
        "graph_facts": len(graph.facts), "new_auth_facts": len(authorization.records_json),
        "statuses": dict(Counter(row["outcome"]["mechanism_status"] for row in rows)),
        "controls": [{"control": r["control"], "method": r["outcome"]["requested_method"],
            "status": r["outcome"]["mechanism_status"], "packed_records": len(r["outcome"]["context_assertion_ids"]),
            "max_packed_hops": max((p["hop"] for p in r["trace"]["packed_paths"]), default=0),
            "new_context_chunks": len(r["trace"]["graph_added_context_chunk_ids"])} for r in rows],
        "receipts_sha256": sha256_bytes(raw), "human_reviewed": 0, "new_model_calls": 0, "aws_calls": 0,
        "new_local_embedding_calls": runtime.encoder.inference_calls, "hidden_fallbacks": 0,
        "native_extraction_claimed": False, "permission_grants_asserted": 0, "serving_published": False}
    if replay:
        if (output / "receipts.jsonl").read_bytes() != raw:
            raise ValueError("AUTH_CONTROLS_TRACE_CONTEXT_PROMPT_MISMATCH")
        (output / "replay.json").write_bytes(canonical_json({"status": "EXACT_REPLAY_MATCH", "receipts": len(rows),
            "receipts_sha256": sha256_bytes(raw), "new_model_calls": 0, "aws_calls": 0}) + b"\n")
    else:
        (output / "receipts.jsonl").write_bytes(raw)
        (output / "report.json").write_bytes(canonical_json(report) + b"\n")
        (output / "timing.json").write_bytes(canonical_json({"backend": "LOCAL_WINDOWS_CPU", "rows": timings,
            "includes_query_embedding": False, "cloud_latency_measured": False}) + b"\n")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--replay", action="store_true")
    args = parser.parse_args()
    check(args.output_dir, replay=args.replay)
