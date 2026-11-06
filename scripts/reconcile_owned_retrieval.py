"""Read-only audit of local owned-run pins, artifacts, sources and replay; no model/AWS."""

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from app.collections import load_collection
from app.contracts import canonical_json, sha256_bytes
from app.owned_runs import OwnedRunRepository
from app.query_auth import AuthContext
from app.query_state import QueryRepository
from app.run_artifacts import LocalRunBlobs, RunArtifactRepository
from app.run_execution import validate_result
from app.run_storage import SQLiteRunStore

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "data/cache/owned-runs")
    parser.add_argument("--pipeline", type=Path, default=ROOT / "results/corpus-expansion-v2/pipeline.json")
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or not (args.state_dir / "state.sqlite3").is_file():
        raise ValueError("AUDIT_REQUIRES_EXISTING_STATE_AND_NEW_RECEIPT")
    pipeline = json.loads(args.pipeline.read_bytes())
    collection = load_collection(Path(pipeline["collection_path"]), tenant_id="demo",
        embedding_model_id="BAAI/bge-small-en-v1.5", cache_shards=4)
    repo = OwnedRunRepository(SQLiteRunStore(args.state_dir / "state.sqlite3", read_only=True))
    artifacts = RunArtifactRepository(LocalRunBlobs(args.state_dir / "artifacts", read_only=True))
    auth = AuthContext("demo", "00000000-0000-4000-8000-000000000001")
    methods, source_count, input_hashes = [], 0, {}
    for path in args.receipts:
        raw = path.read_bytes()
        input_hashes[path.as_posix()] = sha256_bytes(raw)
        receipt = json.loads(raw)
        assert receipt["status"] == "COMPLETED" and receipt["new_model_calls"] == receipt["new_aws_calls"] == 0
        parent = repo.get(auth, receipt["run"]["run_id"])
        assert parent["release"]["collection_manifest_sha256"] == collection.manifest_sha256
        for child in repo.children(auth, parent["run_id"]):
            assert child["status"] == "COMPLETED"
            result = validate_result(parent, child["method_id"], json.loads(
                artifacts.read(auth, parent["run_id"], child["method_id"], child["result_ref"])))
            for source in result.context_sources:
                shard = collection.get_shard(source["shard_id"])
                chunk = shard.chunks[source["chunk_id"]]
                doc = shard.documents[source["document_version_id"]]
                assert source["shard_manifest_sha256"] == shard.manifest_sha256
                assert source["text"] == chunk.text == doc.normalized_text[source["start_char"]:source["end_char"]]
                assert source["text_sha256"] == sha256_bytes(chunk.text.encode("utf-8"))
                assert source["source_uri"] == doc.source_uri and source["title"] == doc.title
                source_count += 1
            methods.append({"run_id": parent["run_id"], "method": child["method_id"],
                "config_sha256": result.config_sha256, "result_sha256": child["result_ref"]["sha256"],
                "context_sources": len(result.context_sources), "model_cost_micro_usd": 0})
    replay_checked = False
    if args.replay:
        raw = args.replay.read_bytes()
        replay = json.loads(raw)
        original = next(json.loads(p.read_bytes()) for p in args.receipts
            if json.loads(p.read_bytes())["run"]["run_id"] == replay["run"]["run_id"])
        assert replay["idempotent_replay"] and replay["published_messages"] == 0
        assert replay["results"] == original["results"] and replay["run"] == original["run"]
        assert replay["new_model_calls"] == replay["new_aws_calls"] == replay["query_onnx_calls"] == replay["reranker_onnx_calls"] == 0
        input_hashes[args.replay.as_posix()] = sha256_bytes(raw)
        replay_checked = True
    receipt = {"status": "PASS_LOCAL_SOURCE_ARTIFACT_STATE_REPLAY", "checked_at": datetime.now(UTC).isoformat(),
        "collection_manifest_sha256": collection.manifest_sha256, "methods": methods,
        "source_appearances": source_count, "input_sha256": input_hashes, "replay_checked": replay_checked,
        "local_quota_remaining": QueryRepository(repo.store).quota_remaining(),
        "local_active_children": repo.store.get(repo.active_keys(auth)[0])["count"],
        "model_calls_this_command": 0, "aws_calls_this_command": 0, "cloud_deployed": False,
        "quality_metrics": None, "human_reviewed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        output.write(canonical_json(receipt) + b"\n")
    print(json.dumps({k: receipt[k] for k in ("status", "source_appearances", "replay_checked", "local_quota_remaining",
        "local_active_children", "model_calls_this_command", "aws_calls_this_command")}))


if __name__ == "__main__":
    main()
