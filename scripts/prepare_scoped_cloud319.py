"""Create a local immutable319 cloud candidate, without AWS/provider calls."""

import argparse
import hashlib
import json
import shutil
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from aws_cdk import App
from aws_cdk.assertions import Template

from app.contracts import canonical_json, sha256_bytes
from app.ingestion.text import TOKENIZER_CACHE, TOKENIZER_SHA256, TOKENIZER_URL
from app.scoped_cloud_assets import materialize_scoped_graph, plan_scoped_graph
from app.scoped_run_cloud import ScopedCloudConfigV2
from app.scoped_run_contracts import ScopedRunReleaseV2
from app.workbench_assets import materialize_workbench_assets, plan_workbench_assets
from infra.scoped_runs import ACCOUNT, STACK_NAME, ScopedRunsStack
from scripts.migrate_collection319_graph import OUTPUT, load_migrated

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/"results/funding-transition/review-v1/query-funding-transition.zip"


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(canonical_json(value))


def repackage(target, config):
    pin = json.loads((ROOT/"results/scoped-runtime-linux-2026-11-05-v3/stage.json").read_bytes())["base_package_sha256"]
    with BASE.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != pin:
            raise ValueError("SCOPED_PACKAGE_BASE_CHANGED")
    sources = {path.relative_to(ROOT).as_posix(): path.read_bytes() for path in (ROOT/"app").rglob("*.py")
        if not path.is_symlink() and path.resolve().is_relative_to((ROOT/"app").resolve())}
    for name in ("configs/models/external-direct-v1.json", "configs/models/bge-small-en-v1.5-onnx.json",
            "configs/models/ms-marco-MiniLM-L2-v2-onnx.json", "configs/extraction/aws-typed-catalog-v7.json", config["prices"]["path"]):
        sources[name] = (ROOT/name).read_bytes()
    asset = TOKENIZER_CACHE/hashlib.sha1(TOKENIZER_URL.encode()).hexdigest()
    if sha256_bytes(asset.read_bytes()) != TOKENIZER_SHA256:
        raise ValueError("SCOPED_PACKAGE_TOKENIZER_PIN")
    sources["data/tokenizer-cache/"+asset.name] = asset.read_bytes()
    config_raw = canonical_json(config)
    config_sha = sha256_bytes(config_raw)
    sources[f"configs/releases/scoped-cloud-{config_sha}.json"] = config_raw
    with zipfile.ZipFile(BASE) as base, zipfile.ZipFile(target, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        members = {}
        for entry in base.infolist():
            name = entry.filename
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts or "\\" in name or entry.flag_bits & 1:
                raise ValueError("SCOPED_PACKAGE_UNSAFE_BASE")
            if entry.is_dir() or relative.parts[0] in {"app", "configs", "data", "bin"} or (
                    "__pycache__" in relative.parts or relative.suffix == ".pyc"):
                continue
            if name in members:
                raise ValueError("SCOPED_PACKAGE_DUPLICATE_BASE")
            members[name] = entry
        if members.keys() & sources.keys() or sum(m.file_size for m in members.values())+sum(map(len, sources.values())) >= 250*1024*1024:
            raise ValueError("SCOPED_PACKAGE_SIZE_OR_COLLISION")
        for name in sorted(members.keys() | sources.keys()):
            raw = sources[name] if name in sources else base.read(members[name])
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system, info.external_attr, info.compress_type = 3, 0o644 << 16, zipfile.ZIP_DEFLATED
            output.writestr(info, raw, compresslevel=6)
    with target.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"sha256": digest, "bytes": target.stat().st_size, "base_dependency_zip_sha256": pin,
        "config_sha256": config_sha, "source_sha256": {name: sha256_bytes(raw) for name, raw in sources.items()},
        "network": "none", "native_checked": False}


def prepare(out):
    out = out.resolve()
    if out.exists() or not out.is_relative_to(ROOT/"results"):
        raise ValueError("SCOPED_CANDIDATE_NEW_RESULTS_DIRECTORY_REQUIRED")
    if shutil.disk_usage(ROOT).free < 2*1024**3:
        raise ValueError("SCOPED_CANDIDATE_DISK_MARGIN")
    out.mkdir(parents=True)
    collection, runtime, graph = load_migrated()
    frozen_scope = json.loads((ROOT/"results/scoped-owned319-preflight-2026-11-05-v1/scope.json").read_bytes())
    if any(sha256_bytes((ROOT/name).read_bytes()) != digest for name, digest in frozen_scope["code_sha256"].items()):
        raise ValueError("SCOPED_CANDIDATE_FROZEN_PROTOCOL_CHANGED")
    base, sources = plan_workbench_assets(runtime["collection_path"], runtime["lexical_manifest_path"],
        ROOT/"data/models/bge-small-en-v1.5", ROOT/"configs/models/bge-small-en-v1.5-onnx.json",
        ROOT/"data/models/ms-marco-MiniLM-L2-v2", ROOT/"configs/models/ms-marco-MiniLM-L2-v2-onnx.json")
    base_path = materialize_workbench_assets(out/"assets"/base.version, base, sources)
    graph_bundle = plan_scoped_graph(OUTPUT, collection, graph_release_sha256=graph.release_sha256)
    graph_path = materialize_scoped_graph(out/"assets"/graph_bundle.version, OUTPUT, graph_bundle, collection)
    old = ScopedRunReleaseV2.model_validate_json((ROOT/"results/scoped-owned319-preflight-2026-11-05-v1/release.json").read_bytes())
    if old.collection_manifest_sha256 != collection.manifest_sha256 or old.graph_fact_count != len(graph.facts):
        raise ValueError("SCOPED_CANDIDATE_RELEASE_SOURCE_CHANGED")
    configs = {key: c.model_copy(update={"fixture_only": False}) for key, c in old.configs.items()}
    release = old.model_copy(update={"budget_run_id": "collection319-scoped-answer-v2", "configs": configs,
        "method_config_sha256": {key: c.fingerprint for key, c in configs.items()}})
    settings = json.loads((ROOT/"results/collection316-serving/review-v2/settings.json").read_bytes())
    bucket = settings["artifacts"]["bucket"]
    prices_path = "configs/prices/external-direct-2026-11-05.json"
    config = ScopedCloudConfigV2.model_validate({"schema_version": 2,
        "queue_url": f"https://sqs.us-east-1.amazonaws.com/{ACCOUNT}/{STACK_NAME}-work",
        "results_bucket": STACK_NAME+"-results-"+ACCOUNT,
        "assets": {"bucket": bucket, "version": base.version, "manifest_sha256": sha256_bytes(base_path.read_bytes()),
            "manifest_bytes": base_path.stat().st_size, "tenant_id": "demo"},
        "graph_assets": {"bucket": bucket, "version": graph_bundle.version, "manifest_sha256": sha256_bytes(graph_path.read_bytes()),
            "manifest_bytes": graph_path.stat().st_size},
        "prices": {"path": prices_path, "sha256": sha256_bytes((ROOT/prices_path).read_bytes())},
        "release": release.model_dump(mode="json"), "public_config": settings["public_config"]}).model_dump(mode="json")
    write(out/"cloud-config.json", config)
    write(out/"deployment-inputs.json", {key: settings[key] for key in ("api_id", "authorizer_id", "deepseek_secret_arn")})
    package = repackage(out/"scoped-runtime.zip", config)
    write(out/"package.json", package)
    app = App(outdir=str(out/"cdk"), analytics_reporting=False)
    stack = ScopedRunsStack(app, config=config, package_sha256=package["sha256"],
        **{key: settings[key] for key in ("api_id", "authorizer_id", "deepseek_secret_arn")})
    template = Template.from_stack(stack).to_json()
    write(out/"template.json", template)
    scope = {"version": "scoped-cloud319-local-candidate-v1", "checked_at": datetime.now(UTC).isoformat(),
        "status": "LOCAL_ONLY_NATIVE_AND_BUDGET_PREFLIGHT_PENDING", "documents": release.document_count,
        "chunks": release.chunk_count, "graph_runtime_facts": release.graph_fact_count,
        "graph_source_documents": release.graph_source_document_count, "base_bundle_bytes": base.total_bytes,
        "graph_bundle_bytes": sum(f.bytes for f in graph_bundle.files), "resources": len(template["Resources"]),
        "package_sha256": package["sha256"], "config_sha256": package["config_sha256"],
        "template_sha256": sha256_bytes(canonical_json(template)), "model_calls": 0, "aws_calls": 0,
        "funding_initialized": False, "cloud_deployed": False, "frontend_changed": False,
        "open_gates": ["exact_package_linux", "live_session_funding_model_prices", "native_owned_canary",
            "frontend_v2_browser", "fresh_resource_iam_review", "user_release_approval"]}
    write(out/"scope.json", scope)
    print(json.dumps(scope), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out)
