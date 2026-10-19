"""Local tools and explicit, gated AWS operations."""

import argparse
import json
import sqlite3
import sys
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from botocore.exceptions import BotoCoreError, ClientError

from app import contracts
from app.contracts import Citation, Claim, QueryRequest, QueryResponse
from app.fixtures import (
    FIXTURE_EMBEDDING_MODEL,
    FIXTURE_TENANT,
    build_fixture,
    fixture_vector,
)
from app.retrieval import retrieve_baseline
from app.snapshot import load_snapshot


def export_schemas(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    names = (
        "DocumentVersion", "Chunk", "Entity", "EvidenceSpan", "Mention", "Qualifier",
        "ExtractionRecipe", "RelationAssertion", "EmbeddingRecipe", "ExternalEmbeddingRecipe", "LocalBgeRecipe", "EmbeddingRecord",
        "SnapshotCounts", "SnapshotFile", "SnapshotManifest", "QueryRequest", "Citation", "Claim",
        "QueryResponse",
    )
    for name in names:
        schema = getattr(contracts, name).model_json_schema()
        (output / f"{name}.schema.json").write_text(
            json.dumps(schema, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8"
        )
    return {"status": "EXPORTED", "schema_version": 2, "models": list(names)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Knowledge graph assistant: local tools and gated AWS operations")
    sub = parser.add_subparsers(dest="command", required=True)
    fixture = sub.add_parser("fixture-demo", help="Synthetic fixture; no real AWS/model answers")
    fixture.add_argument("--output-root", type=Path, default=Path("results/fixture_demo"))
    fixture.add_argument("--question", default="What does Service A use?")
    fixture.add_argument("--mode", choices=("vector", "hybrid"), default="hybrid")
    validate = sub.add_parser("snapshot-validate")
    validate.add_argument("--snapshot-dir", type=Path, required=True)
    validate.add_argument("--tenant", required=True)
    validate.add_argument("--embedding-model", required=True)
    validate.add_argument("--allow-fixture", action="store_true")
    validate.add_argument("--trusted-manifest-sha256")
    schemas = sub.add_parser("export-schemas")
    schemas.add_argument("--output-dir", type=Path, default=Path("schemas/v2"))
    preflight = sub.add_parser("preflight", help="Read-only AWS metadata, no inference or writes")
    preflight.add_argument("--profile", default="kg-dev")
    preflight.add_argument("--region", choices=("us-east-1",), default="us-east-1")
    preflight.add_argument("--output", type=Path, default=Path("results/aws_metadata_preflight.json"))
    quote = sub.add_parser("cost-quote", help="Token-cost forecast only; no model or ledger writes")
    quote.add_argument("--input-tokens", type=int, required=True)
    quote.add_argument("--output-tokens", type=int, default=0)
    quote.add_argument("--embedding", action="store_true")
    quote.add_argument("--price-sheet", type=Path,
                       default=Path("configs/prices/aws-us-east-1-2026-10-16.json"))
    smoke = sub.add_parser("model-smoke", help="Default: offline forecast. Execution requires confirmed funding.")
    smoke.add_argument("--execute", action="store_true")
    smoke.add_argument("--embedding-only", action="store_true")
    sub.add_parser("prepare-tokenizer", help="Explicit one-time download of a hash-pinned proxy tokenizer asset")
    ingest = sub.add_parser("ingest", help="Local AWS-document collector/chunker; no model calls")
    ingest.add_argument("--manifest", type=Path, default=Path("configs/corpus/aws-r72-v1.json"))
    ingest.add_argument("--output-root", type=Path, default=Path("data/ingestion"))
    ingest.add_argument("--tenant", default="demo")
    ingest.add_argument("--limit", type=int)
    ingest.add_argument("--offline", action="store_true", help="Require local raw cache; no source network calls")
    ingest.add_argument("--refresh", action="store_true", help="Explicitly refetch sources; retain immutable versions")
    ingest.add_argument("--report", type=Path, default=Path("results/ingestion_report.json"))
    plan = sub.add_parser("embedding-plan", help="Inspect corpus/cache needs offline; never invokes a model")
    plan.add_argument("--corpus-dir", type=Path, required=True)
    plan.add_argument("--tenant", default="demo")
    plan.add_argument("--cache-dir", type=Path, default=Path("data/embedding-cache"))
    plan.add_argument("--report", type=Path, default=Path("results/embedding_plan.json"))
    extract = sub.add_parser("extraction-plan", help="Inspect extraction cache/cost bounds; no model calls")
    extract.add_argument("--corpus-dir", type=Path, required=True)
    extract.add_argument("--tenant", default="demo")
    extract.add_argument("--cache-dir", type=Path, default=Path("data/extraction-cache"))
    extract.add_argument("--aliases", type=Path, default=Path("configs/extraction/aws-approved-aliases-v1.json"))
    extract.add_argument("--price-sheet", type=Path,
                         default=Path("configs/prices/aws-us-east-1-2026-10-16.json"))
    extract.add_argument("--report", type=Path, default=Path("results/extraction_plan.json"))
    lexical = sub.add_parser("lexical-search", help="Real-corpus BM25 excerpts; no embeddings or generation")
    lexical.add_argument("--corpus-dir", type=Path, required=True)
    lexical.add_argument("--tenant", default="demo")
    lexical.add_argument("--question", required=True)
    lexical.add_argument("--top-k", type=int, default=5)
    lexical.add_argument("--report", type=Path, default=Path("results/lexical_search.json"))
    for name in ("archive-discover", "archive-fetch", "archive-status", "archive-search",
                 "archive-export", "archive-retry", "archive-refresh"):
        command = sub.add_parser(name, help="Large official-doc archive; independent of MVP count caps")
        command.add_argument("--policy", type=Path, default=Path("configs/corpus/aws-official-archive-v1.json"))
        command.add_argument("--root", type=Path, default=Path("data/aws-official-archive"))
        command.add_argument("--report", type=Path)
        if name in {"archive-discover", "archive-fetch"}:
            command.add_argument("--max-resources", type=int, default=25 if name == "archive-discover" else 100,
                                 help="Per-run batch; 0 drains the known queue. Not a total corpus limit.")
            command.add_argument("--max-seconds", type=int, default=600,
                                 help="Time budget per run, checked between resources; repeat to resume.")
            command.add_argument("--cached-only", action="store_true", help="Reprocess only cached queued resources; no source HTTP")
        if name == "archive-search":
            command.add_argument("--question", required=True)
            command.add_argument("--top-k", type=int, default=5)
        if name == "archive-export":
            command.add_argument("--output", type=Path, required=True)
            command.add_argument("--inventory", action="store_true", help="Export all URL states, including excluded/raw-only")
        if name in {"archive-retry", "archive-refresh"}:
            command.add_argument("--kind", choices=("SITEMAP", "PAGE"), required=True)
        if name == "archive-retry":
            command.add_argument("--refetch", action="store_true", help="Fetch failed resources again instead of retrying raw cache")
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare-tokenizer":
            from app.ingestion.text import prepare_tokenizer

            output = prepare_tokenizer()
        elif args.command == "ingest":
            from app.ingestion.pipeline import ingest as run_ingest

            report = run_ingest(args.manifest, args.output_root, tenant_id=args.tenant,
                                limit=args.limit, offline=args.offline, refresh=args.refresh)
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            output = {key: report[key] for key in (
                "status", "corpus_version", "corpus_directory", "sources_requested",
                "sources_validated", "unique_documents", "chunks", "max_chunk_proxy_tokens",
                "source_cache_hits", "http_requests_this_run", "failures", "report_path",
                "inference_calls", "semantic_source_review",
            )}
            output["report"] = str(args.report)
        elif args.command == "extraction-plan":
            from app.corpus import load_corpus
            from app.cost import PriceSheet
            from app.extraction import AliasDictionary
            from app.extraction_cache import ExtractionCache, plan_extraction

            corpus = load_corpus(args.corpus_dir, tenant_id=args.tenant)
            prices = PriceSheet.model_validate_json(args.price_sheet.read_bytes())
            output = plan_extraction(corpus, ExtractionCache(args.cache_dir),
                                     AliasDictionary.load(args.aliases), prices, now=datetime.now(UTC))
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            output["report"] = str(args.report)
        elif args.command in {"embedding-plan", "lexical-search"}:
            from app.corpus import load_corpus

            corpus = load_corpus(args.corpus_dir, tenant_id=args.tenant)
            if args.command == "embedding-plan":
                from app.embeddings import EmbeddingCache, plan_embeddings, titan_recipe

                output = plan_embeddings(corpus, EmbeddingCache(args.cache_dir), titan_recipe())
            else:
                from app.retrieval import LexicalIndex

                request = QueryRequest(question=args.question, mode="hybrid")
                index = LexicalIndex(corpus.documents, corpus.chunks, tenant_id=corpus.tenant_id)
                ranked = index.search(request.question, k=args.top_k)
                docs = {d.document_version_id: d for d in corpus.documents}
                chunks = {c.chunk_id: c for c in corpus.chunks}
                output = {"status": "LEXICAL_ONLY", "corpus_version": corpus.version,
                          "inference_calls": 0, "question": request.question, "hits": [
                              {"chunk_id": key, "score": score,
                               "source_uri": docs[chunks[key].document_version_id].source_uri,
                               "title": docs[chunks[key].document_version_id].title,
                               "section_path": list(chunks[key].section_path),
                               "start_char": chunks[key].start_char,
                               "excerpt": chunks[key].text[:600]} for key, score in ranked
                          ], "notice": "Source excerpts only; no generated answer or semantic quality claim."}
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
            if args.command == "embedding-plan":
                output = {k: v for k, v in output.items() if k not in {"missing_keys", "blocked_keys"}}
            output["report"] = str(args.report)
        elif args.command.startswith("archive-"):
            from app.archive import Archive, ArchivePolicy

            policy = ArchivePolicy.model_validate_json(args.policy.read_bytes())
            read_only = args.command in {"archive-status", "archive-search", "archive-export"}
            with Archive(args.root, policy, read_only=read_only) as archive:
                if args.command in {"archive-discover", "archive-fetch"}:
                    def progress(value):
                        print(json.dumps({"archive_progress": value}), file=sys.stderr, flush=True)

                    output = archive.run("SITEMAP" if args.command == "archive-discover" else "PAGE",
                                         max_resources=args.max_resources, max_seconds=args.max_seconds,
                                         progress=progress, cached_only=args.cached_only)
                elif args.command == "archive-search":
                    output = {"status": "ARCHIVE_LEXICAL_EXCERPTS", "inference_calls": 0,
                              "hits": archive.search(args.question, limit=args.top_k),
                              "notice": "SQLite FTS5 search of normalized archive; not the B2 benchmark or a generated answer."}
                elif args.command == "archive-export":
                    output = archive.export(args.output, inventory=args.inventory)
                elif args.command in {"archive-retry", "archive-refresh"}:
                    count = archive.retry_failed(args.kind, refetch=args.refetch) if (
                        args.command == "archive-retry") else archive.refresh(args.kind)
                    output = {"status": "ARCHIVE_RESOURCES_REQUEUED", "count": count, "inference_calls": 0}
                else:
                    output = archive.status()
            if args.report:
                args.report.parent.mkdir(parents=True, exist_ok=True)
                args.report.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
                output["report"] = str(args.report)
            output = {k: v for k, v in output.items() if k != "families"}
        elif args.command == "preflight":
            from app.preflight import probe_aws

            output = probe_aws(args.profile, args.region)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
        elif args.command == "model-smoke":
            from app.model_smoke import run_smoke

            output = run_smoke(execute=args.execute, embedding_only=args.embedding_only)
        elif args.command == "cost-quote":
            from app.cost import PriceSheet, TokenUsage

            sheet = PriceSheet.model_validate_json(args.price_sheet.read_bytes())
            model = "amazon.titan-embed-text-v2:0" if args.embedding else (
                "anthropic.claude-haiku-4-5-20251001-v1:0"
            )
            route = model if args.embedding else "us." + model
            rate = sheet.rate_for(model, route, "us-east-1", now=datetime.now(UTC), live=True)
            amount = rate.quote(TokenUsage(input_tokens=args.input_tokens, output_tokens=args.output_tokens))
            output = {
                "status": "FORECAST_ONLY", "inference_calls": 0,
                "model_id": model, "invocation_id": route, "price_sheet_version": sheet.version,
                "estimated_micro_usd": amount, "estimated_usd": str(Decimal(amount) / 1_000_000),
                "notice": "Token/rate calculation, not an invoice or authorization to invoke",
            }
        elif args.command == "export-schemas":
            output = export_schemas(args.output_dir)
        elif args.command == "snapshot-validate":
            loaded = load_snapshot(
                args.snapshot_dir, tenant_id=args.tenant, embedding_model_id=args.embedding_model,
                allow_fixture=args.allow_fixture,
                trusted_manifest_sha256=args.trusted_manifest_sha256,
            )
            output = {
                "status": "VALIDATED", "snapshot_version": loaded.manifest.snapshot_version,
                "manifest_sha256": loaded.manifest_sha256,
                "authenticated_manifest": args.trusted_manifest_sha256 is not None,
                "counts": loaded.manifest.counts.model_dump(),
            }
        else:
            request = QueryRequest(question=args.question, mode=args.mode)
            path = build_fixture(args.output_root)
            loaded = load_snapshot(
                path, tenant_id=FIXTURE_TENANT, embedding_model_id=FIXTURE_EMBEDDING_MODEL,
                allow_fixture=True,
            )
            ranked = retrieve_baseline(
                loaded, request.question, fixture_vector(request.question), mode=request.mode
            )
            chunk = loaded.chunks[ranked[0]]
            doc = loaded.documents[chunk.document_version_id]
            quote = chunk.text[:600]
            response = QueryResponse(
                request_id="fixture-request", answer=quote,
                claims=(Claim(claim_id="c1", text=quote, citation_ids=("s1",)),),
                citations=(Citation(
                    citation_id="s1", chunk_id=chunk.chunk_id,
                    document_version_id=chunk.document_version_id, title=doc.title,
                    source_uri=doc.source_uri, section_path=chunk.section_path, quote=quote,
                ),),
                answer_status="PARTIAL", mode_requested=request.mode, mode_used=request.mode,
                used_graph=False, snapshot_version=loaded.manifest.snapshot_version,
                warnings=("FIXTURE_ONLY_NO_MODEL",),
            )
            output = {
                "fixture_only": True,
                "notice": "Synthetic content and toy vectors; extracted excerpt, not a real AWS answer",
                "snapshot_directory": str(path), "ranked_chunk_ids": ranked,
                "response": response.model_dump(mode="json"),
            }
    except (ValueError, OSError, sqlite3.Error, BotoCoreError, ClientError) as exc:
        # ValidationErrors may contain private payloads; return only a safe category here.
        print(json.dumps({"error": {"code": type(exc).__name__, "message": "Command failed"}}))
        return 1
    print(json.dumps(output, ensure_ascii=True, indent=2))
    return 0 if output.get("status") not in {"INCOMPLETE", "DEVELOPMENT_IDENTITY_REQUIRED", "QUALITY_SMOKE_FAILED"} else 1


if __name__ == "__main__":
    sys.exit(main())
