"""Balanced offline archive-to-corpus preparation; no downloads or model inference."""

import importlib.metadata
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from app.archive import NORMALIZED_LIMIT, Archive, ArchivePolicy, archive_url
from app.contracts import (
    MAX_SNAPSHOT_BYTES,
    DocumentVersion,
    canonical_json,
    embedding_input,
    sha256_bytes,
    stable_id,
)
from app.corpus import load_corpus
from app.corpus_inventory import _validate_document
from app.ingestion.models import ChunkRecipe, Section
from app.ingestion.pipeline import _jsonl, _write
from app.ingestion.text import chunk_document

EXPANSION_RECIPE = "balanced-official-guide-corpus-v1"
MODEL_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
TOKENIZER_SHA256 = "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66"


class LocalWordPieceTokenizer:
    """Only the pinned tokenizer asset is loaded; no ONNX session is created."""

    def __init__(self, directory: Path, manifest_path: Path):
        from tokenizers import Tokenizer

        config = json.loads(manifest_path.read_bytes())
        if (config.get("model_id") != "BAAI/bge-small-en-v1.5" or config.get("revision") != MODEL_REVISION
                or config.get("max_tokens") != 512):
            raise ValueError("BGE_MANIFEST_INVALID")
        rows = [row for row in config["files"] if row["path"] == "tokenizer.json"]
        path = directory / "tokenizer.json"
        if len(rows) != 1 or not path.is_file() or path.is_symlink() or path.stat().st_size > 4 * 1024 * 1024:
            raise ValueError("BGE_TOKENIZER_ASSET_INVALID")
        raw = path.read_bytes()
        if (len(raw) != rows[0]["bytes"] or sha256_bytes(raw) != rows[0]["sha256"]
                or rows[0]["sha256"] != TOKENIZER_SHA256):
            raise ValueError("BGE_TOKENIZER_CHECKSUM_INVALID")
        self.tokenizer = Tokenizer.from_str(raw.decode("utf-8"))
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        self.asset_sha256 = sha256_bytes(raw)
        self.version = ("tokenizers-" + importlib.metadata.version("tokenizers") +
                        "-bge-wordpiece-no-truncation-v1-" + self.asset_sha256)

    def count(self, text: str) -> int:
        return len(self.tokenizer.encode(text, add_special_tokens=True).ids)


def _bounded_json(path: Path, limit=16 * 1024 * 1024):
    if not path.is_file() or path.stat().st_size > limit:
        raise ValueError("EXPANSION_INPUT_SIZE_LIMIT")
    raw = path.read_bytes()
    return json.loads(raw), sha256_bytes(raw)


def _service_guides(value):
    rows = value.get("service_inventory")
    if not isinstance(rows, list) or not rows:
        raise ValueError("EXPANSION_SERVICE_INVENTORY_REQUIRED")
    services = {}
    for row in rows:
        service, prefixes = row.get("service"), row.get("guide_prefixes")
        if not isinstance(service, str) or not service.strip() or service in services or not prefixes:
            raise ValueError("EXPANSION_SERVICE_INVALID")
        checked = []
        for prefix in prefixes:
            if not isinstance(prefix, str) or not prefix.endswith("/") or archive_url(prefix) != prefix:
                raise ValueError("EXPANSION_GUIDE_PREFIX_INVALID")
            # The prefix list is explicit; never expand it to generic service,
            # SDK or API-reference paths by substring/family matching.
            if re.search(r"/(?:APIReference|API|sdk-for-[^/]+)/", prefix, flags=re.I):
                raise ValueError("EXPANSION_GUIDE_PREFIX_IS_API_OR_SDK")
            checked.append(prefix)
        services[service] = tuple(sorted(set(checked)))
    return services


def _candidate_queues(path, services, priority=None):
    if not path.is_file() or path.stat().st_size > 256 * 1024 * 1024:
        raise ValueError("EXPANSION_CANDIDATE_SIZE_LIMIT")
    queues = {name: [] for name in services}
    reasons, seen = Counter(), set()
    with path.open("rb") as stream:
        for line in stream:
            if len(line) > 128 * 1024:
                raise ValueError("EXPANSION_CANDIDATE_LINE_LIMIT")
            if not line.strip():
                continue
            ref = json.loads(line)
            uri = ref.get("source_uri")
            try:
                if not isinstance(uri, str) or archive_url(uri) != uri:
                    raise ValueError("invalid URI")
            except ValueError:
                reasons["UNSAFE_OR_NONCANONICAL_SOURCE"] += 1
                continue
            if uri in seen:
                reasons["DUPLICATE_SOURCE_REFERENCE"] += 1
                continue
            seen.add(uri)
            if (ref.get("format") != "html" or not uri.lower().endswith((".html", ".htm"))
                    or ref.get("language") != "en"):
                reasons["NOT_ENGLISH_GUIDE_HTML"] += 1
                continue
            matched = [service for service, prefixes in services.items()
                       if any(uri.startswith(prefix) for prefix in prefixes)]
            if len(matched) != 1:
                reasons["OUTSIDE_APPROVED_OR_AMBIGUOUS_GUIDES"] += 1
                continue
            if (ref.get("structural_validation") != "PASS" or ref.get("raw_blob_verified") is not True
                    or ref.get("semantic_review") != "NOT_REVIEWED"):
                reasons["UNVERIFIED_REFERENCE"] += 1
                continue
            if type(ref.get("text_characters")) is not int or not 300 <= ref["text_characters"] <= 150_000:
                reasons["BODY_SIZE_OUTSIDE_SELECTION_BOUNDS"] += 1
                continue
            queues[matched[0]].append(ref)
    priority = priority or {}
    for queue in queues.values():
        queue.sort(key=lambda ref: (priority.get(ref["source_uri"], 1_000_000),
                                   ref["source_uri"], ref.get("document_sha256", "")))
    return queues, reasons


def _prepare_source(archive, ref, tenant_id, tokenizer):
    row = archive.db.execute("SELECT * FROM resources WHERE url=? AND kind='PAGE' AND state='DONE'",
                             (ref["source_uri"],)).fetchone()
    if row is None or row["document_sha256"] != ref.get("document_sha256"):
        raise ValueError("CANDIDATE_ARCHIVE_VERSION_CHANGED")
    doc = json.loads(archive.read_blob(ref["document_sha256"], NORMALIZED_LIMIT))
    _validate_document(doc, row)
    if archive.db.execute("SELECT 1 FROM versions WHERE version_id=? AND document_sha256=? AND url=?", (
            doc["archive_document_version_id"], ref["document_sha256"], doc["source_uri"])).fetchone() is None:
        raise ValueError("CANDIDATE_VERSION_LINK_MISSING")
    for key in ("source_uri", "effective_uri", "archive_document_version_id", "raw_sha256",
                "normalized_sha256", "normalizer_version", "format", "family", "language", "fetched_at"):
        if doc[key] != ref.get(key):
            raise ValueError("CANDIDATE_REFERENCE_MISMATCH")
    if not 300 <= len(doc["text"]) <= 150_000 or len(doc["text"]) != ref["text_characters"]:
        raise ValueError("DOCUMENT_BODY_SIZE_INVALID")
    archive.read_blob(doc["raw_sha256"], archive.policy.max_page_bytes)
    sections = tuple(Section.model_validate_json(canonical_json(s)) for s in doc["sections"])
    if (not sections or sections[0].start_char != 0 or sections[-1].end_char != len(doc["text"])
            or any(a.end_char != b.start_char for a, b in zip(sections, sections[1:]))):
        raise ValueError("SECTION_COVERAGE_FAILED")
    metadata_max = max(tokenizer.count(doc["title"] + "\n" + " > ".join(s.section_path) + "\n\n")
                       for s in sections)
    if metadata_max > 440:
        raise ValueError("EMBEDDING_METADATA_HEADROOM_EXCEEDED")
    document = DocumentVersion.create(
        tenant_id=tenant_id, source_uri=doc["source_uri"], title=doc["title"],
        fetched_at=datetime.fromisoformat(doc["fetched_at"]), raw_sha256=doc["raw_sha256"],
        normalizer_version=doc["normalizer_version"], normalized_text_path="pending",
        normalized_text=doc["text"], source_kind="HTML",
    )
    document = document.model_copy(update={
        "normalized_text_path": "normalized/" + document.document_version_id + "/source.txt"})
    return document, sections, metadata_max


def prepare_expansion(archive_root: Path, policy: ArchivePolicy, candidates_path: Path,
                      guide_sources_path: Path, output_root: Path, *, tenant_id="demo",
                      per_service=20, total_limit=500, shard_documents=100, tokenizer=None) -> dict:
    """Build immutable v2-compatible shards from balanced verified archive sources."""
    if (type(per_service) is not int or not 1 <= per_service <= 1000 or
            type(total_limit) is not int or not 1 <= total_limit <= 10_000 or
            type(shard_documents) is not int or not 1 <= shard_documents <= 100):
        raise ValueError("EXPANSION_SELECTION_LIMIT_INVALID")
    archive_root, output_root = Path(archive_root).resolve(), Path(output_root).resolve()
    if output_root.is_relative_to(archive_root) or archive_root.is_relative_to(output_root):
        raise ValueError("EXPANSION_OUTPUT_OVERLAPS_ARCHIVE")
    guide_data, guide_sha = _bounded_json(Path(guide_sources_path))
    services = _service_guides(guide_data)
    priority = {row["url"]: row["within_service_rank"] for row in guide_data.get("sources", [])
                if isinstance(row.get("url"), str) and type(row.get("within_service_rank")) is int}
    queues, skipped = _candidate_queues(Path(candidates_path), services, priority)
    tokenizer = tokenizer or LocalWordPieceTokenizer(
        Path("data/models/bge-small-en-v1.5"), Path("configs/models/bge-small-en-v1.5-onnx.json"))
    positions = Counter()
    selected_counts = Counter()
    selected, duplicates, failures = [], [], []
    seen_hash = {}
    with Archive(archive_root, policy, read_only=True) as archive:
        archive.db.execute("PRAGMA query_only=ON")
        archive.db.execute("BEGIN")
        # One candidate per service per round keeps the total cap from favoring
        # whichever service has the largest queue or alphabetically first URLs.
        while len(selected) < total_limit:
            advanced = False
            for service, queue in queues.items():
                if selected_counts[service] >= per_service or len(selected) >= total_limit:
                    continue
                while positions[service] < len(queue):
                    ref = queue[positions[service]]
                    positions[service] += 1
                    advanced = True
                    try:
                        document, sections, metadata_max = _prepare_source(archive, ref, tenant_id, tokenizer)
                        digest = document.normalized_sha256
                        if digest in seen_hash:
                            duplicates.append({"source_uri": ref["source_uri"],
                                               "same_normalized_text_as": seen_hash[digest],
                                               "normalized_sha256": digest,
                                               "provenance": ref})
                            continue
                        seen_hash[digest] = ref["source_uri"]
                        selected.append((service, ref, document, sections, metadata_max))
                        selected_counts[service] += 1
                        break
                    except (ValueError, OSError, KeyError, TypeError) as exc:
                        code = str(exc) if re.fullmatch(r"[A-Z][A-Z0-9_]{1,95}", str(exc)) else type(exc).__name__
                        failures.append({"source_uri": ref["source_uri"], "reason": code})
            if not advanced:
                break
        archive.db.rollback()
    max_metadata = max((row[4] for row in selected), default=0)
    max_body = min(448, 512 - max_metadata - 8)
    recipe = ChunkRecipe(target_tokens=min(320, max_body), max_tokens=max_body, overlap_tokens=32)
    prepared = []
    actual_max = 0
    for service, ref, document, sections, _ in selected:
        try:
            chunks = chunk_document(document, sections, recipe, tokenizer)
            counts = [tokenizer.count(embedding_input(document, chunk)) for chunk in chunks]
            if any(n > 512 for n in counts) or any(len(embedding_input(document, c)) > 50_000 for c in chunks):
                raise ValueError("EMBEDDING_INPUT_BOUND_FAILED")
            actual_max = max(actual_max, *counts)
            prepared.append((service, ref, document, sections, chunks))
        except (ValueError, OSError, KeyError, TypeError) as exc:
            code = str(exc) if re.fullmatch(r"[A-Z][A-Z0-9_]{1,95}", str(exc)) else type(exc).__name__
            failures.append({"source_uri": ref["source_uri"], "reason": code})
    base_recipe = {
        "manifest_version": EXPANSION_RECIPE, "chunk_recipe": recipe.model_dump(),
        "tokenizer_version": tokenizer.version, "tokenizer_asset_sha256": tokenizer.asset_sha256,
        "embedding_input_recipe": "title-section-text-v1", "embedding_model_id": "BAAI/bge-small-en-v1.5",
        "embedding_model_revision": MODEL_REVISION, "metadata_prefix_max_tokens": max_metadata,
        "embedding_input_max_tokens": 512, "embedding_safety_margin_tokens": 8,
        "archive_policy_fingerprint": policy.fingerprint, "guide_sources_sha256": guide_sha,
        "source_snapshot_sha256": sha256_bytes(canonical_json([
            {"source_uri": ref["source_uri"], "document_sha256": ref["document_sha256"]}
            for _, ref, _, _, _ in prepared])),
        "semantic_source_review": "NOT_REVIEWED", "selection_policy": {
            "per_service": per_service, "total_limit": total_limit, "shard_documents": shard_documents,
            "min_text_characters": 300, "max_text_characters": 150_000,
            "only_format": "html", "exact_text_duplicates": "ONE_TEXT_WITH_ALL_SOURCE_PROVENANCE",
        },
    }
    grouped = defaultdict(list)
    for row in prepared:
        grouped[row[2].normalizer_version].append(row)
    shards = []
    for normalizer, rows in sorted(grouped.items()):
        shard, chunk_count, byte_count = [], 0, 0
        for row in rows:
            _, ref, document, sections, chunks = row
            source = {"source_id": stable_id("archive-source", source_uri=ref["source_uri"]),
                      "service": row[0], **ref}
            extra = (len(_jsonl([document])) + len(_jsonl(chunks)) + len(_jsonl([source])) +
                     len(document.normalized_text.encode("utf-8")) +
                     len(canonical_json([s.model_dump(mode="json") for s in sections])))
            if shard and (len(shard) >= shard_documents or chunk_count + len(chunks) > 5000 or
                          byte_count + extra + 65536 > MAX_SNAPSHOT_BYTES):
                shards.append(_publish_shard(output_root, shard, base_recipe, normalizer, tenant_id))
                shard, chunk_count, byte_count = [], 0, 0
            if extra + 65536 > MAX_SNAPSHOT_BYTES or len(chunks) > 5000:
                raise ValueError("EXPANSION_DOCUMENT_EXCEEDS_SHARD_BOUND")
            shard.append(row)
            chunk_count += len(chunks)
            byte_count += extra
        if shard:
            shards.append(_publish_shard(output_root, shard, base_recipe, normalizer, tenant_id))
    final_counts = Counter(row[0] for row in prepared)
    desired_total = min(total_limit, len(services) * per_service)
    report = {
        "schema_version": 1, "recipe_version": EXPANSION_RECIPE,
        "status": "PREPARED" if len(prepared) == desired_total else "PREPARED_WITH_SHORTAGES",
        "tenant_id": tenant_id, "selected_documents": len(prepared),
        "selected_chunks": sum(s["chunks"] for s in shards), "requested_total": desired_total,
        "total_shortage": desired_total - len(prepared),
        "service_counts": dict(sorted(final_counts.items())),
        "service_inventory": [{"service": name, "requested": per_service,
            "metadata_eligible_candidates": len(queues[name]), "selected": final_counts[name],
            "shortage_to_service_quota": max(0, per_service - final_counts[name])} for name in services],
        "metadata_filters": dict(sorted(skipped.items())), "exact_text_duplicates": duplicates,
        "failures": failures, "shards": shards,
        "corpus_directories": [s["corpus_directory"] for s in shards],
        "normalizer_versions": sorted(grouped), "chunking_version": recipe.version(tokenizer.version),
        "tokenizer_version": tokenizer.version, "max_embedding_input_tokens": actual_max,
        "metadata_prefix_max_tokens": max_metadata,
        "semantic_source_review": "NOT_REVIEWED", "embeddings_generated": False,
        "source_http_requests": 0, "inference_calls": 0,
        "notice": "Prepared source/chunk integrity only. Service quotas are selection limits, not proof of topic or answer coverage. No vectors, graph, website deployment, or semantic approval produced.",
    }
    _write(output_root, "preparation_report.json", canonical_json(report), immutable=False)
    return report


def _publish_shard(root, rows, base_recipe, normalizer, tenant_id):
    documents = sorted((row[2] for row in rows), key=lambda d: d.document_version_id)
    chunks = sorted((c for row in rows for c in row[4]), key=lambda c: c.chunk_id)
    sources = sorted(({"source_id": stable_id("archive-source", source_uri=row[1]["source_uri"]),
                       "service": row[0], **row[1]} for row in rows), key=lambda r: r["source_id"])
    recipe = {**base_recipe, "normalizer_version": normalizer}
    artifacts = {"documents.jsonl": _jsonl(documents), "chunks.jsonl": _jsonl(chunks),
                 "sources.jsonl": _jsonl(sources), "recipe.json": canonical_json(recipe)}
    if sum(len(raw) for raw in artifacts.values()) > MAX_SNAPSHOT_BYTES:
        raise ValueError("EXPANSION_SHARD_SIZE_BOUND")
    version = stable_id("corpus", files={name: sha256_bytes(raw) for name, raw in artifacts.items()})
    relative = "corpora/" + version
    for name, raw in artifacts.items():
        _write(root, relative + "/" + name, raw)
    for _, _, document, sections, _ in rows:
        _write(root, relative + "/" + document.normalized_text_path, document.normalized_text.encode("utf-8"))
        _write(root, relative + "/normalized/" + document.document_version_id + "/sections.json",
               canonical_json([s.model_dump(mode="json") for s in sections]))
    checked = load_corpus(root / relative, tenant_id=tenant_id)
    return {"corpus_version": version, "corpus_directory": str(root / relative),
            "normalizer_version": normalizer, "chunking_version": chunks[0].chunking_version,
            "tokenizer_version": chunks[0].tokenizer_version, "documents": len(checked.documents),
            "chunks": len(checked.chunks), "service_counts": dict(sorted(Counter(row[0] for row in rows).items())),
            "artifact_sha256": {name: sha256_bytes(raw) for name, raw in artifacts.items()}}
