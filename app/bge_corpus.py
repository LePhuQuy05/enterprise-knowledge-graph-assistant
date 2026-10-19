"""Rechunk an immutable local corpus with BGE WordPiece and metadata headroom."""

import json
from pathlib import Path

from app.contracts import canonical_json, embedding_input, stable_id
from app.corpus import load_corpus
from app.ingestion.models import ChunkRecipe, Section
from app.ingestion.text import chunk_document


def rechunk_for_bge(parent_directory: Path, encoder) -> tuple[Path, dict]:
    parent = load_corpus(parent_directory, tenant_id="demo")
    root = parent_directory.parent.parent.resolve()
    sections_by_doc, prefix_max = {}, 0
    for doc in parent.documents:
        path = (root / Path(doc.normalized_text_path).parent / "sections.json").resolve()
        if not path.is_relative_to(root) or path.stat().st_size > 1024 * 1024:
            raise ValueError("BGE_SECTIONS_PATH_OR_SIZE")
        sections = tuple(Section.model_validate_json(canonical_json(row)) for row in json.loads(path.read_bytes()))
        sections_by_doc[doc.document_version_id] = sections
        for section in sections:
            prefix = doc.title + "\n" + " > ".join(section.section_path) + "\n\n"
            prefix_max = max(prefix_max, encoder.count(prefix))
    raw_cap = min(448, 512 - prefix_max - 8)
    if raw_cap < 64:
        raise ValueError("BGE_SOURCE_METADATA_LEAVES_TOO_LITTLE_CONTEXT")
    recipe = ChunkRecipe(target_tokens=min(320, raw_cap), max_tokens=raw_cap, overlap_tokens=32)

    class BareTokenizer:
        version = "bge-wordpiece-v1-" + encoder.recipe("document").tokenizers_version + "-" + encoder.manifest_sha256

        def count(self, text):
            return max(0, encoder.count(text) - 2) if text.strip() else 0

    tokenizer = BareTokenizer()
    chunks, full_counts = [], []
    for doc in parent.documents:
        own = chunk_document(doc, sections_by_doc[doc.document_version_id], recipe, tokenizer)
        for chunk in own:
            n = encoder.count(embedding_input(doc, chunk))
            if n > 512:
                raise ValueError("BGE_COMPLETE_EMBEDDING_INPUT_EXCEEDS_CONTEXT")
            full_counts.append(n)
        chunks.extend(own)
    metadata = {"parent_corpus_version": parent.version, "normalizer_version": parent.recipe["normalizer_version"],
                "chunk_recipe": recipe.model_dump(mode="json"), "tokenizer_version": tokenizer.version,
                "tokenizer_asset_sha256": encoder.config["files"][1]["sha256"],
                "embedding_input_max_tokens": 512, "largest_metadata_tokens_including_specials": prefix_max,
                "metadata_guard_tokens": 8, "artifact_manifest_sha256": encoder.manifest_sha256}
    payloads = {name: (parent_directory / name).read_bytes() for name in ("documents.jsonl", "sources.jsonl")}
    payloads["chunks.jsonl"] = b"".join(canonical_json(c.model_dump(mode="json")) + b"\n"
                                      for c in sorted(chunks, key=lambda c: c.chunk_id))
    payloads["recipe.json"] = canonical_json(metadata)
    from app.contracts import sha256_bytes

    version = stable_id("corpus", files={name: sha256_bytes(raw) for name, raw in payloads.items()})
    destination = parent_directory.parent / version
    destination.mkdir(exist_ok=True)
    for name, raw in payloads.items():
        path = destination / name
        if path.exists() and path.read_bytes() != raw:
            raise ValueError("BGE_CORPUS_IMMUTABLE_CONFLICT")
        if not path.exists():
            path.write_bytes(raw)
    loaded = load_corpus(destination, tenant_id=parent.tenant_id)
    return destination, {"parent_chunks": len(parent.chunks), "chunks": len(loaded.chunks),
        "documents": len(loaded.documents), "max_complete_input_tokens": max(full_counts),
        "complete_input_tokens": sum(full_counts), "chunking_version": recipe.version(tokenizer.version),
        "tokenizer_version": tokenizer.version, "normalizer_version": parent.recipe["normalizer_version"],
        "source_slice_and_coverage": "PASS", "silent_truncations": 0}
