"""Rechunk the three-source supplement to the exact parent BGE input recipe.

The initial supplement used the ingestion proxy tokenizer. It cannot be mixed
with the parent's WordPiece chunks or silently truncated by the BGE encoder.
Original sources, chunks, graph and evaluation artifacts stay unchanged.
"""

import json
from pathlib import Path

from app.contracts import DocumentVersion, canonical_json, embedding_input, sha256_bytes
from app.corpus import load_corpus
from app.corpus_expansion import LocalWordPieceTokenizer, _publish_shard
from app.ingestion.models import ChunkRecipe, Section
from app.ingestion.text import chunk_document
from scripts.pilot_reviewed_source_answers import immutable
from scripts.prepare_authorization_evidence import ROOT, load_sources

OUTPUT = ROOT / "results/collection319-local"
DATA = ROOT / "data/collection319-local"


def prepare():
    collection, original_chunks, original_docs = load_sources()
    prior_report = json.loads((ROOT / "data/corpus-expansion-v2/preparation_report.json").read_bytes())
    prior = [load_corpus(Path(p), tenant_id="demo") for p in prior_report["corpus_directories"]]
    if ({d.document_version_id for c in prior for d in c.documents} != set(original_docs)
            or {c.chunk_id for corpus in prior for c in corpus.chunks} != set(original_chunks)):
        raise ValueError("COLLECTION319_PARENT_PREPARATION_MISMATCH")
    report = json.loads((ROOT / "results/permission-source-supplement-v1/report.json").read_bytes())
    corpus = Path(report["corpus_directory"])
    for name, digest in report["artifact_sha256"].items():
        if sha256_bytes((corpus / name).read_bytes()) != digest:
            raise ValueError("COLLECTION319_SUPPLEMENT_SOURCE_CHANGED")
    documents = [DocumentVersion.model_validate_json(line) for line in (corpus / "documents.jsonl").read_bytes().splitlines()]
    tokenizer = LocalWordPieceTokenizer(ROOT / "data/models/bge-small-en-v1.5", ROOT / "configs/models/bge-small-en-v1.5-onnx.json")
    recipe = ChunkRecipe.model_validate_json(canonical_json(prior[0].recipe["chunk_recipe"]))
    if any((c.recipe["normalizer_version"], c.chunks[0].chunking_version, c.chunks[0].tokenizer_version) !=
            (collection.manifest.normalizer_version, recipe.version(tokenizer.version), tokenizer.version) for c in prior):
        raise ValueError("COLLECTION319_PARENT_RECIPE_CHANGED")
    rows, counts = [], []
    for doc in documents:
        if doc.normalizer_version != collection.manifest.normalizer_version or doc.document_version_id in original_docs:
            raise ValueError("COLLECTION319_SOURCE_NORMALIZER_OR_DUPLICATE")
        sections_path = ROOT / "data/permission-source-supplement-v1/normalized" / doc.document_version_id / "sections.json"
        sections = tuple(Section.model_validate_json(canonical_json(s)) for s in json.loads(sections_path.read_bytes()))
        chunks = chunk_document(doc, sections, recipe, tokenizer)
        lengths = [tokenizer.count(embedding_input(doc, c)) for c in chunks]
        if any(n > 512 for n in lengths):
            raise ValueError("COLLECTION319_NO_TRUNCATION_INPUT_EXCEEDS_512")
        counts.extend(lengths)
        rows.append(("Lambda", {"source_uri": doc.source_uri, "document_version_id": doc.document_version_id,
            "raw_sha256": doc.raw_sha256, "document_sha256": doc.normalized_sha256,
            "source_kind": "OFFICIAL_HTML_SOURCE_SUPPLEMENT", "semantic_review": "AI_SOURCE_DEPENDENCY_REVIEW_NOT_INDEPENDENT_GOLD"},
            doc, sections, chunks))
    shard = _publish_shard(DATA, rows, {"manifest_version": "permission-supplement-parent-rechunk-v1",
        "chunk_recipe": recipe.model_dump(mode="json"), "tokenizer_version": tokenizer.version,
        "tokenizer_asset_sha256": tokenizer.asset_sha256, "embedding_input_recipe": "title-section-text-v1",
        "embedding_input_max_tokens": 512, "parent_collection_manifest_sha256": collection.manifest_sha256,
        "source_supplement_inventory_sha256": sha256_bytes((ROOT / "results/permission-source-supplement-v1/inventory.json").read_bytes())},
        collection.manifest.normalizer_version, "demo")
    all_chunks = [c for corpus in prior for c in corpus.chunks] + [c for row in rows for c in row[4]]
    all_docs = [d for corpus in prior for d in corpus.documents] + documents
    preparation = {"version": "gap-driven-collection319-compatible-preparation-v1",
        "corpus_directories": prior_report["corpus_directories"] + [shard["corpus_directory"]],
        "parent_collection_path": str(collection.path), "parent_manifest_sha256": collection.manifest_sha256,
        "source_corpus_versions": [c.version for c in prior] + [shard["corpus_version"]],
        "expected_documents": len(all_docs), "expected_chunk_occurrences": len(all_chunks),
        "expected_unique_body_texts": len({c.text_sha256 for c in all_chunks}),
        "initial_supplement_proxy_chunks": report["chunks"], "compatible_bge_supplement_chunks": shard["chunks"],
        "supplement_max_embedding_input_tokens": max(counts), "chunking_version": recipe.version(tokenizer.version),
        "tokenizer_version": tokenizer.version, "normalizer_version": collection.manifest.normalizer_version,
        "original_supplement_chunks_retained": True, "new_model_api_calls": 0, "aws_calls": 0,
        "source_qualification_annotations_need_interval_revalidation_on_new_chunks": True,
        "code_sha256": {name: sha256_bytes((ROOT / name).read_bytes()) for name in ("scripts/prepare_collection319_candidate.py",
            "app/corpus_expansion.py", "app/ingestion/text.py", "app/collection_indexing.py", "app/local_bge.py")},
        "serving_published": False, "graph_migrated": False}
    immutable(OUTPUT / "preparation.json", preparation)
    print(json.dumps({k: v for k, v in preparation.items() if k not in {"corpus_directories", "code_sha256"}}))


if __name__ == "__main__":
    prepare()
