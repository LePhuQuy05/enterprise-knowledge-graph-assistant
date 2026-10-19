"""Synthetic, unlicensed-content-free fixtures. Never an AWS knowledge corpus."""

import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from app.contracts import (
    Chunk,
    DocumentVersion,
    EmbeddingRecipe,
    Entity,
    EvidenceSpan,
    ExtractionRecipe,
    Mention,
    Qualifier,
    RelationAssertion,
    embedding_input,
    text_hash,
)
from app.snapshot import SnapshotContent, build_snapshot

FIXTURE_TENANT = "tenant_fixture"
FIXTURE_EMBEDDING_MODEL = "fixture-hash-v1"


def fixture_vector(text: str) -> np.ndarray:
    """Deterministic toy encoding; explicitly not a Bedrock/model-quality baseline."""
    vector = np.zeros(1024, dtype=np.float32)
    for token in re.findall(r"\w+", text.casefold()):
        vector[int(text_hash(token)[:8], 16) % 1024] += 1
    if not vector.any():
        vector[0] = 1
    return vector / np.linalg.norm(vector)


def fixture_recipes() -> tuple[ExtractionRecipe, EmbeddingRecipe]:
    extraction = ExtractionRecipe(
        version="extract-recipe-v2", model_id="fixture-extractor", invocation_id="fixture-extractor",
        source_region="local-fixture", parameters={"temperature": 0.0, "max_tokens": 2048},
        prompt_version="extract-v1", output_schema_version="extraction-v2",
        ontology_version="ontology-v1", canonicalization_version="canonical-v1",
    )
    embedding = EmbeddingRecipe(
        version="embedding-v1", model_id=FIXTURE_EMBEDDING_MODEL,
        invocation_id=FIXTURE_EMBEDDING_MODEL, source_region="local-fixture",
    )
    return extraction, embedding


def fixture_content() -> SnapshotContent:
    texts = (
        "Service A uses Service B.\nService B uses Resource C.\n"
        "A note mentions Service Z. Unicode: tiếng Việt 🚀\u2028kept.",
        "Resource C requires Permission D in region demo-1.\n"
        "Service A does not support Permission D.",
    )
    documents = []
    chunks = []
    by_line = []
    for index, text in enumerate(texts):
        document = DocumentVersion.create(
            tenant_id=FIXTURE_TENANT, source_uri=f"https://example.invalid/fixture/{index}",
            title=f"Synthetic fixture {index}", fetched_at=datetime(2026, 10, 16, tzinfo=UTC),
            raw_sha256=text_hash(text), normalizer_version="normalize-v1",
            normalized_text_path=f"sources/fixture-{index}.txt", normalized_text=text,
            source_kind="TEXT",
        )
        documents.append(document)
        start = 0
        for position, line in enumerate(text.split("\n")):
            chunk = Chunk.create(
                tenant_id=FIXTURE_TENANT, document_version_id=document.document_version_id,
                chunking_version="chunk-v1", section_path=("Synthetic fixture",), position=position,
                start_char=start, end_char=start + len(line), text=line,
                token_count=max(1, len(re.findall(r"\w+", line))), tokenizer_version="fixture-words-v1",
            )
            chunks.append(chunk)
            by_line.append(chunk)
            start += len(line) + 1
    names = ("Service A", "Service B", "Resource C", "Permission D", "Service Z")
    entities = tuple(Entity.create(
        tenant_id=FIXTURE_TENANT, canonical_key=f"fixture/{name.casefold().replace(' ', '-')}",
        canonical_name=name, entity_type="CONCEPT", aliases=(name,),
        canonicalization_version="canonical-v1",
    ) for name in names)
    entity_map = {entity.canonical_name: entity for entity in entities}
    mentions = []
    for chunk in chunks:
        for entity in entities:
            position = chunk.text.find(entity.canonical_name)
            if position >= 0:
                start = chunk.start_char + position
                mentions.append(Mention.create(
                    tenant_id=FIXTURE_TENANT, entity_id=entity.entity_id, chunk_id=chunk.chunk_id,
                    document_version_id=chunk.document_version_id, start_char=start,
                    end_char=start + len(entity.canonical_name), quote=entity.canonical_name,
                    canonicalization_version="canonical-v1",
                ))
    extraction, _ = fixture_recipes()
    facts = (
        (by_line[0], "Service A", "USES", "Service B", "POSITIVE", ()),
        (by_line[1], "Service B", "USES", "Resource C", "POSITIVE", ()),
        (by_line[3], "Resource C", "DEPENDS_ON", "Permission D", "POSITIVE",
         (Qualifier(kind="REGION", text="region demo-1"),)),
        (by_line[4], "Service A", "SUPPORTS", "Permission D", "NEGATIVE", ()),
    )
    assertions = tuple(RelationAssertion.create(
        recipe=extraction, tenant_id=FIXTURE_TENANT, subject_id=entity_map[subject].entity_id,
        object_id=entity_map[obj].entity_id, predicate=predicate, chunk_id=chunk.chunk_id,
        document_version_id=chunk.document_version_id,
        evidence=EvidenceSpan(start_char=chunk.start_char, end_char=chunk.end_char, quote=chunk.text),
        polarity=polarity, qualifiers=qualifiers,
    ) for chunk, subject, predicate, obj, polarity, qualifiers in facts)
    return SnapshotContent(tuple(documents), tuple(chunks), entities, tuple(mentions), assertions)


def build_fixture(output_root: Path) -> Path:
    content = fixture_content()
    extraction, embedding = fixture_recipes()
    docs = {d.document_version_id: d for d in content.documents}
    vectors = {c.chunk_id: fixture_vector(embedding_input(docs[c.document_version_id], c))
               for c in content.chunks}
    return build_snapshot(
        output_root, content, vectors, tenant_id=FIXTURE_TENANT, purpose="FIXTURE",
        extraction_recipe=extraction, embedding_recipe=embedding,
        corpus_version="synthetic-v1", normalizer_version="normalize-v1",
        chunking_version="chunk-v1", tokenizer_version="fixture-words-v1",
    )
