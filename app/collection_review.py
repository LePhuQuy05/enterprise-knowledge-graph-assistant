"""Source-preserving retrieval worksheets; no answers, gold labels or quality claims."""

from datetime import UTC, datetime
from typing import Annotated, Literal, Self

import numpy as np
from pydantic import Field, StringConstraints, field_validator, model_validator

from app.collection_lexical import GlobalLexicalIndex
from app.collections import SnapshotCollection
from app.contracts import EmbeddingRecipe, LocalBgeRecipe, Record, canonical_json, sha256_bytes
from app.retrieval import reciprocal_rank_fusion

Text = Annotated[str, StringConstraints(min_length=1, max_length=2000)]
Label = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class ReviewScenario(Record):
    id: Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9-]{0,79}$")]
    kind: Label
    services: Annotated[tuple[Label, ...], Field(min_length=1, max_length=10)]
    en: Text
    vi: Text

    @field_validator("en", "vi")
    @classmethod
    def nonblank_question(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Review questions cannot be blank")
        return value


class ReviewScenarioSet(Record):
    schema_version: Literal[1]
    version: Label
    purpose: Literal["DEVELOPMENT_WORKFLOW_FIXTURES_NOT_GOLD_OR_HOLDOUT"]
    semantic_review: Literal["NOT_REVIEWED"]
    quality_metrics_available: Literal[False]
    cases: Annotated[tuple[ReviewScenario, ...], Field(min_length=1, max_length=100)]
    notice: Text

    @model_validator(mode="after")
    def unique_cases(self) -> Self:
        if len({case.id for case in self.cases}) != len(self.cases):
            raise ValueError("Duplicate review case IDs")
        return self


def _validate_context(collection, lexical, tenant_id, document_recipe, query_recipe, allow_fixture):
    manifest = collection.manifest
    if manifest.tenant_id != tenant_id:
        raise ValueError("REVIEW_TENANT_MISMATCH")
    fixture = manifest.purpose == "FIXTURE"
    if fixture and not allow_fixture:
        raise ValueError("REVIEW_FIXTURE_REQUIRES_OPT_IN")
    if document_recipe != manifest.embedding_recipe:
        raise ValueError("REVIEW_DOCUMENT_RECIPE_MISMATCH")
    if isinstance(document_recipe, LocalBgeRecipe):
        if (not isinstance(query_recipe, LocalBgeRecipe) or document_recipe.input_role != "document"
                or query_recipe.input_role != "query"
                or document_recipe.model_dump(exclude={"input_role", "input_recipe_version"})
                != query_recipe.model_dump(exclude={"input_role", "input_recipe_version"})):
            raise ValueError("REVIEW_QUERY_RECIPE_MISMATCH")
    elif not fixture or query_recipe != document_recipe:
        raise ValueError("REVIEW_UNSUPPORTED_ENCODER_RECIPE")
    expected = {
        "collection_version": manifest.collection_version,
        "collection_manifest_sha256": collection.manifest_sha256,
        "corpus_version": manifest.corpus_version,
        "embedding_recipe_sha256": document_recipe.fingerprint,
    }
    if any(getattr(lexical.manifest, key) != value for key, value in expected.items()):
        raise ValueError("REVIEW_LEXICAL_COLLECTION_MISMATCH")


def build_review_worksheet(
    collection: SnapshotCollection, lexical: GlobalLexicalIndex,
    scenarios: ReviewScenarioSet, query_vectors: dict[str, np.ndarray], *,
    tenant_id: str, document_recipe: EmbeddingRecipe, query_recipe: EmbeddingRecipe,
    allow_fixture: bool = False,
) -> dict:
    """Keep EN/VI queries independent; reviewers must supply semantic judgments.

    Direct Vietnamese retrieval with English BGE is a diagnostic candidate,
    not an assertion of multilingual support or the production query rewrite.
    """
    _validate_context(collection, lexical, tenant_id, document_recipe, query_recipe, allow_fixture)
    expected_ids = {f"{case.id}:{language}" for case in scenarios.cases for language in ("en", "vi")}
    if set(query_vectors) != expected_ids:
        raise ValueError("REVIEW_QUERY_VECTOR_COVERAGE_MISMATCH")

    def source_row(chunk_id, rank):
        shard_id = lexical.shard_for_chunk(chunk_id)
        snapshot = collection.get_shard(shard_id)
        chunk = snapshot.chunks[chunk_id]
        document = snapshot.documents[chunk.document_version_id]
        if (chunk.tenant_id != tenant_id or document.tenant_id != tenant_id
                or document.normalized_text[chunk.start_char:chunk.end_char] != chunk.text):
            raise ValueError("REVIEW_SOURCE_SPAN_MISMATCH")
        return {
            "rank": rank, "shard_id": shard_id, "chunk_id": chunk.chunk_id,
            "document_version_id": document.document_version_id,
            "source_uri": document.source_uri, "title": document.title,
            "section_path": list(chunk.section_path), "start_char": chunk.start_char,
            "end_char": chunk.end_char, "text": chunk.text,
            "text_sha256": sha256_bytes(chunk.text.encode("utf-8")),
            "review_status": "NOT_REVIEWED", "relevance_label": None,
            "supported_facts": None, "semantic_support": None, "reviewer_notes": None,
        }

    queries = []
    for case in scenarios.cases:
        for language in ("en", "vi"):
            question_id = f"{case.id}:{language}"
            question = getattr(case, language)
            vector = collection.search_vector(query_vectors[question_id], k=20)
            lexical_hits = lexical.search(question, k=20)
            vector_ids, lexical_ids = [hit.chunk_id for hit in vector], [key for key, _ in lexical_hits]
            rankings = {
                "vector": vector_ids, "bm25": lexical_ids,
                "hybrid": reciprocal_rank_fusion([vector_ids, lexical_ids]),
            }
            queries.append({
                "question_id": question_id, "case_id": case.id, "language": language,
                "question": question, "kind": case.kind, "services": list(case.services),
                "retrieval_query_transform": "NONE_DIRECT_DIAGNOSTIC",
                "review_status": "NOT_REVIEWED", "expected_answerability": None,
                "required_facts": None, "missing_evidence": None, "reviewer": None,
                "methods": {method: [source_row(key, rank) for rank, key in enumerate(ids[:5], 1)]
                            for method, ids in rankings.items()},
            })
    return {
        "schema_version": 1, "kind": "LOCAL_RETRIEVAL_REVIEW_WORKSHEET",
        "created_at": datetime.now(UTC).isoformat(), "purpose": collection.manifest.purpose,
        "collection_version": collection.manifest.collection_version,
        "collection_manifest_sha256": collection.manifest_sha256,
        "corpus_version": collection.manifest.corpus_version,
        "lexical_index_version": lexical.manifest.index_version,
        "document_recipe": document_recipe.model_dump(mode="json"),
        "query_recipe": query_recipe.model_dump(mode="json"),
        "scenario_version": scenarios.version,
        "scenario_sha256": sha256_bytes(canonical_json(scenarios.model_dump(mode="json"))),
        "scenario_purpose": scenarios.purpose,
        "case_count": len(scenarios.cases), "query_count": len(queries),
        "ranking_count": len(queries) * 3, "human_reviewed_queries": 0,
        "semantic_review": "NOT_REVIEWED", "contains_answers": False,
        "quality_metrics": None, "paid_model_calls": 0, "queries": queries,
        "limitations": [
            "Development cases are not independent gold or holdout evaluation.",
            "Ranks and exact source spans do not prove relevance, completeness or claim support.",
            "VI queries are direct diagnostics with English BGE and ASCII BM25; no translation is run.",
            "No model generates answers, labels, required facts or semantic judgments in this worksheet.",
        ],
    }
