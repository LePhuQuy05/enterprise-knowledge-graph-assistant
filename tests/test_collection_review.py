from types import SimpleNamespace

import numpy as np
import pytest
from pydantic import ValidationError

from app.collection_lexical import build_global_lexical_index, load_global_lexical_index
from app.collection_review import ReviewScenarioSet, build_review_worksheet
from app.collections import build_collection, load_collection
from app.contracts import LocalBgeRecipe, canonical_json, text_hash
from app.fixtures import FIXTURE_TENANT, fixture_content, fixture_recipes
from app.snapshot import build_snapshot


def scenario_payload():
    return {
        "schema_version": 1, "version": "review-fixture-v1",
        "purpose": "DEVELOPMENT_WORKFLOW_FIXTURES_NOT_GOLD_OR_HOLDOUT",
        "semantic_review": "NOT_REVIEWED", "quality_metrics_available": False,
        "cases": [{"id": "dev-01", "kind": "fixture", "services": ["Service B"],
                   "en": "Service B", "vi": "Dịch vụ B"}],
        "notice": "Synthetic fixture, not AWS knowledge or approved evidence.",
    }


@pytest.fixture
def review_context(tmp_path):
    content = fixture_content()
    document_recipe = LocalBgeRecipe(
        artifact_manifest_sha256="a" * 64, onnxruntime_version="fixture",
        tokenizers_version="fixture", input_role="document",
        input_recipe_version="title-section-text-v1",
    )
    query_recipe = document_recipe.model_copy(update={
        "input_role": "query", "input_recipe_version": "question-text-v1",
    })
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    snapshot = build_snapshot(
        tmp_path / "index", content, {chunk.chunk_id: vector for chunk in content.chunks},
        tenant_id=FIXTURE_TENANT, purpose="FIXTURE", extraction_recipe=fixture_recipes()[0],
        embedding_recipe=document_recipe, corpus_version="synthetic-v1", normalizer_version="normalize-v1",
        chunking_version="chunk-v1", tokenizer_version="fixture-words-v1",
    )
    manifest = build_collection(tmp_path / "index", [snapshot], tenant_id=FIXTURE_TENANT,
                                embedding_model_id=document_recipe.model_id, allow_fixture=True)
    collection = load_collection(manifest, tenant_id=FIXTURE_TENANT,
                                 embedding_model_id=document_recipe.model_id, allow_fixture=True)
    lexical_manifest = build_global_lexical_index(collection, tmp_path / "lexical")
    scenarios = ReviewScenarioSet.model_validate_json(canonical_json(scenario_payload()))
    with load_global_lexical_index(lexical_manifest, collection) as lexical:
        yield collection, lexical, scenarios, {"dev-01:en": vector, "dev-01:vi": vector}, {
            "tenant_id": FIXTURE_TENANT, "document_recipe": document_recipe,
            "query_recipe": query_recipe, "allow_fixture": True,
        }


def test_worksheet_preserves_exact_unicode_source_spans_and_has_no_labels(review_context):
    collection, lexical, scenarios, vectors, kwargs = review_context
    report = build_review_worksheet(collection, lexical, scenarios, vectors, **kwargs)
    assert report["query_count"] == 2 and report["ranking_count"] == 6
    assert report["purpose"] == "FIXTURE" and report["human_reviewed_queries"] == 0
    assert report["quality_metrics"] is None and report["contains_answers"] is False
    assert report["paid_model_calls"] == 0
    for query in report["queries"]:
        assert query["expected_answerability"] is None and query["required_facts"] is None
        assert query["retrieval_query_transform"] == "NONE_DIRECT_DIAGNOSTIC"
        assert set(query["methods"]) == {"vector", "bm25", "hybrid"}
        for rows in query["methods"].values():
            assert 1 <= len(rows) <= 5
            for row in rows:
                snapshot = collection.get_shard(row["shard_id"])
                document = snapshot.documents[row["document_version_id"]]
                assert row["text"] == document.normalized_text[row["start_char"]:row["end_char"]]
                assert row["source_uri"] == document.source_uri
                assert row["text_sha256"] == text_hash(row["text"])
                assert row["review_status"] == "NOT_REVIEWED"
                assert all(row[key] is None for key in (
                    "relevance_label", "supported_facts", "semantic_support", "reviewer_notes",
                ))
    assert any("tiếng Việt" in row["text"] for q in report["queries"]
               for rows in q["methods"].values() for row in rows)


@pytest.mark.parametrize("field,value,error", [
    ("tenant_id", "wrong", "TENANT"),
    ("allow_fixture", False, "FIXTURE"),
])
def test_context_rejected_before_review(review_context, field, value, error):
    collection, lexical, scenarios, vectors, kwargs = review_context
    kwargs[field] = value
    with pytest.raises(ValueError, match=error):
        build_review_worksheet(collection, lexical, scenarios, vectors, **kwargs)


@pytest.mark.parametrize("role", ["document", "query"])
def test_model_assets_or_input_recipe_mismatch_is_rejected(review_context, role):
    collection, lexical, scenarios, vectors, kwargs = review_context
    key = f"{role}_recipe"
    kwargs[key] = kwargs[key].model_copy(update={"artifact_manifest_sha256": "b" * 64})
    with pytest.raises(ValueError, match=f"REVIEW_{role.upper()}_RECIPE_MISMATCH"):
        build_review_worksheet(collection, lexical, scenarios, vectors, **kwargs)


def test_wrong_lexical_collection_is_rejected_before_looking_up_sources(review_context):
    collection, lexical, scenarios, vectors, kwargs = review_context
    wrong = SimpleNamespace(manifest=lexical.manifest.model_copy(update={
        "collection_manifest_sha256": "b" * 64,
    }))
    with pytest.raises(ValueError, match="LEXICAL_COLLECTION"):
        build_review_worksheet(collection, wrong, scenarios, vectors, **kwargs)


def test_missing_query_vector_is_not_silently_skipped(review_context):
    collection, lexical, scenarios, vectors, kwargs = review_context
    del vectors["dev-01:vi"]
    with pytest.raises(ValueError, match="VECTOR_COVERAGE"):
        build_review_worksheet(collection, lexical, scenarios, vectors, **kwargs)


@pytest.mark.parametrize("mutation", ["duplicate", "oversized", "labelled", "blank", "answers"])
def test_unbounded_or_prelabelled_scenarios_are_rejected(mutation):
    payload = scenario_payload()
    if mutation == "duplicate":
        payload["cases"] *= 2
    elif mutation == "oversized":
        payload["cases"] = [{**payload["cases"][0], "id": f"dev-{i}"} for i in range(101)]
    elif mutation == "labelled":
        payload["quality_metrics_available"] = True
    elif mutation == "blank":
        payload["cases"][0]["en"] = "   "
    else:
        payload["cases"][0]["answer"] = "Should not enter unreviewed source worksheets"
    with pytest.raises(ValidationError):
        ReviewScenarioSet.model_validate_json(canonical_json(payload))
