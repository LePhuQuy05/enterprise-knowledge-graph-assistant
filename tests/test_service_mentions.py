import dataclasses
from pathlib import Path

import pytest

from app.contracts import Chunk, DocumentVersion, text_hash
from app.extraction import AliasDictionary
from app.fixtures import FIXTURE_EMBEDDING_MODEL, fixture_recipes, fixture_vector
from app.graph_retrieval import GraphIndex
from app.service_mentions import derive_service_mentions
from app.snapshot import SnapshotContent, build_snapshot, load_snapshot

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def aliases():
    return AliasDictionary.load(ROOT / "configs/extraction/aws-approved-aliases-v1.json")


@pytest.fixture
def service_base(loaded, tmp_path):
    template = loaded.content.documents[0]
    text = "前文 🚀 AWS Lambda uses Amazon S3 and AWS KMS.\nLambdaish S3Express DynamoDB2.\nLambda Event Function and lambda.amazonaws.com."
    document = DocumentVersion.create(tenant_id=template.tenant_id, source_uri="https://example.invalid/services",
        title="Synthetic service mentions", fetched_at=template.fetched_at, raw_sha256=text_hash(text),
        normalizer_version="normalize-v1", normalized_text=text, normalized_text_path="fixture",
        source_kind="TEXT")
    chunks = []
    start = 0
    for index, line in enumerate(text.splitlines()):
        chunks.append(Chunk.create(tenant_id=document.tenant_id, document_version_id=document.document_version_id,
            chunking_version="chunk-v1", section_path=("fixture",), position=index, start_char=start,
            end_char=start+len(line), text=line, token_count=20, tokenizer_version="fixture-words-v1"))
        start += len(line) + 1
    extraction, embedding = fixture_recipes()
    path = build_snapshot(tmp_path / "base", SnapshotContent((document,), tuple(chunks)),
        {c.chunk_id:fixture_vector(c.text) for c in chunks}, tenant_id=document.tenant_id, purpose="FIXTURE",
        extraction_recipe=extraction, embedding_recipe=embedding, corpus_version="services-fixture-v1",
        normalizer_version="normalize-v1", chunking_version="chunk-v1", tokenizer_version="fixture-words-v1")
    return load_snapshot(path,tenant_id=document.tenant_id,embedding_model_id=FIXTURE_EMBEDDING_MODEL,allow_fixture=True)


def test_longest_alias_and_unicode_source_offsets_are_exact(service_base, aliases):
    result = derive_service_mentions(service_base, aliases)
    quotes = [m.quote for m in result.content.mentions]
    assert quotes.count("AWS Lambda") == quotes.count("Amazon S3") == quotes.count("AWS KMS") == 1
    # The nested aliases Lambda, S3 and KMS are not counted again in these spans.
    first = next(c for c in service_base.content.chunks if "前文" in c.text)
    own = [m for m in result.content.mentions if m.chunk_id == first.chunk_id]
    assert len(own) == 3
    for mention in result.content.mentions:
        doc = service_base.documents[mention.document_version_id]
        assert doc.normalized_text[mention.start_char:mention.end_char] == mention.quote
    assert len(result.covered_chunk_ids) == 2 and result.inspected_chunks == 3


def test_word_boundaries_do_not_invent_services_from_longer_names(service_base, aliases):
    result = derive_service_mentions(service_base, aliases)
    bad = next(c for c in service_base.content.chunks if "Lambdaish" in c.text)
    assert bad.chunk_id not in result.covered_chunk_ids
    assert not any(m.chunk_id == bad.chunk_id for m in result.content.mentions)


def test_resource_phrase_is_not_merged_as_a_service_and_no_relations_are_created(service_base, aliases):
    result = derive_service_mentions(service_base, aliases)
    assert not result.content.assertions
    assert {e.canonical_key for e in result.content.entities} == {"aws/lambda", "aws/s3", "aws/kms"}
    assert not any(e.canonical_name == "Lambda Event Function" for e in result.content.entities)
    assert not any(m.quote == "Lambda Event Function" for m in result.content.mentions)
    assert result.recipe.parameters["assertions_enabled"] is False


def test_records_are_stable_under_chunk_iteration_order_and_preserve_baseline(service_base, aliases):
    before = service_base.content
    first = derive_service_mentions(service_base, aliases)
    reversed_content = dataclasses.replace(before,chunks=tuple(reversed(before.chunks)))
    second = derive_service_mentions(dataclasses.replace(service_base,content=reversed_content),aliases)
    assert first.content.mentions == second.content.mentions and first.content.entities == second.content.entities
    assert first.content.documents == before.documents and first.content.chunks == before.chunks
    assert service_base.content == before and not service_base.content.mentions


@pytest.mark.parametrize("change", ["tenant", "text", "missing-document"])
def test_source_or_tenant_mismatch_fails_closed(service_base, aliases, change):
    original = service_base.content.chunks[0]
    if change == "missing-document":
        changed = dataclasses.replace(service_base, documents={})
    else:
        chunk = original.model_copy(update={"tenant_id":"other"} if change == "tenant" else {"text":"S3"})
        changed = dataclasses.replace(service_base, content=dataclasses.replace(service_base.content,
            chunks=(chunk,*service_base.content.chunks[1:])))
    with pytest.raises(ValueError,match="SOURCE_OR_TENANT_MISMATCH"):
        derive_service_mentions(changed,aliases)


def test_raw_graph_is_rejected_instead_of_implicitly_cleaned(loaded, aliases):
    with pytest.raises(ValueError,match="GRAPH_FREE_BASELINE"):
        derive_service_mentions(loaded,aliases)


def test_snapshot_roundtrip_and_expansion_have_no_assertion_paths(service_base, aliases, tmp_path):
    result = derive_service_mentions(service_base,aliases)
    m = service_base.manifest
    path = build_snapshot(tmp_path / "graph", result.content,
        {key:service_base.vectors[i] for i,key in enumerate(service_base.chunk_ids)},
        tenant_id=m.tenant_id,purpose="FIXTURE",extraction_recipe=result.recipe,embedding_recipe=m.embedding_recipe,
        corpus_version=m.corpus_version,normalizer_version=m.normalizer_version,
        chunking_version=m.chunking_version,tokenizer_version=m.tokenizer_version)
    graph = load_snapshot(path,tenant_id=m.tenant_id,embedding_model_id=FIXTURE_EMBEDDING_MODEL,allow_fixture=True)
    assert graph.content == result.content and not graph.content.assertions
    retrieved = GraphIndex(graph).retrieve("AWS Lambda",fixture_vector("AWS Lambda"),hops=1)
    assert retrieved.used_graph and retrieved.assertion_visits == 0
    assert all(not c.assertion_path and c.hop == 0 and not c.supporting_chunk_ids for c in retrieved.graph_candidates)


def test_nonservice_aliases_are_not_promoted(service_base, aliases):
    from app.extraction import ApprovedAlias

    concept = ApprovedAlias(canonical_key="concept/uses",canonical_name="uses",entity_type="CONCEPT",aliases=("uses",))
    value = AliasDictionary(version="test-v2",entries=(*aliases.entries,concept))
    result = derive_service_mentions(service_base,value)
    assert all(e.entity_type == "AWS_SERVICE" for e in result.content.entities)
    assert not any(m.quote == "uses" for m in result.content.mentions)
