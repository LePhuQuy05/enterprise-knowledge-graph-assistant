import gzip
import json
import re
from pathlib import Path

import pytest

from app.archive import Archive, ArchivePolicy, location, resource_format
from app.contracts import canonical_json, embedding_input, sha256_bytes, stable_id, text_hash
from app.corpus import load_corpus
from app.corpus_expansion import LocalWordPieceTokenizer, prepare_expansion
from app.corpus_inventory import audit_archive

S3 = "https://docs.aws.amazon.com/AmazonS3/latest/userguide/"
IAM = "https://docs.aws.amazon.com/IAM/latest/UserGuide/"


class FixtureTokenizer:
    version = "fixture-wordpiece-v1-no-truncation"
    asset_sha256 = "a" * 64

    def count(self, text):
        return len(re.findall(r"\w+|[^\w\s]", text)) + 2


@pytest.fixture
def setup(tmp_path):
    root = tmp_path / "archive"
    policy = ArchivePolicy(version="expansion-fixture-v1", min_free_disk_bytes=0)
    with Archive(root, policy):
        pass
    guides = tmp_path / "guides.json"
    guides.write_bytes(canonical_json({"service_inventory": [
        {"service": "S3", "guide_prefixes": [S3]}, {"service": "IAM", "guide_prefixes": [IAM]}]}))
    return root, policy, tmp_path / "candidates.jsonl", guides, tmp_path / "output"


def add(root, policy, url, *, text=None, title="Permissions guide", section_path=None, gap=False,
        normalizer="fixture-source-normalizer-v1"):
    text = text or ("A policy grants a principal access only when all conditions hold. " * 12 + url)
    with Archive(root, policy) as archive:
        archive._add(url, "PAGE", "fixture", None)
        raw_sha = archive._put(("RAW FIXTURE\n" + text).encode())
        language, family = location(url)
        doc = {
            "schema_version": 1,
            "archive_document_version_id": stable_id("archdoc", source_uri=url,
                raw_sha256=raw_sha, normalizer=normalizer),
            "source_uri": url, "effective_uri": url, "family": family,
            "language": language, "format": resource_format(url), "version_hint": "latest",
            "fetched_at": "2026-10-23T00:00:00+00:00", "raw_sha256": raw_sha,
            "normalized_sha256": text_hash(text), "normalizer_version": normalizer,
            "title": title, "text": text, "sections": [{"start_char": 1 if gap else 0,
                "end_char": len(text), "section_path": section_path or ["Access conditions"],
                "anchor": "conditions"}], "warnings": [], "semantic_review": "NOT_REVIEWED",
        }
        digest = archive._put(canonical_json(doc))
        archive.db.execute("INSERT INTO versions VALUES (?,?,?)",
            (doc["archive_document_version_id"], url, digest))
        archive.db.execute("UPDATE resources SET state='DONE',document_sha256=?,raw_sha256=?,"
            "effective_url=?,fetched_at=? WHERE url=?", (digest, raw_sha, url, doc["fetched_at"], url))
        archive.db.commit()
    return doc, digest


def prepare(setup, **kwargs):
    root, policy, candidates, guides, output = setup
    audit_archive(root, policy, candidate_output=candidates)
    return prepare_expansion(root, policy, candidates, guides, output,
                             tokenizer=FixtureTokenizer(), **kwargs)


def test_balanced_round_robin_caps_and_immutable_repeat_identity(setup):
    root, policy, _, _, _ = setup
    for i in range(4):
        add(root, policy, S3 + f"s3-{i}.html")
        add(root, policy, IAM + f"iam-{i}.html")
    first = prepare(setup, per_service=2, total_limit=3)
    assert first["status"] == "PREPARED"
    assert first["service_counts"] == {"IAM": 1, "S3": 2}
    second = prepare(setup, per_service=2, total_limit=3)
    assert first == second
    assert first["source_http_requests"] == first["inference_calls"] == 0
    assert first["embeddings_generated"] is False
    assert first["semantic_source_review"] == "NOT_REVIEWED"
    for shard in first["shards"]:
        corpus = load_corpus(Path(shard["corpus_directory"]), tenant_id="demo")
        assert corpus.version == shard["corpus_version"]
        assert corpus.recipe["normalizer_version"] == shard["normalizer_version"]
        for doc in corpus.documents:
            source_path = Path(shard["corpus_directory"]) / doc.normalized_text_path
            assert source_path.read_text() == doc.normalized_text


def test_more_than_100_documents_split_into_compatible_shards(setup):
    root, policy, _, _, _ = setup
    for i in range(103):
        add(root, policy, S3 + f"page-{i:03}.html")
    result = prepare(setup, per_service=110, total_limit=103, shard_documents=100)
    assert result["selected_documents"] == 103
    assert [s["documents"] for s in result["shards"]] == [100, 3]
    assert len({s["chunking_version"] for s in result["shards"]}) == 1
    assert len({s["tokenizer_version"] for s in result["shards"]}) == 1
    assert len({s["corpus_version"] for s in result["shards"]}) == 2
    assert all(s["chunks"] <= 5000 for s in result["shards"])


def test_html_preference_and_exact_hash_dedup_keep_all_provenance(setup):
    root, policy, _, _, _ = setup
    body = "A principal requires permission only when the bucket policy permits it. " * 12
    add(root, policy, S3 + "same.html", text=body)
    add(root, policy, S3 + "same.md", text=body + "Markdown is independently sourced.")
    add(root, policy, S3 + "duplicate.html", text=body)
    result = prepare(setup, per_service=2)
    assert result["selected_documents"] == 1
    assert result["metadata_filters"]["NOT_ENGLISH_GUIDE_HTML"] == 1
    assert len(result["exact_text_duplicates"]) == 1
    assert result["exact_text_duplicates"][0]["provenance"]["raw_blob_verified"] is True
    assert result["status"] == "PREPARED_WITH_SHORTAGES"
    assert result["total_shortage"] == 3


def test_only_explicit_guide_prefixes_not_sdk_api_or_other_guides(setup):
    root, policy, _, _, _ = setup
    add(root, policy, S3 + "allowed.html")
    add(root, policy, "https://docs.aws.amazon.com/AmazonS3/latest/API/API_GetObject.html")
    add(root, policy, "https://docs.aws.amazon.com/sdk-for-java/latest/developer-guide/s3.html")
    result = prepare(setup, per_service=1)
    assert result["selected_documents"] == 1
    assert result["metadata_filters"]["OUTSIDE_APPROVED_OR_AMBIGUOUS_GUIDES"] == 2
    assert result["service_inventory"][1]["shortage_to_service_quota"] == 1


def test_gap_and_short_sources_rejected_with_explicit_shortages(setup):
    root, policy, _, _, _ = setup
    add(root, policy, S3 + "gap.html", gap=True)
    add(root, policy, S3 + "tiny.html", text="Too short.")
    result = prepare(setup, per_service=1)
    assert result["selected_documents"] == 0
    assert result["total_shortage"] == 2
    assert result["failures"][0]["reason"] == "SECTION_COVERAGE_FAILED"
    assert result["metadata_filters"]["BODY_SIZE_OUTSIDE_SELECTION_BOUNDS"] == 1
    assert not result["corpus_directories"]


@pytest.mark.parametrize("tamper", ["reference", "blob"])
def test_archive_or_reference_tamper_excluded(setup, tamper):
    root, policy, candidates, guides, output = setup
    _, digest = add(root, policy, S3 + "policy.html")
    audit_archive(root, policy, candidate_output=candidates)
    if tamper == "reference":
        ref = json.loads(candidates.read_bytes())
        ref["raw_sha256"] = "0" * 64
        candidates.write_bytes(canonical_json(ref) + b"\n")
    else:
        (root / "blobs" / digest[:2] / (digest + ".gz")).write_bytes(gzip.compress(b"{}"))
    result = prepare_expansion(root, policy, candidates, guides, output, tokenizer=FixtureTokenizer())
    assert result["selected_documents"] == 0
    assert len(result["failures"]) == 1


def test_metadata_headroom_and_every_actual_embedding_input_checked(setup):
    root, policy, _, _, _ = setup
    original = "Permission requires all policy conditions. " * 600 + "Never omit the final negation: access is NOT allowed."
    add(root, policy, S3 + "long.html", text=original, section_path=["condition " * 180])
    result = prepare(setup, per_service=1)
    assert result["selected_documents"] == 1
    assert result["metadata_prefix_max_tokens"] > 180
    assert result["max_embedding_input_tokens"] <= 512
    shard = result["shards"][0]
    corpus = load_corpus(Path(shard["corpus_directory"]), tenant_id="demo")
    assert len(corpus.chunks) > 1
    assert corpus.documents[0].normalized_text == original
    by_id = {d.document_version_id: d for d in corpus.documents}
    assert all(FixtureTokenizer().count(embedding_input(by_id[c.document_version_id], c)) <= 512
               for c in corpus.chunks)
    assert any("NOT allowed" in c.text for c in corpus.chunks)
    assert corpus.recipe["chunk_recipe"]["max_tokens"] < 448


def test_targeted_topic_rank_precedes_alphabetical_filler(setup):
    root, policy, _, guides, _ = setup
    add(root, policy, S3 + "aaa-first.html")
    add(root, policy, S3 + "zz-troubleshooting.html")
    value = json.loads(guides.read_bytes())
    value["sources"] = [{"url": S3 + "zz-troubleshooting.html", "within_service_rank": 1}]
    guides.write_bytes(canonical_json(value))
    result = prepare(setup, per_service=1, total_limit=1)
    corpus = load_corpus(Path(result["corpus_directories"][0]), tenant_id="demo")
    assert corpus.documents[0].source_uri == S3 + "zz-troubleshooting.html"


def test_distinct_normalizer_versions_never_share_a_shard(setup):
    root, policy, _, _, _ = setup
    add(root, policy, S3 + "a.html", normalizer="normalizer-a")
    add(root, policy, IAM + "b.html", normalizer="normalizer-b")
    result = prepare(setup, per_service=1)
    assert result["normalizer_versions"] == ["normalizer-a", "normalizer-b"]
    assert len(result["shards"]) == 2
    assert len({s["chunking_version"] for s in result["shards"]}) == 1
    for shard in result["shards"]:
        corpus = load_corpus(Path(shard["corpus_directory"]), tenant_id="demo")
        assert {d.normalizer_version for d in corpus.documents} == {shard["normalizer_version"]}


def test_actual_composed_embedding_bound_is_independently_enforced(setup):
    root, policy, candidates, guides, output = setup
    add(root, policy, S3 + "a.html")
    audit_archive(root, policy, candidate_output=candidates)

    class AdversarialCompositionTokenizer(FixtureTokenizer):
        def count(self, text):
            if "\n\n" in text and not text.endswith("\n\n"):
                return 513
            return super().count(text)

    result = prepare_expansion(root, policy, candidates, guides, output,
                               tokenizer=AdversarialCompositionTokenizer())
    assert result["selected_documents"] == 0
    assert result["failures"][0]["reason"] == "EMBEDDING_INPUT_BOUND_FAILED"
    assert not result["shards"]


def test_tokenizer_manifest_asset_tamper_rejected_without_inference(tmp_path):
    asset = tmp_path / "tokenizer.json"
    asset.write_bytes(b"untrusted tokenizer bytes")
    manifest = tmp_path / "manifest.json"
    manifest.write_bytes(canonical_json({"model_id": "BAAI/bge-small-en-v1.5",
        "revision": "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a", "max_tokens": 512,
        "files": [{"path": "tokenizer.json", "bytes": asset.stat().st_size,
                   "sha256": sha256_bytes(asset.read_bytes())}]}))
    with pytest.raises(ValueError, match="TOKENIZER_CHECKSUM_INVALID"):
        LocalWordPieceTokenizer(tmp_path, manifest)


def test_output_must_not_overlap_archive(setup):
    root, policy, candidates, guides, _ = setup
    with pytest.raises(ValueError, match="OUTPUT_OVERLAPS_ARCHIVE"):
        prepare_expansion(root, policy, candidates, guides, root / "bad", tokenizer=FixtureTokenizer())
