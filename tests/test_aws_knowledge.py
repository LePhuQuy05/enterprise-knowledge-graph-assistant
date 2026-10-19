import copy
import json
from datetime import UTC, datetime

import pytest

from app.aws_knowledge import (
    BUNDLE_LIMIT,
    ENDPOINT,
    LEGACY_RECIPE,
    RECIPE,
    AwsKnowledgeClient,
    KnowledgeError,
    create_bundle,
    load_bundle,
    normalize_excerpt,
    official_url,
    pack_bundle,
)
from app.contracts import canonical_json, sha256_bytes
from app.query_language import answer_language


def item(url="https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/USER_ReadRepl.html"):
    return {"url": url, "title": "Working with read replicas", "context":
        "Amazon RDS uses asynchronous replication to update a read replica with changes to the source DB instance."}


def wire(items):
    return canonical_json({"jsonrpc": "2.0", "id": 1, "result": {"isError": False, "content": [{
        "type": "text", "text": json.dumps({"content": {"result": items}})}]}})


def test_fixed_search_tool_and_filtered_deduplicated_official_sources():
    sent = []
    def transport(body, timeout):
        sent.append(json.loads(body))
        assert 0 < timeout <= 6
        return wire([item("https://attacker.invalid/aws"), item(), item(), item("https://aws.amazon.com/blogs/database/read-replicas/")])
    results = AwsKnowledgeClient(transport=transport).search("RDS read replica")
    assert len(results) == 2
    assert sent[0]["params"]["name"] == "aws___search_documentation"
    assert sent[0]["params"]["arguments"]["limit"] == 8
    assert all("title" in value and "text" in value for value in results)


@pytest.mark.parametrize("url", ["http://docs.aws.amazon.com/a", "https://docs.aws.amazon.com.evil/a",
    "https://evil@docs.aws.amazon.com/a", "https://docs.aws.amazon.com:8443/a",
    "https://docs.aws.amazon.com/a?url=http://169.254.169.254", "https://127.0.0.1/a",
    "https://docs.aws.amazon.com/%2e%2e/a", "https://docs.aws.amazon.com/%0aevil"])
def test_untrusted_urls_reject_ssrf(url):
    with pytest.raises(KnowledgeError):
        official_url(url)


@pytest.mark.parametrize("raw", [b"not json", b'{"jsonrpc":"2.0","jsonrpc":"2.0"}',
    canonical_json({"jsonrpc": "2.0", "id": 1, "error": {"message": "secret"}}), b"x" * (512 * 1024 + 1)],
    ids=["invalid-json", "duplicate-key", "server-error", "oversize"])
def test_malformed_oversize_and_server_errors_fail_closed(raw):
    with pytest.raises(KnowledgeError):
        AwsKnowledgeClient(transport=lambda *_: raw).search("EC2")


def bundle():
    items = AwsKnowledgeClient(transport=lambda *_: wire([item()])).search("RDS")
    return create_bundle(items, tenant_id="demo", snapshot_version="snap-policy", owner_id="owner-a",
        request_id="q-request", search_phrase="RDS read replica", fetched_at=datetime(2026, 10, 18, tzinfo=UTC))


def test_bundle_integrity_owner_source_and_packed_bounds():
    value = bundle()
    packed = pack_bundle(value)
    assert len(packed.chunk_ids) == 1 and packed.proxy_tokens <= 4096
    assert not value.record["full_corpus_downloaded"] and value.record["endpoint"] == ENDPOINT
    chunk = value.chunks[packed.chunk_ids[0]]
    assert chunk.text == value.documents[chunk.document_version_id].normalized_text
    for field in ("tenant_id", "owner_id", "request_id", "snapshot_version"):
        kwargs = dict(tenant_id="demo", owner_id="owner-a", request_id="q-request", snapshot_version="snap-policy")
        kwargs[field] = "foreign"
        with pytest.raises(KnowledgeError):
            load_bundle(value.record, value.sha256, **kwargs)
    bad = copy.deepcopy(value.record)
    bad["chunks"][0]["text"] += " modified"
    for digest in (value.sha256, sha256_bytes(canonical_json(bad))):
        with pytest.raises(KnowledgeError):
            load_bundle(bad, digest, tenant_id="demo", owner_id="owner-a", request_id="q-request", snapshot_version="snap-policy")


def test_language_detection_and_explicit_selection():
    assert answer_language("RDS Multi-AZ khác gì Read Replica?") == "vi"
    assert answer_language("How does RDS replication work?") == "en"
    assert answer_language("RDS", "vi") == "vi"
    assert answer_language("Tiếng Việt", "en") == "en"


def test_soft_wrap_is_normalized_before_offsets_and_raw_provenance_is_retained():
    raw = "You\ncan't use a standby replica to serve read traffic."
    result = create_bundle([{"url": item()["url"], "title": "RDS", "text": raw}],
        tenant_id="demo", snapshot_version="snap", owner_id="owner", request_id="request", search_phrase="RDS")
    doc = next(iter(result.documents.values()))
    assert doc.normalized_text == "You can't use a standby replica to serve read traffic."
    assert doc.raw_sha256 == sha256_bytes(raw.encode())
    assert result.record["raw_excerpts"][0]["raw_text"] == raw
    assert result.record["version"] == RECIPE
    bad = copy.deepcopy(result.record)
    bad["raw_excerpts"][0]["raw_text"] = "You can use a standby replica to serve read traffic."
    with pytest.raises(KnowledgeError):
        load_bundle(bad, sha256_bytes(canonical_json(bad)), tenant_id="demo", snapshot_version="snap",
                    owner_id="owner", request_id="request")


@pytest.mark.parametrize("block", [
    '```json\n{"Effect": "Deny",\n "Action": "s3:GetObject"}\n```',
    '~~~python\nprint("a")\nprint("b")\n~~~',
    '    code line\n    second line',
    '| Name | Meaning |\n| --- | --- |\n| a | b |',
    '- No decrypt\n  without a key policy.',
    '1. First\n2. Second',
    '# Heading\nAnother heading line',
    '> quoted line\n> second line',
    'Hard break  \nnext line',
    'Hard break\\\nnext line',
    'Heading\n=======',
    '{\n"Effect": "Deny"\n}',
])
def test_markdown_and_policy_structures_are_preserved(block):
    text = "A wrapped\nprose paragraph.\n\n" + block + "\n\nAnother\nparagraph."
    assert normalize_excerpt(text) == "A wrapped prose paragraph.\n\n" + block + "\n\nAnother paragraph."
    assert normalize_excerpt(normalize_excerpt(text)) == normalize_excerpt(text)


def test_v1_history_remains_readable_without_normalization():
    raw = "Original wrapped\nsource passage."
    from app.contracts import Chunk, DocumentVersion
    doc = DocumentVersion.create(tenant_id="demo", source_uri=item()["url"], title="RDS",
        fetched_at=datetime(2026, 10, 18, tzinfo=UTC), raw_sha256="1" * 64,
        normalizer_version=LEGACY_RECIPE, normalized_text_path="request-evidence/old", normalized_text=raw,
        source_kind="TEXT")
    chunk = Chunk.create(tenant_id="demo", document_version_id=doc.document_version_id,
        chunking_version=LEGACY_RECIPE, section_path=("Official AWS search excerpt",), position=0,
        start_char=0, end_char=len(raw), text=raw, token_count=8, tokenizer_version="cl100k_base")
    record = {"version": LEGACY_RECIPE, "tenant_id": "demo", "owner_id": "owner", "request_id": "old",
        "snapshot_version": "snap", "endpoint": ENDPOINT, "source_kind": "OFFICIAL_SEARCH_EXCERPT",
        "full_corpus_downloaded": False, "documents": [doc.model_dump(mode="json")],
        "chunks": [chunk.model_dump(mode="json")]}
    result = load_bundle(record, sha256_bytes(canonical_json(record)), tenant_id="demo", snapshot_version="snap",
        owner_id="owner", request_id="old")
    assert next(iter(result.chunks.values())).text == raw


def test_unicode_bundle_budget_is_applied_before_adding_an_excerpt():
    items = [{"url": item()["url"] + str(i), "title": "RDS", "text": "\u1ef1" * 2400} for i in range(6)]
    result = create_bundle(items, tenant_id="demo", snapshot_version="snap", owner_id="owner",
        request_id="request", search_phrase="RDS")
    assert 0 < len(result.chunks) < 6
    assert len(canonical_json(result.record)) <= BUNDLE_LIMIT


def test_adapter_hashes_the_persisted_prefix_not_an_unavailable_full_excerpt():
    source = item()
    source["context"] = "Word " * 1000
    result = AwsKnowledgeClient(transport=lambda *_: wire([source])).search("RDS")[0]
    assert len(result["text"]) <= 2400
    assert result["excerpt_sha256"] == sha256_bytes(result["text"].encode())
