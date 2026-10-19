import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from app.contracts import DocumentVersion, sha256_bytes
from app.ingestion import pipeline
from app.ingestion.collector import Collector, Fetched, FetchError, HttpResponse
from app.ingestion.models import ChunkRecipe, SourceManifest, source_uri
from app.ingestion.pipeline import Registry, ingest
from app.ingestion.text import ProxyTokenizer, chunk_document, normalize_html, normalizer_version

URL = "https://docs.aws.amazon.com/lambda/latest/dg/with-s3.html"
URL2 = "https://docs.aws.amazon.com/kms/latest/developerguide/key-policies.html"
HTML = b'<html><nav>OUTSIDE_NAV</nav><main><h1 id="intro">Test title</h1><p>Body evidence.</p></main></html>'


@pytest.fixture(scope="module")
def tokenizer():
    # Explicit setup: python -m app.cli prepare-tokenizer. This fixture never downloads.
    return ProxyTokenizer()


def manifest(tmp_path, urls=(URL, URL2)):
    data = {"version": "test-corpus-v1", "language": "en", "sources": [
        {"source_id": "source-" + str(i), "service": "Lambda" if i == 0 else "KMS",
         "url": url, "expected_title": "Expected", "evidence_intent": ["E1"],
         "selection_reason": "Synthetic fixture", "usage_note": "Test content only"}
        for i, url in enumerate(urls)
    ]}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class FixtureCollector:
    def __init__(self, bodies=None):
        self.bodies = bodies or {URL: HTML, URL2: HTML.replace(b"Body evidence", b"Different evidence")}
        self.request_count = 0
        self.delay_seconds = 5
        self.robots_bytes = None

    def fetch(self, url):
        self.request_count += 1
        return Fetched(url, url, datetime(2026, 10, 17, tzinfo=UTC)
                       + timedelta(seconds=self.request_count), "text/html; charset=utf-8",
                       self.bodies[url], (), 1)


class FakeClock:
    def __init__(self):
        self.now = 0
        self.waits = []

    def time(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def collector_with(responses, robots=b"User-agent: *\nDisallow:\n\nCrawl-delay: 7\n"):
    calls = []
    clock = FakeClock()
    sequence = iter(responses)

    def transport(url, limit, timeout):
        calls.append(url)
        if url.endswith("/robots.txt"):
            return HttpResponse(200, {}, robots)
        return next(sequence)

    return Collector(transport, sleep=clock.sleep, clock=clock.time), calls, clock


@pytest.mark.parametrize("url", [
    "http://docs.aws.amazon.com/lambda/latest/dg/page.html",
    "https://docs.aws.amazon.com.evil.invalid/lambda/latest/dg/page.html",
    "https://user@docs.aws.amazon.com/lambda/latest/dg/page.html",
    "https://docs.aws.amazon.com:444/lambda/latest/dg/page.html",
    "https://127.0.0.1/lambda/latest/dg/page.html",
    "https://docs.aws.amazon.com/search/page.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/%2e%2e/page.html",
])
def test_manifest_blocks_wrong_hosts_ports_schemes_and_paths(url):
    with pytest.raises(ValueError):
        source_uri(url)


def test_manifest_rejects_canonical_duplicate_urls(tmp_path):
    path = manifest(tmp_path, urls=(URL, URL + "#section"))
    with pytest.raises(ValueError, match="DUPLICATE_MANIFEST_SOURCE"):
        SourceManifest.model_validate_json(path.read_bytes())


def test_html_preserves_inline_spaces_table_headers_code_and_unicode():
    raw = '''<nav>OUTSIDE_NAV</nav><main><h1 id="root">Title</h1>
    <p>A<span> B</span> and <code>s3:GetObject</code>.</p>
    <h2 id="permissions">Permissions</h2><p>Không được bỏ “not”. 🚀</p>
    <table><tr><th>Action</th><th>Principal</th></tr>
    <tr><td>kms:Decrypt</td><td>Role B</td></tr></table>
    <pre>if enabled:\n    print("same  spaces")\n\t# tab stays</pre>
    <script>EVIL_SCRIPT</script><awsdocs-thumb-feedback>FEEDBACK</awsdocs-thumb-feedback>
    </main>'''.encode()
    n = normalize_html(raw, "Fallback")
    assert "A B and s3:GetObject." in n.text
    assert "Action: kms:Decrypt | Principal: Role B" in n.text
    assert '    print("same  spaces")\n\t# tab stays' in n.text
    assert "Không được bỏ “not”. 🚀" in n.text
    assert all(s not in n.text for s in ["OUTSIDE_NAV", "EVIL_SCRIPT", "FEEDBACK"])
    assert n.sections[0].section_path == ("Title",)
    assert n.sections[1].section_path == ("Title", "Permissions")
    assert n.sections[1].anchor == "permissions"
    assert n.sections[0].end_char == n.sections[1].start_char
    assert n.sections[-1].end_char == len(n.text)
    assert "bs4-" in normalizer_version()


@pytest.mark.parametrize("raw", [b"<p>No article</p>", b"<main>\xff</main>",
                               b"<main>\x00binary</main>", b"<main><script>only</script></main>"])
def test_invalid_or_boilerplate_only_source_is_not_a_document(raw):
    with pytest.raises(ValueError):
        normalize_html(raw, "Fallback")


def test_complex_table_and_images_are_reported_for_review():
    n = normalize_html(b'<main><h1>T</h1><img src="figure.png"><table><tr><td colspan="2">A</td></tr></table></main>', "T")
    assert n.warnings == ("COMPLEX_TABLE_SPANS_REVIEW", "IMAGES_NOT_TRANSCRIBED")
    assert "A" in n.text


def test_definition_lists_and_direct_line_breaks_do_not_merge_distinct_text():
    n = normalize_html(b'<main><h1>T</h1>A<br>B<dl><dt>Term</dt><dd>Definition</dd></dl>'
                       b'<div class="highlights">NAVIGATION_ONLY</div>'
                       b'<awsdocs-language-banner>LANGUAGE_UI</awsdocs-language-banner></main>', 'T')
    assert 'A B' in n.text and 'Term\n\nDefinition' in n.text
    assert 'NAVIGATION_ONLY' not in n.text and 'LANGUAGE_UI' not in n.text


def test_chunker_preserves_every_nonwhitespace_character_and_bounds_overlap(tokenizer):
    body = ("Tiếng Việt 🚀 中文 — do not grant every permission. " * 75) + "TAIL_NOT_DROPPED"
    n = normalize_html(('<main><h1>Title</h1><p>' + body + '</p></main>').encode(), "T")
    doc = DocumentVersion.create(
        tenant_id="demo", source_uri=URL, title=n.title, fetched_at=datetime(2026, 10, 17, tzinfo=UTC),
        raw_sha256=sha256_bytes(body.encode()), normalizer_version=normalizer_version(),
        normalized_text_path="fixture.txt", normalized_text=n.text, source_kind="HTML",
    )
    recipe = ChunkRecipe(target_tokens=24, max_tokens=32, overlap_tokens=4)
    chunks = chunk_document(doc, n.sections, recipe, tokenizer)
    covered = [False] * len(doc.normalized_text)
    for i, chunk in enumerate(chunks):
        assert chunk.text == n.text[chunk.start_char:chunk.end_char]
        assert chunk.token_count == tokenizer.count(chunk.text) <= 32
        assert "\ufffd" not in chunk.text
        covered[chunk.start_char:chunk.end_char] = [True] * len(chunk.text)
        if i and chunks[i-1].end_char > chunk.start_char:
            assert tokenizer.count(n.text[chunk.start_char:chunks[i-1].end_char]) <= 4
    assert all(ok or char.isspace() for char, ok in zip(n.text, covered))
    assert n.text.endswith("TAIL_NOT_DROPPED")
    changed = chunk_document(doc, n.sections, ChunkRecipe(target_tokens=24, max_tokens=32,
                                                         overlap_tokens=0), tokenizer)
    assert chunks[0].chunk_id != changed[0].chunk_id
    assert chunks[0].chunking_version != changed[0].chunking_version


def test_proxy_tokenizer_never_interprets_source_special_tokens(tokenizer):
    assert tokenizer.count("untrusted <|endoftext|> source") > 0


def test_missing_tokenizer_cache_never_downloads_implicitly(tmp_path, monkeypatch):
    from app.ingestion import text

    monkeypatch.setattr(text, "TOKENIZER_CACHE", tmp_path)
    monkeypatch.setattr(text.urllib.request, "urlopen", lambda *a, **k: pytest.fail("Implicit network"))
    with pytest.raises(ValueError, match="TOKENIZER_NOT_PREPARED"):
        ProxyTokenizer()


def test_collector_honors_trailing_delay_and_paces_every_request():
    c, calls, clock = collector_with([HttpResponse(200, {"content-type": "text/html"}, HTML)] * 2)
    c.fetch(URL)
    c.fetch(URL2)
    assert c.delay_seconds == 7
    assert c.request_count == len(calls) == 3
    assert clock.now == 14


def test_external_redirect_is_rejected_before_connecting():
    c, calls, _ = collector_with([HttpResponse(302, {"location": "https://evil.invalid/"}, b"")])
    with pytest.raises(ValueError, match="SOURCE_HOST_NOT_ALLOWED"):
        c.fetch(URL)
    assert len(calls) == 2


def test_robots_denial_stops_before_source_request():
    c, calls, _ = collector_with([], robots=b"User-agent: *\nDisallow: /lambda/\n")
    with pytest.raises(FetchError, match="ROBOTS_DENIED"):
        c.fetch(URL)
    assert len(calls) == 1


def test_http_retries_are_bounded_and_never_succeed_with_empty_result():
    c, calls, _ = collector_with([HttpResponse(503, {}, b"temporarily unavailable")] * 3)
    with pytest.raises(FetchError, match="HTTP_RETRIES_EXHAUSTED"):
        c.fetch(URL)
    assert len(calls) == 4


@pytest.mark.parametrize("headers,body,error", [
    ({"content-type": "application/pdf"}, HTML, "HTML_CONTENT_TYPE"),
    ({"content-type": "text/html; charset=shift-jis"}, HTML, "UTF8_CONTENT_TYPE"),
    ({"content-type": "text/html"}, b"", "EMPTY_RESPONSE"),
])
def test_collector_rejects_wrong_format_or_encoding(headers, body, error):
    c, _, _ = collector_with([HttpResponse(200, headers, body)])
    with pytest.raises(FetchError, match=error):
        c.fetch(URL)


def test_oversized_source_and_long_retry_after_fail_closed():
    c, _, _ = collector_with([HttpResponse(200, {"content-type": "text/html"}, b"x"*(10*1024*1024+1))])
    with pytest.raises(FetchError, match="SOURCE_SIZE_LIMIT"):
        c.fetch(URL)
    c, calls, _ = collector_with([HttpResponse(429, {"retry-after": "3600"}, b"")])
    with pytest.raises(FetchError, match="RETRY_AFTER_REQUIRES_LATER_RUN"):
        c.fetch(URL)
    assert len(calls) == 2


def test_cached_replay_offline_has_identical_artifacts_and_zero_fetches(tmp_path):
    path = manifest(tmp_path)
    root = tmp_path / "local"
    first = ingest(path, root, collector=FixtureCollector())
    assert first["status"] == "INGESTED" and first["unique_documents"] == 2
    second = ingest(path, root, offline=True, collector=FixtureCollector())
    assert second["source_cache_hits"] == 2 and second["http_requests_this_run"] == 0
    assert first["corpus_version"] == second["corpus_version"]
    assert first["artifact_sha256"] == second["artifact_sha256"]
    assert second["semantic_source_review"] == "NOT_REVIEWED"


def test_exact_content_duplicates_preserve_alias_inventory_without_inflating_docs(tmp_path):
    path = manifest(tmp_path)
    r = ingest(path, tmp_path / "local", collector=FixtureCollector({URL: HTML, URL2: HTML}))
    assert r["sources_validated"] == 2 and r["unique_documents"] == 1
    assert len(r["duplicates"]) == 1
    inventory = (Path(r["corpus_directory"]) / "sources.jsonl").read_text()
    assert inventory.count('"source_id"') == 2 and '"duplicate_of"' in inventory


def test_refresh_same_bytes_keeps_first_observation_and_corpus_identity(tmp_path):
    path, root = manifest(tmp_path), tmp_path / "local"
    first = ingest(path, root, collector=FixtureCollector())
    c = FixtureCollector()
    c.request_count = 100  # Later fetch timestamps, identical bytes.
    second = ingest(path, root, collector=c, refresh=True)
    assert first["corpus_version"] == second["corpus_version"]
    assert first["artifact_sha256"] == second["artifact_sha256"]


def test_corrupt_raw_cache_records_failure_instead_of_complete(tmp_path):
    path, root = manifest(tmp_path), tmp_path / "local"
    first = ingest(path, root, collector=FixtureCollector())
    raw_path = root / first["sources"][0]["raw_path"]
    raw_path.write_bytes(raw_path.read_bytes() + b"corruption")
    second = ingest(path, root, offline=True, collector=FixtureCollector())
    assert second["status"] == "INCOMPLETE" and len(second["failures"]) == 1
    assert second["failures"][0]["error_category"] == "RAW_CACHE_INTEGRITY_FAILED"
    with sqlite3.connect(root / "jobs.sqlite3") as db:
        assert db.execute("SELECT count(*) FROM jobs WHERE status='FAILED'").fetchone()[0] == 1


def test_corrupt_chunk_artifact_is_not_silently_overwritten(tmp_path):
    path, root = manifest(tmp_path), tmp_path / "local"
    first = ingest(path, root, collector=FixtureCollector())
    broken = root / first["sources"][0]["chunks_path"]
    broken.write_bytes(b"broken")
    second = ingest(path, root, offline=True, collector=FixtureCollector())
    assert second["status"] == "INCOMPLETE"
    assert broken.read_bytes() == b"broken"


def test_cold_offline_run_and_conflicting_flags_never_fetch(tmp_path):
    path, root = manifest(tmp_path), tmp_path / "local"
    c = FixtureCollector()
    r = ingest(path, root, collector=c, offline=True)
    assert r["status"] == "INCOMPLETE" and c.request_count == 0
    with pytest.raises(ValueError, match="OFFLINE_REFRESH_CONFLICT"):
        ingest(path, root, collector=c, offline=True, refresh=True)


def test_single_writer_lock_and_crash_recovery(tmp_path, monkeypatch):
    root = tmp_path / "local"
    root.mkdir()
    with Registry(root):
        with pytest.raises(ValueError, match="WRITER_ALREADY_ACTIVE"):
            with Registry(root):
                pytest.fail("Second writer acquired lock")
    path = manifest(tmp_path)
    original = pipeline.normalize_html

    def crash(*args):
        raise KeyboardInterrupt("Simulated process loss after raw checkpoint")

    monkeypatch.setattr(pipeline, "normalize_html", crash)
    with pytest.raises(KeyboardInterrupt):
        ingest(path, root, collector=FixtureCollector())
    monkeypatch.setattr(pipeline, "normalize_html", original)
    # The first source raw exists; the second source had not been started.
    recovery = ingest(path, root, collector=FixtureCollector())
    assert recovery["status"] == "INGESTED" and recovery["recovered_jobs"] == 1
    assert recovery["source_cache_hits"] == 1 and recovery["http_requests_this_run"] == 1


def test_changed_source_bytes_and_normalizer_create_new_versions_without_erasing_old(tmp_path, monkeypatch):
    path, root = manifest(tmp_path), tmp_path / "local"
    first = ingest(path, root, collector=FixtureCollector())
    changed = FixtureCollector({URL: HTML.replace(b"Body evidence", b"Changed source version"),
                                URL2: HTML.replace(b"Body evidence", b"Different evidence")})
    second = ingest(path, root, collector=changed, refresh=True)
    assert second["corpus_version"] != first["corpus_version"]
    assert Path(first["corpus_directory"]).is_dir()
    assert first["sources"][0]["document_version_id"] != second["sources"][0]["document_version_id"]
    monkeypatch.setattr(pipeline, "normalizer_version", lambda: "new-normalizer-fixture-v2")
    third = ingest(path, root, collector=FixtureCollector(), offline=True)
    assert third["status"] == "INGESTED" and third["http_requests_this_run"] == 0
    assert third["corpus_version"] != second["corpus_version"]
    assert third["sources"][0]["document_version_id"] != second["sources"][0]["document_version_id"]


@pytest.mark.parametrize("path", ["../outside", "/absolute", "C:\\private", "\\\\server\\share", "raw/../../secret"])
def test_artifact_paths_reject_escape_before_filesystem_resolution(tmp_path, path):
    with pytest.raises(ValueError, match="ARTIFACT_PATH_ESCAPES_ROOT"):
        pipeline._inside(tmp_path.resolve(), path)


def test_cli_returns_safe_failure_for_corrupt_local_registry(tmp_path, capsys):
    from app.cli import main

    path, root = manifest(tmp_path), tmp_path / "local"
    root.mkdir()
    (root / "jobs.sqlite3").write_bytes(b"not a SQLite database")
    assert main(["ingest", "--manifest", str(path), "--output-root", str(root),
                 "--offline"]) == 1
    response = json.loads(capsys.readouterr().out)
    assert response["error"]["code"] == "DatabaseError"
    assert "not a SQLite" not in json.dumps(response)
