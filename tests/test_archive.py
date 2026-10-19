import gzip
import json
from xml.sax.saxutils import escape

import pytest

from app.archive import ROOT_SITEMAP, Archive, ArchivePolicy, archive_url, location, parse_sitemap
from app.cli import main
from app.ingestion.collector import Collector, HttpResponse

EC2_MAP = "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/sitemap.xml"
IAM_MAP = "https://docs.aws.amazon.com/IAM/latest/UserGuide/sitemap.xml"
JA_MAP = "https://docs.aws.amazon.com/ja_jp/IAM/latest/UserGuide/sitemap.xml"
EC2_PAGE = "https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/cpu.html"
IAM_PAGE = "https://docs.aws.amazon.com/IAM/latest/UserGuide/roles.html"
PDF_PAGE = "https://docs.aws.amazon.com/pdfs/IAM/latest/UserGuide/iam.pdf"
MD_PAGE = "https://docs.aws.amazon.com/IAM/latest/UserGuide/roles.md"


def sitemap(kind, urls):
    entry = "sitemap" if kind == "sitemapindex" else "url"
    return ("<" + kind + ">" + "".join("<" + entry + "><loc>" + escape(u) +
            "</loc><lastmod>2026-10-17</lastmod></" + entry + ">" for u in urls) +
            "</" + kind + ">").encode()


def html(title, text, links=()):
    return ("<div id='main-col-body'><h1>" + title + "</h1><p>" + text + "</p>" +
            "".join("<a href='" + u + "'>Source</a>" for u in links) + "</div>").encode()


@pytest.fixture
def policy():
    return ArchivePolicy(version="test-v1", max_storage_bytes=128 * 1024 * 1024,
                         min_free_disk_bytes=0, follow_document_links=True)


@pytest.fixture
def source():
    bodies = {
        "https://docs.aws.amazon.com/robots.txt": ("text/plain", b"User-agent: *\nDisallow: /blocked/\nCrawl-delay: 5\n"),
        ROOT_SITEMAP: ("text/xml", sitemap("sitemapindex", [EC2_MAP, IAM_MAP, JA_MAP, ROOT_SITEMAP])),
        EC2_MAP: ("application/xml", sitemap("urlset", [EC2_PAGE])),
        IAM_MAP: ("text/xml", sitemap("urlset", [IAM_PAGE])),
        EC2_PAGE: ("text/html", html("EC2 CPU", "Processor CPU credits", [PDF_PAGE, "https://evil.invalid/doc.html"])),
        IAM_PAGE: ("text/html", html("IAM role", "Execution role permissions", [MD_PAGE, "#anchor"])),
        MD_PAGE: ("text/markdown", b"# IAM\r\nRole permissions"),
        PDF_PAGE: ("application/pdf", b"%PDF-1.7 synthetic fixture"),
    }
    requests, waits, clock = [], [], [0.0]

    def transport(url, limit, timeout):
        requests.append(url)
        mime, body = bodies[url]
        return HttpResponse(200, {"content-type": mime}, body)

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    return bodies, requests, Collector(transport, sleep=sleep, clock=lambda: clock[0]), waits


def discover(root, policy, collector):
    with Archive(root, policy, collector=collector) as archive:
        return archive.run("SITEMAP", max_resources=0)


def test_discovers_every_in_scope_map_without_mvp_manifest_cap(tmp_path, policy, source):
    bodies, requests, collector, waits = source
    urls = [f"https://docs.aws.amazon.com/service{i}/latest/guide/sitemap.xml" for i in range(150)]
    bodies[ROOT_SITEMAP] = ("text/xml", sitemap("sitemapindex", urls + [JA_MAP]))
    with Archive(tmp_path, policy, collector=collector) as archive:
        result = archive.run("SITEMAP", max_resources=1)
    assert result["counts"]["SITEMAP"]["QUEUED"] == 150
    assert result["counts"]["SITEMAP"]["EXCLUDED"] == 1
    assert not result["catalog_complete_for_known_scope"]
    assert len(requests) == 2 and all(wait >= 5 for wait in waits)


def test_resume_sitemaps_round_robin_pages_links_search_and_exports(tmp_path, policy, source):
    _, requests, collector, _ = source
    root = tmp_path / "archive"
    first = discover(root, policy, collector)
    assert first["catalog_complete_for_known_scope"] and first["counts"]["PAGE"]["QUEUED"] == 2
    assert JA_MAP not in requests
    with Archive(root, policy, collector=collector) as archive:
        archive.run("PAGE", max_resources=2)
        result = archive.run("PAGE", max_resources=0)
        assert result["counts"]["PAGE"]["DONE"] == 3
        assert result["counts"]["PAGE"]["RAW_ONLY"] == 1
        assert result["download_complete_for_known_scope"]
        assert not result["text_complete_for_known_scope"]
        assert not result["absolute_aws_completeness_claimed"]
        assert archive.search("execution")[0]["url"] == IAM_PAGE
        assert not archive.search("zzzznotmatched")
        exported = archive.export(tmp_path / "documents.jsonl.gz")
        assert exported["records"] == 3
        inventory = archive.export(tmp_path / "inventory.jsonl.gz", inventory=True)
        assert inventory["records"] > 3
    with gzip.open(tmp_path / "documents.jsonl.gz", "rb") as stream:
        records = [json.loads(line) for line in stream]
    assert {r["source_uri"] for r in records} == {EC2_PAGE, IAM_PAGE, MD_PAGE}
    assert all(r["semantic_review"] == "NOT_REVIEWED" for r in records)
    assert "https://evil.invalid/doc.html" not in requests


@pytest.mark.parametrize("url", [
    "http://docs.aws.amazon.com/a.html", "https://docs.aws.amazon.com.evil.invalid/a.html",
    "https://docs.aws.amazon.com:8443/a.html", "https://user@docs.aws.amazon.com/a.html",
    "https://docs.aws.amazon.com/%2e%2e/secrets.html", "https://docs.aws.amazon.com/a/%00.html",
    "https://docs.aws.amazon.com/a/%zz.html", "https://docs.aws.amazon.com/a.html?cursor=1",
])
def test_url_boundary_rejects_host_path_query_traps(url):
    with pytest.raises(ValueError):
        archive_url(url)


def test_canonicalization_preserves_reserved_encoding_and_language():
    assert archive_url("https://DOCS.AWS.AMAZON.COM:443/a%2fb.html#x") == "https://docs.aws.amazon.com/a%2Fb.html"
    assert location(JA_MAP) == ("ja", "IAM")
    assert location(EC2_MAP) == ("en", "AWSEC2")


@pytest.mark.parametrize("raw", [
    b"<!DOCTYPE x [<!ENTITY x 'data'>]><urlset/>", b"<bad/>",
    b"<urlset><url><lastmod>today</lastmod></url></urlset>", b"\xff<urlset/>",
    b"\x1f\x8btruncated",
])
def test_xml_rejects_dtd_missing_loc_wrong_root_encoding_and_bad_gzip(raw):
    with pytest.raises((ValueError, OSError)):
        parse_sitemap(raw)


def test_sitemap_gzip_expansion_is_bounded(monkeypatch):
    monkeypatch.setattr("app.archive.XML_LIMIT", 1024)
    with pytest.raises(ValueError, match="EXPANSION"):
        parse_sitemap(gzip.compress(b"x" * 2048))


def test_existing_raw_recovery_is_offline_and_idempotent(tmp_path, policy, source):
    _, requests, collector, _ = source
    with Archive(tmp_path, policy, collector=collector) as archive:
        archive.run("SITEMAP", max_resources=1)
        archive.db.execute("UPDATE resources SET state='FETCHING' WHERE url=?", (ROOT_SITEMAP,))
        archive.db.commit()
    count = len(requests)
    with Archive(tmp_path, policy, collector=collector) as archive:
        result = archive.run("SITEMAP", max_resources=1)
        assert result["cache_hits_this_run"] == 1
        assert result["http_requests_this_run"] == 0
        assert archive.db.execute("SELECT count(*) FROM resources WHERE url=?", (EC2_MAP,)).fetchone()[0] == 1
    assert len(requests) == count


def test_corrupt_blob_marks_failed_without_implicit_refetch(tmp_path, policy, source):
    _, requests, collector, _ = source
    with Archive(tmp_path, policy, collector=collector) as archive:
        archive.run("SITEMAP", max_resources=1)
        row = archive.db.execute("SELECT raw_sha256 FROM resources WHERE url=?", (ROOT_SITEMAP,)).fetchone()
        archive._blob_path(row[0]).write_bytes(gzip.compress(b"corrupt"))
        archive.db.execute("UPDATE resources SET state='QUEUED' WHERE url=?", (ROOT_SITEMAP,))
        archive.db.commit()
        count = len(requests)
        result = archive.run("SITEMAP", max_resources=1)
        assert result["counts"]["SITEMAP"]["FAILED"] == 1
        assert not result["catalog_complete_for_known_scope"]
    assert len(requests) == count


def test_storage_guard_stops_before_request_and_can_raise_operational_budget(tmp_path, policy, source):
    _, requests, collector, _ = source
    smaller = policy.model_copy(update={"max_storage_bytes": 1})
    with Archive(tmp_path, smaller, collector=collector) as archive:
        result = archive.run("SITEMAP")
        assert result["stop_reason"] == "ARCHIVE_STORAGE_BUDGET_REACHED"
    assert not requests
    with Archive(tmp_path, policy, collector=collector) as archive:
        assert archive.run("SITEMAP", max_resources=1)["processed_this_run"] == 1


def test_scope_change_requires_separate_root_and_one_writer_is_enforced(tmp_path, policy):
    with Archive(tmp_path, policy):
        with pytest.raises(ValueError, match="WRITER"):
            with Archive(tmp_path, policy):
                pass
        with Archive(tmp_path, policy, read_only=True) as reader:
            assert not reader.status()["catalog_complete_for_known_scope"]
    with pytest.raises(ValueError, match="POLICY_MISMATCH"):
        with Archive(tmp_path, policy.model_copy(update={"languages": ("*",)})):
            pass


def test_pdf_raw_only_and_html_parse_failure_are_not_empty_success(tmp_path, policy, source):
    bodies, _, collector, _ = source
    bodies[EC2_PAGE] = ("text/html", b"<nav>No main article</nav>")
    discover(tmp_path, policy, collector)
    with Archive(tmp_path, policy, collector=collector) as archive:
        report = archive.run("PAGE", max_resources=0)
        assert report["counts"]["PAGE"]["FAILED"] == 1
        assert "MAIN_ARTICLE_NOT_FOUND" in report["reasons"]
        assert not report["download_complete_for_known_scope"]


def test_refresh_same_content_keeps_document_version_and_failed_retry_is_explicit(tmp_path, policy, source):
    _, requests, collector, _ = source
    discover(tmp_path, policy, collector)
    with Archive(tmp_path, policy, collector=collector) as archive:
        archive.run("PAGE", max_resources=0)
        versions = archive.db.execute("SELECT count(*) FROM versions").fetchone()[0]
        count = len(requests)
        assert archive.refresh("PAGE") == 4
        archive.run("PAGE", max_resources=0)
        assert len(requests) == count + 4
        assert archive.db.execute("SELECT count(*) FROM versions").fetchone()[0] == versions
        archive.db.execute("UPDATE resources SET state='FAILED' WHERE url=?", (IAM_PAGE,))
        archive.db.commit()
        assert archive.retry_failed("PAGE") == 1
        count = len(requests)
        archive.run("PAGE", max_resources=0)
        assert len(requests) == count  # Reparse raw, no refetch unless explicitly requested.


def test_robots_denied_is_reported_and_cross_host_redirect_never_connected(tmp_path, policy, source):
    bodies, requests, collector, _ = source
    blocked = "https://docs.aws.amazon.com/blocked/secret.html"
    bodies[EC2_MAP] = ("text/xml", sitemap("urlset", [blocked, EC2_PAGE]))
    original = collector.transport

    def redirected(url, limit, timeout):
        if url == EC2_PAGE:
            requests.append(url)
            return HttpResponse(302, {"location": "https://evil.invalid/doc.html"}, b"")
        return original(url, limit, timeout)

    collector.transport = redirected
    discover(tmp_path, policy, collector)
    with Archive(tmp_path, policy, collector=collector) as archive:
        report = archive.run("PAGE", max_resources=0)
        assert report["reasons"]["ROBOTS_DENIED"] == 1
        assert report["counts"]["PAGE"]["FAILED"] >= 1
    assert blocked not in requests and not any("evil.invalid" in u for u in requests)


def test_archive_read_only_cli_no_aws_or_source_requests(tmp_path, policy, monkeypatch, capsys):
    monkeypatch.setattr("boto3.Session", lambda **kw: pytest.fail("No AWS allowed"))
    policy_path = tmp_path / "policy.json"
    policy_path.write_text(policy.model_dump_json(), encoding="utf-8")
    root = tmp_path / "archive"
    with Archive(root, policy):
        pass
    assert main(["archive-status", "--root", str(root), "--policy", str(policy_path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["inference_calls"] == 0 and not report["absolute_aws_completeness_claimed"]


def test_changed_authoritative_sitemap_hint_requeues_only_affected_page(tmp_path, policy, source):
    bodies, _, collector, _ = source
    discover(tmp_path, policy, collector)
    with Archive(tmp_path, policy, collector=collector) as archive:
        archive.run("PAGE", max_resources=0)
        bodies[IAM_MAP] = ("text/xml", sitemap("urlset", [IAM_PAGE]).replace(b"2026-10-17", b"2026-10-18"))
        archive.refresh("SITEMAP")
        archive.run("SITEMAP", max_resources=0)
        iam = archive.db.execute("SELECT state,raw_sha256 FROM resources WHERE url=?", (IAM_PAGE,)).fetchone()
        ec2 = archive.db.execute("SELECT state FROM resources WHERE url=?", (EC2_PAGE,)).fetchone()
        assert iam[0] == "QUEUED" and iam[1] is None
        assert ec2[0] == "DONE"


@pytest.mark.parametrize("target", ["roles.html", "https://evil.invalid/doc.html"])
def test_meta_refresh_stubs_queue_safe_target_without_executing_scripts(tmp_path, policy, source, target):
    bodies, requests, collector, _ = source
    bodies[EC2_PAGE] = ("text/html", ("<html><meta http-equiv='refresh' content='0;URL=" + target +
                                    "'><script>throw 'MUST NEVER EXECUTE';</script></html>").encode())
    discover(tmp_path, policy, collector)
    with Archive(tmp_path, policy, collector=collector) as archive:
        archive.run("PAGE", max_resources=1)
        row = archive.db.execute("SELECT state,reason FROM resources WHERE url=?", (EC2_PAGE,)).fetchone()
        assert row[0] == ("REDIRECTED" if target == "roles.html" else "FAILED")
        assert not any("evil.invalid" in u for u in requests)


def test_cached_only_reparse_uses_no_transport_and_preserves_pending_queue(tmp_path, policy, source):
    _, requests, collector, _ = source
    with Archive(tmp_path, policy, collector=collector) as archive:
        archive.run("SITEMAP", max_resources=1)
        archive.db.execute("UPDATE resources SET state='QUEUED' WHERE url=?", (ROOT_SITEMAP,))
        archive.db.commit()
        count = len(requests)
        report = archive.run("SITEMAP", max_resources=0, cached_only=True)
        assert report["processed_this_run"] == 1
        assert report["http_requests_this_run"] == 0 and len(requests) == count
        assert report["stop_reason"] == "CACHED_QUEUE_DRAINED"
        assert report["counts"]["SITEMAP"]["QUEUED"] == 2


def test_stop_file_pauses_without_http_and_is_not_marked_complete(tmp_path, policy, source):
    _, requests, collector, _ = source
    with Archive(tmp_path, policy, collector=collector) as archive:
        (tmp_path / "STOP").touch()
        result = archive.run("SITEMAP", max_resources=0)
        assert result["stop_reason"] == "USER_STOP_REQUESTED"
        assert result["processed_this_run"] == 0 and not requests
        assert not result["download_complete_for_known_scope"]


def test_new_collector_honors_previous_process_request_clock(tmp_path, policy, source):
    bodies, _, collector, _ = source
    with Archive(tmp_path, policy, collector=collector) as archive:
        archive.run("SITEMAP", max_resources=1)
    marker = json.loads((tmp_path / "request-clock.json").read_text())
    marker["at"] = "2099-01-01T00:00:00+00:00"  # Conservative clock-skew branch, deterministic.
    (tmp_path / "request-clock.json").write_text(json.dumps(marker), encoding="utf-8")
    waits, clock = [], [0.0]

    def sleep(seconds):
        waits.append(seconds)
        clock[0] += seconds

    def get(url, limit, timeout):
        mime, body = bodies[url]
        return HttpResponse(200, {"content-type": mime}, body)

    fresh = Collector(get, sleep=sleep, clock=lambda: clock[0])
    with Archive(tmp_path, policy, collector=fresh) as archive:
        archive.run("SITEMAP", max_resources=1)
    assert waits[0] >= 5.0


def test_normalized_export_exceeds_one_hundred_documents(tmp_path, policy, source):
    bodies, _, collector, _ = source
    urls = [f"https://docs.aws.amazon.com/new-service/latest/guide/p{i:03}.html" for i in range(150)]
    bodies[ROOT_SITEMAP] = ("text/xml", sitemap("urlset", urls))
    for i, url in enumerate(urls):
        bodies[url] = ("text/html", html("Official synthetic source", f"Unique fixture record {i}"))
    with Archive(tmp_path, policy, collector=collector) as archive:
        archive.run("SITEMAP", max_resources=0)
        report = archive.run("PAGE", max_resources=0)
        assert report["counts"]["PAGE"]["DONE"] == 150
        assert report["text_complete_for_known_scope"]
        result = archive.export(tmp_path / "large-fixture.jsonl.gz")
        assert result["records"] == 150
    with gzip.open(tmp_path / "large-fixture.jsonl.gz", "rb") as stream:
        assert sum(1 for _ in stream) == 150


def test_meta_refresh_cycle_never_becomes_complete_text(tmp_path, policy, source):
    bodies, _, collector, _ = source
    bodies[EC2_PAGE] = ("text/html", ("<meta http-equiv='refresh' content='0;URL=" + IAM_PAGE + "'>").encode())
    bodies[IAM_PAGE] = ("text/html", ("<meta http-equiv='refresh' content='0;URL=" + EC2_PAGE + "'>").encode())
    discover(tmp_path, policy, collector)
    with Archive(tmp_path, policy, collector=collector) as archive:
        result = archive.run("PAGE", max_resources=0)
        assert result["reasons"]["ARCHIVE_HTML_REFRESH_LOOP"] == 1
        assert not result["text_complete_for_known_scope"]
