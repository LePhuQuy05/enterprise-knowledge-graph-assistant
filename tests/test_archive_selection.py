import json

import pytest

from app.archive import Archive, ArchivePolicy
from app.ingestion.collector import Collector, HttpResponse

BASE = "https://docs.aws.amazon.com/lambda/latest/dg/"


def make_archive(tmp_path):
    requests, clock = [], [0.0]

    def transport(url, limit, timeout):
        requests.append(url)
        if url.endswith("robots.txt"):
            return HttpResponse(200, {"content-type": "text/plain"},
                                b"User-agent: *\nCrawl-delay: 5\n")
        body = f'<main><h1>Guide</h1><p>Evidence {url}</p></main>'.encode()
        return HttpResponse(200, {"content-type": "text/html"}, body)

    def sleep(seconds):
        clock[0] += seconds

    collector = Collector(transport, sleep=sleep, clock=lambda: clock[0])
    policy = ArchivePolicy(version="selection-fixture", min_free_disk_bytes=0)
    return Archive(tmp_path, policy, collector=collector), requests


def test_selection_processes_only_requested_pages_in_order_and_resumes(tmp_path):
    archive, requests = make_archive(tmp_path)
    urls = tuple(BASE + name + ".html" for name in ("a", "b", "c"))
    with archive:
        for url in urls:
            archive._add(url, "PAGE", None, None)
        archive.db.commit()
        first = archive.run("PAGE", selected_urls=(urls[2], urls[1]), max_resources=1)
        assert first["processed_this_run"] == 1
        assert archive.db.execute("SELECT state FROM resources WHERE url=?", (urls[0],)).fetchone()[0] == "QUEUED"
        second = archive.run("PAGE", selected_urls=(urls[2], urls[1]), max_resources=0)
        assert second["stop_reason"] == "SELECTION_DRAINED"
        assert second["processed_this_run"] == 1
        assert second["absolute_aws_completeness_claimed"] is False
        assert [u for u in requests if u.endswith(".html")] == [urls[2], urls[1]]
        assert second["selection_sha256"] == first["selection_sha256"]


@pytest.mark.parametrize("selection", [(), ("https://evil.invalid/a.html",), (BASE + "missing.html",),
                                       (BASE + "a.html", BASE + "a.html"), ({"url": "invalid"},)])
def test_selection_rejects_before_network_or_state_changes(tmp_path, selection):
    archive, requests = make_archive(tmp_path)
    with archive:
        archive._add(BASE + "a.html", "PAGE", None, None)
        archive.db.commit()
        with pytest.raises(ValueError, match="ARCHIVE_(SELECTION|HOST)"):
            archive.run("PAGE", selected_urls=selection)
        assert requests == []
        assert archive.db.execute("SELECT state FROM resources WHERE url=?", (BASE + "a.html",)).fetchone()[0] == "QUEUED"


def test_selected_failed_page_not_silently_retried(tmp_path):
    archive, requests = make_archive(tmp_path)
    with archive:
        url = BASE + "a.html"
        archive._add(url, "PAGE", None, None)
        archive.db.execute("UPDATE resources SET state='FAILED',reason='TEST' WHERE url=?", (url,))
        archive.db.commit()
        report = archive.run("PAGE", selected_urls=(url,))
        assert report["processed_this_run"] == 0
        assert report["counts"]["PAGE"]["FAILED"] == 1
        assert requests == []
        json.dumps(report)
