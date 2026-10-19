"""Single-worker, robots-aware, bounded HTTPS fetches of allowlisted AWS docs."""

import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin
from urllib.robotparser import RobotFileParser

from app.ingestion.models import source_uri

USER_AGENT = "EKGPortfolioIngest/1.0 (public AWS documentation; local educational index)"
ROBOTS_URL = "https://docs.aws.amazon.com/robots.txt"
MAX_SOURCE_BYTES = 10 * 1024 * 1024


class FetchError(ValueError):
    """Only controlled categories are persisted; not raw URLs/provider messages."""


@dataclass(frozen=True)
class HttpResponse:
    status: int
    headers: dict[str, str]
    body: bytes


@dataclass(frozen=True)
class Fetched:
    requested_url: str
    effective_url: str
    fetched_at: datetime
    content_type: str
    body: bytes
    redirects: tuple[str, ...]
    attempts: int


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def http_get(url: str, limit: int, timeout: float) -> HttpResponse:
    opener = urllib.request.build_opener(_NoRedirect())
    request = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT, "Accept": "text/html,text/plain;q=0.5",
        "Accept-Encoding": "identity",
    })
    try:
        response = opener.open(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        response = exc
    with response:
        headers = {k.lower(): v for k, v in response.headers.items()}
        if headers.get("content-encoding", "identity").lower() not in ("", "identity"):
            raise FetchError("COMPRESSED_RESPONSE_REJECTED")
        length = headers.get("content-length")
        if length is not None and (not length.isdecimal() or int(length) > limit):
            raise FetchError("SOURCE_SIZE_LIMIT")
        body = response.read(limit + 1)
        if len(body) > limit:
            raise FetchError("SOURCE_SIZE_LIMIT")
        return HttpResponse(response.status, headers, body)


class Collector:
    def __init__(self, transport=http_get, *, sleep=time.sleep, clock=time.monotonic):
        self.transport, self.sleep, self.clock = transport, sleep, clock
        self.last_start: float | None = None
        self.delay_seconds = 5.0
        self.robot: RobotFileParser | None = None
        self.robots_bytes: bytes | None = None
        self.robots_fetched_at: datetime | None = None
        self.request_count = 0

    def _request(self, url: str, limit: int) -> HttpResponse:
        if self.last_start is not None:
            wait = self.delay_seconds - (self.clock() - self.last_start)
            if wait > 0:
                self.sleep(wait)
        self.last_start = self.clock()
        self.request_count += 1
        response = self.transport(url, limit, 15.0)
        if len(response.body) > limit:
            raise FetchError("SOURCE_SIZE_LIMIT")
        return response

    def _load_robots(self):
        if self.robot is not None:
            return
        response = self._request(ROBOTS_URL, 128 * 1024)
        if response.status != 200:
            raise FetchError("ROBOTS_UNAVAILABLE")
        text = response.body.decode("utf-8", errors="strict")
        robot = RobotFileParser()
        robot.parse(text.splitlines())
        # AWS currently has a blank line before its trailing Crawl-delay. Python's
        # RobotFileParser may then return None. Honor every numeric delay anyway,
        # conservatively, instead of accidentally falling back to a faster rate.
        delays = [float(n) for n in re.findall(
            r"(?im)^crawl-delay:\s*(\d+(?:\.\d+)?)\s*$", text
        )]
        self.delay_seconds = max(5.0, robot.crawl_delay(USER_AGENT) or 0, *delays)
        if self.delay_seconds > 60:
            raise FetchError("ROBOTS_DELAY_REQUIRES_REVIEW")
        self.robot, self.robots_bytes = robot, response.body
        self.robots_fetched_at = datetime.now(UTC)

    def fetch(self, url: str) -> Fetched:
        return self.fetch_resource(url, validator=source_uri, content_types=("text/html",),
                                   max_bytes=MAX_SOURCE_BYTES)

    def fetch_resource(self, url: str, *, validator, content_types: tuple[str, ...],
                       max_bytes: int) -> Fetched:
        """Explicit server-owned policy for archival resources; fetch() stays narrow."""
        requested = validator(url)
        self._load_robots()
        current = requested
        redirects = []
        attempts = 0
        errors = 0
        while True:
            current = validator(current)  # Validate before every connection.
            if not self.robot.can_fetch(USER_AGENT, current):
                raise FetchError("ROBOTS_DENIED")
            attempts += 1
            try:
                response = self._request(current, max_bytes)
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                if errors >= 2:
                    raise FetchError("NETWORK_RETRIES_EXHAUSTED") from exc
                errors += 1
                self.sleep(float(2 ** (errors - 1)))
                continue
            if response.status in (301, 302, 303, 307, 308):
                if len(redirects) >= 3 or "location" not in response.headers:
                    raise FetchError("REDIRECT_LIMIT_OR_MISSING_LOCATION")
                target = validator(urljoin(current, response.headers["location"]))
                if target in redirects or target == current:
                    raise FetchError("REDIRECT_LOOP")
                redirects.append(target)
                current = target
                continue
            if response.status == 429 or 500 <= response.status <= 599:
                if errors >= 2:
                    raise FetchError("HTTP_RETRIES_EXHAUSTED")
                after = response.headers.get("retry-after", "")
                if after and (not after.isdecimal() or int(after) > 30):
                    raise FetchError("RETRY_AFTER_REQUIRES_LATER_RUN")
                errors += 1
                self.sleep(max(float(after or 0), float(2 ** (errors - 1))))
                continue
            if response.status != 200:
                raise FetchError("HTTP_NON_SUCCESS")
            content_type = response.headers.get("content-type", "").lower()
            if content_type.split(";", 1)[0].strip() not in content_types:
                category = "HTML_CONTENT_TYPE_REQUIRED" if content_types == ("text/html",) else (
                    "RESOURCE_CONTENT_TYPE_REQUIRED")
                raise FetchError(category)
            if "charset=" in content_type and not re.search(
                r"charset=[\"']?(utf-8|us-ascii)(?:[\"';\s]|$)", content_type
            ):
                raise FetchError("UTF8_CONTENT_TYPE_REQUIRED")
            if not response.body:
                raise FetchError("EMPTY_RESPONSE")
            return Fetched(requested, current, datetime.now(UTC), content_type,
                           response.body, tuple(redirects), attempts)
