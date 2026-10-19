import json

import pytest
from pydantic import SecretStr

from app import external_http as http

KEY = "FIXTURE_ONLY_DO_NOT_USE_1234567890"


def test_private_key_loader_checks_provider_size_and_redacts(tmp_path):
    directory = tmp_path / "results/private/model-api-keys"
    directory.mkdir(parents=True)
    path = directory / "deepseek.json"
    path.write_text(json.dumps({"provider": "deepseek", "api_key": KEY}))
    key = http.load_private_key(tmp_path, "DEEPSEEK")
    assert key.get_secret_value() == KEY
    assert KEY not in repr(key)
    path.write_text(json.dumps({"provider": "voyage", "api_key": KEY}))
    with pytest.raises(http.ExternalHttpError, match="INVALID_KEY_FILE") as exc:
        http.load_private_key(tmp_path, "DEEPSEEK")
    assert KEY not in str(exc.value)
    path.write_bytes(b"X" * 4097)
    with pytest.raises(http.ExternalHttpError, match="KEY_NOT_STORED"):
        http.load_private_key(tmp_path, "DEEPSEEK")


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'[]', b'not JSON', b'\xff'])
def test_json_rejects_ambiguous_payloads_without_echo(raw):
    with pytest.raises(http.ExternalHttpError, match="INVALID_JSON") as exc:
        http.strict_json(raw)
    assert str(exc.value) == "External API failed: INVALID_JSON"


class Response:
    def __init__(self, status=200, body=b'{"ok":true}'):
        self.status, self.body = status, body

    def read1(self, count):
        value, self.body = self.body[:count], self.body[count:]
        return value


class Connection:
    def __init__(self, response):
        self.response, self.calls, self.sock, self.closed = response, [], None, False

    def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))

    def getresponse(self):
        return self.response

    def close(self):
        self.closed = True


def transport(monkeypatch, response):
    connection = Connection(response)
    monkeypatch.setattr(http.http.client, "HTTPSConnection", lambda *a, **kw: connection)
    return http.OfficialHttpsTransport({"DEEPSEEK": SecretStr(KEY)}), connection


def test_https_dispatches_once_and_validates_route_before_key_use(monkeypatch):
    gateway, conn = transport(monkeypatch, Response())
    for method, path in (("GET", "https://other.test/"), ("POST", "/models"), ("GET", "/models?x=1")):
        with pytest.raises(http.ExternalHttpError, match="UNAPPROVED_ROUTE"):
            gateway.request("DEEPSEEK", method, path)
    assert not conn.calls
    assert gateway.request("DEEPSEEK", "GET", "/models") == {"ok": True}
    assert len(conn.calls) == 1 and conn.closed
    assert KEY not in repr(gateway)


def test_header_wait_uses_remaining_budget_after_request_send(monkeypatch):
    gateway, conn = transport(monkeypatch, Response())
    clock, timeouts = [0.0], []
    conn.sock = type("Socket", (), {"settimeout": lambda _, value: timeouts.append(value)})()
    original = conn.request
    def slow_request(*args, **kwargs):
        original(*args, **kwargs)
        clock[0] = 18.0
    conn.request = slow_request
    monkeypatch.setattr(http.time, "monotonic", lambda: clock[0])
    assert gateway.request("DEEPSEEK", "GET", "/models") == {"ok": True}
    assert timeouts[0] == 2.0 and len(conn.calls) == 1


def test_absolute_deadline_includes_time_spent_before_transport_dispatch(monkeypatch):
    conn = Connection(Response())
    clock, socket_timeouts, connection_timeouts = [8.0], [], []
    conn.sock = type("Socket", (), {"settimeout": lambda _, value: socket_timeouts.append(value)})()
    original = conn.request

    def slow_send(*args, **kwargs):
        original(*args, **kwargs)
        clock[0] = 9.0

    def connect(*args, **kwargs):
        connection_timeouts.append(kwargs["timeout"])
        return conn

    conn.request = slow_send
    monkeypatch.setattr(http.http.client, "HTTPSConnection", connect)
    monkeypatch.setattr(http.time, "monotonic", lambda: clock[0])
    transport = http.OfficialHttpsTransport({"DEEPSEEK": SecretStr(KEY)}, timeout_seconds=18,
                                            deadline_monotonic=10.0)
    assert transport.request("DEEPSEEK", "GET", "/models") == {"ok": True}
    assert connection_timeouts == [2.0] and socket_timeouts[0] == 1.0 and len(conn.calls) == 1


def test_expired_absolute_deadline_does_not_create_connection_or_dispatch(monkeypatch):
    monkeypatch.setattr(http.time, "monotonic", lambda: 11.0)
    monkeypatch.setattr(http.http.client, "HTTPSConnection", lambda *a, **kw: pytest.fail("No connection allowed"))
    transport = http.OfficialHttpsTransport({"DEEPSEEK": SecretStr(KEY)}, deadline_monotonic=10.0)
    with pytest.raises(http.ExternalHttpError, match="TIMEOUT"):
        transport.request("DEEPSEEK", "GET", "/models")
    assert transport.dispatch_attempts == 0


@pytest.mark.parametrize("status,category", [
    (301, "REDIRECT_BLOCKED"), (401, "AUTHENTICATION"), (402, "BALANCE"),
    (403, "PERMISSION"), (429, "RATE_LIMIT"), (500, "PROVIDER_ERROR"),
])
def test_errors_never_follow_redirect_retry_or_return_body(monkeypatch, status, category):
    gateway, conn = transport(monkeypatch, Response(status, KEY.encode()))
    with pytest.raises(http.ExternalHttpError, match=category) as exc:
        gateway.request("DEEPSEEK", "GET", "/models")
    assert len(conn.calls) == 1 and conn.closed
    assert KEY not in str(exc.value)


def test_size_and_total_read_deadline_are_bounded(monkeypatch):
    gateway, conn = transport(monkeypatch, Response(body=b" " * (http.MAX_HTTP_BYTES + 1)))
    with pytest.raises(http.ExternalHttpError, match="RESPONSE_TOO_LARGE"):
        gateway.request("DEEPSEEK", "GET", "/models")
    assert conn.closed
    gateway, conn = transport(monkeypatch, Response())
    clock = iter((0, 30))
    monkeypatch.setattr(http.time, "monotonic", lambda: next(clock))
    with pytest.raises(http.ExternalHttpError, match="TIMEOUT"):
        gateway.request("DEEPSEEK", "GET", "/models")
    assert conn.closed


def test_read_only_preflight_uses_only_two_gets_and_does_not_claim_inference():
    class Stub:
        def __init__(self):
            self.calls = []

        def request(self, provider, method, path):
            self.calls.append((provider, method, path))
            if path == "/models":
                return {"data": [{"id": "deepseek-flash", "context_window": 1048576}]}
            return {"is_available": True, "balance_infos": [{"currency": "USD", "total_balance": "10.00"}]}

    stub = Stub()
    report = http.deepseek_preflight(stub)
    assert [call[1] for call in stub.calls] == ["GET", "GET"]
    assert report["balances"] == [{"currency": "USD", "total_balance": "10.00"}]
    assert report["inference_verified"] is False


def test_local_batch_pacing_waits_after_completion_and_never_retries(monkeypatch):
    clock, waits, connections = [0.0], [], []

    def sleep(delay):
        waits.append(delay)
        clock[0] += delay

    def connect(*args, **kwargs):
        connection = Connection(Response())
        connections.append(connection)
        return connection

    monkeypatch.setattr(http.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(http.time, "sleep", sleep)
    monkeypatch.setattr(http.http.client, "HTTPSConnection", connect)
    gateway = http.OfficialHttpsTransport({"VOYAGE": SecretStr(KEY)}, minimum_intervals={"VOYAGE": 21})
    for _ in range(4):
        gateway.request("VOYAGE", "POST", "/v1/embeddings", b"{}")
    assert waits == [21, 21, 21] and gateway.dispatch_attempts == 4
    assert all(len(connection.calls) == 1 for connection in connections)


@pytest.mark.parametrize("intervals", [{"VOYAGE": 0}, {"VOYAGE": 60}, {"VOYAGE": True}, {"OTHER": 21}])
def test_invalid_local_pacing_rejected_before_request(intervals):
    with pytest.raises(http.ExternalHttpError, match="INVALID_PACING"):
        http.OfficialHttpsTransport({"VOYAGE": SecretStr(KEY)}, minimum_intervals=intervals)
