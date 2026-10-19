"""Single-dispatch HTTPS and private key loading for approved external providers.

No configurable base URLs, proxy environment, redirects, SDK retries or raw errors.
"""

import http.client
import json
import math
import ssl
import time
from pathlib import Path
from typing import Literal

from pydantic import SecretStr

from app.contracts import canonical_json, sha256_bytes
from app.model_gateway import GatewayError

ProviderName = Literal["DEEPSEEK", "VOYAGE"]
MAX_HTTP_BYTES = 1_048_576
HOSTS = {"DEEPSEEK": "api.deepseek.com", "VOYAGE": "api.voyageai.com"}
ROUTES = {
    ("DEEPSEEK", "GET", "/models"),
    ("DEEPSEEK", "GET", "/user/balance"),
    ("DEEPSEEK", "POST", "/chat/completions"),
    ("VOYAGE", "POST", "/v1/embeddings"),
}


class ExternalHttpError(GatewayError):
    def __init__(self, category: str):
        self.category = category
        super().__init__("External API failed: " + category)


def strict_json(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    def constant(_):
        raise ValueError("nonfinite")

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=constant)
        if type(value) is not dict:
            raise ValueError("object required")
        return value
    except (ValueError, UnicodeError, RecursionError):
        raise ExternalHttpError("INVALID_JSON") from None


def load_private_key(project_root: Path, provider: ProviderName) -> SecretStr:
    """Read only the intended bounded secret file; never return diagnostics with its content."""
    if provider not in HOSTS:
        raise ExternalHttpError("UNAPPROVED_PROVIDER")
    root = project_root.resolve()
    parts = ("results", "private", "model-api-keys", provider.lower() + ".json")
    path = root
    try:
        for part in parts:
            path = path / part
            if path.is_symlink() or path.is_junction():
                raise ExternalHttpError("UNSAFE_SECRET_PATH")
        if not path.is_file() or path.stat().st_size > 4_096:
            raise ExternalHttpError("KEY_NOT_STORED")
        with path.open("rb") as stream:
            raw = stream.read(4_097)
        if len(raw) > 4_096:
            raise ExternalHttpError("INVALID_KEY_FILE")
        record = strict_json(raw)
        key = record.get("api_key")
        if set(record) != {"provider", "api_key"} or record["provider"] != provider.lower() or (
            type(key) is not str or not 16 <= len(key) <= 512
            or any(not 0x21 <= ord(char) <= 0x7E for char in key)
        ):
            raise ExternalHttpError("INVALID_KEY_FILE")
        return SecretStr(key)
    except ExternalHttpError:
        raise
    except OSError:
        raise ExternalHttpError("KEY_READ_DENIED") from None


class OfficialHttpsTransport:
    live = True

    def __init__(self, keys: dict[ProviderName, SecretStr], *, timeout_seconds: int = 20,
                 minimum_intervals: dict[ProviderName, int] | None = None,
                 deadline_monotonic: float | None = None):
        if not keys or set(keys) - set(HOSTS) or any(type(k) is not SecretStr for k in keys.values()):
            raise ExternalHttpError("INVALID_KEY_CONFIGURATION")
        if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 25:
            raise ExternalHttpError("INVALID_TIMEOUT")
        self._keys, self.timeout_seconds = dict(keys), timeout_seconds
        if deadline_monotonic is not None and (
            type(deadline_monotonic) is not float or not math.isfinite(deadline_monotonic) or deadline_monotonic < 0
        ):
            raise ExternalHttpError("INVALID_DEADLINE")
        self._deadline_monotonic = deadline_monotonic
        self.dispatch_attempts = 0
        intervals = minimum_intervals or {}
        if set(intervals) - set(HOSTS) or any(type(v) is not int or not 1 <= v <= 59 for v in intervals.values()):
            raise ExternalHttpError("INVALID_PACING_CONFIGURATION")
        self._intervals, self._completed_at = dict(intervals), {}

    def request(self, provider: ProviderName, method: str, path: str, body: bytes | None = None) -> dict:
        if (provider, method, path) not in ROUTES or provider not in self._keys:
            raise ExternalHttpError("UNAPPROVED_ROUTE")
        if (method == "GET" and body is not None) or (
            method == "POST" and (type(body) is not bytes or not 0 < len(body) <= MAX_HTTP_BYTES)
        ):
            raise ExternalHttpError("INVALID_REQUEST_BODY")
        # Optional local batch pacing, measured after the previous request completes.
        # Cloud serving does not sleep here: its admission policy is a separate gate.
        if provider in self._intervals and provider in self._completed_at:
            delay = self._intervals[provider] - (time.monotonic() - self._completed_at[provider])
            if delay > 0:
                time.sleep(delay)
        # http.client ignores HTTP(S)_PROXY and does not follow redirects or retry requests.
        started = time.monotonic()
        deadline = started + self.timeout_seconds
        if self._deadline_monotonic is not None:
            deadline = min(deadline, self._deadline_monotonic)
        if deadline <= started:
            raise ExternalHttpError("TIMEOUT")  # No HTTP dispatch; ledger remains conservative if begin_call already ran.
        connection = http.client.HTTPSConnection(
            HOSTS[provider], timeout=deadline - started, context=ssl.create_default_context(),
        )
        try:
            self.dispatch_attempts += 1
            connection.request(method, path, body=body, headers={
                "Authorization": "Bearer " + self._keys[provider].get_secret_value(),
                "Content-Type": "application/json", "Accept": "application/json",
            })
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise ExternalHttpError("TIMEOUT")
            if connection.sock is not None:
                connection.sock.settimeout(remaining)
            response = connection.getresponse()
            if response.status != 200:
                category = {401: "AUTHENTICATION", 402: "BALANCE", 403: "PERMISSION",
                            429: "RATE_LIMIT"}.get(response.status, "PROVIDER_ERROR")
                if 300 <= response.status < 400:
                    category = "REDIRECT_BLOCKED"
                # Never read/print provider error bodies: they can contain echoed secrets/inputs.
                raise ExternalHttpError(category)
            raw = bytearray()
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise ExternalHttpError("TIMEOUT")
                if connection.sock is not None:
                    connection.sock.settimeout(remaining)
                chunk = response.read1(min(65_536, MAX_HTTP_BYTES + 1 - len(raw)))
                raw.extend(chunk)
                if len(raw) > MAX_HTTP_BYTES:
                    raise ExternalHttpError("RESPONSE_TOO_LARGE")
                if not chunk:
                    break
            return strict_json(bytes(raw))
        except ExternalHttpError:
            raise
        except (OSError, http.client.HTTPException, ValueError):
            raise ExternalHttpError("TRANSPORT_UNKNOWN") from None
        finally:
            connection.close()
            if provider in self._intervals:
                self._completed_at[provider] = time.monotonic()


def deepseek_preflight(transport) -> dict:
    """Two read-only calls; intentionally not a generation/embedding access proof."""
    models = transport.request("DEEPSEEK", "GET", "/models")
    balance = transport.request("DEEPSEEK", "GET", "/user/balance")
    data = models.get("data")
    if type(data) is not list or len(data) > 100:
        raise ExternalHttpError("INVALID_MODEL_METADATA")
    selected = [item for item in data if type(item) is dict and item.get("id") == "deepseek-flash"]
    if len(selected) != 1:
        raise ExternalHttpError("MODEL_NOT_AVAILABLE")
    model = selected[0]
    # Copy only known public metadata and balance fields, never the raw provider payload.
    result = {key: model[key] for key in ("id", "name", "context_window", "max_output_tokens") if key in model}
    entries = balance.get("balance_infos")
    if type(balance.get("is_available")) is not bool or type(entries) is not list or len(entries) > 10:
        raise ExternalHttpError("INVALID_BALANCE_METADATA")
    amounts = []
    for item in entries:
        if type(item) is not dict or item.get("currency") not in {"USD", "CNY"}:
            raise ExternalHttpError("INVALID_BALANCE_METADATA")
        amount = item.get("total_balance")
        if type(amount) is not str or len(amount) > 32 or any(c not in "0123456789.-" for c in amount):
            raise ExternalHttpError("INVALID_BALANCE_METADATA")
        amounts.append({"currency": item["currency"], "total_balance": amount})
    return {"provider": "DEEPSEEK", "model": result, "available": balance["is_available"],
            "balances": amounts, "inference_verified": False,
            "metadata_sha256": sha256_bytes(canonical_json(result))}
