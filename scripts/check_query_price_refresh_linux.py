"""Offline smoke of the exact two-entry deployed-package patch; no live credentials."""

import json
import os
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

with zipfile.ZipFile("/build/query-price-refresh.zip") as package:
    package.extractall("/tmp/price-refresh")
sys.path.insert(0, "/tmp/price-refresh")
os.chdir("/tmp/price-refresh")
os.environ.update(MODEL_ENABLED="true", KG_DEEPSEEK_SECRET_ARN="synthetic-no-secrets-mounted")

from boto3.dynamodb.types import TypeSerializer  # noqa: E402
from pydantic import SecretStr  # noqa: E402

from app import query_handler  # noqa: E402
from app.contracts import canonical_json  # noqa: E402
from app.external_gateway import ExternalPriceSheet  # noqa: E402
from app.external_http import OfficialHttpsTransport  # noqa: E402

fixtures = json.loads(Path("/build/counter-fixture.json").read_bytes())


class FixtureDynamo:
    meta = SimpleNamespace(endpoint_url="https://dynamodb.us-east-1.amazonaws.com")

    def get_item(self, **kwargs):
        key = kwargs["Key"]["PK"]["S"]
        row = next(r for r in fixtures.values() if r["PK"] == key)
        return {"Item": {k: TypeSerializer().serialize(v) for k, v in row.items()}}


def forbidden(*args, **kwargs):
    raise AssertionError("MODEL_OR_NETWORK_DISPATCH_NOT_ALLOWED")


OfficialHttpsTransport.request = forbidden
query_handler.clients = lambda: {"dynamodb": FixtureDynamo(), "secretsmanager": None}
query_handler.read_deepseek_secret = lambda *args, **kwargs: SecretStr("synthetic-key-not-used-by-any-api")
gateway = query_handler.gateway(5)
gateway.prices.rate_for("deepseek-flash", "deepseek-flash", "external-default", now=datetime.now(UTC), live=True)
old = ExternalPriceSheet.model_validate_json(Path("configs/prices/external-direct-2026-10-17.json").read_bytes())
try:
    old.rate_for("deepseek-flash", "deepseek-flash", "external-default", now=datetime.now(UTC), live=True)
except ValueError:
    pass
else:
    raise ValueError("OLD_PRICE_EXPIRY_NOT_REPRODUCED")
page = query_handler.handler({"rawPath": "/", "requestContext": {"http": {"method": "GET"}}}, None)
if page["statusCode"] != 200:
    raise ValueError("PRICE_REFRESH_STATIC_PAGE_REGRESSION")
report = {"status": "PASS", "python": sys.version.split()[0], "old_price_rejection_reproduced": True,
    "gateway_constructed_with_current_sheet": True, "new_price_version": gateway.prices.version,
    "static_home_status": 200, "aws_calls": 0, "model_calls": 0, "network": "none", "real_credentials_mounted": False}
Path("/build/native-report.json").write_bytes(canonical_json(report) + b"\n")
print(json.dumps(report))
