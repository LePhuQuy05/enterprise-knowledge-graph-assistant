"""Inspect the exact combined web ZIP offline; never mount credentials or call models."""

import copy
import hashlib
import json
import os
import sys
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace


def main():
    source = Path("/input")
    pins = json.loads((source/"pins.json").read_bytes())
    package = source/"scoped-web-current-rates.zip"
    if hashlib.sha256(package.read_bytes()).hexdigest() != pins["package_sha256"]:
        raise ValueError("SCOPED_WEB_PROBE_PACKAGE_PIN")
    target = Path("/tmp/scoped-web")
    with zipfile.ZipFile(package) as z:
        for n in z.namelist():
            if Path(n).is_absolute() or ".." in Path(n).parts or "\\" in n:
                raise ValueError("SCOPED_WEB_PROBE_UNSAFE_ZIP")
        z.extractall(target)
    os.environ.update(MODEL_ENABLED="true", KG_FUNDING_POLICY="AUDITED_2_TO_8_USD_V2",
                      KG_DEEPSEEK_SECRET_ARN="synthetic-not-used",
                      PUBLIC_CONFIG_JSON=json.dumps(pins["public_config"]|{"scoped_runs_enabled": True}))
    os.chdir(target)
    sys.path.insert(0, str(target))
    from boto3.dynamodb.types import TypeSerializer
    from pydantic import SecretStr

    from app import query_handler
    from app.contracts import canonical_json
    from app.external_gateway import ExternalPriceSheet
    from app.external_http import OfficialHttpsTransport
    rows = json.loads((source/"managed-counter-fixture.json").read_bytes())
    active = copy.deepcopy(rows)

    class ReadOnlyDynamo:
        meta = SimpleNamespace(endpoint_url="https://dynamodb.us-east-1.amazonaws.com")

        def get_item(self, **kwargs):
            key = (kwargs["Key"]["PK"]["S"], kwargs["Key"]["SK"]["S"])
            row = next((r for r in active.values() if isinstance(r, dict) and (r.get("PK"), r.get("SK")) == key), None)
            return {"Item": {k: TypeSerializer().serialize(v) for k, v in row.items()}} if row else {}

    def forbidden(*args, **kwargs):
        raise AssertionError("NETWORK_MODEL_OR_COUNTER_MUTATION_FORBIDDEN")

    OfficialHttpsTransport.request = forbidden
    query_handler.clients = lambda: {"dynamodb": ReadOnlyDynamo(), "secretsmanager": None,
                                    "cognito-idp": None, "s3": None}
    query_handler.read_deepseek_secret = lambda *args, **kwargs: SecretStr("synthetic-unused")
    checks = []
    gateway = query_handler.gateway(2)
    gateway.prices.rate_for("deepseek-flash", "deepseek-flash", "external-default", now=datetime.now(UTC), live=True)
    checks.append("8USD funded gateway composes with current prices and no provider request")
    old = ExternalPriceSheet.model_validate_json((target/pins["old_price_path"]).read_bytes())
    try:
        old.rate_for("deepseek-flash", "deepseek-flash", "external-default", now=datetime.now(UTC), live=True)
    except ValueError:
        checks.append("historical price expiry reproduced; old asset preserved")
    else:
        raise ValueError("SCOPED_WEB_OLD_PRICE_MUST_BE_EXPIRED")
    manifest = json.loads((target/"configs/frontend-static.json").read_bytes())
    if len(manifest) != 119:
        raise ValueError("SCOPED_WEB_PUBLIC_ASSET_COUNT")
    for path, item in manifest.items():
        response = query_handler.handler({"rawPath": path, "requestContext": {"http": {"method": "GET"}}}, None)
        if response["statusCode"] != 200 or response["body"] != item["body"]:
            raise ValueError("SCOPED_WEB_STATIC_ASSET_REGRESSION")
    checks.append("all119 static routes return exact UTF8 body")
    response = query_handler.handler({"rawPath": "/config.json", "requestContext": {"http": {"method": "GET"}}}, None)
    if response["statusCode"] != 200 or json.loads(response["body"])["scoped_runs_enabled"] is not True:
        raise ValueError("SCOPED_WEB_PUBLIC_FLAG_BINDING")
    checks.append("public opt-in flag exposed without overwriting auth settings")
    response = query_handler.handler({"rawPath": "/v1/status", "requestContext": {"http": {"method": "GET"}}},
                                     SimpleNamespace(get_remaining_time_in_millis=lambda: 60000))
    if response["statusCode"] != 401:
        raise ValueError("SCOPED_WEB_UNAUTHENTICATED_REGRESSION")
    checks.append("legacy authenticated endpoint still rejects unsigned caller")
    for mutation in ("audit", "halted", "wrong_cap"):
        active.clear()
        active.update(copy.deepcopy(rows))
        if mutation == "audit":
            active.pop("audit")
        elif mutation == "halted":
            active["funding"]["halted"] = True
        else:
            active["funding"]["cap_micro_usd"] = 9000000
        try:
            query_handler.gateway(2)
        except query_handler.ApiError:
            checks.append("funding fail-closed: "+mutation)
        else:
            raise ValueError("SCOPED_WEB_FUNDING_GUARD_REGRESSION")
    report = {"status": "PASS_EXACT_COMBINED_WEB_LINUX", "package_sha256": pins["package_sha256"],
              "checks": checks, "static_routes": len(manifest), "python": sys.version.split()[0],
              "new_price_version": gateway.prices.version, "model_calls": 0, "aws_calls": 0,
              "counter_mutations": 0, "network": "none", "credentials_mounted": False,
              "limitation": "Fresh managed rows replayed offline; not real JWT/IAM, provider inference or answer quality"}
    Path("/output/native.json").write_bytes(canonical_json(report))
    print(json.dumps({k: report[k] for k in ("status", "package_sha256", "static_routes", "model_calls", "aws_calls")}))


if __name__ == "__main__":
    main()
