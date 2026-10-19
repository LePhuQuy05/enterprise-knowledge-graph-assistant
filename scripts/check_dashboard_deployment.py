"""Read back public dashboard assets and negative HTTP auth, with no inference."""

import argparse
import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.contracts import sha256_bytes
from app.tracer import HEADERS
from scripts.foundation import _session

ROOT = Path(__file__).resolve().parents[1]
URL = "https://cyg0uvml94.execute-api.us-east-1.amazonaws.com"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/dashboard/deployed_asset_readback.json")
    args = parser.parse_args()
    build = json.loads((ROOT / "results/query_package_build.json").read_bytes())
    function = _session("kg-dev").client("lambda").get_function_configuration(FunctionName="kg-r72-query")
    if (base64.b64decode(function["CodeSha256"], validate=True).hex() != build["sha256"]
            or function["LastUpdateStatus"] != "Successful" or function["State"] != "Active"):
        raise ValueError("DASHBOARD_PACKAGE_NOT_DEPLOYED")
    assets = json.loads((ROOT / "data/build/query-runtime-v1/source/configs/frontend-static.json").read_bytes())
    if len(assets) != 12:
        raise ValueError("DASHBOARD_ASSET_SET_MISMATCH")
    checks = []
    for route, expected in assets.items():
        request = Request(URL + route, headers={"Cache-Control": "no-cache"})
        with urlopen(request, timeout=20) as response:
            body = response.read(512 * 1024 + 1)
            if (response.status != 200 or body != expected["body"].encode("utf-8")
                    or response.headers.get("Content-Type") != expected["content_type"]
                    or any(response.headers.get(name) != value for name, value in HEADERS.items())):
                raise ValueError("DASHBOARD_ASSET_READBACK_MISMATCH:" + route)
        checks.append({"route": route, "bytes": len(body), "sha256": sha256_bytes(body),
                       "mime": expected["content_type"], "security_headers_match": True})
    negatives = []
    for route in ("/v1/status", "/v1/queries"):
        try:
            with urlopen(URL + route, timeout=15):
                raise ValueError("UNAUTHENTICATED_ROUTE_MUST_REJECT")
        except HTTPError as exc:
            if exc.code != 401:
                raise ValueError("UNEXPECTED_UNAUTHENTICATED_STATUS") from exc
        negatives.append({"route": route, "http_status": 401})
    report = {"status": "PASS_DEPLOYED_DASHBOARD_READBACK", "checked_at": datetime.now(UTC).isoformat(),
              "package_sha256": build["sha256"], "url": URL + "/", "assets": checks,
              "negative_http_auth": negatives, "model_calls": 0,
              "scope": "Public asset identity/MIME/CSP and negative HTTP auth; not positive reviewer login or answer quality."}
    path = args.output
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "assets": len(checks), "negative_http_auth": len(negatives)}))


if __name__ == "__main__":
    main()
