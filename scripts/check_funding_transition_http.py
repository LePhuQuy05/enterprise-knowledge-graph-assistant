"""Read-only byte parity for unchanged deployed assets and negative API auth."""

import json
import zipfile
from concurrent.futures import ThreadPoolExecutor
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from app.contracts import sha256_bytes
from scripts.release_funding_transition import DIRECTORY, read, write

ORIGIN = "https://cyg0uvml94.execute-api.us-east-1.amazonaws.com"


def main():
    plan = read("release-plan.json")
    path = DIRECTORY / "query-funding-transition.zip"
    if sha256_bytes(path.read_bytes()) != plan["package_sha256"]:
        raise ValueError("HTTP_CHECK_PACKAGE_DRIFT")
    with zipfile.ZipFile(path) as archive:
        manifest = json.loads(archive.read("configs/frontend-static.json"))
    if len(manifest) != 13 or any(p != "/" and not p.startswith("/assets/") for p in manifest):
        raise ValueError("HTTP_CHECK_EXPECTED_DEPLOYED_STATIC_MANIFEST")

    def check(item):
        route, asset = item
        with urlopen(ORIGIN + route, timeout=15) as response:
            raw = response.read(4 * 1024 * 1024 + 1)
            expected = asset["body"].encode("utf-8")
            passed = (response.status == 200 and raw == expected
                and response.headers.get("Cache-Control") == "no-store"
                and response.headers.get("X-Content-Type-Options") == "nosniff")
            return {"path": route, "status": response.status, "byte_content_matches": raw == expected,
                    "sha256": sha256_bytes(raw), "passed": passed}
    with ThreadPoolExecutor(max_workers=3) as pool:
        checks = list(pool.map(check, manifest.items()))
    for route, token in (("/v1/status", None), ("/v1/queries", "Bearer synthetic.invalid.jwt")):
        request = Request(ORIGIN + route, headers={"Authorization": token} if token else {})
        try:
            with urlopen(request, timeout=15) as response:
                status = response.status
        except HTTPError as exc:
            status = exc.code
            exc.close()
        checks.append({"path": route, "status": status, "passed": status == 401})
    result = {"status": "PASS" if all(row["passed"] for row in checks) else "FAIL", "checks": checks,
              "static_assets": 13, "negative_http_auth_checks": 2, "model_calls": 0,
              "browser_visual_or_positive_oauth_validation": False}
    write("http-static-auth.json", result)
    print(json.dumps({k: result[k] for k in ("status", "static_assets", "negative_http_auth_checks", "model_calls")}))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
